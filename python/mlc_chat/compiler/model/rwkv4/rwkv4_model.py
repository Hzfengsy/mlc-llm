"""
Implementation for Mistral architecture.
"""
import dataclasses
from typing import Any, Dict, Optional

from tvm import relax as rx
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op
from tvm.script import tir as T


from ....support import logging
from ....support.config import ConfigBase

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class RWKVConfig(ConfigBase):  # pylint: disable=too-many-instance-attributes
    """Configuration of the Mistral model."""

    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    vocab_size: int
    tensor_parallel_shards: int = 1
    rescale_every: int = 0
    layer_norm_epsilon: float = 1e-5
    context_window_size: int = -1  # RWKV does not have context window limitation.
    prefill_chunk_size: int = -1  # RWKV does not have prefill chunk size.
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        if self.tensor_parallel_shards != 1:
            raise ValueError("Only support single device at this moment.")


# pylint: disable=invalid-name,missing-docstring
def create_wkv_func(hidden_size: int, dtype: str, out_dtype: str):
    @T.prim_func
    def wkv_func(
        k: T.handle,
        v: T.handle,
        time_decay: T.handle,
        time_first: T.handle,
        saved_a: T.handle,
        saved_b: T.handle,
        saved_p: T.handle,
        wkv: T.handle,
        out_a: T.handle,
        out_b: T.handle,
        out_p: T.handle,
    ):
        T.func_attr({"op_pattern": 8, "tir.noalias": True, "tir.is_scheduled": 1})
        context_length = T.int64()
        K = T.match_buffer(k, (context_length, hidden_size), dtype=dtype)
        V = T.match_buffer(v, (context_length, hidden_size), dtype=dtype)
        TimeDecay = T.match_buffer(time_decay, (hidden_size,), dtype=dtype)
        TimeFirst = T.match_buffer(time_first, (hidden_size,), dtype=dtype)
        SavedA = T.match_buffer(saved_a, (1, hidden_size), dtype=dtype)
        SavedB = T.match_buffer(saved_b, (1, hidden_size), dtype=dtype)
        SavedP = T.match_buffer(saved_p, (1, hidden_size), dtype=dtype)
        WKV = T.match_buffer(wkv, (context_length, hidden_size), dtype=out_dtype)
        OutA = T.match_buffer(out_a, (1, hidden_size), dtype=dtype)
        OutB = T.match_buffer(out_b, (1, hidden_size), dtype=dtype)
        OutP = T.match_buffer(out_p, (1, hidden_size), dtype=dtype)

        P = T.alloc_buffer((hidden_size,), dtype=dtype, scope="local")
        E1 = T.alloc_buffer((hidden_size,), dtype=dtype, scope="local")
        E2 = T.alloc_buffer((hidden_size,), dtype=dtype, scope="local")
        A_local = T.alloc_buffer((hidden_size,), dtype=dtype, scope="local")
        B_local = T.alloc_buffer((hidden_size,), dtype=dtype, scope="local")
        P_local = T.alloc_buffer((hidden_size,), dtype=dtype, scope="local")

        for bx in T.thread_binding(hidden_size // 32, thread="blockIdx.x"):
            for tx in T.thread_binding(32, thread="threadIdx.x"):
                with T.block("init"):
                    vi = T.axis.S(hidden_size, bx * 32 + tx)
                    A_local[vi] = SavedA[0, vi]
                    B_local[vi] = SavedB[0, vi]
                    P_local[vi] = SavedP[0, vi]
                for j in range(context_length):
                    with T.block("main"):
                        vi = T.axis.S(hidden_size, bx * 32 + tx)
                        vj = T.axis.opaque(context_length, j)
                        P[vi] = T.max(P_local[vi], K[vj, vi] + TimeFirst[vi])
                        E1[vi] = T.exp(P_local[vi] - P[vi])
                        E2[vi] = T.exp(K[vj, vi] + TimeFirst[vi] - P[vi])
                        WKV[vj, vi] = T.cast(
                            (E1[vi] * A_local[vi] + E2[vi] * V[vj, vi])
                            / (E1[vi] * B_local[vi] + E2[vi]),
                            out_dtype,
                        )

                        P[vi] = T.max(P_local[vi] + TimeDecay[vi], K[vj, vi])
                        E1[vi] = T.exp(P_local[vi] + TimeDecay[vi] - P[vi])
                        E2[vi] = T.exp(K[vj, vi] - P[vi])
                        A_local[vi] = E1[vi] * A_local[vi] + E2[vi] * V[vj, vi]
                        B_local[vi] = E1[vi] * B_local[vi] + E2[vi]
                        P_local[vi] = P[vi]

                with T.block("write_back"):
                    vi = T.axis.S(hidden_size, bx * 32 + tx)
                    OutA[0, vi] = A_local[vi]
                    OutB[0, vi] = B_local[vi]
                    OutP[0, vi] = P_local[vi]

    return wkv_func


class RWKV_Embedding(nn.Module):
    """RWKV Embedding."""

    def __init__(self, config: RWKVConfig):
        super().__init__()
        self.weight = nn.Parameter((config.vocab_size, config.hidden_size))

    def forward(self, x: Tensor):
        x = op.reshape(x, (-1,))
        return op.take(self.weight, x, axis=0)


class RWKV_FNN(nn.Module):
    def __init__(self, config: RWKVConfig):
        super().__init__()
        self.time_mix_key = nn.Parameter((config.hidden_size,))
        self.time_mix_receptance = nn.Parameter((config.hidden_size,))
        self.key = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.receptance = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.value = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.state = nn.StateCache(rx.op.zeros((1, config.hidden_size), dtype="float32"))

    def forward(self, x: Tensor):
        state_x = self.state.view()
        state_x = op.concat([state_x, x[:-1]], dim=0)
        xk = x * self.time_mix_key + state_x * (1.0 - self.time_mix_key)
        xr = x * self.time_mix_receptance + state_x * (1.0 - self.time_mix_receptance)
        self.state.update(x[-1])
        r = op.sigmoid(self.receptance(xr))
        xv = op.square(op.relu(self.key(xk)))
        return r * self.value(xv)


class RWKV_Attention(nn.Module):
    """Attention layer for RWKV."""

    def __init__(self, config: RWKVConfig):
        super().__init__()
        self.time_decay = nn.Parameter((config.hidden_size,))
        self.time_first = nn.Parameter((config.hidden_size,))
        self.time_mix_key = nn.Parameter((config.hidden_size,))
        self.time_mix_value = nn.Parameter((config.hidden_size,))
        self.time_mix_receptance = nn.Parameter((config.hidden_size,))
        self.key = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.value = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.receptance = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.output = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.state_A = nn.StateCache(rx.op.zeros((1, config.hidden_size), dtype="float32"))
        self.state_B = nn.StateCache(rx.op.zeros((1, config.hidden_size), dtype="float32"))
        self.state_P = nn.StateCache(
            rx.op.full((1, config.hidden_size), rx.const(-1e30), dtype="float32")
        )
        self.state_X = nn.StateCache(rx.op.zeros((1, config.hidden_size), dtype="float32"))
        self.dtype = "float32"

    def forward(self, x: Tensor):
        state_A = self.state_A.view()
        state_B = self.state_B.view()
        state_P = self.state_P.view()
        state_X = self.state_X.view()
        seq_len, hidden_size = x.shape
        x = op.concat([state_X, x[:-1]], dim=0)
        xk = x * self.time_mix_key + state_X * (1.0 - self.time_mix_key)
        xv = x * self.time_mix_value + state_X * (1.0 - self.time_mix_value)
        xr = x * self.time_mix_receptance + state_X * (1.0 - self.time_mix_receptance)

        r = op.sigmoid(self.receptance(xr))
        k = self.key(xk)
        v = self.value(xv)
        # call wkv
        wkv, state_A, state_B, state_P = op.call_tir_op(
            create_wkv_func(hidden_size, "float32", self.dtype),
            [k, v, self.time_decay, self.time_first, state_A, state_B, state_P],
            [
                nn.Tensor.placeholder((seq_len, hidden_size), self.dtype),
                nn.Tensor.placeholder((1, hidden_size), "float32"),
                nn.Tensor.placeholder((1, hidden_size), "float32"),
                nn.Tensor.placeholder((1, hidden_size), "float32"),
            ],
            name_hint="wkv",
        )

        self.state_X.update(x[-1])
        self.state_A.update(state_A)
        self.state_B.update(state_B)
        self.state_P.update(state_P)
        return self.output(r * wkv)

    def to(self, dtype: Optional[str] = None):
        # RWKV uses special dtype, so we need to convert it.
        if dtype is not None:
            self.dtype = dtype

        self.time_mix_key.to(dtype)
        self.time_mix_value.to(dtype)
        self.time_mix_receptance.to(dtype)
        self.key.to(dtype)
        self.value.to(dtype)
        self.receptance.to(dtype)
        self.output.to(dtype)
        self.state_X.to(dtype)


class RWKVLayer(nn.Module):
    def __init__(self, config: RWKVConfig, index: int):
        super().__init__()
        if index == 0:
            self.pre_ln = nn.LayerNorm(
                config.hidden_size,
                eps=config.layer_norm_epsilon,
            )
        self.ln1 = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_epsilon,
        )
        self.ln2 = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_epsilon,
        )
        self.attention = RWKV_Attention(config)
        self.feed_forward = RWKV_FNN(config)
        self.index = index
        self.config = config

    def forward(self, x: Tensor):
        if self.index == 0:
            x = self.pre_ln(x)
        x = self.attention(self.ln1(x)) + x
        x = self.feed_forward(self.ln2(x)) + x
        if self.config.rescale_every > 0 and (self.index + 1) % self.config.rescale_every == 0:
            x = x / 2.0
        return x


class RWKVModel(nn.Module):
    """Exact same as LlamaModel."""

    def __init__(self, config: RWKVConfig):
        super().__init__()
        self.embedding = RWKV_Embedding(config)
        self.blocks = nn.ModuleList([RWKVLayer(config, i) for i in range(config.num_hidden_layers)])
        self.ln_out = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_epsilon,
        )

    def forward(self, input_ids: Tensor):
        """Forward pass of the model, passing through all decoder layers."""
        hidden_states = self.embedding(input_ids)
        for block in self.blocks:
            hidden_states = block(hidden_states)
        return self.ln_out(hidden_states[-1])


class RWKVForCasualLM(nn.Module):
    """Same as LlamaForCausalLM, except for the use of sliding window attention."""

    def __init__(self, config: RWKVConfig):
        self.model = RWKVModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.vocab_size = config.vocab_size
        self.dtype = "float32"

    def to(self, dtype: Optional[str] = None):
        super().to(dtype=dtype)
        if dtype is not None:
            self.dtype = dtype

    def forward(self, inputs: Tensor):
        """Forward pass."""
        hidden_states = self.model(inputs)
        logits = self.lm_head(hidden_states)
        logits = op.reshape(logits, (1, 1, self.vocab_size))
        return logits

    def prefill(self, inputs: Tensor):
        """Prefilling the prompt."""
        return self.forward(inputs)

    def decode(self, inputs: Tensor):
        """Decoding step."""
        return self.forward(inputs)

    def softmax_with_temperature(self, logits: Tensor, temperature: Tensor):
        """Softmax."""
        return op.softmax(logits / temperature, axis=-1)

    def get_default_spec(self):
        """Needed for ``export_tvm()``."""
        batch_size = 1
        mod_spec = {
            "prefill": {
                "inputs": nn.spec.Tensor([batch_size, "seq_len"], "int32"),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "packed",
                },
            },
            "decode": {
                "inputs": nn.spec.Tensor([batch_size, 1], "int32"),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "packed",
                },
            },
            "softmax_with_temperature": {
                "logits": nn.spec.Tensor([1, 1, "vocab_size"], "float32"),
                "temperature": nn.spec.Tensor([], "float32"),
                "$": {
                    "param_mode": "none",
                    "effect_mode": "none",
                },
            },
        }
        return nn.spec.ModuleSpec.from_raw(mod_spec, self)
