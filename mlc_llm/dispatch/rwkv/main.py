import tvm
from tvm.script import tir as T


def wkv_func_before(
    hidden_size: int,
    dtype: str = "float32",
    out_dtype: str = "float16",
):
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
        T.func_attr({"op_pattern": 8, "tir.noalias": True})
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

        for i in range(hidden_size):
            with T.block("init"):
                vi = T.axis.S(hidden_size, i)
                A_local[vi] = SavedA[0, vi]
                B_local[vi] = SavedB[0, vi]
                P_local[vi] = SavedP[0, vi]
            for j in range(context_length):
                with T.block("main"):
                    vi = T.axis.S(hidden_size, i)
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
                vi = T.axis.S(hidden_size, i)
                OutA[0, vi] = A_local[vi]
                OutB[0, vi] = B_local[vi]
                OutP[0, vi] = P_local[vi]

    return wkv_func


def wkv_func_after(
    hidden_size: int,
    dtype: str = "float32",
    out_dtype: str = "float16",
):
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

def fused_multiply_subtract_multiply_add_before(hidden_size: int):
    # fmt: off
    @T.prim_func
    def fused_multiply_subtract_multiply_add(p_lv3: T.handle, att_0_time_mix_k: T.Buffer((T.int64(hidden_size),), "float16"), param_0: T.Buffer((T.int64(hidden_size),), "float16"), p_lv13: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        lv3 = T.match_buffer(p_lv3, (n, T.int64(hidden_size)), "float16")
        lv13 = T.match_buffer(p_lv13, (n, T.int64(hidden_size)), "float16")
        var_T_add_intermediate = T.match_buffer(p_output0, (n, T.int64(hidden_size)), "float16")
        var_T_multiply_intermediate = T.alloc_buffer((n, T.int64(hidden_size)), "float16")
        var_T_subtract_intermediate = T.alloc_buffer((T.int64(hidden_size),), "float16")
        var_T_multiply_intermediate_1 = T.alloc_buffer((n, T.int64(hidden_size)), "float16")
        for ax0, ax1 in T.grid(n, T.int64(hidden_size)):
            with T.block("T_multiply"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(lv3[v_ax0, v_ax1], att_0_time_mix_k[v_ax1])
                T.writes(var_T_multiply_intermediate[v_ax0, v_ax1])
                var_T_multiply_intermediate[v_ax0, v_ax1] = lv3[v_ax0, v_ax1] * att_0_time_mix_k[v_ax1]
        for ax0 in range(T.int64(hidden_size)):
            with T.block("T_subtract"):
                v_ax0 = T.axis.spatial(T.int64(hidden_size), ax0)
                T.reads(param_0[v_ax0], att_0_time_mix_k[v_ax0])
                T.writes(var_T_subtract_intermediate[v_ax0])
                var_T_subtract_intermediate[v_ax0] = param_0[v_ax0] - att_0_time_mix_k[v_ax0]
        for ax0, ax1 in T.grid(n, T.int64(hidden_size)):
            with T.block("T_multiply_1"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(lv13[v_ax0, v_ax1], var_T_subtract_intermediate[v_ax1])
                T.writes(var_T_multiply_intermediate_1[v_ax0, v_ax1])
                var_T_multiply_intermediate_1[v_ax0, v_ax1] = lv13[v_ax0, v_ax1] * var_T_subtract_intermediate[v_ax1]
        for ax0, ax1 in T.grid(n, T.int64(hidden_size)):
            with T.block("T_add"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(var_T_multiply_intermediate[v_ax0, v_ax1], var_T_multiply_intermediate_1[v_ax0, v_ax1])
                T.writes(var_T_add_intermediate[v_ax0, v_ax1])
                var_T_add_intermediate[v_ax0, v_ax1] = var_T_multiply_intermediate[v_ax0, v_ax1] + var_T_multiply_intermediate_1[v_ax0, v_ax1]

    # fmt: on
    return fused_multiply_subtract_multiply_add


################################################


def get_dict_key(func):
    return tvm.ir.structural_hash(func), func


tir_dispatch_dict = {}
for hidden_size in [2048, 2560, 4096]:
    tir_dispatch_dict.update(
        {
            get_dict_key(wkv_func_before(hidden_size)): wkv_func_after(hidden_size),
        }
    )


def lookup_func(func):
    for (hash_value, func_before), f_after in tir_dispatch_dict.items():
        if tvm.ir.structural_hash(func) == hash_value and tvm.ir.structural_equal(
            func, func_before
        ):
            return f_after
    return None
