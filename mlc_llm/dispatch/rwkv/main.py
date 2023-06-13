import tvm
from tvm import tir
from tvm.script import tir as T


def wkv(
    hidden_size: int,
    dtype: str = "float32",
    out_dtype: str = "float16",
):
    assert hidden_size % 32 == 0

    @T.prim_func
    def before(
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

    @T.prim_func
    def after(
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

    return before, after


################################################
def decode_matmul_NT(hidden_size: int):
    # fmt: off
    @T.prim_func
    def fused_decode1_fused_NT_matmul3_add1(lv1210: T.Buffer((T.int64(256), T.int64(hidden_size)), "uint32"), lv1211: T.Buffer((T.int64(64), T.int64(hidden_size)), "float16"), p_lv270: T.handle, p_lv2: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        lv270 = T.match_buffer(p_lv270, (n, T.int64(hidden_size)), "float16")
        lv2 = T.match_buffer(p_lv2, (n, T.int64(hidden_size)), "float16")
        p_output0_intermediate = T.match_buffer(p_output0, (n, T.int64(hidden_size)), "float16")
        var_decode_intermediate = T.alloc_buffer((T.int64(hidden_size), T.int64(hidden_size)), "float16")
        var_NT_matmul_intermediate = T.alloc_buffer((n, T.int64(hidden_size)), "float16")
        for i, j in T.grid(T.int64(hidden_size), T.int64(hidden_size)):
            with T.block("decode"):
                v_i, v_j = T.axis.remap("SS", [i, j])
                T.reads(lv1210[v_i // T.int64(8), v_j], lv1211[v_i // T.int64(32), v_j])
                T.writes(var_decode_intermediate[v_i, v_j])
                var_decode_intermediate[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv1210[v_i // T.int64(8), v_j], T.Cast("uint32", v_i % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv1211[v_i // T.int64(32), v_j]
        for i0, i1, k in T.grid(n, T.int64(hidden_size), T.int64(hidden_size)):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
                T.reads(lv270[v_i0, v_k], var_decode_intermediate[v_i1, v_k])
                T.writes(var_NT_matmul_intermediate[v_i0, v_i1])
                with T.init():
                    var_NT_matmul_intermediate[v_i0, v_i1] = T.float16(0)
                var_NT_matmul_intermediate[v_i0, v_i1] = var_NT_matmul_intermediate[v_i0, v_i1] + lv270[v_i0, v_k] * var_decode_intermediate[v_i1, v_k]
        for ax0, ax1 in T.grid(n, T.int64(hidden_size)):
            with T.block("T_add"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(lv2[v_ax0, v_ax1], var_NT_matmul_intermediate[v_ax0, v_ax1])
                T.writes(p_output0_intermediate[v_ax0, v_ax1])
                p_output0_intermediate[v_ax0, v_ax1] = lv2[v_ax0, v_ax1] + var_NT_matmul_intermediate[v_ax0, v_ax1]
    # fmt: on


################################################


def last_matmul(hidden_size: int):
    @T.prim_func
    def before(
        A: T.Buffer((T.int64(1), T.int64(hidden_size)), "float16"),
        B: T.Buffer((T.int64(hidden_size), T.int64(50277)), "float16"),
        matmul: T.Buffer((T.int64(1), T.int64(50277)), "float16"),
    ):
        T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
        for i0, i1, k in T.grid(T.int64(1), T.int64(50277), T.int64(hidden_size)):
            with T.block("matmul"):
                v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
                T.reads(A[v_i0, v_k], B[v_k, v_i1])
                T.writes(matmul[v_i0, v_i1])
                with T.init():
                    matmul[v_i0, v_i1] = T.float16(0)
                matmul[v_i0, v_i1] = matmul[v_i0, v_i1] + A[v_i0, v_k] * B[v_k, v_i1]

    sch = tir.Schedule(before)
    b0 = sch.get_block(name="matmul")
    sch.pad_einsum(b0, [1, 64, 64])
    b1 = sch.get_producers(b0)[-1]
    b2 = sch.get_consumers(b0)[-1]
    i, j, k = sch.get_loops(b0)
    j0, j1, j2, j3, j4 = sch.split(j, [None, 1, 64, 1, 2])
    k0, k1, k2 = sch.split(k, [None, 4, 4])
    sch.reorder(i, j0, j1, j2, k0, k1, j3, k2, j4)
    bx = sch.fuse(i, j0)
    vtx, tx = j1, j2
    sch.bind(bx, thread_axis="blockIdx.x")
    sch.bind(vtx, thread_axis="vthread.x")
    sch.bind(tx, thread_axis="threadIdx.x")
    C_local = sch.cache_write(b0, 0, "local")
    sch.reverse_compute_at(C_local, tx)

    def schedule_coop_fetch(index: int, vector_len: int):
        local_block = sch.cache_read(b0, index, "shared")
        sch.compute_at(local_block, k0, preserve_unit_loops=True)
        t = sch.fuse(*sch.get_loops(local_block)[-2:])
        t0, _, t2 = sch.split(t, [64, None, vector_len])
        sch.bind(t0, thread_axis="threadIdx.x")
        if vector_len > 1:
            sch.vectorize(t2)
        offset = 8
        sch.storage_align(local_block, 0, axis=-2, factor=32, offset=offset)
        return local_block

    schedule_coop_fetch(0, 1)
    schedule_coop_fetch(1, 8)
    sch.decompose_reduction(b0, k0)
    sch.compute_inline(b1)
    sch.reverse_compute_inline(b2)

    return before, sch.mod["main"]


################################################


def update_dict(func):
    updated = {}
    for hidden_size in [2048, 2560, 4096]:
        before, after = func(hidden_size)
        updated[(tvm.ir.structural_hash(before), before)] = after
    return updated


def lookup_func(func):
    tir_dispatch_dict = {}
    tir_dispatch_dict.update(update_dict(wkv))
    tir_dispatch_dict.update(update_dict(last_matmul))

    for (hash_value, func_before), f_after in tir_dispatch_dict.items():
        if tvm.ir.structural_hash(func) == hash_value and tvm.ir.structural_equal(
            func, func_before
        ):
            return f_after
    return None
