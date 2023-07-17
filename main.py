# fmt: off
# pylint: disable=missing-docstring
import os
import pickle
import subprocess
import sys
from typing import Dict, List, Optional

import numpy as np

import tvm
from tvm import  tir, dlight
from tvm.target import Target
from tvm.contrib import nvcc
from tvm.script import tir as T
from hashlib import sha256

TARGET = Target("nvidia/geforce-rtx-3080")
DEVICE = tvm.cuda(0)
USE_MANUAL_CODE = False

def tvm_callback_cuda_compile(code, target):
    compute_version = "".join(
        nvcc.get_target_compute_version(Target.current(allow_none=True)).split(".")
    )
    arch = ["-gencode", f"arch=compute_{compute_version},code=sm_{compute_version}"]
    sha_hash = sha256(code.encode("utf-8")).hexdigest()
    path = f"dist/dump/model/{sha_hash}"
    os.makedirs(path, exist_ok=True)
    target_format = "ptx"
    file_code = f"{path}/my_kernel.cu"
    file_target = f"{path}/my_kernel.{target_format}"

    with open(file_code, "w", encoding="utf-8") as out_file:
        out_file.write(code)

    cmd = ["nvcc"]
    cmd += [f"--{target_format}", "-O3"]
    if isinstance(arch, list):
        cmd += arch
    elif isinstance(arch, str):
        cmd += ["-arch", arch]
    cmd += ["-lineinfo"]

    cmd += ["-o", file_target, file_code]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    (out, _) = proc.communicate()

    if proc.returncode != 0:
        msg = code
        msg += "\nCompilation error:\n"
        msg += out.decode("utf-8")
        raise RuntimeError(msg)

    with open(file_target, "rb") as f:
        data = bytearray(f.read())
        if not data:
            raise RuntimeError("Compilation error: empty result is generated")
        return data

@tvm.register_func
def tvm_callback_cuda_postproc(code, target):
    if USE_MANUAL_CODE:
        code = open("dist/dump/manual/generated.cu", "r", encoding="utf-8").read()
    return code

# fmt: off

@T.prim_func
def fused_fused_decode4_NT_matmul9(lv19: T.Buffer((T.int64(22016), T.int64(512)), "uint32"), lv20: T.Buffer((T.int64(22016), T.int64(128)), "float16"), lv1654: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"), var_NT_matmul_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(22016)), "float16")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    p_output0_intermediate = T.alloc_buffer((T.int64(22016), T.int64(4096)), "float16")
    for i, j in T.grid(T.int64(22016), T.int64(4096)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv19[v_i, v_j // T.int64(8)], lv20[v_i, v_j // T.int64(32)])
            T.writes(p_output0_intermediate[v_i, v_j])
            p_output0_intermediate[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv19[v_i, v_j // T.int64(8)], T.Cast("uint32", v_j % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv20[v_i, v_j // T.int64(32)]
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(22016), T.int64(4096)):
        with T.block("NT_matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv1654[v_i0, v_i1, v_k], p_output0_intermediate[v_i2, v_k])
            T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + lv1654[v_i0, v_i1, v_k] * p_output0_intermediate[v_i2, v_k]

@T.prim_func
def fused_fused_decode2_NT_matmul6(lv555: T.Buffer((T.int64(12288), T.int64(512)), "uint32"), lv556: T.Buffer((T.int64(12288), T.int64(128)), "float16"), lv1615: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"), var_NT_matmul_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(12288)), "float16")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    p_output0_intermediate = T.alloc_buffer((T.int64(12288), T.int64(4096)), "float16")
    for i, j in T.grid(T.int64(12288), T.int64(4096)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv555[v_i, v_j // T.int64(8)], lv556[v_i, v_j // T.int64(32)])
            T.writes(p_output0_intermediate[v_i, v_j])
            p_output0_intermediate[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv555[v_i, v_j // T.int64(8)], T.Cast("uint32", v_j % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv556[v_i, v_j // T.int64(32)]
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(12288), T.int64(4096)):
        with T.block("NT_matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv1615[v_i0, v_i1, v_k], p_output0_intermediate[v_i2, v_k])
            T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + lv1615[v_i0, v_i1, v_k] * p_output0_intermediate[v_i2, v_k]


@T.prim_func
def fused_fused_decode5_fused_NT_matmul10_add1(lv575: T.Buffer((T.int64(4096), T.int64(1376)), "uint32"), lv576: T.Buffer((T.int64(4096), T.int64(344)), "float16"), lv574: T.Buffer((T.int64(1), T.int64(1), T.int64(11008)), "float16"), lv570: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"), p_output0_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    p_output0_intermediate_1 = T.alloc_buffer((T.int64(4096), T.int64(11008)), "float16")
    var_NT_matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16")
    for i, j in T.grid(T.int64(4096), T.int64(11008)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv575[v_i, v_j // T.int64(8)], lv576[v_i, v_j // T.int64(32)])
            T.writes(p_output0_intermediate_1[v_i, v_j])
            p_output0_intermediate_1[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv575[v_i, v_j // T.int64(8)], T.Cast("uint32", v_j % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv576[v_i, v_j // T.int64(32)]
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(4096), T.int64(11008)):
        with T.block("NT_matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv574[v_i0, v_i1, v_k], p_output0_intermediate_1[v_i2, v_k])
            T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + lv574[v_i0, v_i1, v_k] * p_output0_intermediate_1[v_i2, v_k]
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(4096)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(lv570[v_ax0, v_ax1, v_ax2], var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2])
            T.writes(p_output0_intermediate[v_ax0, v_ax1, v_ax2])
            p_output0_intermediate[v_ax0, v_ax1, v_ax2] = lv570[v_ax0, v_ax1, v_ax2] + var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2]


@T.prim_func
def fused_fused_decode3_fused_NT_matmul8_add1(lv567: T.Buffer((T.int64(4096), T.int64(512)), "uint32"), lv568: T.Buffer((T.int64(4096), T.int64(128)), "float16"), lv566: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"), lv1613: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"), p_output0_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    p_output0_intermediate_1 = T.alloc_buffer((T.int64(4096), T.int64(4096)), "float16")
    var_NT_matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16")
    for i, j in T.grid(T.int64(4096), T.int64(4096)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv567[v_i, v_j // T.int64(8)], lv568[v_i, v_j // T.int64(32)])
            T.writes(p_output0_intermediate_1[v_i, v_j])
            p_output0_intermediate_1[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv567[v_i, v_j // T.int64(8)], T.Cast("uint32", v_j % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv568[v_i, v_j // T.int64(32)]
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(4096), T.int64(4096)):
        with T.block("NT_matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv566[v_i0, v_i1, v_k], p_output0_intermediate_1[v_i2, v_k])
            T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + lv566[v_i0, v_i1, v_k] * p_output0_intermediate_1[v_i2, v_k]
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(4096)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(lv1613[v_ax0, v_ax1, v_ax2], var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2])
            T.writes(p_output0_intermediate[v_ax0, v_ax1, v_ax2])
            p_output0_intermediate[v_ax0, v_ax1, v_ax2] = lv1613[v_ax0, v_ax1, v_ax2] + var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2]

# fmt: on


def prepare_args(
    func: tir.PrimFunc, var_dict: Dict[str, int], load_file: Optional[str] = None
):
    if load_file is not None:
        with open(load_file, "rb") as f:
            args = pickle.load(f)
        args = [tvm.nd.array(arg, DEVICE) for arg in args]
        return args

    args: List[np.ndarray] = []
    analyzer = tvm.arith.Analyzer()
    for param in func.params:
        buffer = func.buffer_map[param]
        shape = []
        for dim in buffer.shape:
            if isinstance(dim, tir.IntImm):
                shape.append(dim.value)
            elif isinstance(dim, tir.Var):
                assert dim.name in var_dict
                value = var_dict[dim.name]
                shape.append(value)
                analyzer.bind(dim, value)
            else:
                raise ValueError(f"Unknown shape: {buffer.shape}")
        np_array = np.random.uniform(size=shape).astype(buffer.dtype)
        tvm_array = tvm.nd.array(np_array, DEVICE)
        args.append(tvm_array)

    return args


def evaluate(func: tir.PrimFunc, args, run_only: bool = False):
    rt_mod = tvm.build(func, target=TARGET)
    rt_mod(*args)
    ret = args[-1]
    if not run_only:
        DEVICE.sync()
        time_eval = rt_mod.time_evaluator(
            rt_mod.entry_name, DEVICE, number=100, cache_flush_bytes=8192 * 1024
        )
        total_bytes = sum(arg.numpy().size * arg.numpy().itemsize for arg in args)
        DEVICE.sync()
        time = time_eval(*args).mean * 1e3
        bandwidth = total_bytes / time / (1024**2)
        print(
            f"Time (ms): {time:.6f}",
            f"Total Bytes (MB): {total_bytes / (1024**2):.6f}",
            f"Memory (GB/s): {bandwidth:.6f}",
            sep="\t",
        )
    return ret


def export_source(mod, name="gemv"):
    os.makedirs(f"dist/dump/{name}", exist_ok=True)
    lib = tvm.build(mod, target=TARGET)
    cuda_c = lib.imported_modules[0].get_source()
    ptx = nvcc.compile_cuda(cuda_c, target_format="ptx")
    with open(f"dist/dump/{name}/tir.py", "w", encoding="utf-8") as f:
        f.write(mod.script())
    with open(f"dist/dump/{name}/lowerd.py", "w", encoding="utf-8") as f:
        f.write(tvm.lower(mod, simple_mode=True).script())
    with open(f"dist/dump/{name}/generated.ptx", "wb") as f:
        f.write(ptx)
    with open(f"dist/dump/{name}/generated.cu", "w", encoding="utf-8") as f:
        f.write(cuda_c)


def apply_best(mod):
    tx_len = 32
    ty_len = 2
    vec_len = 8
    sch = tir.Schedule(mod)
    b0 = sch.get_block(name="root", func_name="main")
    b1, b2, *epilogue = sch.get_child_blocks(b0)
    sch.transform_block_layout(
        block=b1,
        index_map=lambda v_i, v_j: (
            v_i,
            v_j,
        ),
    )
    sch.transform_block_layout(
        block=b2,
        index_map=lambda v_i0, v_i1, v_i2, v_k: (
            v_i2,
            v_k,
        ),
    )
    sch.compute_inline(block=b1)

    i, j = sch.get_loops(block=b2)
    jo, j0, tx, inner = sch.split(
        j, [None, 4, tx_len, vec_len], preserve_unit_iters=True
    )
    rf = sch.rfactor(tx, factor_axis=0)
    i, jo, j0, tx, inner = sch.get_loops(rf)
    sch.reorder(i, tx, jo, j0)
    bx, ty = sch.split(i, [None, ty_len], preserve_unit_iters=True)
    sch.bind(bx, "blockIdx.x")
    sch.bind(ty, "threadIdx.y")
    sch.bind(tx, "threadIdx.x")
    sch.annotate(block_or_loop=tx, ann_key="pragma_auto_unroll_max_step", ann_val=256)
    sch.annotate(block_or_loop=tx, ann_key="pragma_unroll_explicit", ann_val=1)

    shared_A = sch.cache_read(rf, 0, "shared")
    sch.compute_at(shared_A, tx)
    fused = sch.fuse(*sch.get_loops(shared_A)[3:])
    _, _ty, _tx, _vec = sch.split(fused, factors=[None, ty_len, tx_len, vec_len])
    sch.bind(_ty, "threadIdx.y")
    sch.bind(_tx, "threadIdx.x")
    sch.vectorize(_vec)

    local_A = sch.cache_read(rf, 0, "local")
    sch.compute_at(local_A, j0)
    sch.vectorize(sch.get_loops(local_A)[-1])

    shared_S = sch.cache_read(rf, 2, "shared")
    sch.compute_at(shared_S, tx)
    fused = sch.fuse(*sch.get_loops(shared_S)[3:])
    _, _ty, _tx, _vec = sch.split(fused, factors=[None, ty_len, tx_len, 4])
    sch.bind(_ty, "threadIdx.y")
    sch.bind(_tx, "threadIdx.x")
    sch.vectorize(_vec)

    sch.set_scope(rf, 0, "local")
    sch.decompose_reduction(rf, jo)
    sch.reverse_compute_at(block=b2, loop=ty)
    sch.bind(sch.get_loops(b2)[-1], "threadIdx.x")

    if len(epilogue) > 0:
        assert len(epilogue) == 1
        (epilogue,) = epilogue
        sch.reverse_compute_at(epilogue, ty)

        sch.set_scope(b2, 0, "local")

    return sch.mod


def apply_update(mod):
    tx_len = 32
    ty_len = 2
    vec_len = 8
    sch = tir.Schedule(mod)
    b0 = sch.get_block(name="root", func_name="main")
    b1, b2, *epilogue = sch.get_child_blocks(b0)
    sch.transform_block_layout(
        block=b1,
        index_map=lambda v_i, v_j: (
            v_i,
            v_j,
        ),
    )
    sch.transform_block_layout(
        block=b2,
        index_map=lambda v_i0, v_i1, v_i2, v_k: (
            v_i2,
            v_k,
        ),
    )
    sch.compute_inline(block=b1)

    i, j = sch.get_loops(block=b2)
    jo, j0, tx, inner = sch.split(
        j, [None, 4, tx_len, vec_len], preserve_unit_iters=True
    )
    rf = sch.rfactor(tx, factor_axis=0)
    i, jo, j0, tx, inner = sch.get_loops(rf)
    sch.reorder(i, tx, jo, j0)
    bx, ty = sch.split(i, [None, ty_len], preserve_unit_iters=True)
    sch.bind(bx, "blockIdx.x")
    sch.bind(ty, "threadIdx.y")
    sch.bind(tx, "threadIdx.x")
    sch.annotate(block_or_loop=tx, ann_key="pragma_auto_unroll_max_step", ann_val=256)
    sch.annotate(block_or_loop=tx, ann_key="pragma_unroll_explicit", ann_val=1)

    shared_A = sch.cache_read(rf, 0, "shared")
    sch.compute_at(shared_A, jo)
    fused = sch.fuse(*sch.get_loops(shared_A)[4:])
    _, _ty, _tx, _vec = sch.split(fused, factors=[None, ty_len, tx_len, vec_len])
    sch.bind(_ty, "threadIdx.y")
    sch.bind(_tx, "threadIdx.x")
    sch.vectorize(_vec)

    local_A = sch.cache_read(rf, 0, "local")
    sch.compute_at(local_A, j0)
    sch.vectorize(sch.get_loops(local_A)[-1])

    shared_S = sch.cache_read(rf, 2, "shared")
    sch.compute_at(shared_S, tx)
    fused = sch.fuse(*sch.get_loops(shared_S)[3:])
    _, _ty, _tx, _vec = sch.split(fused, factors=[None, ty_len, tx_len, 4])
    sch.bind(_ty, "threadIdx.y")
    sch.bind(_tx, "threadIdx.x")
    sch.vectorize(_vec)

    sch.set_scope(rf, 0, "local")
    sch.decompose_reduction(rf, jo)
    sch.reverse_compute_at(block=b2, loop=ty)
    sch.bind(sch.get_loops(b2)[-1], "threadIdx.x")

    if len(epilogue) > 0:
        assert len(epilogue) == 1
        (epilogue,) = epilogue
        sch.reverse_compute_at(epilogue, ty)

        sch.set_scope(b2, 0, "local")

    return sch.mod


def main():
    funcs = {
        "fused_fused_decode4_NT_matmul9": fused_fused_decode4_NT_matmul9,
        "fused_fused_decode2_NT_matmul6": fused_fused_decode2_NT_matmul6,
        # "fused_fused_decode5_fused_NT_matmul10_add1": fused_fused_decode5_fused_NT_matmul10_add1,
        "fused_fused_decode3_fused_NT_matmul8_add1": fused_fused_decode3_fused_NT_matmul8_add1,
    }
    arg_list = sys.argv[1:]
    run_only = True if "--run_only" in arg_list else False
    if run_only:
        tvm.register_func(
            "tvm_callback_cuda_compile", tvm_callback_cuda_compile, override=True
        )
    for name, func in funcs.items():
        print("func_name:", name)
        best_mod = apply_best(func)["main"]
        dlight_mod = dlight.gpu.DecodeGEMV().apply(func, TARGET, False).mod["main"]
        update_mod = apply_update(func)["main"]
        # if name == "fused_fused_decode5_fused_NT_matmul10_add1":
        #     best_mod = manual
        data_path = f"dist/dump/data/{name}/args.pkl"
        print("dlight:", end="\t")
        ret_dlight = evaluate(
            dlight_mod,
            prepare_args(dlight_mod, {"n": 256}, data_path),
            run_only,
        )
        print("best:", end="\t")
        ret_best = evaluate(
            best_mod,
            prepare_args(best_mod, {"n": 256}, data_path),
            run_only,
        )
        export_source(best_mod, name)
        # np.testing.assert_allclose(
        #     ret_dlight.numpy(), ret_best.numpy(), atol=1e-2, rtol=1e-3
        # )
        print("update:", end="\t")
        evaluate(
            update_mod,
            prepare_args(update_mod, {"n": 256}, data_path),
            run_only,
        )
        # export_source(update_mod["main"])


if __name__ == "__main__":
    main()
