# pylint: disable=missing-docstring
import os
import subprocess
import sys
from typing import Dict, List

import numpy as np

import tvm
from tvm import tir, te
from tvm.target import Target
from tvm.contrib import nvcc
from hashlib import sha256

TARGET = Target("nvidia/geforce-rtx-3080")
DEVICE = tvm.cuda(0)

# TARGET = Target("vulkan -supports_float16=1 -supports_16bit_buffer=1")
# DEVICE = tvm.vulkan(0)
# USE_MANUAL_CODE = False


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


def create_func(n: int, k: int):
    _k = te.reduce_axis((0, k), name="k")
    A = te.placeholder((1, 1, k), name="A", dtype="float16")
    B = te.placeholder((n, k), name="B", dtype="float16")
    C = te.compute(
        (1, 1, n), lambda b, i, j: te.sum(A[b, i, _k] * B[j, _k], axis=_k), name="C"
    )
    return te.create_prim_func([A, B, C])


def prepare_args(func: tir.PrimFunc, var_dict: Dict[str, int]):
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
        time_eval = rt_mod.time_evaluator(rt_mod.entry_name, DEVICE, number=100, cache_flush_bytes=8192*1024)
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
    vec_len = 4
    sch = tir.Schedule(mod)
    b0 = sch.get_block(name="root", func_name="main")
    (main_block,) = sch.get_child_blocks(b0)
    sch.transform_block_layout(
        block=main_block,
        index_map=lambda v_i0, v_i1, v_i2, v_k: (
            v_i2,
            v_k,
        ),
    )
    i, j = sch.get_loops(block=main_block)
    jo, j0, tx, inner = sch.split(j, [None, 4, tx_len, 8], preserve_unit_iters=True)
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
    _, vec = sch.split(sch.get_loops(local_A)[-1], [None, vec_len])
    sch.vectorize(vec)

    local_B0 = sch.cache_read(rf, 1, "local")
    sch.compute_at(local_B0, j0)
    _, vec = sch.split(sch.get_loops(local_B0)[-1], [None, vec_len])
    sch.vectorize(vec)

    sch.set_scope(rf, 0, "local")
    sch.decompose_reduction(rf, jo)
    sch.reverse_compute_at(block=main_block, loop=ty)
    sch.bind(sch.get_loops(main_block)[-1], "threadIdx.x")

    return tvm.tir.transform.RemoveWeightLayoutRewriteBlock()(sch.mod)

def main():
    func = create_func(4096, 4096)
    best_mod = apply_best(func)
    args_list = sys.argv[1:]
    run_only = True if "--run_only" in args_list else False
    if run_only:
        tvm.register_func(
            "tvm_callback_cuda_compile", tvm_callback_cuda_compile, override=True
        )
    print("best:")
    evaluate(best_mod["main"], prepare_args(best_mod["main"], {"n": 256}), run_only)
    # export_source(best_mod["main"], "fp16")


if __name__ == "__main__":
    main()
