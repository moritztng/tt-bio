"""Elementwise rates at the BindCraft 2 pair shape, with the operands in L1 and in DRAM (bc4-bet-elem step 1).

A standalone ttnn.exp on a DRAM tensor is bandwidth-bound, so it prices DRAM, not the SFPU (triatt_sdpa.py's
ABLATE note). With input and output interleaved over the L1 of every core there is no DRAM in the op, so
time(op, L1) - time(clone, L1) is the math the op costs over moving its tiles once: the part a fused kernel
that already holds the tile in L1 still pays. Per op it reports ms for one [1, N, N, C] pass and ns per tile.

Serial check: a compute-bound matmul with its output in L1, plain and with a fused sigmoid epilogue. If the
math RISC runs the SFPU after the FPU (serial), fused - plain ~ sigmoid(L1) - clone(L1) on the output's tiles;
if they overlap, fused ~ plain.

    TT_VISIBLE_DEVICES=<c> timeout 900 python3 perf/bc4_elem/prims.py OUT.json [N]
"""
import json
import subprocess
import sys
import time

import torch

from tt_bio.main import ensure_p300_mesh_descriptor

ensure_p300_mesh_descriptor()
import ttnn  # noqa: E402

import tt_bio.tenstorrent as T  # noqa: E402

dev = T.get_device()
REPS = 30
N = int(sys.argv[2]) if len(sys.argv) > 2 else 224
C = 128
L1, DRAM = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG


def up(shape, mem, scale=1.0):
    return ttnn.from_torch((torch.randn(*shape) * scale).bfloat16(), layout=ttnn.TILE_LAYOUT,
                           dtype=ttnn.bfloat16, device=dev, memory_config=mem)


def timed(fn):
    for _ in range(3):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(dev)
    t = time.perf_counter()
    for _ in range(REPS):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(dev)
    return (time.perf_counter() - t) / REPS * 1e3


def hifi(approx=False):
    return ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                                  math_approx_mode=approx, fp32_dest_acc_en=True)


def prims(mem):
    x = up((1, N, N, C), mem, 2.0)
    y = up((1, N, N, C), mem)
    w, b = up((C,), mem), up((C,), mem)
    ops = {
        "clone": lambda: ttnn.clone(x, memory_config=mem),
        "exp_fast": lambda: ttnn.exp(x, fast_and_approximate_mode=True, memory_config=mem),
        "exp_exact": lambda: ttnn.exp(x, fast_and_approximate_mode=False, memory_config=mem),
        "sigmoid": lambda: ttnn.sigmoid(x, memory_config=mem),
        "sigmoid_approx": lambda: ttnn.sigmoid(x, vector_mode=4, fast_and_approximate_mode=True,
                                               memory_config=mem),
        "reciprocal": lambda: ttnn.reciprocal(x, memory_config=mem),
        "mul": lambda: ttnn.multiply(x, y, memory_config=mem),
        "add": lambda: ttnn.add(x, y, memory_config=mem),
        "layer_norm": lambda: ttnn.layer_norm(x, weight=w, bias=b, epsilon=1e-5, memory_config=mem,
                                              compute_kernel_config=hifi()),
    }
    out = {}
    for name, fn in ops.items():
        try:
            out[name] = timed(fn)
        except Exception as e:  # an op this wheel lacks (an approx variant) is a missing row, not a failure
            out[name] = f"error: {type(e).__name__}: {str(e)[:160]}"
    for t in (x, y, w, b):
        ttnn.deallocate(t)
    return out


def serial():
    # [M, K] x [K, K] with K = 1024: 2*M*K*K FLOP against ~2*(M*K + K*K + M*K) bytes, compute-bound
    M, K = 4096, 1024
    a, wt = up((1, 1, M, K), L1, 0.03), up((1, 1, K, K), L1, 0.03)
    o = {"shape": [M, K, K], "tiles_out": M * K // 1024}
    o["plain"] = timed(lambda: ttnn.linear(a, wt, memory_config=L1, compute_kernel_config=hifi()))
    o["fused_sigmoid"] = timed(lambda: ttnn.linear(a, wt, activation="sigmoid", memory_config=L1,
                                                   compute_kernel_config=hifi()))
    y = ttnn.linear(a, wt, memory_config=L1, compute_kernel_config=hifi())
    o["sigmoid_L1"] = timed(lambda: ttnn.sigmoid(y, memory_config=L1))
    o["clone_L1"] = timed(lambda: ttnn.clone(y, memory_config=L1))
    o["tflops_plain"] = 2 * M * K * K / o["plain"] / 1e9
    for t in (a, wt, y):
        ttnn.deallocate(t)
    return o


if __name__ == "__main__":
    res = {"n": N, "c": C, "reps": REPS, "tiles": N * N * C // 1024, "utc": time.strftime("%FT%TZ", time.gmtime()),
           "sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}
    res["L1"] = prims(L1)
    res["DRAM"] = prims(DRAM)
    try:
        res["serial"] = serial()
    except Exception as e:
        res["serial"] = f"error: {type(e).__name__}: {str(e)[:300]}"
    clone = res["L1"]["clone"]
    res["L1_net_ns_per_tile"] = {k: (v - clone) * 1e6 / res["tiles"] for k, v in res["L1"].items()
                                 if isinstance(v, float)}
    print(json.dumps(res, indent=1))
    with open(sys.argv[1], "w") as f:
        json.dump(res, f, indent=1)
