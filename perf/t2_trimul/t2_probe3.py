#!/usr/bin/env python3
"""T2 probe 3 — the gate's SFPU cost, and a measured core count for the triangle matmul.

S  The per-chunk gate is multiply_(p, g, activations=[SIGMOID]) on an L1 chunk. A plain multiply
   on the same bytes is 2.2x faster. Hypothesis: the extra time is SFPU ISSUE RATE, i.e. it scales
   with the instruction count of the unary applied to operand b, not with bytes.
   Falsifier: a cheaper unary (RELU) costs the same as SIGMOID.

C  The triangle contraction's program config asks for per_core_M = per_core_N = 1 on an 11x10
   grid, i.e. 100 of 110 cores with one output tile each and the batch of 32 run as a serial loop
   inside the kernel. Measured by shrinking the grid: if 100 cores really are engaged, time is
   flat while grid_cores >= 100 and rises once it drops below.
"""
import argparse
import json
import statistics as st
import time

import torch
import ttnn

from tt_bio.tenstorrent import get_device, COMPUTE_GRID_MAIN

L1 = ttnn.L1_MEMORY_CONFIG


def timed(dev, fn, warm=4, pipe=5, reps=7):
    for _ in range(warm):
        r = fn()
        del r
    ttnn.synchronize_device(dev)
    out = []
    for _ in range(reps):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        keep = [fn() for _ in range(pipe)]
        ttnn.synchronize_device(dev)
        out.append((time.perf_counter() - t0) / pipe)
        del keep
    return st.median(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = get_device()
    gx, gy = COMPUTE_GRID_MAIN
    res = {"grid": f"{gx}x{gy}"}
    N, C = 320, 32
    nb = N * N * C * 2
    nelem = N * N * C

    print("=== S gate unary cost, [1,320,320,32] L1 ===", flush=True)
    a = ttnn.from_torch(torch.randn(1, N, N, C), layout=ttnn.TILE_LAYOUT, device=dev,
                        dtype=ttnn.bfloat16, memory_config=L1)
    b = ttnn.from_torch(torch.randn(1, N, N, C), layout=ttnn.TILE_LAYOUT, device=dev,
                        dtype=ttnn.bfloat16, memory_config=L1)
    S = {}
    base = None
    for lbl, act in (("none", None), ("RELU", ttnn.UnaryOpType.RELU),
                     ("SIGMOID", ttnn.UnaryOpType.SIGMOID), ("EXP", ttnn.UnaryOpType.EXP),
                     ("GELU", ttnn.UnaryOpType.GELU)):
        try:
            kw = {} if act is None else {"input_tensor_b_activations": [act]}
            s = timed(dev, lambda: ttnn.multiply(a, b, memory_config=L1, **kw))
            if base is None:
                base = s
            marg = (s - base) * 1e6
            S[lbl] = {"us": round(s * 1e6, 2), "gbps": round(3 * nb / s / 1e9, 1),
                      "marginal_us": round(marg, 2),
                      "elem_per_us": round(nelem / (marg * 1e3), 3) if marg > 0.5 else None}
            print(f"  {lbl:8s} {s*1e6:8.2f} us  {3*nb/s/1e9:7.1f} GB/s  "
                  f"marginal {marg:7.2f} us", flush=True)
        except Exception as e:
            S[lbl] = {"err": str(e)[:90]}
            print(f"  {lbl} ERR {str(e)[:90]}", flush=True)
    res["S_gate_unary"] = S
    ttnn.deallocate(a)
    ttnn.deallocate(b)

    print("=== C triangle matmul core count, [1,32,320,320] @ [1,32,320,320] L1 ===", flush=True)
    ab = ttnn.from_torch(torch.randn(1, C, N, N), layout=ttnn.TILE_LAYOUT, device=dev,
                         dtype=ttnn.bfloat16, memory_config=L1)
    bb = ttnn.from_torch(torch.randn(1, C, N, N), layout=ttnn.TILE_LAYOUT, device=dev,
                         dtype=ttnn.bfloat16, memory_config=L1)
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    st_ = (N + 31) // 32
    occ = []
    for ggx, ggy in ((11, 10), (10, 10), (9, 10), (8, 10), (7, 10), (5, 10), (10, 5), (5, 5)):
        pcm = -(-st_ // ggy)
        pcn = -(-st_ // ggx)
        bw = max(d for d in range(min(10, st_), 0, -1) if st_ % d == 0)
        pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(ggx, ggy), in0_block_w=bw, out_subblock_h=1,
            out_subblock_w=1, out_block_h=pcm, out_block_w=pcn, per_core_M=pcm, per_core_N=pcn,
            transpose_mcast=False, fused_activation=None, fuse_batch=False)
        try:
            s = timed(dev, lambda: ttnn.matmul(ab, bb, compute_kernel_config=ckc,
                                               memory_config=L1, program_config=pc,
                                               dtype=ttnn.bfloat16), warm=3, pipe=4, reps=5)
            tiles_x = -(-st_ // pcn)
            tiles_y = -(-st_ // pcm)
            occ.append({"grid": f"{ggx}x{ggy}", "grid_cores": ggx * ggy,
                        "cfg_cores": tiles_x * tiles_y, "per_core_M": pcm, "per_core_N": pcn,
                        "us": round(s * 1e6, 1),
                        "tflops": round(C * 2 * N ** 3 / s / 1e12, 2)})
            print("  " + json.dumps(occ[-1]), flush=True)
        except Exception as e:
            occ.append({"grid": f"{ggx}x{ggy}", "err": str(e)[:70]})
            print(f"  {ggx}x{ggy} ERR {str(e)[:70]}", flush=True)
    res["C_trimatmul_occupancy"] = occ
    ttnn.deallocate(ab)
    ttnn.deallocate(bb)

    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print("WROTE " + args.out, flush=True)


if __name__ == "__main__":
    main()
