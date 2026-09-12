#!/usr/bin/env python3
"""Measured bf16 HiFi4 matmul ceiling on this part, at the K depths the trimul actually uses.

The fused chain's floor is arithmetic, not bytes, so the denominator has to be the matmul rate
this device reaches with the production compute kernel config -- measured, at a depth where the
operands are large enough that the answer is not a launch artifact.
"""
import argparse, json, statistics, time
from pathlib import Path
import torch, ttnn
from tt_bio import tenstorrent as TT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=7)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    dev = TT.get_device()
    g = dev.compute_with_storage_grid_size()
    kcls = (ttnn.types.WormholeComputeKernelConfig if dev.arch() == ttnn.Arch.WORMHOLE_B0
            else ttnn.types.BlackholeComputeKernelConfig)
    ckc = kcls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
               fp32_dest_acc_en=True, packer_l1_acc=True)
    rows = []
    # (M, K, N, label). 262144x128x512 is the trimul in-projection; 2048-cube is a deep square
    # matmul, the same shape the campaign's op-cost curve used; the batched one is the triangle
    # matmul itself.
    cases = [("in_proj 262144x128x512", (1, 1, 262144, 128), (128, 512), None),
             ("square 2048^3", (1, 1, 2048, 2048), (2048, 2048), None),
             ("square 4096^3", (1, 1, 4096, 4096), (4096, 4096), None),
             ("tri 128x512x512x512", (1, 128, 512, 512), (1, 128, 512, 512), None)]
    for label, sa, sb, _ in cases:
        x = ttnn.from_torch(torch.randn(*sa) * 0.05, layout=ttnn.TILE_LAYOUT, device=dev,
                            dtype=ttnn.bfloat16)
        w = ttnn.from_torch(torch.randn(*sb) * 0.05, layout=ttnn.TILE_LAYOUT, device=dev,
                            dtype=ttnn.bfloat16)
        M = sa[-2] * (sa[1] if len(sa) > 3 else 1)
        K, N = sa[-1], sb[-1]
        flop = 2 * M * K * N
        fn = lambda: ttnn.matmul(x, w, compute_kernel_config=ckc,
                                 memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)
        ttnn.deallocate(fn())
        ts = []
        for _ in range(a.iters):
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            o = fn()
            ttnn.synchronize_device(dev)
            ts.append(time.perf_counter() - t0)
            ttnn.deallocate(o)
        med, lo = statistics.median(ts), min(ts)
        rows.append({"case": label, "GFLOP": flop / 1e9, "median_ms": med * 1e3,
                     "TFLOPs_median": flop / med / 1e12, "TFLOPs_peak": flop / lo / 1e12})
        ttnn.deallocate(x); ttnn.deallocate(w)
    res = {"arch": str(dev.arch()), "grid": f"{g.x}x{g.y}", "fidelity": "HiFi4 fp32_dest_acc",
           "rows": rows, "mm_roof_TFLOPs": max(r["TFLOPs_peak"] for r in rows)}
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))
    for r in rows:
        print(f"{r['case']:26s} {r['GFLOP']:9.2f} GFLOP  {r['median_ms']:8.3f} ms  "
              f"{r['TFLOPs_median']:7.2f} TFLOP/s  peak {r['TFLOPs_peak']:7.2f}")
    print(f"\nMEASURED HiFi4 MATMUL CEILING {res['arch']} {res['grid']}: "
          f"{res['mm_roof_TFLOPs']:.2f} TFLOP/s")


if __name__ == "__main__":
    main()
