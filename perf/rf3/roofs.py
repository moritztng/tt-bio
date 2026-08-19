#!/usr/bin/env python3
"""Measure the two roofs this port is actually priced against, on this card.

Not "% of peak": the spec sheet number for a p150a is FP8 and the model runs bf16 under
`MathFidelity.HiFi4` with `fp32_dest_acc_en`, which is a different machine. Both roofs are
measured here at the model's own compute-kernel config and at the model's own shapes, and
the fidelity sweep is reported alongside so a fidelity lever can be priced rather than
guessed (it would not be bit-exact, so it is gated, not free).

Batched, not isolated: each rate is the median of `--iters` back-to-back calls between two
syncs, so per-call dispatch is amortised the way it is in a fold
(`tt-bio-isolated-op-timing-oversync-inflates-cost`).
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

import torch
import ttnn


def timed(fn, iters, device, warmup=2):
    for _ in range(warmup):
        fn()
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    ttnn.synchronize_device(device)
    return (time.perf_counter() - t0) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from tt_bio.tenstorrent import get_device
    device = get_device()

    fids = {"HiFi4": (ttnn.MathFidelity.HiFi4, True),
            "HiFi2": (ttnn.MathFidelity.HiFi2, True),
            "LoFi": (ttnn.MathFidelity.LoFi, False)}
    cfgs = {name: ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=mf, fp32_dest_acc_en=acc, packer_l1_acc=True)
        for name, (mf, acc) in fids.items()}

    out = {"arch": str(device.arch()), "iters": args.iters, "matmul": [], "dram": []}

    # --- matmul roof, at the shapes the model actually runs -----------------------
    # (M, K, N): the pair-track transition (c_z=128 -> 512), the trimul projection and a
    # big square, so the roof is reported where the model sits rather than at one point.
    shapes = [(1024 * 1024, 128, 512), (512 * 512, 128, 512), (256 * 256, 128, 512),
              (4096, 4096, 4096), (8192, 512, 512)]
    for M, K, N in shapes:
        a = ttnn.from_torch(torch.randn(1, M, K), layout=ttnn.TILE_LAYOUT,
                            device=device, dtype=ttnn.bfloat16)
        b = ttnn.from_torch(torch.randn(K, N), layout=ttnn.TILE_LAYOUT,
                            device=device, dtype=ttnn.bfloat16)
        row = {"M": M, "K": K, "N": N, "gflop": 2 * M * K * N / 1e9}
        for name, cfg in cfgs.items():
            try:
                s = timed(lambda: ttnn.linear(a, b, compute_kernel_config=cfg),
                          args.iters, device)
                row[name] = {"s": s, "tflops": 2 * M * K * N / s / 1e12}
            except Exception as e:  # a shape the allocator refuses is a fact, not a stop
                row[name] = {"error": str(e)[:120]}
        ttnn.deallocate(a); ttnn.deallocate(b)
        out["matmul"].append(row)
        print(f"matmul M={M} K={K} N={N}: "
              + "  ".join(f"{n}={row[n].get('tflops', float('nan')):.2f} TF/s"
                          for n in cfgs), flush=True)

    # --- DRAM roof: elementwise add, 3 tensor traversals per call ------------------
    for numel in (64 << 20, 16 << 20, 4 << 20):
        n = int(numel ** 0.5) // 32 * 32
        a = ttnn.from_torch(torch.randn(1, n, n), layout=ttnn.TILE_LAYOUT,
                            device=device, dtype=ttnn.bfloat16)
        b = ttnn.from_torch(torch.randn(1, n, n), layout=ttnn.TILE_LAYOUT,
                            device=device, dtype=ttnn.bfloat16)
        s = timed(lambda: ttnn.add(a, b), args.iters, device)
        gb = 3 * n * n * 2 / 1e9
        out["dram"].append({"n": n, "GB_per_call": gb, "s": s, "GB_s": gb / s})
        print(f"dram add n={n}: {gb / s:.1f} GB/s", flush=True)
        ttnn.deallocate(a); ttnn.deallocate(b)

    best_mm = max((r[k]["tflops"] for r in out["matmul"] for k in ("HiFi4",)
                   if "tflops" in r[k]), default=0.0)
    out["roof_matmul_hifi4_tflops"] = best_mm
    out["roof_dram_GB_s"] = max(d["GB_s"] for d in out["dram"])
    print(f"\nROOF (this card, this session): bf16 matmul HiFi4 "
          f"{best_mm:.2f} TFLOP/s, DRAM eltwise {out['roof_dram_GB_s']:.1f} GB/s")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=2)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
