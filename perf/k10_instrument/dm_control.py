#!/usr/bin/env python3
"""Known-answer control for the data-movement stall accumulators.

Two stock ttnn ops whose binding limit is decidable from their arithmetic intensity alone:

  starved  ttnn.add on two 8192x8192 bf16 DRAM tensors. 384 MB moved, 67 Mflop. The reader
           cannot possibly keep the math thread fed, so it must be sitting in the NoC read
           barrier while the compute cluster waits.
  compute  ttnn.matmul 2048^3 bf16 at HiFi4. 25 MB moved, 17.2 Gflop, roughly 7x compute-bound
           against the measured 444.9 GB/s / 178.6 TFLOP/s roofs. The reader has nothing left to
           fetch, so it must be sitting in cb_reserve_back with a full CB.

If the instrument cannot separate those two it is unfit and the discriminator is worthless.

Run one case per process under tracy:
    python -m tracy -r -o OUT --enable-sum-profiling -- dm_control.py --case starved --out J
"""
from __future__ import annotations

import argparse, json, os, statistics as st, time
from pathlib import Path

FENCE_N, FENCE_DIM = 3, 32


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--case", required=True, choices=("starved", "compute"))
    ap.add_argument("--reps", type=int, default=10)
    a = ap.parse_args()

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    import tt_bio.tenstorrent as T

    dev = T.get_device()
    out = {"case": a.case, "reps": a.reps,
           "card": os.environ.get("TT_VISIBLE_DEVICES"),
           "loadavg": open("/proc/loadavg").read().split()[:3],
           "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    try:
        f = ttnn.from_torch(torch.ones(1, 1, FENCE_DIM, FENCE_DIM), layout=ttnn.TILE_LAYOUT,
                            dtype=ttnn.bfloat16, device=dev)

        def fence():
            for _ in range(FENCE_N):
                ttnn.exp(f)
            ttnn.synchronize_device(dev)

        if a.case == "starved":
            n = 8192
            x = ttnn.from_torch(torch.randn(n, n), layout=ttnn.TILE_LAYOUT,
                                dtype=ttnn.bfloat16, device=dev,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG)
            y = ttnn.from_torch(torch.randn(n, n), layout=ttnn.TILE_LAYOUT,
                                dtype=ttnn.bfloat16, device=dev,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG)
            call = lambda: ttnn.add(x, y)
            out["bytes"] = 3 * n * n * 2
            out["flops"] = n * n
        else:
            n = 2048
            x = ttnn.from_torch(torch.randn(n, n), layout=ttnn.TILE_LAYOUT,
                                dtype=ttnn.bfloat16, device=dev,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG)
            y = ttnn.from_torch(torch.randn(n, n), layout=ttnn.TILE_LAYOUT,
                                dtype=ttnn.bfloat16, device=dev,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG)
            cfg = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4)
            call = lambda: ttnn.matmul(x, y, compute_kernel_config=cfg)
            out["bytes"] = 3 * n * n * 2
            out["flops"] = 2 * n ** 3

        for _ in range(3):
            call()
        ttnn.synchronize_device(dev)
        solo = []
        for _ in range(a.reps):
            t0 = time.perf_counter()
            call()
            ttnn.synchronize_device(dev)
            solo.append(time.perf_counter() - t0)
        fence()
        t0 = time.perf_counter()
        for _ in range(a.reps):
            call()
        ttnn.synchronize_device(dev)
        out["back_to_back_ms"] = round(1e3 * (time.perf_counter() - t0) / a.reps, 4)
        fence()
        out["solo_synced_ms"] = round(1e3 * st.median(solo), 4)
        out["implied_GBps"] = round(out["bytes"] / (out["back_to_back_ms"] * 1e-3) / 1e9, 1)
        out["implied_TFLOPs"] = round(out["flops"] / (out["back_to_back_ms"] * 1e-3) / 1e12, 2)
    finally:
        pass

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    print("  " + json.dumps(out), flush=True)
    print("DONE", a.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
