#!/usr/bin/env python3
"""How many cores are actually engaged by the MSA/template matmuls, measured rather than derived.

The tt-metal device profiler does not work on these hosts (the wheel ships `tracy` without the
capture binaries), so there is no CORE COUNT column to read. This substitutes an A/B: run the exact
production call with `core_grid` restricted to g cores for g = 1..110 and time it. If an op really
engages all 110, its time falls as ~1/g right up to 110. If it saturates at g*, the cores past g*
are receiving no useful work and g* is the engaged count.

Shapes are the fold's own, taken from the per-op record (perf/msa_template/ops_*.json), padded:
N=298 tokens -> batch 298 x M=320 rows.

    TT_VISIBLE_DEVICES=0 PYTHONPATH=$PWD python3 perf/msa_template/occupancy_ab.py \
        --out perf/msa_template/occupancy_pc0.json
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

import torch
import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tt_bio.tenstorrent import get_device  # noqa: E402

DRAM = ttnn.DRAM_MEMORY_CONFIG

# (label, batch, M, K, N, site) -- every one is a real row in the ledger.
CASES = [
    ("pwa_z_bias  K256 nt1", 298, 320, 256, 32,  "tenstorrent.py:2807"),
    ("tpl_z_proj  K256 nt2", 298, 320, 256, 64,  "protenix.py:306"),
    ("tri_out     K256 nt8", 298, 320, 256, 256, "tenstorrent.py:595"),
    ("tpl_tri_out K64  nt2", 298, 320, 64,  64,  "tenstorrent.py:595 (template)"),
]
GRIDS = [(1, 1), (2, 2), (4, 2), (4, 4), (6, 4), (8, 4), (8, 6), (11, 6), (11, 8), (11, 10)]


def timed(dev, fn, warm=2, pipe=3, reps=5):
    for _ in range(warm):
        fn()
    ttnn.synchronize_device(dev)
    o = []
    for _ in range(reps):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(pipe):
            fn()
        ttnn.synchronize_device(dev)
        o.append((time.perf_counter() - t0) / pipe)
    return st.median(o)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    out = {"grids": [f"{x}x{y}" for x, y in GRIDS], "cases": {}}

    for label, batch, M, K, N, site in CASES:
        a = ttnn.from_torch(torch.randn(1, batch, M, K) * 0.1, dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
        w = ttnn.from_torch(torch.randn(K, N) * 0.1, dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
        row = {"site": site, "batch": batch, "M": M, "K": K, "N": N,
               "k_tiles": K // 32, "n_tiles": N // 32, "us": {}}
        for gx, gy in GRIDS:
            g = ttnn.CoreGrid(x=gx, y=gy)
            try:
                s = timed(dev, lambda: ttnn.deallocate(
                    ttnn.linear(a, w, compute_kernel_config=ckc, memory_config=DRAM,
                                dtype=ttnn.bfloat16, core_grid=g)))
                row["us"][f"{gx}x{gy}"] = round(s * 1e6, 1)
            except Exception as e:                                     # noqa: BLE001
                row["us"][f"{gx}x{gy}"] = str(e)[:70]
        ttnn.deallocate(a)
        ttnn.deallocate(w)
        # Engaged-core estimate: smallest grid within 5 % of the best time measured.
        good = {k: v for k, v in row["us"].items() if isinstance(v, float)}
        if good:
            best = min(good.values())
            sat = min((k for k, v in good.items() if v <= best * 1.05),
                      key=lambda k: int(k.split("x")[0]) * int(k.split("x")[1]))
            row["best_us"] = best
            row["saturates_at"] = sat
            row["cores_engaged"] = int(sat.split("x")[0]) * int(sat.split("x")[1])
        out["cases"][label] = row
        print(f"{label:22s} {json.dumps(row['us'])}  -> saturates {row.get('saturates_at')} "
              f"({row.get('cores_engaged')} cores of 110)", flush=True)

    args.out.write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
