#!/usr/bin/env python3
"""The four arms at the configs the sweep liked, paired and interleaved, n=9.

`mm_sweep.py` found `minimal_matmul` with K_block = 2 BEATING `ttnn.linear` (1.0822x at
(8,2,1,4,1) on 8x8) -- two K blocks give the operand chain something to pipeline over, which the
shipped K_block = 4 config does not have. It also showed the generic_op transcription failing to
follow native at some of those configs, which the factory source does not explain: the compile-time
args, the CB depths, the semaphore order and the NOC choice are all transcribed exactly.

One asymmetry the sweep did have: the generic arm wrote into one persistent L1 tensor allocated
before everything else, while native allocated a fresh output per call. `generic_fresh` removes it.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--rows", type=int, default=16)
ap.add_argument("--n", type=int, default=512)
ap.add_argument("--c", type=int, default=128)
ap.add_argument("--hidden", type=int, default=512)
ap.add_argument("--reps", type=int, default=9)
ap.add_argument("--inner", type=int, default=32)
ap.add_argument("--configs", type=str,
                default="8,2,1,4,1@8x8;8,2,2,2,2@8x8;4,2,2,2,2@8x9;8,2,2,2,2@8x9;4,4,1,4,1@8x9")
a = ap.parse_args()

import torch                                                                  # noqa: E402
torch.set_grad_enabled(False)
import ttnn                                                                   # noqa: E402
import tt_bio.tenstorrent as T                                                # noqa: E402
import tt_bio.mm_generic as MG                                                # noqa: E402

dev = T.get_device()
from tt_bio.af2 import compute_kernel_config                                  # noqa: E402
ckc = T.trunk_compute_kernel_config(compute_kernel_config())
ckc4 = T._mm_ckc(ckc)
MMC = getattr(ttnn.experimental, "MinimalMatmulConfig", None) or ttnn.MinimalMatmulConfig

torch.manual_seed(0)
up = lambda t, mc: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                   device=dev, memory_config=mc)
x = up(torch.randn(1, a.rows, a.n, a.c, dtype=torch.bfloat16) * 0.5, ttnn.L1_MEMORY_CONFIG)
w = up(torch.randn(a.c, a.hidden, dtype=torch.bfloat16) * 0.05, ttnn.DRAM_MEMORY_CONFIG)
shape = ttnn.Shape([1, a.rows, a.n, a.hidden])

CFGS = []
for spec in a.configs.split(";"):
    b, g = spec.split("@")
    CFGS.append((tuple(int(v) for v in b.split(",")), tuple(int(v) for v in g.split("x"))))


def linear():
    ttnn.deallocate(ttnn.linear(x, w, compute_kernel_config=ckc,
                                memory_config=ttnn.L1_MEMORY_CONFIG,
                                core_grid=T.CORE_GRID_MAIN))


def arms_for(block, grid):
    mmc = MMC(M_block_size=block[0], K_block_size=block[1], N_block_size=block[2],
              subblock_h=block[3], subblock_w=block[4],
              compute_with_storage_grid_size=ttnn.CoreCoord(grid[0], grid[1]))
    fixed = ttnn.allocate_tensor_on_device(shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, dev,
                                           ttnn.L1_MEMORY_CONFIG)

    def native():
        ttnn.deallocate(ttnn.experimental.minimal_matmul(
            x, w, compute_kernel_config=ckc, memory_config=ttnn.L1_MEMORY_CONFIG, config=mmc))

    def generic_fresh():
        o = ttnn.allocate_tensor_on_device(shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, dev,
                                           ttnn.L1_MEMORY_CONFIG)
        MG.generic_minimal_matmul(dev, x, w, [o], (block, grid), ckc4)
        ttnn.deallocate(o)

    def generic_fixed():
        MG.generic_minimal_matmul(dev, x, w, [fixed], (block, grid), ckc4)

    return [("native", native), ("generic_fresh", generic_fresh),
            ("generic_fixed", generic_fixed)], fixed


def timed(fn):
    for _ in range(2):
        fn()
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(a.inner):
        fn()
    ttnn.synchronize_device(dev)
    return 1e3 * (time.perf_counter() - t0) / a.inner


rows = []
for block, grid in CFGS:
    arms, fixed = arms_for(block, grid)
    got = {n: [] for n, _ in arms}
    got["linear"] = []
    for _ in range(a.reps):
        got["linear"].append(timed(linear))
        for name, fn in arms:
            got[name].append(timed(fn))
    base = st.median(got["linear"])
    row = {"block": list(block), "grid": list(grid), "linear_ms": round(base, 5)}
    for name, _ in arms:
        m = got[name]
        row[name] = {"ms": round(st.median(m), 5), "vs_linear": round(base / st.median(m), 4),
                     "spread_pct": round(100 * (max(m) - min(m)) / st.median(m), 2)}
    rows.append(row)
    print("%s %s linear %.5f | %s" % (block, grid, base,
          "  ".join("%s %.5f (%.4fx, %.1f%%)" % (n, row[n]["ms"], row[n]["vs_linear"],
                                                 row[n]["spread_pct"]) for n, _ in arms)),
          flush=True)
    ttnn.deallocate(fixed)

a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps({"inner": a.inner, "reps": a.reps, "rows": rows}, indent=1))
print("wrote", a.out)
