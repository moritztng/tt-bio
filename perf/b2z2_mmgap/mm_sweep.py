#!/usr/bin/env python3
"""Close the residual: the best generic_op block config at the production shape, host hidden.

`host_split.py` leaves 1.077-1.084x between the generic_op transcription and `ttnn.linear` once the
timing window is deep enough to hide dispatch. The sibling row swept eight block configs at four
calls per window, which measured the host; this sweeps them (plus the K_block variants it never
tried) at 32, where the arms are device-bound. K_block is the interesting axis: at K_tiles = 4 the
shipped config takes the whole contraction in one block, so the operand chain has nothing to
pipeline over.
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
ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--inner", type=int, default=32)
ap.add_argument("--grids", type=str, default="")
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
GRID = tuple(T.COMPUTE_GRID_MAIN)
MMC = getattr(ttnn.experimental, "MinimalMatmulConfig", None) or ttnn.MinimalMatmulConfig

torch.manual_seed(0)
up = lambda t, mc: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                   device=dev, memory_config=mc)
x = up(torch.randn(1, a.rows, a.n, a.c, dtype=torch.bfloat16) * 0.5, ttnn.L1_MEMORY_CONFIG)
w = up(torch.randn(a.c, a.hidden, dtype=torch.bfloat16) * 0.05, ttnn.DRAM_MEMORY_CONFIG)
shape = ttnn.Shape([1, a.rows, a.n, a.hidden])
OUT = ttnn.allocate_tensor_on_device(shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, dev,
                                     ttnn.L1_MEMORY_CONFIG)

BLOCKS = [(4, 4, 1, 4, 1), (2, 4, 1, 2, 1), (8, 4, 1, 4, 1), (16, 4, 1, 4, 1), (32, 4, 1, 4, 1),
          (4, 4, 2, 2, 2), (8, 4, 2, 2, 2), (16, 4, 2, 2, 2), (2, 4, 2, 2, 2),
          (4, 2, 1, 4, 1), (8, 2, 1, 4, 1), (16, 2, 1, 4, 1), (4, 1, 1, 4, 1), (8, 1, 1, 4, 1),
          (4, 2, 2, 2, 2), (8, 2, 2, 2, 2)]
GRIDS = ([tuple(int(v) for v in g.split("x")) for g in a.grids.split(",")] if a.grids
         else [GRID, (GRID[0], 8)])


def linear():
    ttnn.deallocate(ttnn.linear(x, w, compute_kernel_config=ckc,
                                memory_config=ttnn.L1_MEMORY_CONFIG,
                                core_grid=T.CORE_GRID_MAIN))


def generic_for(block, grid):
    def f():
        MG.generic_minimal_matmul(dev, x, w, [OUT], (block, grid), ckc4)
    return f


def native_for(block, grid):
    mmc = MMC(M_block_size=block[0], K_block_size=block[1], N_block_size=block[2],
              subblock_h=block[3], subblock_w=block[4],
              compute_with_storage_grid_size=ttnn.CoreCoord(grid[0], grid[1]))

    def f():
        ttnn.deallocate(ttnn.experimental.minimal_matmul(
            x, w, compute_kernel_config=ckc, memory_config=ttnn.L1_MEMORY_CONFIG, config=mmc))
    return f


def timed(fn):
    for _ in range(2):
        fn()
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(a.inner):
        fn()
    ttnn.synchronize_device(dev)
    return 1e3 * (time.perf_counter() - t0) / a.inner


ref = ttnn.to_torch(ttnn.linear(x, w, compute_kernel_config=ckc,
                                memory_config=ttnn.L1_MEMORY_CONFIG,
                                core_grid=T.CORE_GRID_MAIN)).float()


def pcc_of(v):
    return float(((ref - ref.mean()) * (v - v.mean())).mean()
                 / (ref.std(unbiased=False) * v.std(unbiased=False)))


rows = []
for grid in GRIDS:
    for block in BLOCKS:
        for kind, mk in (("generic", generic_for), ("native", native_for)):
            f = mk(block, grid)
            try:
                if kind == "generic":
                    f()
                    v = ttnn.to_torch(OUT).float()
                else:
                    t = ttnn.experimental.minimal_matmul(
                        x, w, compute_kernel_config=ckc, memory_config=ttnn.L1_MEMORY_CONFIG,
                        config=MMC(M_block_size=block[0], K_block_size=block[1],
                                   N_block_size=block[2], subblock_h=block[3],
                                   subblock_w=block[4],
                                   compute_with_storage_grid_size=ttnn.CoreCoord(*grid)))
                    v = ttnn.to_torch(t).float()
                    ttnn.deallocate(t)
                pcc = pcc_of(v)
                g = st.median([timed(f) for _ in range(a.reps)])
                b = st.median([timed(linear) for _ in range(a.reps)])
            except Exception as e:                                            # noqa: BLE001
                print("%-22s %-8s FAILED %s" % (str(block) + str(grid), kind, str(e)[:110]),
                      flush=True)
                continue
            rows.append({"block": list(block), "grid": list(grid), "kind": kind,
                         "ms": round(g, 5), "linear_ms": round(b, 5),
                         "vs_linear": round(b / g, 4), "pcc": pcc})
            print("%-22s %-8s %.5f ms  linear %.5f  vs_linear %.4fx  pcc %.7f"
                  % (str(block) + str(grid), kind, g, b, b / g, pcc), flush=True)

rows.sort(key=lambda r: -r["vs_linear"])
print("\nbest:", json.dumps(rows[0]) if rows else "none")
a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps({"inner": a.inner, "reps": a.reps, "grid_main": list(GRID),
                             "rows": rows}, indent=1))
print("wrote", a.out)
