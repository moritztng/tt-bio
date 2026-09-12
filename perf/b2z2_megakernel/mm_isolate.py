#!/usr/bin/env python3
"""Split the fused SwiGLU's loss into its two halves: the fusion, and the matmul it is built on.

The fused kernel is a `minimal_matmul` transcription with an epilogue. It deletes real traffic and
measures SLOWER, which has now happened three times on this chain (F1 `trimul_tail`,
`TRIMUL_INPROJ_ROWBLOCK`, and this). There are exactly two candidates and they pull in opposite
directions:

  fusion       deletes 16,384 L1 tile passes a chunk       should be faster
  the matmul   `minimal_matmul` through `generic_op` vs `ttnn.linear` at this shape

So time the matmul on its own, same shape, same block config, same compute kernel config, against
the `ttnn.linear` the fused kernel replaces. If the bare matmul is already slower by more than the
fused kernel loses, the fusion is a win sitting underneath a slower matmul and the answer is to
keep ttnn's matmul; if it is not, the fusion itself does not pay.
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
a = ap.parse_args()

import torch                                                                  # noqa: E402
torch.set_grad_enabled(False)
import ttnn                                                                   # noqa: E402
import tt_bio.tenstorrent as T                                                # noqa: E402
import tt_bio.mm_generic as MG                                                # noqa: E402
import tt_bio.transition_swiglu as TS                                         # noqa: E402

dev = T.get_device()
from tt_bio.af2 import compute_kernel_config                                  # noqa: E402
ckc = T.trunk_compute_kernel_config(compute_kernel_config())
ckc4 = T._mm_ckc(ckc)
grid = tuple(T.COMPUTE_GRID_MAIN)
# (M_block, K_block, N_block, subblock_h, subblock_w). K_block is fixed at 4 -- kt = 4 is the whole
# contraction -- and subblock_h * subblock_w must fit the fp32 half-sync DST budget of 4 tiles
# (b2z2-dst-resident-fusion: DEST_NUM_TILES_FP16 = 16, half sync, fp32 => 4).
BLOCKS = [(4, 4, 1, 4, 1), (4, 4, 2, 2, 2), (4, 4, 4, 1, 4), (8, 4, 1, 4, 1), (8, 4, 2, 2, 2),
          (8, 4, 4, 1, 4), (16, 4, 1, 4, 1), (16, 4, 2, 2, 2)]

torch.manual_seed(0)
xt = torch.randn(1, a.rows, a.n, a.c, dtype=torch.bfloat16) * 0.5
wt = torch.randn(a.c, a.hidden, dtype=torch.bfloat16) * 0.05
up = lambda t, mc: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                   device=dev, memory_config=mc)
x = up(xt, ttnn.L1_MEMORY_CONFIG)
w = up(wt, ttnn.DRAM_MEMORY_CONFIG)
shape = ttnn.Shape([1, a.rows, a.n, a.hidden])

def ttnn_mm():
    return ttnn.linear(x, w, compute_kernel_config=ckc, memory_config=ttnn.L1_MEMORY_CONFIG,
                       core_grid=T.CORE_GRID_MAIN)

def gen_mm_for(block):
    def f():
        o = ttnn.allocate_tensor_on_device(shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, dev,
                                           ttnn.L1_MEMORY_CONFIG)
        MG.generic_minimal_matmul(dev, x, w, [o], (block, grid), ckc4)
        return o
    return f

def timed(fn, inner=4):
    for _ in range(2):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(inner):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(dev)
    return 1e3 * (time.perf_counter() - t0) / inner

mt = a.rows * a.n // 32
nt = a.hidden // 32
print("mt=%d nt=%d grid=%s" % (mt, nt, grid), flush=True)


def timed(fn, inner=4):
    for _ in range(2):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(inner):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(dev)
    return 1e3 * (time.perf_counter() - t0) / inner


ref = ttnn.to_torch(ttnn_mm()).float()
rows = []
base = timed(ttnn_mm)
for block in BLOCKS:
    M, K, N, sh, sw = block
    if mt % M or nt % N:
        print("%-20s skipped (mt %% M = %d, nt %% N = %d)" % (str(block), mt % M, nt % N), flush=True)
        continue
    f = gen_mm_for(block)
    try:
        r = f()
        v = ttnn.to_torch(r).float()
        ttnn.deallocate(r)
        pcc = float(((ref - ref.mean()) * (v - v.mean())).mean()
                    / (ref.std(unbiased=False) * v.std(unbiased=False)))
        ms = timed(f)
    except Exception as e:                                                    # noqa: BLE001
        print("%-20s FAILED %s" % (str(block), str(e)[:120]), flush=True)
        continue
    rows.append({"block": list(block), "ms": round(ms, 4), "pcc": pcc,
                 "vs_ttnn": round(base / ms, 4)})
    print("%-20s %.4f ms  vs ttnn.linear %.4fx  pcc %.8f" % (str(block), ms, base / ms, pcc),
          flush=True)
aa = timed(ttnn_mm)
print("ttnn.linear %.4f ms (A/A floor %.2f %%)" % (base, 100 * abs(aa - base) / base), flush=True)
a.out.write_text(json.dumps({"grid": list(grid), "mt": mt, "nt": nt,
                             "ttnn_linear_ms": round(base, 4),
                             "aa_floor_pct": round(100 * abs(aa - base) / base, 3),
                             "rows": rows}, indent=1))
