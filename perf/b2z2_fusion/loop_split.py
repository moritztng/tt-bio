#!/usr/bin/env python3
"""Is the fused kernel's cost the two-pass LOOP, or the second pass?

`epilogue_mode.py` measured that deleting the epilogue product entirely still leaves the kernel at
0.14057 ms against the 0.0555 ms its two matmuls cost as separate `generic_op` calls -- 2.5x. That
localises the cost to the loop but does not say which part of it.

This separates them. Both dataflow kernels honour TRIMUL_TAIL_PASSES, so a one-pass build pushes one
pass and pops one pass; with MUL_MODE=2 it packs pass 0 and computes no product. That arm does
exactly what ONE standalone `generic_op` matmul does, through the fused kernel's loop.

  mm_1        one standalone generic_op matmul, the reference
  loop_p1     the fused kernel, one pass, no product.  mm_1 x 1
  loop_p2     the fused kernel, two passes, no product. mm_1 x 2 if the loop is free

FALSIFIER, written before the numbers exist:
  * loop_p1 within ~1.3x of mm_1  -> the loop is fine and the SECOND pass is the cost. Fixable by
    restructuring the pass loop, and the fusion route stays open.
  * loop_p1 at ~2.5x mm_1         -> the loop itself is the cost, it is there at one pass, and no
    amount of restructuring the second pass recovers it. The fused-chain route closes on a measured
    reason, which is a deliverable.
I predict loop_p1 lands at 1.9-2.6x mm_1, i.e. the loop, because the 2.5x is already visible with
the product deleted and a second matmul pass is the one part of this kernel that provably does the
same work a standalone matmul does.
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
ap.add_argument("--block", type=str, default="8,2,2,2,2")
ap.add_argument("--grid", type=str, default="11x8")
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
block = tuple(int(v) for v in a.block.split(","))
grid = tuple(int(v) for v in a.grid.split("x"))
kt, nt = a.c // 32, a.hidden // 32

torch.manual_seed(0)
up = lambda t, mc: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                   device=dev, memory_config=mc)
x = up(torch.randn(1, a.rows, a.n, a.c, dtype=torch.bfloat16) * 0.5, ttnn.L1_MEMORY_CONFIG)
w1 = up(torch.randn(a.c, a.hidden, dtype=torch.bfloat16) * 0.05, ttnn.DRAM_MEMORY_CONFIG)
w2 = up(torch.randn(a.c, a.hidden, dtype=torch.bfloat16) * 0.05, ttnn.DRAM_MEMORY_CONFIG)
hshape = ttnn.Shape([1, a.rows, a.n, a.hidden])
TS.set_enabled(True)
TS.BLOCK_KEYS = {(kt, nt): block}
TS._block_for.cache_clear()
TS.GRID = None


def mm_1():
    o = ttnn.allocate_tensor_on_device(hshape, ttnn.bfloat16, ttnn.TILE_LAYOUT, dev,
                                       ttnn.L1_MEMORY_CONFIG)
    MG.generic_minimal_matmul(dev, x, w1, [o], (block, grid), ckc4)
    ttnn.deallocate(o)


def loop_for(passes):
    def f():
        TS.PASSES, TS.MUL_MODE = passes, 2
        o = TS.fused_swiglu(x, w2, w1, ckc4, grid)
        if o is None:
            raise RuntimeError("declined " + json.dumps({str(k): v for k, v in TS.REJECTS.items()}))
        ttnn.deallocate(o)
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


ARMS = [("mm_1", mm_1), ("mm_1_aa", mm_1),
        ("loop_p1", loop_for(1)), ("loop_p2", loop_for(2))]
for name, fn in ARMS:                                    # build every variant before timing any
    fn()
ttnn.synchronize_device(dev)

got = {n: [] for n, _ in ARMS}
for _ in range(a.reps):
    for name, fn in ARMS:
        got[name].append(timed(fn))

base = st.median(got["mm_1"])
rows = {}
for name, _ in ARMS:
    m = st.median(got[name])
    rows[name] = {"ms": round(m, 5), "x_mm_1": round(m / base, 4),
                  "spread_pct": round(100 * (max(got[name]) - min(got[name])) / m, 2)}
    print("%-9s %.5f ms  %.4fx mm_1  spread %.1f%%"
          % (name, rows[name]["ms"], rows[name]["x_mm_1"], rows[name]["spread_pct"]), flush=True)

p1, p2 = rows["loop_p1"]["ms"], rows["loop_p2"]["ms"]
rows["second_pass_ms"] = round(p2 - p1, 5)
rows["loop_overhead_p1_ms"] = round(p1 - base, 5)
print("\nsecond pass costs %.5f ms (a standalone matmul is %.5f)" % (p2 - p1, base), flush=True)
print("one-pass loop overhead over a standalone matmul: %.5f ms (%.2fx)"
      % (p1 - base, p1 / base), flush=True)
a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps({"block": list(block), "grid": list(grid), "inner": a.inner,
                             "reps": a.reps, "arms": rows}, indent=1))
print("wrote %s" % a.out)
