#!/usr/bin/env python3
"""What the fused kernel's "2.5x two-pass loop" actually is.

`loop_split.py` measured, on qb2 card 0 at 11x10: a ONE-pass fused kernel with the product deleted
costs 0.9975x a standalone `generic_op` matmul. The loop is free. The whole 2.5x is the second pass,
at 0.10968 ms against the first pass's 0.03081.

The two passes differ in exactly one thing: pass 1 applies silu. And
`ttnn.linear(activation="silu") - ttnn.linear` is 0.0877 ms at this shape, which is the same number.
So the "two-pass loop" cost is the silu, and the 2.5x came from comparing an arm that computes silu
against two standalone matmuls that do not.

This harness measures that directly and then prices the fix. On Blackhole `silu_tile` resolves
sigmoid through `_sfpu_exp_fp32_accurate_` plus a 2-iteration Newton reciprocal because DST is fp32
-- and `pack_tile` writes the result as bf16 one instruction later. SILU_MODE 1 computes the same
sigmoid at bf16 precision (`exp_21f`, one Newton step), which is all the pack keeps.

Arms (`ms` is one production chunk, `x_mm1` is against one standalone matmul):

  mm_1        one standalone generic_op matmul, no activation
  p1_none     fused kernel, 1 pass, no silu, no product      -- the loop, alone
  p1_silu0    fused kernel, 1 pass, ttnn's silu, no product  -- the loop + the incumbent silu
  p1_silu1    the same with the bf16-precision silu
  p1_silu1h   the same with the init hoisted out of the tile loop
  p2_none     fused kernel, 2 passes, no silu, no product    -- two matmuls through the loop
  p2_silu0    2 passes, ttnn's silu, no product              -- reproduces loop_split's loop_p2
  incumbent   linear(silu) + linear + multiply_, the three ops the fusion replaces
  fused_s0    the shipping fused arm: 2 passes, MUL_MODE 1, BATCH 4, ttnn's silu
  fused_s1    the same with the bf16-precision silu and the hoisted init

PCC is scored against torch for every arm that computes the real answer; the `*_none` and `p*_silu*`
arms are diagnostics and are not offered as real.
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
ap.add_argument("--reps", type=int, default=15)
ap.add_argument("--inner", type=int, default=64)
ap.add_argument("--block", type=str, default="8,2,2,2,2")
ap.add_argument("--grid", type=str, default="11x10")
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
xt = (torch.randn(1, a.rows, a.n, a.c) * 0.5).bfloat16()
w1t = (torch.randn(a.c, a.hidden) * 0.05).bfloat16()          # the ACTIVATED projection
w2t = (torch.randn(a.c, a.hidden) * 0.05).bfloat16()          # the plain one
ref = (torch.nn.functional.silu(xt.float() @ w1t.float()) * (xt.float() @ w2t.float()))

up = lambda t, mc: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                   device=dev, memory_config=mc)
x = up(xt, ttnn.L1_MEMORY_CONFIG)
w1 = up(w1t, ttnn.DRAM_MEMORY_CONFIG)
w2 = up(w2t, ttnn.DRAM_MEMORY_CONFIG)
hshape = ttnn.Shape([1, a.rows, a.n, a.hidden])
TS.set_enabled(True)
TS.BLOCK_KEYS = {(kt, nt): block}
TS._block_for.cache_clear()
TS.GRID = None


def _pcc(got, want):
    got, want = got.flatten(), want.flatten()
    return float(((got - got.mean()) * (want - want.mean())).mean()
                 / (got.std(unbiased=False) * want.std(unbiased=False)))


def mm_1(keep=False):
    o = ttnn.allocate_tensor_on_device(hshape, ttnn.bfloat16, ttnn.TILE_LAYOUT, dev,
                                       ttnn.L1_MEMORY_CONFIG)
    MG.generic_minimal_matmul(dev, x, w2, [o], (block, grid), ckc4)
    if keep:
        return o
    ttnn.deallocate(o)


def L(w, act=None):
    # core_grid is NOT optional: without it ttnn.linear picks its own config and the incumbent
    # reads 0.716 ms instead of 0.167. Matches perf/b2z2_fusion/silu_probe.py exactly.
    return ttnn.linear(x, w, activation=act, compute_kernel_config=ckc,
                       memory_config=ttnn.L1_MEMORY_CONFIG, core_grid=T.CORE_GRID_MAIN)


def incumbent(keep=False):
    x1 = L(w1, "silu")
    x2 = L(w2)
    r = ttnn.multiply_(x1, x2)
    ttnn.deallocate(x2)
    if keep:
        return r
    ttnn.deallocate(r)


def fused(passes, mul_mode, mul_batch, silu_mode, hoist):
    def f(keep=False):
        TS.PASSES, TS.MUL_MODE, TS.MUL_BATCH = passes, mul_mode, mul_batch
        TS.SILU_MODE, TS.SILU_HOIST = silu_mode, hoist
        o = TS.fused_swiglu(x, w2, w1, ckc4, grid)
        if o is None:
            raise RuntimeError("declined " + json.dumps({str(k): v for k, v in TS.REJECTS.items()}))
        if keep:
            return o
        ttnn.deallocate(o)
    return f


#                 name          fn                                  real answer?
ARMS = [("mm_1",      mm_1,                                   False),
        ("mm_1_aa",   mm_1,                                   False),
        ("p1_none",   fused(1, 2, 1, 3, 0),                   False),
        ("p1_silu0",  fused(1, 2, 1, 0, 0),                   False),
        ("p1_silu1",  fused(1, 2, 1, 1, 0),                   False),
        ("p1_silu1h", fused(1, 2, 1, 1, 1),                   False),
        ("p2_none",   fused(2, 2, 1, 3, 0),                   False),
        ("p2_silu0",  fused(2, 2, 1, 0, 0),                   False),
        ("incumbent", incumbent,                              True),
        ("incumbent_aa", incumbent,                           True),
        ("fused_s0",  fused(2, 1, 4, 0, 0),                   True),
        ("fused_s1",  fused(2, 1, 4, 1, 1),                   True),
        ("fused_s1nh", fused(2, 1, 4, 1, 0),                  True)]


def timed(fn):
    for _ in range(2):
        fn()
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(a.inner):
        fn()
    ttnn.synchronize_device(dev)
    return 1e3 * (time.perf_counter() - t0) / a.inner


for name, fn, _ in ARMS:                          # build every variant before timing any
    fn()
ttnn.synchronize_device(dev)

o = incumbent(keep=True)
dev_ref = ttnn.to_torch(o).float()
ttnn.deallocate(o)

score = {}
for name, fn, real in ARMS:
    if not real:
        continue
    o = fn(keep=True)
    v = ttnn.to_torch(o).float()
    ttnn.deallocate(o)
    score[name] = {"pcc_torch": round(_pcc(v, ref), 7),
                   "pcc_incumbent": round(_pcc(v, dev_ref), 7),
                   "max_abs_vs_incumbent": round(float((dev_ref - v).abs().max()), 6),
                   "bit_exact_vs_incumbent": bool(torch.equal(dev_ref, v))}
    print("PCC %-12s torch %.7f  incumbent %.7f  max_abs %.6f  bitexact %s"
          % (name, score[name]["pcc_torch"], score[name]["pcc_incumbent"],
             score[name]["max_abs_vs_incumbent"], score[name]["bit_exact_vs_incumbent"]),
          flush=True)

got = {n: [] for n, _, _ in ARMS}
for _ in range(a.reps):
    for name, fn, _ in ARMS:
        got[name].append(timed(fn))

base = st.median(got["mm_1"])
inc = st.median(got["incumbent"])
rows = {}
print("\n%-12s %9s %9s %9s %8s  %s" % ("arm", "ms", "x_mm1", "x_incumb", "spread", "PCC"))
for name, _, _ in ARMS:
    m = st.median(got[name])
    rows[name] = {"ms": round(m, 5), "x_mm1": round(m / base, 4), "x_incumbent": round(inc / m, 4),
                  "spread_pct": round(100 * (max(got[name]) - min(got[name])) / m, 2),
                  **(score.get(name) or {})}
    print("%-12s %9.5f %9.4f %9.4f %7.1f%%  %s"
          % (name, rows[name]["ms"], rows[name]["x_mm1"], rows[name]["x_incumbent"],
             rows[name]["spread_pct"],
             ("%.7f" % score[name]["pcc_incumbent"]) if name in score else "-"), flush=True)

d = lambda x, y: round(rows[x]["ms"] - rows[y]["ms"], 5)
derived = {
    "silu0_cost_ms": d("p1_silu0", "p1_none"),
    "silu1_cost_ms": d("p1_silu1", "p1_none"),
    "silu1h_cost_ms": d("p1_silu1h", "p1_none"),
    "second_pass_no_silu_ms": d("p2_none", "p1_none"),
    "loop_overhead_1pass_ms": round(rows["p1_none"]["ms"] - base, 5),
    "twopass_vs_two_matmuls": round(rows["p2_silu0"]["ms"] / (2 * base), 4),
    "twopass_nosilu_vs_two_matmuls": round(rows["p2_none"]["ms"] / (2 * base), 4),
}
print("\n" + json.dumps(derived, indent=1))
a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps({"block": list(block), "grid": list(grid), "inner": a.inner,
                             "reps": a.reps, "arms": rows, "derived": derived}, indent=1))
print("wrote %s" % a.out)
