#!/usr/bin/env python3
"""Can the fused SwiGLU's epilogue be made to cost what `ttnn.multiply_` costs?

`silu_probe.py` put the fused chunk at 1.0214x and `epilogue_ablate.py` showed why it is not more:
the in-kernel product costs >= 0.107 ms against `ttnn.multiply_`'s 0.02692 over the same tiles. The
per-tile DST barrier is worth 0.04 % and the SFPU rounding 8.4 %, so the remainder is the three math
passes `mul_block` makes an output tile -- copy p into DST, copy g into DST, SFPU product -- where
ttnn's binary unpacks both operands in one.

TRIMUL_TAIL_MUL_MODE=1 does that: `mul_tiles` unpacks both straight into the FPU. It also drops the
DST cost to one slot an output tile, so MUL_BATCH can go to 4 under fp32 half sync.

If the epilogue lands within ~1.5x of `ttnn.multiply_`, the fused chunk goes to ~2x and the other
six stages become worth building. If it does not, the fused-chain route closes on a measured reason.
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
ap.add_argument("--variants", type=str, default="2/1/0;2/4/1;0/4/1;2/4/2")
a = ap.parse_args()

import torch                                                                  # noqa: E402
torch.set_grad_enabled(False)
import ttnn                                                                   # noqa: E402
import tt_bio.tenstorrent as T                                                # noqa: E402
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
TS.set_enabled(True)
TS.BLOCK_KEYS = {(kt, nt): block}
TS._block_for.cache_clear()


def L(w, act=None):
    return ttnn.linear(x, w, activation=act, compute_kernel_config=ckc,
                       memory_config=ttnn.L1_MEMORY_CONFIG, core_grid=T.CORE_GRID_MAIN)


def incumbent():
    x1 = L(w1, "silu")
    x2 = L(w2)
    r = ttnn.multiply_(x1, x2)
    ttnn.deallocate(x2)
    return r


def fused():
    o = TS.fused_swiglu(x, w2, w1, ckc4, grid)
    if o is None:
        raise RuntimeError("declined " + json.dumps({str(k): v for k, v in TS.REJECTS.items()}))
    return o


def timed(fn):
    for _ in range(2):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(a.inner):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(dev)
    return 1e3 * (time.perf_counter() - t0) / a.inner


r = incumbent(); ref = ttnn.to_torch(r).float(); ttnn.deallocate(r)
VAR = [tuple(int(v) for v in s.split("/")) for s in a.variants.split(";")]

# Build and SCORE every variant before any of them is timed: a variant that computes the wrong
# answer must not be reported as a fast one.
ok = []
for rnd, b, mode in VAR:
    TS.ROUND, TS.MUL_BATCH, TS.MUL_MODE = rnd, b, mode
    try:
        o = fused(); v = ttnn.to_torch(o).float(); ttnn.deallocate(o)
    except Exception as e:                                                    # noqa: BLE001
        print("r%d b%d m%d BUILD FAILED %s" % (rnd, b, mode, str(e)[:150]), flush=True)
        continue
    pcc = float(((ref - ref.mean()) * (v - v.mean())).mean()
                / (ref.std(unbiased=False) * v.std(unbiased=False)))
    mx = float((ref - v).abs().max())
    print("r%d b%d m%d  pcc %.7f  max_abs %.5f%s"
          % (rnd, b, mode, pcc, mx, "  DIAGNOSTIC, wrong answer by construction" if mode == 2
             else ""), flush=True)
    if mode != 2 and pcc < 0.9999:
        print("  rejected on PCC, not timed", flush=True)
        continue
    ok.append((rnd, b, mode, pcc, mx))


def fused_for(rnd, b, mode):
    def f():
        TS.ROUND, TS.MUL_BATCH, TS.MUL_MODE = rnd, b, mode
        return fused()
    return f


ARMS = [("incumbent", incumbent), ("incumbent_aa", incumbent)]
for rnd, b, mode, _, _ in ok:
    ARMS.append(("r%d_b%d_m%d" % (rnd, b, mode), fused_for(rnd, b, mode)))

got = {n: [] for n, _ in ARMS}
for _ in range(a.reps):
    for name, fn in ARMS:
        got[name].append(timed(fn))

base = st.median(got["incumbent"])
qual = {"r%d_b%d_m%d" % (r_, b_, m_): (p_, x_) for r_, b_, m_, p_, x_ in ok}
rows = {}
for name, _ in ARMS:
    m = got[name]
    rows[name] = {"ms": round(st.median(m), 5), "ratio_vs_incumbent": round(base / st.median(m), 4),
                  "spread_pct": round(100 * (max(m) - min(m)) / st.median(m), 2)}
    if name in qual:
        rows[name]["pcc"], rows[name]["max_abs"] = qual[name]
    print("%-14s %.5f ms  %.4fx  spread %.1f%%  %s"
          % (name, rows[name]["ms"], rows[name]["ratio_vs_incumbent"], rows[name]["spread_pct"],
             ("pcc %.7f max_abs %.5f" % qual[name]) if name in qual else ""), flush=True)

a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps({"block": list(block), "grid": list(grid), "inner": a.inner,
                             "reps": a.reps, "arms": rows}, indent=1))
print("wrote", a.out)
