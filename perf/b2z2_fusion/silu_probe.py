#!/usr/bin/env python3
"""Who actually pays for the Transition chunk on Blackhole, and does the fused arm reproduce?

`epilogue_ablate.py` turned up two things that need settling before any of them is quoted.

1. `ttnn.linear(activation="silu")` measured **0.11432 ms** at the production shape against
   **0.02663 ms** for the identical matmul with no activation. 4.3x for one SFPU pass over the
   output is not an SFPU cost, so the activation almost certainly puts `ttnn.linear` on a different
   program. If a plain `linear` followed by a separate `silu` is cheaper, the Transition has a large
   lever in it that has nothing to do with fusion -- but silu on the bf16 output is NOT silu on the
   fp32 accumulator, so that arm is a numerics change and is scored here, not shipped.
2. The fused arm read 0.9533x in `fused_cfg.py` and 1.0255x here at the same block and grid. The one
   difference between the harnesses is that this one holds two 8.4 MB L1 tensors resident for the
   whole run. Hold them or not, on the same card in the same process, and see which way it moves.

Every arm is paired and interleaved in one process, n=9, with an A/A floor.
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
    """What `Transition` ships: silu fused into the projection, product in place."""
    x1 = L(w1, "silu")
    x2 = L(w2)
    r = ttnn.multiply_(x1, x2)
    ttnn.deallocate(x2)
    return r


def split_silu():
    """Same algebra, silu as its own op on the bf16 projection. NOT the same arithmetic."""
    x1 = L(w1)
    ttnn.silu(x1, output_tensor=x1)
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


def score(fn):
    o = fn(); v = ttnn.to_torch(o).float(); ttnn.deallocate(o)
    pcc = float(((ref - ref.mean()) * (v - v.mean())).mean()
                / (ref.std(unbiased=False) * v.std(unbiased=False)))
    return pcc, float((ref - v).abs().max()), bool(torch.equal(ref, v))


ARMS = [("incumbent", incumbent), ("incumbent_aa", incumbent), ("split_silu", split_silu)]
VAR = [(2, 1), (2, 2), (0, 2)]
built = {}
for rnd, b in VAR:
    TS.ROUND, TS.MUL_BATCH = rnd, b
    o = fused(); ttnn.deallocate(o)                       # build + cache under this key
    built[(rnd, b)] = (rnd, b)


def fused_for(rnd, b):
    def f():
        TS.ROUND, TS.MUL_BATCH = rnd, b
        return fused()
    return f


for rnd, b in VAR:
    ARMS.append(("fused_r%d_b%d" % (rnd, b), fused_for(rnd, b)))

qual = {}
for name, fn in ARMS:
    if name == "incumbent_aa":
        continue
    qual[name] = score(fn)

got = {n: [] for n, _ in ARMS}
for _ in range(a.reps):
    for name, fn in ARMS:
        got[name].append(timed(fn))

base = st.median(got["incumbent"])
rows = {}
for name, _ in ARMS:
    m = got[name]
    rows[name] = {"ms": round(st.median(m), 5), "ratio_vs_incumbent": round(base / st.median(m), 4),
                  "spread_pct": round(100 * (max(m) - min(m)) / st.median(m), 2)}
    if name in qual:
        rows[name]["pcc"], rows[name]["max_abs"], rows[name]["bit_exact"] = qual[name]
    print("%-14s %.5f ms  %.4fx  spread %.1f%%  %s"
          % (name, rows[name]["ms"], rows[name]["ratio_vs_incumbent"], rows[name]["spread_pct"],
             ("pcc %.7f max_abs %.5f bit_exact %s" % qual[name]) if name in qual else ""),
          flush=True)

a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps({"block": list(block), "grid": list(grid), "inner": a.inner,
                             "reps": a.reps, "arms": rows}, indent=1))
print("wrote", a.out)
