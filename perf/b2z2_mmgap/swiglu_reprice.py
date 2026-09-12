#!/usr/bin/env python3
"""Re-price the fused SwiGLU with the host hidden, and say how much of the verdict was the window.

`b2z2-pairformer-megakernel-build` timed this A/B with 4 calls between two device syncs and called
it a 0.8056x loss. `inner_sensitivity.py` (WH, whglx card 0) shows a generic_op arm's wall-clock
ratio walking from 1.302x at one call per window to 1.077x at 64, because the generic_op call path
costs ~2.7x the host time of a `ttnn.linear` call and a shallow window does not hide it. The two
arms here have different op counts (3 vs 1), so the window biases them in opposite directions and
the only honest thing to do is sweep it.

Same arms as `perf/b2z2_megakernel/swiglu_ab.py`, unchanged, so the two are comparable.
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
ap.add_argument("--inners", type=str, default="4,8,16,32,64")
a = ap.parse_args()

import torch                                                                  # noqa: E402
torch.set_grad_enabled(False)
import ttnn                                                                   # noqa: E402
import tt_bio.tenstorrent as T                                                # noqa: E402
import tt_bio.transition_swiglu as TS                                         # noqa: E402

dev = T.get_device()
from tt_bio.af2 import compute_kernel_config                                  # noqa: E402
ckc = T.trunk_compute_kernel_config(compute_kernel_config())
grid = tuple(T.COMPUTE_GRID_MAIN)

torch.manual_seed(0)
up = lambda t, mc: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                   device=dev, memory_config=mc)
x = up(torch.randn(1, a.rows, a.n, a.c, dtype=torch.bfloat16) * 0.5, ttnn.L1_MEMORY_CONFIG)
w1 = up(torch.randn(a.c, a.hidden, dtype=torch.bfloat16) * 0.05, ttnn.DRAM_MEMORY_CONFIG)
w2 = up(torch.randn(a.c, a.hidden, dtype=torch.bfloat16) * 0.05, ttnn.DRAM_MEMORY_CONFIG)


def incumbent():
    x1 = ttnn.linear(x, w1, activation="silu", compute_kernel_config=ckc,
                     memory_config=ttnn.L1_MEMORY_CONFIG, core_grid=T.CORE_GRID_MAIN)
    x2 = ttnn.linear(x, w2, compute_kernel_config=ckc,
                     memory_config=ttnn.L1_MEMORY_CONFIG, core_grid=T.CORE_GRID_MAIN)
    r = ttnn.multiply_(x1, x2)
    ttnn.deallocate(x2)
    return r


TS.set_enabled(True)
TS.BLOCK_KEYS = {**TS.BLOCK_KEYS, **TS.DIAG_KEYS}


def fused():
    r = TS.fused_swiglu(x, w2, w1, T._mm_ckc(ckc), grid)
    if r is None:
        raise RuntimeError("declined: " + json.dumps({str(k): v for k, v in TS.REJECTS.items()}))
    return r


def cmp(u, v):
    d = (u - v).abs()
    num = (u * v).sum() - u.sum() * v.sum() / u.numel()
    den = ((u * u).sum() - u.sum() ** 2 / u.numel()).sqrt() * \
          ((v * v).sum() - v.sum() ** 2 / v.numel()).sqrt()
    return {"equal": bool(torch.equal(u, v)), "max_abs": float(d.max()),
            "mismatched": int((u != v).sum()), "of": int(u.numel()),
            "pcc": float(num / den) if float(den) else None}


r = incumbent(); ref = ttnn.to_torch(r).float(); ttnn.deallocate(r)
r = fused(); got = ttnn.to_torch(r).float(); ttnn.deallocate(r)
parity = cmp(ref, got)
print("parity fused-vs-incumbent " + json.dumps(parity), flush=True)


def timed(fn, inner):
    for _ in range(2):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(inner):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(dev)
    return 1e3 * (time.perf_counter() - t0) / inner


rows = []
for inner in [int(v) for v in a.inners.split(",")]:
    A, B, AA = [], [], []
    for _ in range(a.reps):
        A.append(timed(incumbent, inner))
        B.append(timed(fused, inner))
        AA.append(timed(incumbent, inner))
    ma, mb, maa = st.median(A), st.median(B), st.median(AA)
    rows.append({"inner": inner, "incumbent_ms": round(ma, 5), "fused_ms": round(mb, 5),
                 "ratio": round(ma / mb, 4),
                 "aa_floor_pct": round(100 * abs(maa - ma) / ma, 3),
                 "per_transition_call_delta_ms": round(32 * (ma - mb), 4),
                 "A": [round(v, 5) for v in A], "B": [round(v, 5) for v in B]})
    print("inner=%-3d incumbent %.5f  fused %.5f  ratio %.4fx  A/A %.2f %%  "
          "delta/transition_z call %+.4f ms"
          % (inner, ma, mb, ma / mb, rows[-1]["aa_floor_pct"],
             rows[-1]["per_transition_call_delta_ms"]), flush=True)

a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps({"grid": list(grid), "cores": grid[0] * grid[1], "reps": a.reps,
                             "parity": parity, "rows": rows,
                             "when": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, indent=1))
print("wrote", a.out)
