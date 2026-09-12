#!/usr/bin/env python3
"""Re-price the fused SwiGLU on the block config that closed the matmul gap.

`best_ab.py`: the generic_op matmul at (8,2,1,4,1) on an 8x8 grid runs 1.0702x `ttnn.linear`, where
the shipped (4,4,1,4,1) on 8x9 runs 0.9264x. K_block = 2 is the axis -- two K blocks give the
operand chain something to pipeline over. The fused SwiGLU is built on the same dataflow, so it
should inherit that; whether its epilogue survives K_blocks > 1 is a correctness question and the
PCC leg answers it.
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
                default="4,4,1,4,1@8x9;8,2,1,4,1@8x8;8,2,2,2,2@8x8;8,2,1,4,1@8x9;4,2,1,4,1@8x8")
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

torch.manual_seed(0)
up = lambda t, mc: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                   device=dev, memory_config=mc)
x = up(torch.randn(1, a.rows, a.n, a.c, dtype=torch.bfloat16) * 0.5, ttnn.L1_MEMORY_CONFIG)
w1 = up(torch.randn(a.c, a.hidden, dtype=torch.bfloat16) * 0.05, ttnn.DRAM_MEMORY_CONFIG)
w2 = up(torch.randn(a.c, a.hidden, dtype=torch.bfloat16) * 0.05, ttnn.DRAM_MEMORY_CONFIG)
kt, nt = a.c // 32, a.hidden // 32
TS.set_enabled(True)


def incumbent():
    x1 = ttnn.linear(x, w1, activation="silu", compute_kernel_config=ckc,
                     memory_config=ttnn.L1_MEMORY_CONFIG, core_grid=T.CORE_GRID_MAIN)
    x2 = ttnn.linear(x, w2, compute_kernel_config=ckc,
                     memory_config=ttnn.L1_MEMORY_CONFIG, core_grid=T.CORE_GRID_MAIN)
    r = ttnn.multiply_(x1, x2)
    ttnn.deallocate(x2)
    return r


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
rows = []
for spec in a.configs.split(";"):
    b, g = spec.split("@")
    block = tuple(int(v) for v in b.split(","))
    grid = tuple(int(v) for v in g.split("x"))
    TS.BLOCK_KEYS = {(kt, nt): block}
    TS._block_for.cache_clear()

    def fused():
        out = TS.fused_swiglu(x, w2, w1, ckc4, grid)
        if out is None:
            raise RuntimeError("declined " + json.dumps({str(k): v for k, v in TS.REJECTS.items()}))
        return out

    try:
        r = fused()
        v = ttnn.to_torch(r).float()
        ttnn.deallocate(r)
        pcc = float(((ref - ref.mean()) * (v - v.mean())).mean()
                    / (ref.std(unbiased=False) * v.std(unbiased=False)))
        max_abs = float((ref - v).abs().max())
        A, B = [], []
        for _ in range(a.reps):
            A.append(timed(incumbent))
            B.append(timed(fused))
    except Exception as e:                                                    # noqa: BLE001
        print("%-24s FAILED %s" % (spec, str(e)[:140]), flush=True)
        continue
    ma, mb = st.median(A), st.median(B)
    rows.append({"block": list(block), "grid": list(grid), "incumbent_ms": round(ma, 5),
                 "fused_ms": round(mb, 5), "ratio": round(ma / mb, 4), "pcc": pcc,
                 "max_abs": max_abs,
                 "spread_pct": round(100 * (max(B) - min(B)) / mb, 2),
                 "per_transition_call_delta_ms": round(32 * (ma - mb), 4)})
    print("%-24s incumbent %.5f  fused %.5f  ratio %.4fx  pcc %.7f  max_abs %.5f  "
          "delta/transition_z %+.4f ms"
          % (spec, ma, mb, ma / mb, pcc, max_abs, rows[-1]["per_transition_call_delta_ms"]),
          flush=True)

a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps({"inner": a.inner, "reps": a.reps, "rows": rows}, indent=1))
print("wrote", a.out)
