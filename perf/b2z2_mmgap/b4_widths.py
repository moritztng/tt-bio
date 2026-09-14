#!/usr/bin/env python3
"""B4: is `K_block = 2` a win at the shapes the shipped fold actually presents?

`perf/b2z2_mmgap/mm_sweep.py` answers the ratio question but only at one width. The `_MM_BLOCK`
table is keyed on `(kt, nt)` and every kt=4 key is a separate production call site, so a verdict
taken at nt=16 alone is a verdict about one of four. This walks all of them, and for each candidate
block it also asks `torch.equal` against what the site ships today -- a config that keeps `K_block`
folds the contraction in the shipped order and should be byte-identical, which is a far cheaper
parity claim than an accuracy envelope.
"""
from __future__ import annotations

import argparse, json, statistics as st, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--rows", type=int, default=512)
ap.add_argument("--n", type=int, default=512)
ap.add_argument("--reps", type=int, default=7)
ap.add_argument("--inner", type=int, default=16)
a = ap.parse_args()

import torch                                                                  # noqa: E402
torch.set_grad_enabled(False)
import ttnn                                                                   # noqa: E402
import tt_bio.tenstorrent as T                                                # noqa: E402
import tt_bio.mm_generic as MG                                                # noqa: E402

dev = T.get_device()
from tt_bio.af2 import compute_kernel_config                                  # noqa: E402
ckc = T.trunk_compute_kernel_config(compute_kernel_config())
ckc4 = MG.ckc_args(ckc)
GRID = tuple(T.COMPUTE_GRID_MAIN)
DRAM = ttnn.DRAM_MEMORY_CONFIG

# (kt, nt) -> the entry `_MM_BLOCK` ships today, and the candidates worth timing against it.
# K_block == kt on every shipped entry, so a candidate with the same K_block is a bit-exactness
# candidate and one with K_block == kt // 2 is the B4 lever.
CASES = []
for (kt, nt), ship in sorted(T._MM_BLOCK.items()):
    cands = {tuple(ship)}
    for M in (4, 8):
        for N, sh, sw in ((1, 4, 1), (2, 2, 2)):
            if nt % N:
                continue
            cands.add((M, kt, N, sh, sw))          # K_block kept -> same accumulation order
            if kt % 2 == 0:
                cands.add((M, kt // 2, N, sh, sw))  # the B4 lever
    # the shipped entry runs FIRST: it is the `torch.equal` and ratio base, and a candidate
    # timed before it has nothing to compare against.
    cands.discard(tuple(ship))
    CASES.append((kt, nt, tuple(ship), [tuple(ship)] + sorted(cands)))

torch.manual_seed(0)
up = lambda t, mc: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                   device=dev, memory_config=mc)


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
for kt, nt, ship, cands in CASES:
    c, hidden = kt * 32, nt * 32
    x = up(torch.randn(1, a.rows, a.n, c, dtype=torch.bfloat16) * 0.5, DRAM)
    w = up(torch.randn(c, hidden, dtype=torch.bfloat16) * 0.05, DRAM)
    OUT = ttnn.allocate_tensor_on_device(ttnn.Shape([1, a.rows, a.n, hidden]), ttnn.bfloat16,
                                         ttnn.TILE_LAYOUT, dev, DRAM)
    mt = a.rows * a.n // 32
    base_t = base_ms = None
    for block in cands:
        gate = "served" if not (kt % block[1] or mt % block[0] or nt % block[2]) else "GUARD_NONE"
        f = lambda b=block: MG.generic_minimal_matmul(dev, x, w, [OUT], (b, GRID), ckc4)
        try:
            f()
            v = ttnn.to_torch(OUT).clone()
            ms = st.median([timed(f) for _ in range(a.reps)])
        except Exception as e:                                                # noqa: BLE001
            print("kt=%d nt=%-2d %-18s FAILED %s" % (kt, nt, block, str(e)[:90]), flush=True)
            continue
        if block == ship:
            base_t, base_ms = v, ms
        eq = bool(torch.equal(v, base_t)) if base_t is not None else None
        mx = float((v.float() - base_t.float()).abs().max()) if base_t is not None else None
        rows.append({"kt": kt, "nt": nt, "ship": list(ship), "block": list(block), "ms": round(ms, 5),
                     "vs_ship": round(base_ms / ms, 4) if base_ms else None,
                     "bit_exact_vs_ship": eq, "max_abs_vs_ship": mx, "gate": gate})
        print("kt=%d nt=%-2d %-18s %8.4f ms  vs_ship %s  eq=%s maxabs=%s %s"
              % (kt, nt, str(block), ms,
                 ("%.4fx" % (base_ms / ms)) if base_ms else "  base ", eq, mx,
                 "" if gate == "served" else gate), flush=True)
    ttnn.deallocate(x); ttnn.deallocate(w); ttnn.deallocate(OUT)

a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps({"grid": list(GRID), "rows_M": a.rows, "n": a.n, "inner": a.inner,
                             "reps": a.reps, "rows": rows}, indent=1))
print("wrote", a.out)
