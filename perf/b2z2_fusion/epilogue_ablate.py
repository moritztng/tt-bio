#!/usr/bin/env python3
"""Decompose the fused SwiGLU chunk on Blackhole, then ablate its epilogue.

`fused_cfg.py` on qb2 card 1 says the fused kernel is a LOSS at every block config (best 0.9533x),
where the same arms on WH read 1.0337x. The matmul handicap cannot explain it: at the best BH
config `generic_op` is 0.974x `ttnn.linear`, so two of them cost 1.5 us against a 8.2 us loss.

So time the chunk's parts against each other rather than arguing about it. The incumbent splits into
two `ttnn.linear` calls and one `ttnn.multiply_`; the fused kernel splits into the same two matmuls
(now `generic_op`) and its own in-kernel epilogue. If the epilogue costs more than `multiply_` does
while moving a quarter of the bytes, the loss is the epilogue and not the deleted movement.

The epilogue knobs: TRIMUL_TAIL_MUL_BATCH folds several output tiles per DST acquire (the derived
kernel takes one, so every output tile pays a full math/pack barrier), and TRIMUL_TAIL_ROUND picks
the product's rounding to bf16 (2 = round-to-nearest-even by hand in the SFPU, 0 = leave it to the
packer, which is a numerics change and is timed here only to price the SFPU pass).
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
ap.add_argument("--variants", type=str, default="2/1;2/2;0/1;0/2")
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


def timed(fn):
    for _ in range(2):
        fn()
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(a.inner):
        fn()
    ttnn.synchronize_device(dev)
    return 1e3 * (time.perf_counter() - t0) / a.inner


def med(fn):
    v = [timed(fn) for _ in range(a.reps)]
    return st.median(v), round(100 * (max(v) - min(v)) / st.median(v), 2)


def lin(w, act):
    def f():
        ttnn.deallocate(ttnn.linear(x, w, activation=act, compute_kernel_config=ckc,
                                    memory_config=ttnn.L1_MEMORY_CONFIG,
                                    core_grid=T.CORE_GRID_MAIN))
    return f


def generic():
    o = ttnn.allocate_tensor_on_device(hshape, ttnn.bfloat16, ttnn.TILE_LAYOUT, dev,
                                       ttnn.L1_MEMORY_CONFIG)
    MG.generic_minimal_matmul(dev, x, w1, [o], (block, grid), ckc4)
    ttnn.deallocate(o)


h1 = ttnn.linear(x, w1, activation="silu", compute_kernel_config=ckc,
                 memory_config=ttnn.L1_MEMORY_CONFIG, core_grid=T.CORE_GRID_MAIN)
h2 = ttnn.linear(x, w2, compute_kernel_config=ckc,
                 memory_config=ttnn.L1_MEMORY_CONFIG, core_grid=T.CORE_GRID_MAIN)


def mul_only():
    """`multiply_` on two resident L1 tensors: the op the fused epilogue replaces, alone."""
    o = ttnn.multiply(h1, h2, memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(o)


def incumbent():
    x1 = ttnn.linear(x, w1, activation="silu", compute_kernel_config=ckc,
                     memory_config=ttnn.L1_MEMORY_CONFIG, core_grid=T.CORE_GRID_MAIN)
    x2 = ttnn.linear(x, w2, compute_kernel_config=ckc,
                     memory_config=ttnn.L1_MEMORY_CONFIG, core_grid=T.CORE_GRID_MAIN)
    r = ttnn.multiply_(x1, x2)
    ttnn.deallocate(x2)
    return r


r = incumbent(); ref = ttnn.to_torch(r).float(); ttnn.deallocate(r)

parts = {}
for name, fn in (("linear_silu", lin(w1, "silu")), ("linear_plain", lin(w2, None)),
                 ("multiply_alone", mul_only), ("generic_matmul", generic)):
    m, sp = med(fn)
    parts[name] = {"ms": round(m, 5), "spread_pct": sp}
    print("%-16s %.5f ms  (spread %.1f%%)" % (name, m, sp), flush=True)

TS.BLOCK_KEYS = {(kt, nt): block}
TS._block_for.cache_clear()
rows = []
for spec in a.variants.split(";"):
    rnd, batch = (int(v) for v in spec.split("/"))
    TS.ROUND, TS.MUL_BATCH = rnd, batch

    def fused():
        o = TS.fused_swiglu(x, w2, w1, ckc4, grid)
        if o is None:
            raise RuntimeError("declined " + json.dumps({str(k): v for k, v in TS.REJECTS.items()}))
        return o

    def fused_t():
        ttnn.deallocate(fused())

    try:
        o = fused(); v = ttnn.to_torch(o).float(); ttnn.deallocate(o)
        pcc = float(((ref - ref.mean()) * (v - v.mean())).mean()
                    / (ref.std(unbiased=False) * v.std(unbiased=False)))
        max_abs = float((ref - v).abs().max())
        A, B = [], []
        for _ in range(a.reps):
            A.append(timed(incumbent_t := (lambda: ttnn.deallocate(incumbent()))))
            B.append(timed(fused_t))
    except Exception as e:                                                    # noqa: BLE001
        print("round=%d batch=%d FAILED %s" % (rnd, batch, str(e)[:160]), flush=True)
        continue
    ma, mb = st.median(A), st.median(B)
    epi = mb - 2 * parts["generic_matmul"]["ms"]
    rows.append({"round": rnd, "mul_batch": batch, "incumbent_ms": round(ma, 5),
                 "fused_ms": round(mb, 5), "ratio": round(ma / mb, 4), "pcc": pcc,
                 "max_abs": max_abs,
                 "epilogue_upper_ms": round(epi, 5),
                 "spread_pct": round(100 * (max(B) - min(B)) / mb, 2),
                 "per_transition_call_delta_ms": round(32 * (ma - mb), 4)})
    print("round=%d batch=%d  incumbent %.5f  fused %.5f  ratio %.4fx  pcc %.7f  "
          "epilogue<=%.5f  delta/transition_z %+.4f ms"
          % (rnd, batch, ma, mb, ma / mb, pcc, epi, rows[-1]["per_transition_call_delta_ms"]),
          flush=True)

a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps({"block": list(block), "grid": list(grid), "inner": a.inner,
                             "reps": a.reps, "parts": parts, "rows": rows}, indent=1))
print("wrote", a.out)
