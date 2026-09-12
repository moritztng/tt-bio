#!/usr/bin/env python3
"""Where the generic_op-vs-ttnn.linear matmul gap lives, split into its candidate terms.

`b2z2-pairformer-megakernel-build` measured the fused SwiGLU riding on a matmul 1.81x slower than
`ttnn.linear` at the production shape, which is what kills the fusion route. `tt_bio/mm_generic.py`
is a transcription of `minimal_matmul_program_factory.cpp`, so the first thing to separate is
whether the gap belongs to `ttnn.generic_op` (the host) or to `minimal_matmul` (the program it
hosts). Arms, all at the same shape, same operands, same compute kernel config, interleaved in one
process:

  linear        ttnn.linear with the production core_grid -- the incumbent
  linear_auto   ttnn.linear with no core_grid, so ttnn picks the program config itself
  mm_native     ttnn.experimental.minimal_matmul, library defaults
  mm_native_cfg ttnn.experimental.minimal_matmul with the block config the transcription uses
  mm_generic    the generic_op transcription with that same block config

mm_native_cfg vs mm_generic is the host term. mm_native vs linear is the program term.
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
ap.add_argument("--reps", type=int, default=7)
ap.add_argument("--inner", type=int, default=8)
ap.add_argument("--block", type=str, default="4,4,1,4,1")
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
grid = tuple(T.COMPUTE_GRID_MAIN)
BLOCK = tuple(int(v) for v in a.block.split(","))

torch.manual_seed(0)
xt = torch.randn(1, a.rows, a.n, a.c, dtype=torch.bfloat16) * 0.5
wt = torch.randn(a.c, a.hidden, dtype=torch.bfloat16) * 0.05
up = lambda t, mc: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                   device=dev, memory_config=mc)
x = up(xt, ttnn.L1_MEMORY_CONFIG)
w = up(wt, ttnn.DRAM_MEMORY_CONFIG)
shape = ttnn.Shape([1, a.rows, a.n, a.hidden])

mt, kt, nt = a.rows * a.n // 32, a.c // 32, a.hidden // 32
gx, gy = grid
transpose = (a.rows * a.n) > a.hidden
in0_axis, in1_axis = (gx, gy) if transpose else (gy, gx)
pad_m = -(-mt // in0_axis) * in0_axis
pad_n = -(-nt // in1_axis) * in1_axis
geom = {"grid": [gx, gy], "M_tiles": mt, "K_tiles": kt, "N_tiles": nt, "transpose": transpose,
        "padded_M_tiles": pad_m, "padded_N_tiles": pad_n,
        "pad_waste": round(pad_m * pad_n / (mt * nt), 4),
        "M_tiles_per_core": pad_m // in0_axis, "N_tiles_per_core": pad_n // in1_axis}
print("geometry:", json.dumps(geom), flush=True)

MMC = ttnn.experimental.MinimalMatmulConfig if hasattr(ttnn.experimental, "MinimalMatmulConfig") \
    else getattr(ttnn, "MinimalMatmulConfig", None)


def arm_linear():
    return ttnn.linear(x, w, compute_kernel_config=ckc, memory_config=ttnn.L1_MEMORY_CONFIG,
                       core_grid=T.CORE_GRID_MAIN)


def arm_linear_auto():
    return ttnn.linear(x, w, compute_kernel_config=ckc, memory_config=ttnn.L1_MEMORY_CONFIG)


def arm_mm_native():
    return ttnn.experimental.minimal_matmul(x, w, compute_kernel_config=ckc,
                                            memory_config=ttnn.L1_MEMORY_CONFIG)


def _mmc(block, g):
    M, K, N, sh, sw = block
    return MMC(M_block_size=M, K_block_size=K, N_block_size=N, subblock_h=sh, subblock_w=sw,
               compute_with_storage_grid_size=ttnn.CoreCoord(g[0], g[1]))


def arm_mm_native_cfg():
    return ttnn.experimental.minimal_matmul(x, w, compute_kernel_config=ckc,
                                            memory_config=ttnn.L1_MEMORY_CONFIG,
                                            config=_mmc(BLOCK, grid))


def arm_mm_generic():
    o = ttnn.allocate_tensor_on_device(shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, dev,
                                       ttnn.L1_MEMORY_CONFIG)
    MG.generic_minimal_matmul(dev, x, w, [o], (BLOCK, grid), ckc4)
    return o


ARMS = [("linear", arm_linear), ("linear_auto", arm_linear_auto), ("mm_native", arm_mm_native),
        ("mm_native_cfg", arm_mm_native_cfg), ("mm_generic", arm_mm_generic),
        ("linear_aa", arm_linear)]


def timed(fn):
    for _ in range(2):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(a.inner):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(dev)
    return 1e3 * (time.perf_counter() - t0) / a.inner


ref = ttnn.to_torch(arm_linear()).float()
res = {}
for name, fn in ARMS:
    try:
        r = fn()
        v = ttnn.to_torch(r).float()
        ttnn.deallocate(r)
        res[name] = {"exact": bool(torch.equal(ref, v)),
                     "pcc": float(((ref - ref.mean()) * (v - v.mean())).mean()
                                  / (ref.std(unbiased=False) * v.std(unbiased=False))),
                     "max_abs": float((ref - v).abs().max()), "ms": []}
    except Exception as e:                                                    # noqa: BLE001
        print("%-14s UNAVAILABLE %s" % (name, str(e)[:160]), flush=True)
        res[name] = {"error": str(e)[:400], "ms": []}

live = [(n, f) for n, f in ARMS if "error" not in res[n]]
for rep in range(a.reps):
    for name, fn in live:
        res[name]["ms"].append(timed(fn))
    print("rep %d: %s" % (rep, " ".join("%s=%.4f" % (n, res[n]["ms"][-1]) for n, _ in live)),
          flush=True)

base = st.median(res["linear"]["ms"])
print("\n%-14s %8s %8s %9s %9s %s" % ("arm", "ms", "vs lin", "spread%", "pcc", "bit-exact"))
for name, _ in live:
    m = res[name]["ms"]
    med = st.median(m)
    res[name].update(median_ms=round(med, 5), vs_linear=round(base / med, 4),
                     spread_pct=round(100 * (max(m) - min(m)) / med, 2))
    print("%-14s %8.4f %8.4fx %8.2f %9.7f %s"
          % (name, med, base / med, res[name]["spread_pct"], res[name].get("pcc", float("nan")),
             res[name].get("exact")))
aa = st.median(res["linear_aa"]["ms"])
print("A/A floor: %.2f %%" % (100 * abs(aa - base) / base))
a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps({"geom": geom, "block": list(BLOCK), "inner": a.inner,
                             "reps": a.reps, "aa_floor_pct": round(100 * abs(aa - base) / base, 3),
                             "arms": res}, indent=1))
print("wrote", a.out)
