#!/usr/bin/env python3
"""Split the generic_op arm's residual against native minimal_matmul at the SAME program.

`gap_split.py` measured (WH, whglx card 0, 8x9): ttnn.linear 0.0822 ms, native minimal_matmul with
block (4,4,1,4,1) 0.0871 ms, the generic_op transcription of that same program 0.1144 ms. The
program is therefore nearly free of the gap and the residual is the HOST. Two candidates in the
transcription's call path, separated here:

  generic_rebind  allocate a fresh output every call -> address changes -> `rebind()` rewrites a
                  few hundred runtime args AND reconstructs the whole ProgramDescriptor
  generic_fixed   one output tensor allocated once and reused -> addresses stable -> no rebind
  generic_noalloc as above, and no per-call allocate/deallocate either

If generic_fixed lands on native, the gap is rebinding and it is a call-path fix, not a kernel one.
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
ap.add_argument("--inner", type=int, default=16)
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
x = ttnn.from_torch(torch.randn(1, a.rows, a.n, a.c, dtype=torch.bfloat16) * 0.5,
                    dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                    memory_config=ttnn.L1_MEMORY_CONFIG)
w = ttnn.from_torch(torch.randn(a.c, a.hidden, dtype=torch.bfloat16) * 0.05,
                    dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG)
shape = ttnn.Shape([1, a.rows, a.n, a.hidden])
MMC = getattr(ttnn.experimental, "MinimalMatmulConfig", None) or ttnn.MinimalMatmulConfig
mmc = MMC(M_block_size=BLOCK[0], K_block_size=BLOCK[1], N_block_size=BLOCK[2],
          subblock_h=BLOCK[3], subblock_w=BLOCK[4],
          compute_with_storage_grid_size=ttnn.CoreCoord(grid[0], grid[1]))

OUT = ttnn.allocate_tensor_on_device(shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, dev,
                                     ttnn.L1_MEMORY_CONFIG)


def linear(keep=False):
    r = ttnn.linear(x, w, compute_kernel_config=ckc, memory_config=ttnn.L1_MEMORY_CONFIG,
                    core_grid=T.CORE_GRID_MAIN)
    return r if keep else ttnn.deallocate(r)


def native(keep=False):
    r = ttnn.experimental.minimal_matmul(x, w, compute_kernel_config=ckc,
                                         memory_config=ttnn.L1_MEMORY_CONFIG, config=mmc)
    return r if keep else ttnn.deallocate(r)


def generic_rebind(keep=False):
    o = ttnn.allocate_tensor_on_device(shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, dev,
                                       ttnn.L1_MEMORY_CONFIG)
    MG.generic_minimal_matmul(dev, x, w, [o], (BLOCK, grid), ckc4)
    return o if keep else ttnn.deallocate(o)


def generic_fixed(keep=False):
    MG.generic_minimal_matmul(dev, x, w, [OUT], (BLOCK, grid), ckc4)
    return OUT


ARMS = [("linear", linear), ("native", native), ("generic_rebind", generic_rebind),
        ("generic_fixed", generic_fixed), ("linear_aa", linear)]


def timed(fn):
    for _ in range(2):
        fn()
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(a.inner):
        fn()
    ttnn.synchronize_device(dev)
    return 1e3 * (time.perf_counter() - t0) / a.inner


def host_only(fn):
    """Per-call host time with the device queue drained first and no sync inside the loop.

    If this lands on the timed() number the arm is host-bound; if it is well below, device-bound.
    """
    fn()
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(a.inner):
        fn()
    dt = 1e3 * (time.perf_counter() - t0) / a.inner
    ttnn.synchronize_device(dev)
    return dt


ref = ttnn.to_torch(linear(keep=True)).float()
res = {}
for name, fn in ARMS:
    v = ttnn.to_torch(fn(keep=True)).float()
    res[name] = {"exact": bool(torch.equal(ref, v)),
                 "max_abs": float((ref - v).abs().max()), "ms": [], "host_ms": []}

for rep in range(a.reps):
    for name, fn in ARMS:
        res[name]["ms"].append(timed(fn))
    print("rep %d: %s" % (rep, " ".join("%s=%.4f" % (n, res[n]["ms"][-1]) for n, _ in ARMS)),
          flush=True)
for rep in range(3):
    for name, fn in ARMS:
        res[name]["host_ms"].append(host_only(fn))

base = st.median(res["linear"]["ms"])
print("\n%-15s %8s %8s %9s %9s %s" % ("arm", "ms", "vs lin", "spread%", "host_ms", "bit-exact"))
for name, _ in ARMS:
    m = res[name]["ms"]
    med, hmed = st.median(m), st.median(res[name]["host_ms"])
    res[name].update(median_ms=round(med, 5), vs_linear=round(base / med, 4),
                     spread_pct=round(100 * (max(m) - min(m)) / med, 2),
                     host_median_ms=round(hmed, 5))
    print("%-15s %8.4f %8.4fx %8.2f %9.4f %s"
          % (name, med, base / med, res[name]["spread_pct"], hmed, res[name]["exact"]))
aa = st.median(res["linear_aa"]["ms"])
print("A/A floor: %.2f %%" % (100 * abs(aa - base) / base))
a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps({"grid": list(grid), "block": list(BLOCK), "inner": a.inner,
                             "aa_floor_pct": round(100 * abs(aa - base) / base, 3),
                             "arms": res}, indent=1))
print("wrote", a.out)
