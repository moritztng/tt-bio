#!/usr/bin/env python3
"""What the generic_op call path costs above the dispatch itself, and where.

`rebind_probe.py` ruled out the ProgramDescriptor reconstruction. What is left in
`generic_minimal_matmul` before `ttnn.generic_op` is `_key()`, which builds a cache key out of
`str(padded_shape)`, `str(dtype)` and `str(memory_config())` for in0, in1 and every output, on
every call. Three arms, fresh output each, and the host-only time next to the wall time.
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
ap.add_argument("--block", type=str, default="8,2,1,4,1")
ap.add_argument("--grid", type=str, default="8x8")
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
BLOCK = tuple(int(v) for v in a.block.split(","))
GRID = tuple(int(v) for v in a.grid.split("x"))

torch.manual_seed(0)
up = lambda t, mc: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                   device=dev, memory_config=mc)
x = up(torch.randn(1, a.rows, a.n, a.c, dtype=torch.bfloat16) * 0.5, ttnn.L1_MEMORY_CONFIG)
w = up(torch.randn(a.c, a.hidden, dtype=torch.bfloat16) * 0.05, ttnn.DRAM_MEMORY_CONFIG)
shape = ttnn.Shape([1, a.rows, a.n, a.hidden])
alloc = lambda: ttnn.allocate_tensor_on_device(shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, dev,
                                               ttnn.L1_MEMORY_CONFIG)

o0 = alloc()
MG.generic_minimal_matmul(dev, x, w, [o0], (BLOCK, GRID), ckc4)
entry = MG._CACHE[MG._key(x, w, [o0], (BLOCK, GRID), ckc4, (), None, None, None)]
ttnn.deallocate(o0)


def api():
    o = alloc()
    MG.generic_minimal_matmul(dev, x, w, [o], (BLOCK, GRID), ckc4)
    ttnn.deallocate(o)


def direct():
    o = alloc()
    addrs = (x.buffer_address(), w.buffer_address(), (o.buffer_address(),))
    if addrs != entry["addrs"]:
        MG.rebind(entry, *addrs)
    ttnn.generic_op([x, w, o], entry["pd"])
    ttnn.deallocate(o)


def keyonly():
    o = alloc()
    MG._key(x, w, [o], (BLOCK, GRID), ckc4, (), None, None, None)
    ttnn.deallocate(o)


def linear():
    ttnn.deallocate(ttnn.linear(x, w, compute_kernel_config=ckc,
                                memory_config=ttnn.L1_MEMORY_CONFIG,
                                core_grid=T.CORE_GRID_MAIN))


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
    fn()
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(a.inner):
        fn()
    dt = 1e3 * (time.perf_counter() - t0) / a.inner
    ttnn.synchronize_device(dev)
    return dt


ARMS = [("linear", linear), ("api", api), ("direct", direct), ("keyonly_host", keyonly)]
res = {n: {"ms": [], "host_ms": []} for n, _ in ARMS}
for _ in range(a.reps):
    for n, f in ARMS:
        res[n]["ms"].append(timed(f))
for _ in range(3):
    for n, f in ARMS:
        res[n]["host_ms"].append(host_only(f))

base = st.median(res["linear"]["ms"])
print("%-14s %9s %9s %9s" % ("arm", "ms", "vs linear", "host_ms"))
for n, _ in ARMS:
    m, h = st.median(res[n]["ms"]), st.median(res[n]["host_ms"])
    res[n].update(median_ms=round(m, 5), host_median_ms=round(h, 5),
                  vs_linear=round(base / m, 4),
                  spread_pct=round(100 * (max(res[n]["ms"]) - min(res[n]["ms"])) / m, 2))
    print("%-14s %9.5f %8.4fx %9.5f" % (n, m, base / m, h))
a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps({"block": list(BLOCK), "grid": list(GRID), "inner": a.inner,
                             "reps": a.reps, "arms": res}, indent=1))
print("wrote", a.out)
