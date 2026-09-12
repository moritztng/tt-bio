#!/usr/bin/env python3
"""The same three arms with the host removed: capture K back-to-back calls into a ttnn trace and
replay it.

`host_split.py` showed the generic_op arm costs 0.0586 ms of HOST time a call against
`ttnn.linear`'s 0.0215 ms, on a device op of ~0.08 ms. A wall-clock A/B with a small inner count
therefore measures the host, not the matmul -- which is what the 1.81x may actually be. Under
trace replay the host issues one command for the whole captured graph, so what is left is device
time only.
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
ap.add_argument("--calls", type=int, default=16, help="calls captured into one trace")
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


def linear():
    return ttnn.linear(x, w, compute_kernel_config=ckc, memory_config=ttnn.L1_MEMORY_CONFIG,
                       core_grid=T.CORE_GRID_MAIN)


def native():
    return ttnn.experimental.minimal_matmul(x, w, compute_kernel_config=ckc,
                                            memory_config=ttnn.L1_MEMORY_CONFIG, config=mmc)


def generic():
    MG.generic_minimal_matmul(dev, x, w, [OUT], (BLOCK, grid), ckc4)
    return OUT


ARMS = [("linear", linear), ("native", native), ("generic", generic), ("linear_aa", linear)]


def capture(fn):
    for _ in range(2):                       # compile + populate every lazy cache
        r = fn()
        if r is not OUT:
            ttnn.deallocate(r)
    ttnn.synchronize_device(dev)
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    keep = [fn() for _ in range(a.calls)]
    ttnn.end_trace_capture(dev, tid, cq_id=0)
    ttnn.synchronize_device(dev)
    return tid, keep


def replay(tid):
    ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(dev)
    return 1e3 * (time.perf_counter() - t0) / a.calls


traces, res = {}, {}
for name, fn in ARMS:
    traces[name] = capture(fn)
    res[name] = {"ms": []}
    print("captured %s" % name, flush=True)

for rep in range(a.reps):
    for name, _ in ARMS:
        res[name]["ms"].append(replay(traces[name][0]))
    print("rep %d: %s" % (rep, " ".join("%s=%.5f" % (n, res[n]["ms"][-1]) for n, _ in ARMS)),
          flush=True)

base = st.median(res["linear"]["ms"])
print("\n%-10s %9s %9s %8s" % ("arm", "ms/call", "vs lin", "spread%"))
for name, _ in ARMS:
    m = res[name]["ms"]
    med = st.median(m)
    res[name].update(median_ms=round(med, 6), vs_linear=round(base / med, 4),
                     spread_pct=round(100 * (max(m) - min(m)) / med, 2))
    print("%-10s %9.5f %8.4fx %8.2f" % (name, med, base / med, res[name]["spread_pct"]))
aa = st.median(res["linear_aa"]["ms"])
print("A/A floor: %.2f %%" % (100 * abs(aa - base) / base))
a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps({"grid": list(grid), "block": list(BLOCK), "calls": a.calls,
                             "aa_floor_pct": round(100 * abs(aa - base) / base, 3),
                             "arms": res}, indent=1))
print("wrote", a.out)
