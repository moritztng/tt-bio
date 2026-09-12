#!/usr/bin/env python3
"""Does the 1.81x move with the number of calls inside the timing window?

`b2z2-pairformer-megakernel-build`'s `mm_isolate.py` timed 4 calls between two device syncs. If the
generic_op call path costs more HOST time per call than the matmul costs DEVICE time, that window
measures the host and the ratio it reports is a dispatch ratio, not a kernel ratio. The signature
is a ratio that walks with the inner count and settles once the queue is deep enough to hide the
host. Same card, same shapes, same arms; only the inner count changes.
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


def linear():
    ttnn.deallocate(ttnn.linear(x, w, compute_kernel_config=ckc,
                                memory_config=ttnn.L1_MEMORY_CONFIG,
                                core_grid=T.CORE_GRID_MAIN))


def generic():
    o = ttnn.allocate_tensor_on_device(shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, dev,
                                       ttnn.L1_MEMORY_CONFIG)
    MG.generic_minimal_matmul(dev, x, w, [o], (BLOCK, grid), ckc4)
    ttnn.deallocate(o)


ARMS = [("linear", linear), ("generic", generic)]
INNERS = [1, 2, 4, 8, 16, 32, 64]


def timed(fn, inner):
    for _ in range(2):
        fn()
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(inner):
        fn()
    ttnn.synchronize_device(dev)
    return 1e3 * (time.perf_counter() - t0) / inner


rows = []
for inner in INNERS:
    got = {n: [] for n, _ in ARMS}
    for rep in range(a.reps):
        for name, fn in ARMS:
            got[name].append(timed(fn, inner))
    med = {n: st.median(v) for n, v in got.items()}
    rows.append({"inner": inner, "linear_ms": round(med["linear"], 5),
                 "generic_ms": round(med["generic"], 5),
                 "generic_vs_linear": round(med["generic"] / med["linear"], 4)})
    print("inner=%-3d linear %.5f  generic %.5f  generic/linear %.3fx"
          % (inner, med["linear"], med["generic"], med["generic"] / med["linear"]), flush=True)

a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps({"grid": list(grid), "block": list(BLOCK), "rows": rows}, indent=1))
print("wrote", a.out)
