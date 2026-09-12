#!/usr/bin/env python3
"""Replay one `generic_minimal_matmul` under the device profiler at a chosen chain split.

The A/B says a shorter chain makes the op SLOWER. The zones say why only if the same arm is
profiled: if splitting the chain really does cut the wait it was aimed at, and the op still slows
down, then that wait was never on the critical path. Run this twice, once per split, and feed each
log to `skew_split.py`.

Fences (`ttnn.exp` on a tiny tensor) bracket the replays so the window is findable in the ops CSV
exactly the way `kernel_census.py` does it.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--split", default="1")
    ap.add_argument("--m", type=int, default=8192)
    ap.add_argument("--n", type=int, default=384)
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()

    os.environ["TT_BIO_MM_CHAIN_SPLIT"] = a.split
    import torch
    import ttnn
    import tt_bio.tenstorrent as T
    import tt_bio.mm_generic as MG

    dev = T.get_device()
    grid = tuple(T.COMPUTE_GRID_MAIN)
    cfg_block = (4, 4, 1, 4, 1)
    ckc = (ttnn.MathFidelity.HiFi4, False, False, False)
    torch.manual_seed(0)
    x = ttnn.from_torch(torch.randn(a.m, a.k).bfloat16(), layout=ttnn.TILE_LAYOUT,
                        dtype=ttnn.bfloat16, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    w = ttnn.from_torch(torch.randn(a.k, a.n).bfloat16(), layout=ttnn.TILE_LAYOUT,
                        dtype=ttnn.bfloat16, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    fence_t = ttnn.from_torch(torch.zeros(32, 32).bfloat16(), layout=ttnn.TILE_LAYOUT,
                              dtype=ttnn.bfloat16, device=dev,
                              memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def fence():
        for _ in range(3):
            ttnn.deallocate(ttnn.exp(fence_t))
        ttnn.synchronize_device(dev)

    def run():
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([a.m, a.n]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG)
        MG.generic_minimal_matmul(dev, x, w, out, (cfg_block, grid), ckc,
                                  kernel_dir=MG._ARMED_KERNEL_DIR)
        ttnn.deallocate(out)

    for _ in range(3):
        run()
    ttnn.synchronize_device(dev)
    fence()
    t0 = time.perf_counter()
    for _ in range(a.reps):
        run()
    ttnn.synchronize_device(dev)
    wall = 1e3 * (time.perf_counter() - t0) / a.reps
    fence()
    res = {"split": a.split, "reps": a.reps, "grid": list(grid), "m": a.m, "n": a.n, "k": a.k,
           "card": os.environ.get("TT_VISIBLE_DEVICES"),
           "chain_split_built": sorted({str(e["dims"].get("chain_split"))
                                        for e in MG._CACHE.values()}),
           "wall_ms_per_call": round(wall, 5)}
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
