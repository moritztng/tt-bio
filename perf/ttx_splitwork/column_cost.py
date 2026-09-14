#!/usr/bin/env python3
"""Is one column of the compute grid slower than the others?

`ab_corehole.py` found the full 11x10 grid faster than a 10x10 sub-grid at 400 groups (1.15-1.21x)
and SLOWER at 900 (0.86-0.89x), both legs agreeing within a shape, with a 0.3% A/A floor. Per-core
depth explains neither: it is 4 and 4, then 9 and 9.

One mechanism fits both signs. Suppose the cores in the 11th column are slower per group. The
column-wise split hands group 1 -- the cores carrying the extra group -- to the LEFT of the grid,
so at 400 groups the 11th column sits in group 2 with 3 groups while columns 0-6 carry 4, and the
slow cores are under-loaded. At 900 groups group 1 is only the first 20 cores, so the 11th column
carries 8 against the fastest cores' 9, and 8 slow groups beat 9 fast ones only if the column is
less than 12.5% slower. Above that the 11th column is the critical path and the whole grid loses.

This measures it directly: the same unit count run on ONE column of 10 cores at a time, eleven
times, interleaved. A slow column shows up as a slow column.

Usage: TT_VISIBLE_DEVICES=1 ... python perf/ttx_splitwork/column_cost.py
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch
import ttnn

from tt_bio import core_split
from tt_bio import reblock_permute as rp
from tt_bio.tenstorrent import get_device


def _plan_from_cores(grid, units, cores):
    """A `_split_plan` tuple over an explicit, ordered list of (x, y) cores."""
    n, n1, w1, w2 = core_split.units_per_core(units, len(cores))
    def crs(sub):
        return ttnn.CoreRangeSet([
            ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in sub
        ])
    used = cores[:n]
    return (grid.x, grid.y, (n, crs(used), crs(used[:n1]), crs(used[n1:]), w1, w2))


def _timed(device, fn, x, iters):
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn(x)
        ttnn.deallocate(out)
    ttnn.synchronize_device(device)
    return (time.perf_counter() - t0) * 1e3 / iters


def _median(v):
    s = sorted(v)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--N", type=int, default=640)
    ap.add_argument("--C", type=int, default=32)
    ap.add_argument("--out", default="perf/ttx_splitwork/column_cost.json")
    a = ap.parse_args()

    device = get_device()
    g = device.compute_with_storage_grid_size()
    dram = ttnn.DRAM_MEMORY_CONFIG
    N, C = a.N, a.C
    nt = N // 32
    units = nt * nt
    x_t = torch.randn([1, N, N, C], dtype=torch.bfloat16)
    x = ttnn.from_torch(x_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                        device=device, memory_config=dram)
    ref_t = ttnn.to_torch(ttnn.permute(x, (0, 3, 1, 2), memory_config=dram))
    op = rp.reblock_permute

    # One candidate core set per measurement: each single column, then the two 100-core halves
    # the sign flip is about, then the whole grid.
    sets = {f"col{cx}": [(cx, cy) for cy in range(g.y)] for cx in range(g.x)}
    sets["cols0-9"] = [(cx, cy) for cx in range(10) for cy in range(g.y)]
    sets["cols1-10"] = [(cx, cy) for cx in range(1, 11) for cy in range(g.y)]
    sets["full"] = [(cx, cy) for cx in range(g.x) for cy in range(g.y)]

    caches = {k: {} for k in sets}
    times = {k: [] for k in sets}
    exact = {}
    try:
        for k, cores in sets.items():
            rp._split_plan = lambda d, u, _c=cores: _plan_from_cores(
                d.compute_with_storage_grid_size(), u, _c)
            rp._CACHE, rp._SPLIT_CACHE = caches[k], {}
            o = op(x, memory_config=dram)
            exact[k] = bool(torch.equal(ttnn.to_torch(o), ref_t))
            ttnn.deallocate(o)
        for _ in range(a.reps):
            for k, cores in sets.items():
                rp._split_plan = lambda d, u, _c=cores: _plan_from_cores(
                    d.compute_with_storage_grid_size(), u, _c)
                rp._CACHE, rp._SPLIT_CACHE = caches[k], {}
                times[k].append(_timed(device, lambda z: op(z, memory_config=dram), x, a.iters))
        cols = [_median(times[f"col{cx}"]) for cx in range(g.x)]
        best = min(cols)
        print(f"fwd N={N} C={C} units={units} on {g.x}x{g.y}, one column at a time "
              f"({g.y} cores, depth {-(-units // g.y)})\n")
        print(f"  {'cores':<10}{'ms':>10}{'vs fastest column':>20}  exact")
        for cx in range(g.x):
            print(f"  col {cx:<6}{cols[cx]:>10.4f}{cols[cx] / best:>20.4f}  "
                  f"{exact[f'col{cx}']}")
        for k in ("cols0-9", "cols1-10", "full"):
            m = _median(times[k])
            print(f"  {k:<10}{m:>10.4f}{'':>20}  {exact[k]}")
        m09, m110 = _median(times["cols0-9"]), _median(times["cols1-10"])
        print(f"\n  columns 1-10 / columns 0-9 = {m110 / m09:.4f}x "
              f"(same 100 cores, shifted one column right)")
    finally:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump({"grid": [g.x, g.y], "N": N, "C": C, "units": units,
                       "reps": a.reps, "iters": a.iters,
                       "ms": {k: _median(v) for k, v in times.items()},
                       "all": times, "bit_exact": exact}, f, indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
