#!/usr/bin/env python3
"""What the full grid is worth on the bands where the wheel's work split used to throw.

Arm A is the shipped behaviour: `ttnn.split_work_to_cores` on the full grid, and when it raises
`TT_FATAL @ work_split.cpp:305`, the largest rectangular sub-grid it will accept. Arm B is
`tt_bio.core_split`, which always gets the whole grid. The two arms compute the same thing on a
different set of cores, so the output is expected bit-identical and is checked with `torch.equal`
against `ttnn.permute` on both arms, not just one.

Arm A is run twice per rep, in the same interleave, so the table carries its own A/A floor: any
B/A ratio inside the A2/A spread is noise.

The prediction under test is the DEPTH ratio, not the core ratio: 1600 groups on a 10x10 sub-grid
is 16 per busiest core, on the full 11x10 grid it is 15, so 1.0667x is the most the band can give.
Where the sub-grid is already as deep as the full grid (400 groups: 4 either way) the prediction is
1.0000x however many cores sat idle -- `core-coverage-ratio-not-recoverable-time`.

Usage: TT_VISIBLE_DEVICES=1 ... python perf/ttx_splitwork/ab_corehole.py [--reps 5] [--iters 20]
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

# (leg, N, C). `back` moves [1,C,N,N] -> [1,N,N,C] and splits Nt*Nt*Ct groups; `fwd` moves
# [1,N,N,C] -> [1,C,N,N] and splits Nt*Nt. Every row below is a unit count the wheel refuses,
# except the two marked as controls.
ROWS = [
    # The production width on the DRAM path is C = 32: above `_trimul_l1_max_seq` the trimul chunk
    # is pinned at TRIANGLE_MULT_CHUNK_SIZE, so these are the shapes a 640 or 960 aa fold issues.
    ("fwd",  512, 32),    #  256 units: NOT a hole, both arms identical -> the session A/A floor
    ("back", 512, 32),    #  256 units: NOT a hole
    ("fwd",  640, 32),    #  400 units, 10x10 sub-grid, depth 4 vs 4  -> depth predicts 1.0000
    ("back", 640, 32),    #  400 units, same split, the other direction
    ("fwd",  960, 32),    #  900 units, 10x10, depth 9 vs 9           -> depth predicts 1.0000
    ("back", 960, 32),    #  900 units, same split
    # Wider chunks are the L1 path, which the window closes at 352, so these are not production
    # shapes on this grid. Kept because they are the largest depth deltas the census found.
    ("back", 640, 128),   # 1600 units, 10x10, depth 16 vs 15
    ("back", 480, 256),   # 1800 units, 10x10, depth 18 vs 17
]



def _shipped_split_plan(device, units):
    """`_split_plan` as it shipped before `core_split`: the wheel, then the sub-grid search."""
    g = device.compute_with_storage_grid_size()

    def hole(cores, height, u):
        r = u % cores
        return u > cores and r != 0 and r % height == 0

    candidates = [(g.x, g.y)] + sorted(
        ((sx, sy) for sy in range(1, g.y + 1) for sx in range(1, g.x + 1)
         if (sx, sy) != (g.x, g.y) and not hole(sx * sy, sy, units)),
        key=lambda s: -s[0] * s[1],
    )
    for sx, sy in candidates:
        crs = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(sx - 1, sy - 1))]
        )
        try:
            return (sx, sy, ttnn.split_work_to_cores(crs, units))
        except Exception:                                              # noqa: BLE001 -- TT_FATAL
            continue
    return None


def _core_split_plan(device, units):
    g = device.compute_with_storage_grid_size()
    return (g.x, g.y, core_split.split_work_to_cores(g, units)) if units > 0 else None


class Arm:
    """One arm's split policy plus its own descriptor caches, so switching arms costs nothing."""

    def __init__(self, name, plan_fn):
        self.name, self.plan_fn = name, plan_fn
        self.fwd_cache, self.back_cache, self.split_cache = {}, {}, {}

    def install(self):
        rp._split_plan = self.plan_fn
        rp._CACHE, rp._CACHE_BACK, rp._SPLIT_CACHE = (
            self.fwd_cache, self.back_cache, self.split_cache
        )


def _grid_of(arm, device, units):
    plan = arm.plan_fn(device, units)
    sx, sy, (n, _all, _g1, _g2, w1, _w2) = plan
    return f"{sx}x{sy}", n, w1


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
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--out", default="perf/ttx_splitwork/ab_corehole.json")
    a = ap.parse_args()

    device = get_device()
    g = device.compute_with_storage_grid_size()
    arm_a = Arm("subgrid", _shipped_split_plan)
    arm_b = Arm("fullgrid", _core_split_plan)
    dram = ttnn.DRAM_MEMORY_CONFIG
    results = []
    try:
        print(f"grid {g.x}x{g.y} = {g.x * g.y} cores, reps={a.reps} iters={a.iters}\n")
        for leg, N, C in ROWS:
            nt, ct = N // 32, C // 32
            units = nt * nt * ct if leg == "back" else nt * nt
            shape = [1, C, N, N] if leg == "back" else [1, N, N, C]
            perm = (0, 2, 3, 1) if leg == "back" else (0, 3, 1, 2)
            op = rp.reblock_permute_back if leg == "back" else rp.reblock_permute

            t = torch.randn(shape, dtype=torch.bfloat16)
            x = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                device=device, memory_config=dram)
            ref_t = ttnn.to_torch(ttnn.permute(x, perm, memory_config=dram))

            grids, exact = {}, {}
            for arm in (arm_a, arm_b):
                arm.install()
                grids[arm.name] = _grid_of(arm, device, units)
                out = op(x, memory_config=dram)
                exact[arm.name] = bool(torch.equal(ttnn.to_torch(out), ref_t))
                ttnn.deallocate(out)

            ta, tb, ta2, tp = [], [], [], []
            for rep in range(a.reps):
                arm_a.install()
                ta.append(_timed(device, lambda z: op(z, memory_config=dram), x, a.iters))
                arm_b.install()
                tb.append(_timed(device, lambda z: op(z, memory_config=dram), x, a.iters))
                arm_a.install()
                ta2.append(_timed(device, lambda z: op(z, memory_config=dram), x, a.iters))
                # `ttnn.permute` is context, not an arm -- it is an order of magnitude slower and
                # would otherwise own the wall clock. Three reps of it is enough for a magnitude.
                if rep < 3:
                    tp.append(_timed(
                        device, lambda z: ttnn.permute(z, perm, memory_config=dram), x, a.iters))
            ttnn.deallocate(x)
            del ref_t, t

            ma, mb, ma2, mp = _median(ta), _median(tb), _median(ta2), _median(tp)
            (ga, na, wa), (gb, nb, wb) = grids["subgrid"], grids["fullgrid"]
            row = {
                "leg": leg, "N": N, "C": C, "units": units,
                "subgrid": ga, "subgrid_cores": na, "subgrid_depth": wa,
                "fullgrid": gb, "fullgrid_cores": nb, "fullgrid_depth": wb,
                "depth_ratio": wa / wb,
                "ms_subgrid": ma, "ms_fullgrid": mb, "ms_subgrid_again": ma2,
                "ms_ttnn_permute": mp,
                "speedup_full_over_sub": ma / mb, "aa_floor": ma2 / ma,
                "leg_over_permute_sub": mp / ma, "leg_over_permute_full": mp / mb,
                "bit_exact_subgrid": exact["subgrid"], "bit_exact_fullgrid": exact["fullgrid"],
                "ms_subgrid_all": ta, "ms_fullgrid_all": tb, "ms_subgrid_again_all": ta2,
            }
            results.append(row)
            print(f"{leg} N={N} C={C} units={units}: "
                  f"A {ga} {na}c depth {wa} = {ma:.4f} ms | "
                  f"B {gb} {nb}c depth {wb} = {mb:.4f} ms | "
                  f"B/A {ma / mb:.4f}x (predicted {wa / wb:.4f}x, A/A {ma2 / ma:.4f}x) | "
                  f"permute {mp:.4f} ms | exact A={exact['subgrid']} B={exact['fullgrid']}")
    finally:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump({"grid": [g.x, g.y], "reps": a.reps, "iters": a.iters,
                       "rows": results}, f, indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
