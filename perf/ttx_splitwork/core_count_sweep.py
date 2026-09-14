#!/usr/bin/env python3
"""Time the channel move against the NUMBER OF CORES it is split over, one shape at a time.

`ab_corehole.py` measured the full grid against the sub-grid the wheel's bug forced us onto, and
the per-core depth ratio predicted neither the sign nor the size: 400 groups gained 1.15-1.21x at
an unchanged depth of 4, and 900 groups lost 11-14% at an unchanged depth of 9. So the core count
is a knob in its own right, and this sweeps it with the distribution rule held fixed.

**The control is a sandwich, and it had to be.** A first version measured the caps in a fixed order
with one control at the end of each rep. At 81 units -- where every cap above 81 clamps to the same
81 cores, so seven points were the IDENTICAL configuration -- those seven read 0.1081, 0.1082,
0.1080, 0.1074, 0.0636, 0.0638, 0.0630 ms. A 1.70x spread on one configuration: position in the
rep, not core count. So every cap here is bracketed by a full-grid control measured immediately
before and after it, and reported against the mean of its own two neighbours. A cap whose two
brackets disagree by more than `--bracket-tol` is printed as unusable rather than as a number.

Usage: TT_VISIBLE_DEVICES=1 ... python perf/ttx_splitwork/core_count_sweep.py
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

# The three legs differ only in what one call looks like; the split and the arms are shared.
GATED_SLICE_C = 128


def _make(leg, N, C, device, dram):
    """``(x, run_one, reference)`` for one leg: the input, a one-call closure, a bit-exact ref.

    The forward and back legs check against `ttnn.permute`, which is their standing claim. The
    gated leg has no one-op equivalent, so its reference is the control arm's own output: what is
    under test here is that the assignment does not move a value, not the kernel's arithmetic,
    which `perf/b2z_sdpa_floor/gategran_512.py` already owns.
    """
    import ttnn as _tt
    if leg == "gated":
        xw = _tt.from_torch(torch.randn([1, N, N, 4 * GATED_SLICE_C], dtype=torch.bfloat16),
                            dtype=_tt.bfloat16, layout=_tt.TILE_LAYOUT, device=device,
                            memory_config=dram)
        return xw, (lambda z: rp.reblock_permute_gated(z, 0, GATED_SLICE_C, GATED_SLICE_C,
                                                       memory_config=dram)), None
    shape = [1, C, N, N] if leg == "back" else [1, N, N, C]
    perm = (0, 2, 3, 1) if leg == "back" else (0, 3, 1, 2)
    op = rp.reblock_permute_back if leg == "back" else rp.reblock_permute
    x = _tt.from_torch(torch.randn(shape, dtype=torch.bfloat16), dtype=_tt.bfloat16,
                       layout=_tt.TILE_LAYOUT, device=device, memory_config=dram)
    return x, (lambda z: op(z, memory_config=dram)), _tt.to_torch(_tt.permute(x, perm,
                                                                             memory_config=dram))


def _units(leg, N, C):
    nt, ct = N // 32, C // 32
    if leg == "back":
        return nt * nt * ct
    if leg == "gated":
        return nt * nt * (GATED_SLICE_C // 32)
    return nt * nt


SHAPES = [
    ("fwd",  640, 32),    # production, 400 units, a hole: the full grid WINS 1.15x over 10x10
    ("fwd",  960, 32),    # production, 900 units, a hole: the full grid LOSES 0.89x
    ("back", 960, 32),    # the same unit count, the other direction, which agreed with it
    ("fwd",  512, 32),    # 256 units, NOT a hole: is the knob a hole-band thing or general?
    # The two shapes a 512 aa Boltz-2 fold ACTUALLY asks for, from
    # `perf/ttx_splitwork/shape_census.py`: 1120 gated calls and 560 back calls, and not one
    # forward call. Both are 1024 groups. Nothing else in the list above is on that path.
    ("gated", 512, 512),
    ("back",  512, 128),
]


def _plan_for(grid, units, cap):
    """The column-wise split restricted to the first ``cap`` cores of ``grid``."""
    height = int(grid.y)
    n, n1, w1, w2 = core_split.units_per_core(units, min(cap, core_split.core_count(grid)))
    crs = ttnn.CoreRangeSet
    return (grid.x, grid.y, (
        n,
        crs(core_split.column_ranges(0, n, height)),
        crs(core_split.column_ranges(0, n1, height)),
        crs(core_split.column_ranges(n1, n - n1, height)),
        w1, w2,
    ))


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
    ap.add_argument("--reps", type=int, default=9)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--caps", default="60,70,80,88,90,95,99,100,104,108,110")
    ap.add_argument("--modes", default="block,stride,rotate",
                    help="group walks to measure; the control is always `block` at the full grid, "
                         "which is what main shipped")
    ap.add_argument("--bracket-tol", type=float, default=0.03)
    ap.add_argument("--out", default="perf/ttx_splitwork/core_count_sweep.json")
    a = ap.parse_args()

    device = get_device()
    g = device.compute_with_storage_grid_size()
    full = core_split.core_count(g)
    dram = ttnn.DRAM_MEMORY_CONFIG
    out_rows = []
    try:
        print(f"grid {g.x}x{g.y} = {full} cores, reps={a.reps} iters={a.iters}, "
              f"bracket tolerance {a.bracket_tol:.1%}")
        for leg, N, C in SHAPES:
            units = _units(leg, N, C)
            # Every cap above `units` is the same split, so keep exactly one of them.
            modes = a.modes.split(",")
            caps = sorted({min(c, units, full) for c in [int(v) for v in a.caps.split(",")]}
                          | {min(units, full)})
            # An arm is (core cap, strided). The control is the whole grid in blocked order: that
            # is exactly what main ships, so every ratio below reads against today's default.
            arms = [(c, w) for w in modes for c in caps]
            ctl = (min(units, full), "block")
            x, call, ref_t = _make(leg, N, C, device, dram)

            caches = {arm: {} for arm in set(arms) | {ctl}}

            def use(arm):
                cap, rp.WALK = arm
                rp._split_plan = lambda d, u, _c=cap: _plan_for(
                    d.compute_with_storage_grid_size(), u, _c)
                rp._CACHE = rp._CACHE_BACK = rp._CACHE_GATED = caches[arm]
                rp._SPLIT_CACHE = {}

            def run():
                return _timed(device, call, x, a.iters)

            exact = {}
            for cap in arms:
                use(cap)
                o = call(x)
                got = ttnn.to_torch(o)
                if ref_t is None:      # the control arm defines the reference for the gated leg
                    ref_t = got
                exact[cap] = bool(torch.equal(got, ref_t))
                ttnn.deallocate(o)

            ratios = {c: [] for c in arms}
            spreads = {c: [] for c in arms}
            ms = {c: [] for c in arms}
            for _ in range(a.reps):
                use(ctl)
                before = run()
                for cap in arms:
                    use(cap)
                    t = run()
                    use(ctl)
                    after = run()
                    mid = 0.5 * (before + after)
                    ratios[cap].append(t / mid)
                    spreads[cap].append(abs(after - before) / mid)
                    ms[cap].append(t)
                    before = after
            ttnn.deallocate(x)
            del ref_t

            hole = units > full and units % full and units % full % g.y == 0
            print(f"\n{leg} N={N} C={C} units={units}   control = {ctl[0]} cores block, "
                  f"{'a hole' if hole else 'not a hole'}")
            print(f"  {'assign':>8}{'cores':>6}{'depth':>7}{'ms':>10}{'vs control':>12}"
                  f"{'bracket':>9}  exact")
            for cap in arms:
                n, _n1, w1, _w2 = core_split.units_per_core(units, cap[0])
                r, spread = _median(ratios[cap]), _median(spreads[cap])
                flag = "  UNUSABLE (bracket drift)" if spread > a.bracket_tol else ""
                print(f"  {cap[1]:>8}{n:>6}{w1:>7}"
                      f"{_median(ms[cap]):>10.4f}{r:>12.4f}{spread:>9.1%}  {exact[cap]}{flag}")
                out_rows.append({"leg": leg, "N": N, "C": C, "units": units, "cores": n,
                                 "walk": cap[1],
                                 "depth": w1, "ms": _median(ms[cap]), "ratio_vs_control": r,
                                 "control_cores": ctl[0], "bracket_spread": spread,
                                 "usable": spread <= a.bracket_tol, "bit_exact": exact[cap],
                                 "all_ratios": ratios[cap]})
    finally:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump({"grid": [g.x, g.y], "reps": a.reps, "iters": a.iters,
                       "bracket_tol": a.bracket_tol, "rows": out_rows}, f, indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
