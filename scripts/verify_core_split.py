#!/usr/bin/env python3
"""`tt_bio.core_split` against the wheel's `ttnn.split_work_to_cores`, host only, no device.

Three questions, all answered without opening a card:

1. Where the wheel returns, does this module return the SAME thing? Range for range, group for
   group, on every unit count on every grid swept.
2. Where the wheel raises `TT_FATAL @ work_split.cpp:305`, does this module return a VALID split?
   Checked structurally: the two groups partition `all_cores` exactly, every core is inside the
   grid, no core appears twice, and `n1*w1 + n2*w2 == units`.
3. What does the census of holes look like on this host's real grid, and what did
   `reblock_permute._split_plan`'s sub-grid search cost in cores on those bands?

Usage: verify_core_split.py [--max-units N] [--grids 11x10,13x10,...]
"""

from __future__ import annotations

import argparse
import sys

import ttnn

from tt_bio import core_split as cs


class _Grid:
    def __init__(self, x, y):
        self.x, self.y = x, y


def _full(grid):
    return ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))]
    )


def _cores(crs):
    """Every core in a CoreRangeSet as a list of (x, y), in the set's own range order."""
    out = []
    for r in crs.ranges():
        for x in range(r.start.x, r.end.x + 1):
            for y in range(r.start.y, r.end.y + 1):
                out.append((x, y))
    return out


def _wheel(grid, units):
    try:
        return ttnn.split_work_to_cores(_full(grid), units)
    except Exception:                                                  # noqa: BLE001 -- TT_FATAL
        return None


def _same(a, b):
    """The wheel's tuple against ours: scalars equal, CoreRangeSets equal as printed."""
    return (
        a[0] == b[0] and a[4] == b[4] and a[5] == b[5]
        and str(a[1]) == str(b[1]) and str(a[2]) == str(b[2]) and str(a[3]) == str(b[3])
    )


def _valid(grid, units, got):
    """Structural invariants, so a served hole is checked on its own terms, not against a crash."""
    n, allc, g1, g2, w1, w2 = got
    c_all, c1, c2 = _cores(allc), _cores(g1), _cores(g2)
    problems = []
    if n != min(units, cs.core_count(grid)):
        problems.append(f"num_cores {n} != min({units}, {cs.core_count(grid)})")
    if len(c_all) != n:
        problems.append(f"all_cores holds {len(c_all)} cores, num_cores says {n}")
    if len(set(c_all)) != len(c_all):
        problems.append("all_cores repeats a core")
    if any(not (0 <= x < grid.x and 0 <= y < grid.y) for x, y in c_all):
        problems.append("a core lies outside the grid")
    if c1 + c2 != c_all:
        problems.append("group 1 ++ group 2 is not all_cores in order")
    if len(c1) * w1 + len(c2) * w2 != units:
        problems.append(f"{len(c1)}*{w1} + {len(c2)}*{w2} != {units}")
    if c2 and w2 != w1 - 1:
        problems.append(f"group 2 carries {w2}, group 1 carries {w1}")
    return problems


def _subgrid_search(grid, units):
    """`reblock_permute._split_plan`'s search, verbatim in effect: the sub-grid it settles on.

    Returns (sx, sy) or None. Uses the wheel, so it prices what the shipped code actually gets.
    """
    def hole(cores, height, u):
        r = u % cores
        return u > cores and r != 0 and r % height == 0

    candidates = [(grid.x, grid.y)] + sorted(
        ((sx, sy) for sy in range(1, grid.y + 1) for sx in range(1, grid.x + 1)
         if (sx, sy) != (grid.x, grid.y) and not hole(sx * sy, sy, units)),
        key=lambda s: -s[0] * s[1],
    )
    for sx, sy in candidates:
        crs = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(sx - 1, sy - 1))]
        )
        try:
            ttnn.split_work_to_cores(crs, units)
        except Exception:                                              # noqa: BLE001
            continue
        return sx, sy
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-units", type=int, default=4000)
    ap.add_argument("--grids", default="11x10,13x10,7x10,9x13,13x13,12x7,8x10,6x6,5x11,1x1,2x3")
    ap.add_argument("--census-grid", default="11x10")
    a = ap.parse_args()

    grids = []
    for tok in a.grids.split(","):
        x, y = tok.split("x")
        grids.append(_Grid(int(x), int(y)))

    agree = served = mismatch = invalid = 0
    bad = []
    for g in grids:
        for u in range(1, a.max_units + 1):
            mine = cs.split_work_to_cores(g, u)
            ref = _wheel(g, u)
            if ref is None:
                served += 1
                p = _valid(g, u, mine)
                if p:
                    invalid += 1
                    bad.append((g.x, g.y, u, "invalid: " + "; ".join(p)))
            elif _same(ref, mine):
                agree += 1
            else:
                mismatch += 1
                bad.append((g.x, g.y, u, f"mismatch\n  wheel {ref}\n  ours  {mine}"))
    print(f"grids: {', '.join(f'{g.x}x{g.y}' for g in grids)}  units 1..{a.max_units}")
    print(f"wheel served and we agree exactly: {agree}")
    print(f"wheel raised TT_FATAL and we served a valid split: {served - invalid}")
    print(f"MISMATCHES: {mismatch}   INVALID: {invalid}")
    for x, y, u, why in bad[:20]:
        print(f"  {x}x{y} units={u}: {why}")

    # even_split against the implementation it replaces, on the same sweep.
    def old_even(n_units, cores):
        per, rem = divmod(n_units, len(cores))
        return [per + 1 if i < rem else per for i in range(len(cores))]

    diffs = 0
    for g in grids:
        cores = [(cx, cy) for cx in range(g.x) for cy in range(g.y)]
        for u in range(0, min(a.max_units, 1200) + 1):
            if cs.even_split(u, cores) != old_even(u, cores):
                diffs += 1
    print(f"even_split vs the rfd3_bias implementation it replaces: {diffs} differences")

    cx, cy = (int(v) for v in a.census_grid.split("x"))
    g = _Grid(cx, cy)
    cores = cs.core_count(g)
    print(f"\n--- census on {cx}x{cy} ({cores} cores) ---")
    print("Every shape any reblock_permute gate screens, restricted to the unit counts the wheel")
    print("refuses. `sub-grid` is what the shipped search settles for; `depth` is the per-core")
    print("group count on the busiest core, which is what the op's critical path actually costs.")
    print("A band where depth does not move has nothing to recover: the sub-grid is already as")
    print("deep as the full grid (core-coverage-ratio-not-recoverable-time).")
    print()
    print(f"{'leg':<8}{'shape':<22}{'units':>7}  {'sub-grid':<10}{'cores':>9}"
          f"{'depth sub':>11}{'depth full':>12}{'ratio':>8}")
    shapes = []
    for nt in range(8, 33):                                # N = 256 .. 1024
        shapes.append(("forward", f"N={nt * 32}", nt * nt))
    for nt in range(8, 33):
        for ct in (1, 2, 4, 6, 8):                         # C = 32 .. 256
            shapes.append(("back", f"N={nt * 32},C={ct * 32}", nt * nt * ct))
    for nt in range(8, 33):
        for ct in (1, 2, 4, 6, 8):
            shapes.append(("gated", f"N={nt * 32},Cg={ct * 32}", nt * nt * ct))
    seen = set()
    rows = 0
    for leg, name, u in shapes:
        if (leg, u) in seen:
            continue
        seen.add((leg, u))
        if _wheel(g, u) is not None:
            continue
        sub = _subgrid_search(g, u)
        used = sub[0] * sub[1] if sub else 0
        if not used:
            print(f"{leg:<8}{name:<22}{u:>7}  {'NONE':<10}{'-> permute':>9}")
            rows += 1
            continue
        d_sub = -(-u // used)
        d_full = -(-u // cores)
        rows += 1
        print(f"{leg:<8}{name:<22}{u:>7}  {f'{sub[0]}x{sub[1]}':<10}"
              f"{f'{used}/{cores}':>9}{d_sub:>11}{d_full:>12}{d_sub / d_full:>8.4f}")
    if not rows:
        print("(no shape in the swept range lands in a hole)")

    return 1 if (mismatch or invalid or diffs) else 0

if __name__ == "__main__":
    sys.exit(main())
