#!/usr/bin/env python3
"""Where do the wrong cells land?

`paircond_repeat.py` now records the union of victim flat-cell indices per stage. The pc
fault is compute-side and matmul-only, so the question this answers is whether the wrong
cells concentrate on a few Tensix cores (marginal silicon in specific cores) or spread
across the whole grid (a global timing/power effect).

The matmul parallelizes over M = N*N rows = M/32 tile-rows, split across the 130 cores of
the 13x10 grid in contiguous blocks. Mapping victim tile-rows through that split gives the
per-core histogram. Uniform-random victims would spread over ~as many cores as there are
victims; a bad-core fault piles them onto a few.

Usage: victim_locality.py <paircond_*.json> [cores]
"""
import collections
import json
import sys

path = sys.argv[1]
NC = int(sys.argv[2]) if len(sys.argv) > 2 else 130

d = json.load(open(path))
N = d["N"]
M = N * N
MT = M // 32
print(f"{path}: host={d['host']} card={d['card']} N={N} prec={d['prec']} K={d['K']}")
print(f"flat rows M={M}  tile-rows={MT}  cores={NC}  tile-rows/core={MT / NC:.2f}")

per_hi = MT // NC + (1 if MT % NC else 0)
rem = MT % NC


def core_of(t):
    if rem == 0:
        return t // (MT // NC)
    hi = rem * per_hi
    return t // per_hi if t < hi else rem + (t - hi) // (MT // NC)


for st in d["stages"]:
    v = st.get("victim_cells_union") or []
    if not v:
        continue
    name = st["stage"]
    ndiff = st["compute_differing"]
    reps = st["repeats"] - 1
    tiles = sorted({i // 32 for i in v})
    cores = collections.Counter(core_of(t) for t in tiles)
    exp = NC * (1 - (1 - 1 / NC) ** len(tiles))
    print(f"\n== {name}  differing {ndiff}/{reps}  union victim cells={st['victim_cells_union_n']}")
    print(f"   distinct victim tile-rows: {len(tiles)}")
    print(f"   distinct cores hit: {len(cores)} of {NC}   (uniform-random would hit ~{exp:.1f})")
    print(f"   top cores: {cores.most_common(12)}")
    print(f"   cores hit: {sorted(cores)}")
    print(f"   tile-rows: {tiles[:48]}{' ...' if len(tiles) > 48 else ''}")

    # Position of each victim tile-row INSIDE its core's contiguous block. A fault tied to
    # the per-core inner loop piles up at one end of the block; a data- or address-keyed
    # fault spreads evenly. per_hi is the block length for the cores that get the extra row.
    off = collections.Counter()
    for t in tiles:
        c = core_of(t)
        base = c * per_hi if c < rem or rem == 0 else rem * per_hi + (c - rem) * (MT // NC)
        off[t - base] += 1
    blk = per_hi
    first_half = sum(n for o, n in off.items() if o < blk / 2)
    print(f"   offset within the core's {blk}-tile-row block: {sorted(off.items())}")
    print(f"   in FIRST half of the block: {first_half}/{len(tiles)}")

    # Pair-matrix rows (flat cell // N). At N=256 a pair-row is 8 tile-rows and a core block
    # is 16, so even pair-row == first half of a core block: the parity IS the block half.
    prows = sorted({i // N for i in v})
    ev = sum(1 for r in prows if r % 2 == 0)
    print(f"   victim pair-rows: {len(prows)}  even/odd {ev}/{len(prows) - ev}")
    print(f"   {prows}")
