#!/usr/bin/env python3
"""Read the upstream 1536 pair by the rule written before it landed, not after.

`results/upstream_draw_count.txt` and `plans/closing_rule.txt` §3-4 fix the reading: m=2
against the device's n=8 rejects at 0.05 only on perfect separation in upstream's favour
(p = 1/C(10,2) = 0.0222). Anything less means the two sides are at the same level, the 1536
quality floor is a BoltzGen property rather than a port defect, and the row closes.

This script only applies that rule; it picks no threshold. PRIMARY is the device cell QUALITY
publishes at 1536/offset 0 -- `out_boltzgen_1536_d8_bgall_fixed`, n=8, median 19.398, the
after-`cc908c377` engine -- because that is the cell the row's 1536 claim rests on. The other
two cells designed against the byte-identical fixture (sha256 ffa258bb...421e9caf) are printed
beside it, since which tree a device cell came from is the confound this row already controlled
once, and the three cells disagree about where their minimum sits: 6.650 / 17.843 / 19.311 A.

    python3 read_upstream_pair.py <draw_a_A> <draw_b_A>
"""
import sys
from itertools import combinations
from math import comb
from statistics import median

# Device n=8 at 1536, crop offset 0, all three on the fixture the upstream pair used.
# Raw values from results/size_1536.jsonl.
DEVICE = [
    ("PRIMARY  after cc908c377  out_boltzgen_1536_d8_bgall_fixed", [
        18.743, 6.650, 19.296, 22.854, 19.501, 21.154, 18.740, 22.924]),
    ("         onetree 72ad980ee  out_*onetree1536o0", [
        18.808, 20.094, 19.285, 20.900, 19.805, 17.843, 21.002, 22.675]),
    ("         pre-fix  out_boltzgen_1536_d8_s400_gpb", [
        20.140, 22.478, 21.340, 20.958, 21.692, 21.752, 19.311, 22.260]),
]


def exact_p_less(up, dev):
    """One-sided exact rank test: P(U >= U_obs) under the null that both sides are one
    distribution, enumerated over every interleaving. U counts upstream-below-device pairs,
    so a larger U means upstream looks better and the tail is the upper one. numpy-free."""
    m, n = len(up), len(dev)

    def u_of(a, b):
        return (sum(1 for x in a for y in b if x < y)
                + 0.5 * sum(1 for x in a for y in b if x == y))

    u_obs = u_of(up, dev)
    pool = sorted(up + dev)
    hits = total = 0
    for idx in combinations(range(m + n), m):
        sel = [pool[i] for i in idx]
        rest = [pool[i] for i in range(m + n) if i not in idx]
        total += 1
        if u_of(sel, rest) >= u_obs:
            hits += 1
    return u_obs, hits / total, total


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    up = sorted(float(a) for a in sys.argv[1:3])
    print(f"upstream 1536, m=2 (designfolding-bb_rmsd, N/CA/C/O): {up[0]:.3f}  {up[1]:.3f} A")
    print(f"floor on p for m=2 against n=8: 1/C(10,2) = {1/comb(10, 2):.4f}")
    print(f"bar crossings: {sum(1 for x in up if x <= 4.0)} of 2 at 4 A, "
          f"{sum(1 for x in up if x <= 2.0)} of 2 at 2 A "
          f"(the device is 0 of 8 at 4 A in all three cells below)\n")
    for name, dev in DEVICE:
        u, p, total = exact_p_less(up, dev)
        print(f"{name}")
        print(f"  device n=8  median {median(dev):.3f}  min {min(dev):.3f}  max {max(dev):.3f}")
        print(f"  U = {u:.1f} of {len(up)*len(dev)}   exact one-sided p = {p:.4f}"
              f"   ({total} interleavings)"
              f"   separation: {'PERFECT' if up[1] < min(dev) else 'no'}")
        print("  -> " + ("upstream BETTER at 0.05: closing_rule clause 3, a 1536 port defect"
                         " to locate" if p <= 0.05 else
                         "no ordering readable: upstream is at the same level, clause 4"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
