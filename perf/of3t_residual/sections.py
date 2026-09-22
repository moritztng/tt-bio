#!/usr/bin/env python3
"""The per-section table for one agreement report, every arm side by side."""
import json
import sys

d = json.load(open(sys.argv[1]))
pairs = d["pairs"]
arms = sys.argv[2].split(",")
sets = [s["set"] for s in pairs[arms[0]]["sets"]]
by = {a: {s["set"]: s for s in pairs[a]["sets"]} for a in arms}
w = max(len(s) for s in sets) + 2
print(f"{'set':<{w}}{'mass%':>9}", end="")
for a in arms:
    print(f"{a[:22]:>24}", end="")
print()
for s in sets:
    m = by[arms[0]][s]["pct_of_model_mass"]
    print(f"{s:<{w}}{m:>9.4f}", end="")
    for a in arms:
        r = by[a].get(s)
        print(f"{r['mass_weighted_rel_l2']:>12.4e}/{r['mass_weighted_cos']:>+7.4f}"
              f"{'':>4}" if r else f"{'--':>24}", end="")
    print()
