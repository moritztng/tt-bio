#!/usr/bin/env python3
"""Where the trunk's error mass actually is: the top pairformer_stack rows by mass, and by
contribution to the mass-weighted rel_l2 (mass_sq * rel_l2^2), which are not the same list."""
import json, sys
all_rows = [r for r in json.load(open(sys.argv[1])) if r["section"] == "pairformer_stack"]
rows = [r for r in all_rows if r["rel_l2"] is not None]
print("A14: %d of %d pairformer rows measurable" % (len(rows), len(all_rows)))
tot = sum(r["mass_sq"] for r in rows)
err = sum(r["mass_sq"] * r["rel_l2"] ** 2 for r in rows)
print("pairformer_stack: %d rows, mass_sq %.6e, mass-weighted rel_l2 %.6e"
      % (len(rows), tot, (err / tot) ** 0.5))
print("\n-- top 12 by SHARE OF THE ERROR (mass_sq * rel_l2^2) --")
for r in sorted(rows, key=lambda r: -r["mass_sq"] * r["rel_l2"] ** 2)[:12]:
    print("%7.3f%%  rel=%10.4e r=%9.4f cos=%+.4f massfrac=%9.3e  %s"
          % (100 * r["mass_sq"] * r["rel_l2"] ** 2 / err, r["rel_l2"], r["r"], r["cos"],
             r["mass_sq"] / tot, r["param"]))
print("\n-- top 12 by MASS --")
for r in sorted(rows, key=lambda r: -r["mass_sq"])[:12]:
    print("%7.3f%%  rel=%10.4e r=%9.4f cos=%+.4f  %s"
          % (100 * r["mass_sq"] / tot, r["rel_l2"], r["r"], r["cos"], r["param"]))
leaf = {}
for r in rows:
    k = r["param"].split(".", 3)[3] if r["param"].count(".") >= 3 else r["param"]
    e = leaf.setdefault(k, [0.0, 0.0, 0])
    e[0] += r["mass_sq"] * r["rel_l2"] ** 2
    e[1] += r["mass_sq"]
    e[2] += 1
print("\n-- by LEAF NAME across all 48 blocks, top 12 by error share --")
for k, (e, m, n) in sorted(leaf.items(), key=lambda kv: -kv[1][0])[:12]:
    print("%7.3f%% of error  %7.3f%% of mass  n=%2d  rel=%10.4e  %s"
          % (100 * e / err, 100 * m / tot, n, (e / m) ** 0.5, k))
