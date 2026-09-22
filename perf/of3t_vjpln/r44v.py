#!/usr/bin/env python3
"""The pre-registered metric for `allref2`, read against of3t-blk4544's falsifier.

    R44 <= 1.33   names the carrier
    R44 >= 1.90   refutes it

Inherited verbatim from `perf/of3t_blk4544/PREDICTION.md` (d6cbfdc67) and re-registered in
`perf/of3t_vjpln/PREDICTION.md` (38a94d7e8) before the first arm of this row. Both legs of the
ratio are the SAME arm at the two widths, each scored against the in-frame float64 reference at
ITS OWN width and masked to the 56 real tokens: a reference is part of a measurement's identity
(D189), so a ratio built from one arm at 384 and a different one at 64 would not be this
quantity.

`of3t-blk4544`'s own rows are printed beside it, read out of its committed COTSCAN_SAT files
rather than retyped.
"""
from __future__ import annotations

import json
import sys

A = json.load(open(sys.argv[1]))                 # COTSCAN_V_384
B = json.load(open(sys.argv[2]))                 # COTSCAN_V_64
S384 = json.load(open("perf/of3t_blk4544/COTSCAN_SAT_384.json"))
S64 = json.load(open("perf/of3t_blk4544/COTSCAN_SAT_64.json"))

ROWS = [("base", S384, S64, "base"), ("all", S384, S64, "all_4544"),
        ("allref", S384, S64, "allref_4544"), ("allref2", A, B, "allref2_4544")]
COV = {"base": "0 of 11856", "all": "816, 6.88 %", "allref": "1296, 10.93 %",
       "allref2": "3696, 31.17 %"}

out = {"rule": "R44 <= 1.33 names the carrier, R44 >= 1.90 refutes it; inherited verbatim from "
               "perf/of3t_blk4544/PREDICTION.md at d6cbfdc67, re-registered at 38a94d7e8",
       "arms": {}}

print("ds, rung 44, masked to the 56 real tokens, in-frame float64 reference per width")
print("%-9s %14s %14s %10s %10s %14s" % ("arm", "ds@384", "ds@64", "R44", "verdict", "reach"))
for nm, a, b, key in ROWS:
    x = a["rungs"]["44"]["ds"][key]["masked"]["rel_l2"]
    y = b["rungs"]["44"]["ds"][key]["masked"]["rel_l2"]
    R = x / y
    v = "CARRIER" if R <= 1.33 else ("REFUTED" if R >= 1.90 else "amber")
    print("%-9s %14.10f %14.10f %10.6f %10s %14s" % (nm, x, y, R, v, COV[nm]))
    out["arms"][nm] = {"ds44_384": x, "ds44_64": y, "R44": R, "verdict": v,
                       "reach": COV[nm],
                       "nr384": a["rungs"]["44"]["ds"][key]["masked"]["norm_ratio"],
                       "nr64": b["rungs"]["44"]["ds"][key]["masked"]["norm_ratio"],
                       "cos384": a["rungs"]["44"]["ds"][key]["masked"]["cos"],
                       "cos64": b["rungs"]["44"]["ds"][key]["masked"]["cos"]}

print()
print("the mass-injection reading of3t-blk4544 left standing: norm_ratio (ours / reference)")
print("%-9s %10s %10s %10s %10s" % ("arm", "nr@384", "nr@64", "cos@384", "cos@64"))
for nm in out["arms"]:
    e = out["arms"][nm]
    print("%-9s %10.6f %10.6f %10.6f %10.6f" % (nm, e["nr384"], e["nr64"], e["cos384"],
                                                e["cos64"]))
print("reference's own masked ds norm @384 %.15e  @64 %.15e"
      % (A["rungs"]["44"]["ds"]["ref_norm_masked"], B["rungs"]["44"]["ds"]["ref_norm_masked"]))

print()
print("every rung, allref2 only, against base at the same width")
print("%-5s %13s %13s %9s %13s %13s %9s" % ("rung", "base@384", "a2@384", "R", "base@64",
                                            "a2@64", "R"))
prof = {}
for k in ("48", "47", "46", "45", "44", "43", "40", "32", "16", "0"):
    b3 = S384["rungs"][k]["ds"]["base"]["masked"]["rel_l2"]
    a3 = A["rungs"][k]["ds"]["allref2_4544"]["masked"]["rel_l2"]
    b6 = S64["rungs"][k]["ds"]["base"]["masked"]["rel_l2"]
    a6 = B["rungs"][k]["ds"]["allref2_4544"]["masked"]["rel_l2"]
    print("%-5s %13.10f %13.10f %9.6f %13.10f %13.10f %9.6f"
          % (k, b3, a3, b3 / b6, b6, a6, a3 / a6))
    prof[k] = {"base_384": b3, "allref2_384": a3, "base_64": b6, "allref2_64": a6,
               "R_base": b3 / b6, "R_allref2": a3 / a6}
out["per_rung_ds"] = prof

if len(sys.argv) > 3:
    json.dump(out, open(sys.argv[3], "w"), indent=1)
    print("\nwrote", sys.argv[3])
