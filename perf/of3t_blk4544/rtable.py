#!/usr/bin/env python3
"""The pre-registered metric, per arm: R = rel_l2 @ padded 384 / rel_l2 @ padded 64.

`PREDICTION.md` fixed the decision rule before any arm ran:

    R44 <= 1.33  names the carrier (removes >= 75 % of the step 2.201 - 1.042)
    R44 >= 1.90  refutes the candidate (removes < 27 %)

Both legs of every ratio are the SAME arm at the two widths, scored against the in-frame float64
reference at that width, masked to the 56 real tokens. A ratio built from one arm at 384 and the
shipped arm at 64 would not be this quantity (D189: a reference is part of a measurement's
identity), so an arm missing either leg is reported as missing rather than crossed.
"""
from __future__ import annotations

import json
import sys

A = json.load(open(sys.argv[1]))          # COTSCAN at padded 384
B = json.load(open(sys.argv[2]))          # COTSCAN at padded 64
SKIP = ("floor_masked", "floor_unmasked", "ref_norm_masked")

arms = [k for k in A["rungs"]["44"]["ds"] if k not in SKIP]
common = [a for a in arms if a in B["rungs"]["44"]["ds"]]
print(f"arms at 384: {arms}")
print(f"arms at  64: {[k for k in B['rungs']['44']['ds'] if k not in SKIP]}")
print(f"scored: {common}\n")

BASE = {"R44": 2.201, "R45": 1.858, "B46": 1.042, "R44dz": 1.952, "R0": 1.724}
print("of3t-tapeattn shipped-path baseline: " + "  ".join(f"{k}={v}" for k, v in BASE.items()))
print()
hdr = (f"{'arm':>14} {'B46':>7} {'R45':>7} {'R44':>7} {'R43':>7} {'R0':>7} "
       f"{'R44dz':>7} {'step44':>7} {'removed':>8} {'verdict':>9}")
print(hdr)
out = {}
for nm in common:
    row = {}
    for k in (48, 47, 46, 45, 44, 43, 32, 16, 0):
        for track in ("ds", "dz"):
            ra = A["rungs"][str(k)].get(track)
            rb = B["rungs"][str(k)].get(track)
            if not ra or not rb or nm not in ra or nm not in rb:
                continue
            a = ra[nm]["masked"]["rel_l2"]
            b = rb[nm]["masked"]["rel_l2"]
            row[f"{track}{k}"] = {"rel384": a, "rel64": b, "R": a / b if b else None,
                                  "nr384": ra[nm]["masked"]["norm_ratio"],
                                  "nr64": rb[nm]["masked"]["norm_ratio"],
                                  "cos384": ra[nm]["masked"]["cos"],
                                  "cos64": rb[nm]["masked"]["cos"]}
    r44 = row.get("ds44", {}).get("R")
    b46 = row.get("ds46", {}).get("R")
    step = (r44 - b46) if (r44 is not None and b46 is not None) else None
    removed = (1 - step / (BASE["R44"] - BASE["B46"])) if step is not None else None
    verdict = ("--" if r44 is None else
               "CARRIER" if r44 <= 1.33 else "REFUTED" if r44 >= 1.90 else "PARTIAL")
    def c(key):
        v = row.get(key, {}).get("R")
        return float("nan") if v is None else v
    print(f"{nm:>14} {c('ds46'):7.3f} {c('ds45'):7.3f} {c('ds44'):7.3f} {c('ds43'):7.3f} "
          f"{c('ds0'):7.3f} {c('dz44'):7.3f} "
          f"{(step if step is not None else float('nan')):7.3f} "
          f"{(removed if removed is not None else float('nan')):8.1%} {verdict:>9}")
    out[nm] = {"rungs": row, "R44": r44, "B46": b46, "step_above_B46": step,
               "share_of_the_step_removed": removed, "verdict": verdict}

if len(sys.argv) > 3:
    json.dump({"rule": "R44<=1.33 names the carrier, R44>=1.90 refutes it; "
                       "pre-registered in perf/of3t_blk4544/PREDICTION.md at d6cbfdc67",
               "baseline": BASE, "arms": out}, open(sys.argv[3], "w"), indent=1)
    print(f"\nwrote {sys.argv[3]}")
