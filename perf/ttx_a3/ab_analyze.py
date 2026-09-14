#!/usr/bin/env python3
"""Paired read of the interleaved 1536 aa fold A/B, with its own floor beside the ratio.

One process per fold, arms alternating off/on/off/on..., so neither arm's kernels sit in the
other's ttnn program cache. Pairs are adjacent folds, which is the tightest condition match the
box allows; the floor is consecutive same-arm folds (fold i and fold i+2).

    python3 perf/ttx_a3/ab_analyze.py --dir perf/ttx_a3/f1536_ab
"""
import argparse
import json
import statistics as st
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--dir", type=Path, required=True)
ap.add_argument("--out", type=Path)
a = ap.parse_args()

folds = []
for f in sorted(a.dir.glob("*.json"), key=lambda p: p.stat().st_mtime):
    d = json.load(open(f))
    folds.append({"tag": f.stem, "arm": d["arm"], "s": d["fold_s"],
                  "served": d.get("triatt_served"), "declined": d.get("triatt_declined"),
                  "plddt": d["plddt"], "sha": d["cif_sha256"][:16],
                  "load": round(d.get("loadavg", [0])[0], 2),
                  "picks": d.get("sdpa_picks")})

print(f"{'#':>2} {'arm':>4} {'fold_s':>8} {'served':>7} {'decl':>5} {'load':>5}  plddt     sha")
for i, x in enumerate(folds):
    print(f"{i:>2} {x['arm']:>4} {x['s']:>8.3f} {x['served']:>7} {x['declined']:>5} "
          f"{x['load']:>5} {x['plddt']:.6f} {x['sha']}")

offs = [x["s"] for x in folds if x["arm"] == "off"]
ons = [x["s"] for x in folds if x["arm"] == "on"]
pairs = [(folds[i]["s"], folds[i + 1]["s"])
         for i in range(len(folds) - 1)
         if folds[i]["arm"] == "off" and folds[i + 1]["arm"] == "on"]
floors = [(folds[i]["s"], folds[i + 2]["s"])
          for i in range(len(folds) - 2) if folds[i]["arm"] == folds[i + 2]["arm"]]

res = {"n_folds": len(folds), "off_s": offs, "on_s": ons,
       "adjacent_pair_ratios": [round(o / n, 4) for o, n in pairs],
       "same_arm_floor_ratios": [round(max(p) / min(p), 4) for p in floors],
       "median_off_s": round(st.median(offs), 3) if offs else None,
       "median_on_s": round(st.median(ons), 3) if ons else None,
       "digests": sorted({x["sha"] + ":" + x["arm"] for x in folds}),
       "folds": folds}
if offs and ons:
    res["median_ratio"] = round(st.median(offs) / st.median(ons), 4)
if pairs:
    res["paired_median_ratio"] = round(st.median([o / n for o, n in pairs]), 4)

print("\n" + json.dumps({k: v for k, v in res.items() if k != "folds"}, indent=1))
if a.out:
    a.out.write_text(json.dumps(res, indent=1) + "\n")
