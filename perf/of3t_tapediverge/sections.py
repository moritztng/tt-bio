#!/usr/bin/env python3
"""Per-section gradient statistics per arm, from of3t-wholemodel's per-tensor sidecars.

The sidecars are the per-tensor arrays behind `MODEL_arms.json`: one row per parameter, with the
section it belongs to, its share of the model's squared gradient norm, and the arm's `rel_l2`
against the named reference. Re-cutting them by section gives the diffusion-module and
`msa_module` gradient readings on every arm without a second device run, from the same tensors
the model headline is assembled from.

A14 near-zero references are excluded exactly as the campaign's other instruments do it: a row
whose `rel_l2` is null has no measurable reference and is counted, not silently dropped.
"""
from __future__ import annotations

import collections
import json
import math
import sys

BASE = "/home/ttuser/of3t_wholemodel/sidecar_arms/"
ARMS = ["shipped", "renorm", "renormf64", "break"]
REFS = ["FLOAT64", "UPSTREAM_BF16"]

out = {}
for ref in REFS:
    for arm in ARMS:
        rows = json.load(open(f"{BASE}{arm}_vs_{ref}.json"))
        by = collections.defaultdict(list)
        for r in rows:
            by[r["section"]].append(r)
        for sec, rs in sorted(by.items()):
            num = sum(r["diff_norm"] ** 2 for r in rs)
            den = sum(r["ref_norm"] ** 2 for r in rs)
            meas = sorted(r["rel_l2"] for r in rs if r["rel_l2"] is not None)
            out[f"{arm}|{ref}|{sec}"] = {
                "arm": arm, "ref": ref, "section": sec, "n": len(rs),
                "n_measurable": len(meas), "n_a14_excluded": len(rs) - len(meas),
                "pct_of_model_mass": sum(r["pct_of_model_mass"] for r in rs),
                "mass_weighted_rel_l2": math.sqrt(num / den) if den else None,
                "median_rel_l2": meas[len(meas) // 2] if meas else None,
                "worst_rel_l2": meas[-1] if meas else None,
                "worst_tensor": max(
                    (r for r in rs if r["rel_l2"] is not None),
                    key=lambda r: r["rel_l2"], default={"param": None})["param"],
            }

path = sys.argv[1] if len(sys.argv) > 1 else "perf/of3t_tapediverge/SECTIONS.json"
json.dump({"source": BASE, "rows": out}, open(path, "w"), indent=1)
for ref in REFS:
    print(f"===== vs {ref}")
    secs = sorted({v["section"] for v in out.values()})
    for sec in secs:
        for arm in ARMS:
            v = out[f"{arm}|{ref}|{sec}"]
            print(f"  {sec:26s} {arm:10s} n={v['n']:4d} ({v['n_a14_excluded']} a14) "
                  f"mass%={v['pct_of_model_mass']:8.4f} "
                  f"mw={v['mass_weighted_rel_l2']:.6e} med={v['median_rel_l2']:.6e}")
        print()
print(f"wrote {path}")
