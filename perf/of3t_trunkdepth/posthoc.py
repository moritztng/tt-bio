#!/usr/bin/env python3
"""POST-HOC, and labelled as such: what the ladder tracks once each depth is divided by its OWN
floor, and what the norm hypothesis has to survive on effect size rather than on rank.

Nothing here was pre-registered. The pre-registered test is in `ladder_report.py` and its answer
stands whatever this says. This exists because "the error does not track X" is only half an
answer, and the brief asks for the other half: report what it tracks instead.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ladder_report import spearman, ols  # noqa: E402

D = Path("perf/of3t_trunkdepth")
d = json.loads((D / "LADDER.json").read_text())
rows = d["rows"]

for r in rows:
    r["Ds_floor"] = 1.0 - r["ds_in_cos_floor"]
    r["Dz_floor"] = 1.0 - r["dz_in_cos_floor"]
    r["Ds_over_floor"] = r["Ds"] / r["Ds_floor"]
    r["Dz_over_floor"] = r["Dz"] / r["Dz_floor"]

print("blk   Ds=1-cos(ds_in)   its own floor    Ds/floor    ours/floor(grad)   |s_in|")
for r in rows:
    print(f"{r['block']:3d}   {r['Ds']:.6e}     {r['Ds_floor']:.6e}   {r['Ds_over_floor']:8.2f}"
          f"   {r['ours_over_floor']:8.2f}           {r['s_in']:.4e}")

sub = [r for r in rows if r["block"] >= 8]
def span(key, rs):
    v = [x[key] for x in rs]
    return max(v) / min(v)

out = {"what": __doc__.strip().splitlines()[0], "rows": rows, "posthoc": {}}
o = out["posthoc"]
o["effect_size_over_blocks_8_to_47"] = {
    "s_in_span": span("s_in", sub), "z_in_span": span("z_in", sub),
    "cot_s_span": span("cot_s", sub), "cot_z_span": span("cot_z", sub),
    "Ds_span": span("Ds", sub), "ours_over_floor_span": span("ours_over_floor", sub),
    "reading": "the norm hypothesis has to explain the dependent's move with the candidate's "
               "move; over this stretch it does not have the room"}
for dep in ("Ds_over_floor", "Dz_over_floor", "Ds", "ours_over_floor"):
    o[dep] = {}
    for cand in ("s_in", "z_in", "cot_s", "cot_z", "single_track_mass_share", "block"):
        cv = [float(r[cand]) for r in rows]
        if min(cv) <= 0:
            cv = [v - min(cv) + 1.0 for v in cv]
        dv = [r[dep] for r in rows]
        b, _ = ols([math.log10(v) for v in cv], [math.log10(v) for v in dv])
        o[dep][cand] = {"spearman_rho": spearman(cv, dv), "loglog_slope": b}
    o[dep]["span_all7"] = span(dep, rows)
    o[dep]["span_8_to_47"] = span(dep, sub)

print("\n--- effect size over blocks 8..47 (the stretch where ||s_in|| is nearly flat) ---")
for k, v in o["effect_size_over_blocks_8_to_47"].items():
    if isinstance(v, float):
        print(f"  {k:26s} {v:10.3f}x")
print("\n--- post-hoc rank correlations, every dependent against every candidate ---")
for dep in ("Ds_over_floor", "Dz_over_floor", "Ds", "ours_over_floor"):
    print(f"  {dep:16s} span all7 {o[dep]['span_all7']:8.2f}x  span 8..47 "
          f"{o[dep]['span_8_to_47']:8.2f}x")
    for cand in ("s_in", "z_in", "cot_s", "cot_z", "single_track_mass_share", "block"):
        print(f"      vs {cand:26s} rho {o[dep][cand]['spearman_rho']:+.4f}  "
              f"slope {o[dep][cand]['loglog_slope']:+.3f}")
(D / "POSTHOC.json").write_text(json.dumps(out, indent=1) + "\n")
print(f"\nwrote {D/'POSTHOC.json'}")
