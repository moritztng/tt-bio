"""Grade a run.sh run: s a design per leg and config, AICLK beside it, adjacent-leg ratios against b4l1, the
per-batch fixed cost from a least-squares fit of design_s on the batch, and design identity across configs.

    PYTHONPATH=<run>/tree python perf/rfd4/cfg512/grade.py <run>   (the tree unpickles designs.pt)
"""
import json
import sys
from pathlib import Path
from statistics import median

import torch

run = Path(sys.argv[1])
FLOOR = 1200
legs = {}
for f in sorted(run.glob("*/runs.jsonl")):
    rows = [json.loads(line) for line in open(f)]
    meas = [r for r in rows if r.get("phase") == "measure"]
    ok = [r for r in meas if (r.get("aiclk_mhz") or 0) >= FLOOR and "discard" not in r]
    if not meas:
        continue
    legs[f.parent.name] = {
        "batch": meas[0]["batch"], "batches": len(meas), "clean": len(ok),
        "s_per_design": round(sum(r["wall_s"] for r in ok) / max(1, sum(r["designs"] for r in ok)), 4) if ok else None,
        "design_s_per_design": round(sum(r["design_s"] for r in ok) / max(1, sum(r["designs"] for r in ok)), 4) if ok else None,
        "featurize_s": round(median(r["featurize_s"] for r in meas), 3), "write_s": round(median(r["write_s"] for r in meas), 3),
        "aiclk_median": [r["aiclk_mhz"] for r in meas], "aiclk_min": min(r["aiclk_min_mhz"] or 0 for r in meas),
        "discard": [r.get("discard") for r in meas if "discard" in r],
        "rows": [(r["batch"], r["design_s"], r["wall_s"]) for r in ok],
    }
cfg = {}
for name, leg in legs.items():
    if leg["s_per_design"]:
        cfg.setdefault(leg["batch"], []).append(leg["s_per_design"])
base = cfg.get(4)
out = {"legs": legs, "s_per_design": {b: round(sum(v) / len(v), 4) for b, v in cfg.items()}}
if base:
    out["vs_b4l1"] = {b: round(sum(base) / len(base) / (sum(v) / len(v)), 4) for b, v in cfg.items()}
# Adjacent pairs: each non-A leg against the A leg of the same round (A1 with B1/C1, A2 with B2/C2).
pairs = {}
for name, leg in legs.items():
    a = legs.get("A" + name[1:])
    if name[0] != "A" and a and a["s_per_design"] and leg["s_per_design"]:
        pairs[name] = round(a["s_per_design"] / leg["s_per_design"], 4)
out["pair_vs_A_same_round"] = pairs
# design_s = F + B * d over every clean batch: F is what a batch pays once, whatever its size.
pts = [(b, ds) for leg in legs.values() for b, ds, _ in leg["rows"]]
if len({b for b, _ in pts}) > 1:
    n = len(pts); mb = sum(b for b, _ in pts) / n; my = sum(y for _, y in pts) / n
    d = sum((b - mb) * (y - my) for b, y in pts) / sum((b - mb) ** 2 for b, _ in pts)
    out["fit_design_s"] = {"per_design_s": round(d, 4), "per_batch_s": round(my - d * mb, 4), "points": n}
# Identity: seed 1010's designs 0-3 in every leg against A1's.
ref, same, seen = None, [], 0
for f in sorted(run.glob("*/b*l1/measure-s1010/designs.pt")):
    o = torch.load(f, weights_only=False)
    rows = {k: getattr(o, k)[:4] for k in ("coords_nm", "residue_type", "atom_mask")}
    seen += 1
    if ref is None:
        ref = rows
        continue
    same.append(all(torch.equal(rows[k], ref[k]) for k in rows))
out["seed1010_d0_3_identical"] = {"files": seen, "all_equal": all(same) if same else None}
print(json.dumps(out, indent=1))
