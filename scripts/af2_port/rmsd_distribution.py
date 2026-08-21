"""Step 6B: how far the whole population sits from the bound-unbound RMSD bar of 3.5.

`flip_rate4.json` only reports criterion 4 on the designs a conjunction can move, which is the 7 that
pass the three confidence criteria on at least one arm. That leaves the other 43 with no RMSD at all,
so the population's distance from the bar is unstated and the pooled flip rate has no denominator
for its fourth criterion. This is that denominator.

The number that matters is not the median: it is how many designs sit within one device-reference
delta of 3.5, because those are the designs one more precision wobble would flip.

    PYTHONPATH=. python3 scripts/af2_port/rmsd_distribution.py \\
        --pop scripts/af2_port/parity_artifacts/designpop_pxd196 --out rmsd_distribution.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

BAR = 3.5
KEY = "bound_unbound_rmsd"


def rows(path: Path) -> dict:
    if not path.exists():
        return {}
    return {json.loads(l)["id"]: json.loads(l) for l in path.read_text().splitlines() if l.strip()}


def stats(vals: list[float]) -> dict:
    if not vals:
        return {"n": 0}
    s = sorted(vals)
    return {"n": len(s), "min": round(s[0], 4), "median": round(s[len(s) // 2], 4),
            "max": round(s[-1], 4), "below_bar": sum(1 for v in s if round(v, 2) < BAR)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", required=True)
    ap.add_argument("--reference", default="rmsd_reference_all.jsonl")
    ap.add_argument("--device", default="rmsd_device_all.jsonl")
    ap.add_argument("--scores-reference", default="scores_host.jsonl")
    ap.add_argument("--scores-device", default="scores_device.jsonl")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    A = Path(a.pop)

    ref, dev = rows(A / a.reference), rows(A / a.device)
    sref, sdev = rows(A / a.scores_reference), rows(A / a.scores_device)
    # the 3-criterion union: the only designs where criterion 4 can change a decision
    union = {i for i in sref
             if all(sref[i]["ref_pass"].values()) or all(sdev.get(i, sref[i])["ref_pass"].values())}

    paired, deltas = [], []
    for rid in sorted(set(ref) & set(dev)):
        rv, dv = ref[rid].get(KEY), dev[rid].get(KEY)
        if rv is None or dv is None:
            continue
        paired.append(rid)
        deltas.append(abs(dv - rv))

    d_max = max(deltas) if deltas else 0.0
    near = [(rid, ref[rid][KEY]) for rid in paired if abs(ref[rid][KEY] - BAR) <= d_max]
    out = {
        "population": A.name, "bar": BAR,
        "population_rows": len(sref), "paired": len(paired),
        "union_3criteria": sorted(union), "reject_on_both_3criteria": len(sref) - len(union),
        "reference": stats([ref[r][KEY] for r in paired]),
        "device": stats([dev[r][KEY] for r in paired]),
        "delta": {"median": round(sorted(deltas)[len(deltas) // 2], 6) if deltas else None,
                  "max": round(d_max, 6)},
        # a design this close to the bar is one precision wobble from a decision flip
        "within_one_delta_of_bar": [{"id": r, "reference_rmsd": round(v, 4),
                                     "margin": round(abs(v - BAR), 4)} for r, v in near],
        "non_union_below_bar": sorted(
            r for r in paired if r not in union and round(ref[r][KEY], 2) < BAR),
        "decision_flips_criterion4_only": sorted(
            r for r in paired
            if (round(ref[r][KEY], 2) < BAR) != (round(dev[r][KEY], 2) < BAR)),
    }
    out["non_union_below_bar_count"] = len(out["non_union_below_bar"])
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("population", "paired", "reference", "device", "delta",
                                          "non_union_below_bar_count",
                                          "decision_flips_criterion4_only")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
