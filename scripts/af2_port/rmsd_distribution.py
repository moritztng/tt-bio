"""Step 6B: how far the whole population sits from the bound-unbound RMSD bar of 3.5.

`flip_rate4.json` only reports criterion 4 on the designs a conjunction can move, which is the 5 that
pass the three confidence criteria on at least one arm (`four_criterion.coverage.n_union`, 5 in
pxd196 and 0 in bg119). It has 7 paired RMSD rows, the union plus two designs pass 14 looked at for
other reasons, so 43 of the 50 have no RMSD at all and 45 reject on the three confidence criteria
alone. The population's distance from the bar is therefore unstated and the pooled flip rate has no
denominator for its fourth criterion. This is that denominator.

The number that matters is not the median: it is how many designs sit within one device-reference
delta of 3.5, because those are the designs one more precision wobble would flip.

`--pop` repeats. The two populations are scored separately and pooled from the raw values, because a
median of two medians is not a median and a pooled count is not a sum of rounded ones.

    PYTHONPATH=. python3 scripts/af2_port/rmsd_distribution.py \\
        --pop scripts/af2_port/parity_artifacts/designpop_pxd196 \\
        --pop scripts/af2_port/parity_artifacts/designpop_bg119:scores_reference_bf16.jsonl \\
        --out rmsd_distribution.json
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


def score_population(spec: str, a) -> dict:
    """`spec` is a population directory, optionally `dir:scores_reference.jsonl` where that
    population's reference scores are not in the default file."""
    path, _, scores_reference = spec.partition(":")
    A = Path(path)
    ref, dev = rows(A / a.reference), rows(A / a.device)
    sref = rows(A / (scores_reference or a.scores_reference))
    sdev = rows(A / a.scores_device)
    # the 3-criterion union: the only designs where criterion 4 can change a decision
    union = {i for i in sref
             if all(sref[i]["ref_pass"].values()) or all(sdev.get(i, sref[i])["ref_pass"].values())}

    paired, deltas = [], []
    for rid in sorted(set(ref) & set(dev)):
        if ref[rid].get(KEY) is None or dev[rid].get(KEY) is None:
            continue
        paired.append(rid)
        deltas.append(abs(dev[rid][KEY] - ref[rid][KEY]))

    d_max = max(deltas) if deltas else 0.0
    near = [(rid, ref[rid][KEY]) for rid in paired if abs(ref[rid][KEY] - BAR) <= d_max]
    out = {
        "population": A.name, "bar": BAR,
        "population_rows": len(sref), "paired": len(paired),
        "missing_pair": sorted(set(sref) - set(paired)),
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
    out["_raw"] = {r: [ref[r][KEY], dev[r][KEY]] for r in paired}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", required=True, action="append",
                    help="population directory, or DIR:scores_reference.jsonl")
    ap.add_argument("--reference", default="rmsd_reference_all.jsonl")
    ap.add_argument("--device", default="rmsd_device_all.jsonl")
    ap.add_argument("--scores-reference", default="scores_host.jsonl")
    ap.add_argument("--scores-device", default="scores_device.jsonl")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    per = [score_population(spec, a) for spec in a.pop]
    raw = {}
    for p in per:
        raw.update(p.pop("_raw"))
    ref_vals = [v[0] for v in raw.values()]
    deltas = [abs(v[1] - v[0]) for v in raw.values()]
    d_max = max(deltas) if deltas else 0.0
    pooled = {
        "paired": len(raw),
        "reference": stats(ref_vals),
        "device": stats([v[1] for v in raw.values()]),
        "delta": {"median": round(sorted(deltas)[len(deltas) // 2], 6) if deltas else None,
                  "max": round(d_max, 6)},
        "within_one_delta_of_bar": sorted(
            r for r, v in raw.items() if abs(v[0] - BAR) <= d_max),
        "reject_on_both_3criteria": sum(p["reject_on_both_3criteria"] for p in per),
        "non_union_below_bar_count": sum(p["non_union_below_bar_count"] for p in per),
        "decision_flips_criterion4_only": sorted(
            r for p in per for r in p["decision_flips_criterion4_only"]),
    }
    out = {"pooled": pooled, "populations": per}
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(pooled, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
