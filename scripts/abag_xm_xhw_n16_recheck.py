#!/usr/bin/env python3
"""Cross-hardware N=16 flavor-clean recheck (datasheet section 9 evidence).

Compares the restated N=16 rung (p2-era galaxy structures re-scored with the
ARK-interface DockQ scorer, deepn/n16_ark/) against tier_a (qb1, same scorer
flavor) on oracle@16 per target. Hardware and seeds differ; the scorer flavor
does not -- so this isolates the cross-hardware fold-difference component at
the panel's shallowest rung, complementing the N=64 gate (phase0_n64_gate.py).

Per model: median |delta|, median signed delta, fractions above 0.1/0.2, the
both-arms-solve stratum (oracle@16 >= 0.23 on both arms -- where agreement is
expected to be tight), and the largest solve/unsolved flips. esmfold2 excludes
9loz/9w14 (p2-era pipeline mis-folds; N16_ARK_EXCLUDE in the analysis script).

Runs on qb1 (the data trees live there). Read-only. Prints a markdown table
and writes JSON for the datasheet.

  python3 scripts/abag_xm_xhw_n16_recheck.py [--out xhw_n16_recheck.json]
"""
import argparse, json, statistics as st
from pathlib import Path

BASE = Path.home() / "abag_xm"
N16_ARK = BASE / "deepn" / "n16_ark"
TIER_A_LABELS = BASE / "tier_a" / "labels"
# Same exclusion as abag_xm_deepn_analysis.N16_ARK_EXCLUDE (keep in sync).
EXCLUDE = {"esmfold2": {"9loz", "9w14"}}
MODELS = {"protenix-v2": ("protenix", "protenix_v2"),
          "boltz2": ("boltz2", "boltz2"),
          "esmfold2": ("esmfold2", "esmfold2")}
SOLVE = 0.23  # DockQ "acceptable" threshold, matches the analysis THR ladder


def dockqs(labels_json: Path):
    """Per-sample DockQ values from a labels.json, None on any gap."""
    try:
        d = json.loads(labels_json.read_text())
    except Exception:
        return None
    out = []
    for s in d.get("samples", []):
        dq = s.get("dockq")
        v = dq.get("dockq") if isinstance(dq, dict) else dq
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if v == v:  # NaN guard
            out.append(v)
    return out or None


def model_rows(prefix: str, md: str):
    rows = []
    for lj in sorted((N16_ARK / prefix).glob("*/labels.json")):
        t = lj.parent.name.split("_n")[0]
        a = dockqs(lj)
        b = dockqs(TIER_A_LABELS / f"{md}_{t}.json")
        if not a or not b or len(a) < 16 or len(b) < 16:
            continue
        rows.append({"target": t,
                     "ark_oracle16": max(a[:16]),
                     "tier_a_oracle16": max(b[:16]),
                     "tier_a_oracle50": max(b),
                     "delta": max(a[:16]) - max(b[:16])})
    return rows


def summarize(rows):
    d = [r["delta"] for r in rows]
    ad = [abs(x) for x in d]
    both = [r for r in rows
            if r["ark_oracle16"] >= SOLVE and r["tier_a_oracle16"] >= SOLVE]
    bd = [abs(r["delta"]) for r in both]
    flips = sorted(rows, key=lambda r: -abs(r["delta"]))[:6]
    return {
        "n": len(rows),
        "med_abs_delta": st.median(ad),
        "med_signed_delta": st.median(d),
        "frac_abs_gt_0.1": sum(x > 0.1 for x in ad) / len(ad),
        "frac_abs_gt_0.2": sum(x > 0.2 for x in ad) / len(ad),
        "both_solve": {"n": len(both),
                       "med_abs_delta": st.median(bd) if bd else None,
                       "frac_abs_gt_0.2": (sum(x > 0.2 for x in bd) / len(bd)) if bd else None},
        "worst_flips": flips,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    report = {}
    for model, (prefix, md) in MODELS.items():
        rows = [r for r in model_rows(prefix, md)
                if r["target"] not in EXCLUDE.get(model, ())]
        if rows:
            report[model] = summarize(rows)

    lines = ["| model | n | med |delta| | med signed | frac>0.1 | frac>0.2 |"
             " both-solve n | both-solve med |d| | both-solve frac>0.2 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for m, s in report.items():
        bs = s["both_solve"]
        lines.append(
            f"| {m} | {s['n']} | {s['med_abs_delta']:.4f} | {s['med_signed_delta']:+.4f} |"
            f" {s['frac_abs_gt_0.1']:.3f} | {s['frac_abs_gt_0.2']:.3f} |"
            f" {bs['n']} | {bs['med_abs_delta']:.4f} | {bs['frac_abs_gt_0.2']:.3f} |")
    print("\n".join(lines))
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=1))
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
