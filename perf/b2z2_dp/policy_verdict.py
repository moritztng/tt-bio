#!/usr/bin/env python3
"""Attach the OMP-wait-policy comparison to a scaling table scaling.py has already written.

scaling.py builds the table (widths, efficiency against the width-1 arm, the A/A floor, the digest
check) and is not reimplemented here. This adds the second axis the policy question needs:

  * the ACTIVE control arms measured in the SAME session as the passive ones, so the policy
    difference cannot be a session-to-session load difference. The baseline curve in scaling.json
    is a different session and is carried alongside as a cross-check, not as the control.
  * per width, passive vs active on the three numbers that decide it: folds per hour (what the
    fleet gets), median fold seconds (what one user waits) and host cores per fold (why).
  * the verdict string, GO or NO-GO, on shipping OMP_WAIT_POLICY=PASSIVE as a DP dispatch default.

    policy_verdict.py --table ~/.coworker/state/b2z2_dp/scaling_passive.json \
        --active perf/b2z2_dp/pol/active --baseline ~/.coworker/state/b2z2_dp/scaling.json \
        --verdict "..."
"""
import argparse
import json
from pathlib import Path

KEYS = ("n_timed_folds", "window_s", "folds_per_hour", "median_fold_s", "min_fold_s",
        "max_fold_s", "median_cores_per_fold", "total_cores", "median_step_ms",
        "cif_digests", "digest_unanimous")


def arms(d: Path):
    out = {}
    for p in sorted(d.glob("w*.json")):
        a = json.loads(p.read_text())
        if not a.get("all_children_ok"):
            print(f"skip {p.name}: children not ok")
            continue
        r = a["result"]
        rec = {k: r[k] for k in KEYS}
        rec["total_host_cores"] = rec.pop("total_cores")
        rec["artifact"] = p.name
        rec["started_utc"] = a["started_utc"]
        rec["loadavg_at_start"] = a["loadavg_at_start"]
        out.setdefault(str(a["width"]), rec)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=Path, required=True)
    ap.add_argument("--active", type=Path, required=True)
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--verdict", required=True)
    args = ap.parse_args()

    t = json.loads(args.table.read_text())
    ctrl = arms(args.active)
    base = json.loads(args.baseline.read_text())["widths"]

    cmp_ = {}
    for w, p in t["widths"].items():
        row = {"passive": {k: p[k] for k in ("folds_per_hour", "median_fold_s",
                                             "median_cores_per_fold", "total_host_cores")}}
        for name, src in (("active_same_session", ctrl.get(w)), ("active_baseline_session",
                                                                 base.get(w))):
            if not src:
                continue
            row[name] = {k: src[k] for k in ("folds_per_hour", "median_fold_s",
                                             "median_cores_per_fold", "total_host_cores")}
            row[f"folds_per_hour_ratio_vs_{name}"] = round(
                p["folds_per_hour"] / src["folds_per_hour"], 4)
            row[f"median_fold_ratio_vs_{name}"] = round(
                p["median_fold_s"] / src["median_fold_s"], 4)
        cmp_[w] = row

    t["policy"] = {"OMP_WAIT_POLICY": "PASSIVE", "GOMP_SPINCOUNT": "0", "KMP_BLOCKTIME": "0"}
    t["control_active_arms"] = ctrl
    t["passive_vs_active"] = cmp_
    t["baseline_table"] = str(args.baseline)
    t["verdict"] = args.verdict
    args.table.write_text(json.dumps(t, indent=1))

    for w in sorted(cmp_, key=int):
        r = cmp_[w]
        line = f"w={w:>2} passive {r['passive']['folds_per_hour']:>9.2f} f/h"
        for name in ("active_same_session", "active_baseline_session"):
            if name in r:
                line += (f"   vs {name}: {r[name]['folds_per_hour']:>9.2f} f/h "
                         f"x{r[f'folds_per_hour_ratio_vs_{name}']:.4f}")
        print(line)
    print(t["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
