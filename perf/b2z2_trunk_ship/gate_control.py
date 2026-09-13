#!/usr/bin/env python3
"""Control this branch's 44-leg parity gate against the current-main baseline, leg by leg.

The baseline is perf/b2z2_gate/gate_asg.json: main @ 4e607e07 carries a tt_bio/ and scripts/
tree byte-identical to the one that report scored (git diff 2f5072d88 origin/main -- tt_bio/
scripts/ is empty), so it IS the current-main control and does not need a second 141-minute run.

Verdicts are derived with the gate's OWN extract_verdict(), not a reimplementation of its
thresholds, so this control cannot drift from the instrument it is controlling. Two modes:

  final    perf/b2z2_gate/gate_trunkship.json exists -> compare its verdict+detail per leg.
  interim  it does not -> re-derive verdict+detail from the per-leg report JSONs the running
           gate has already written into its workdir, and control those.

A leg that does not reproduce the baseline is a NEW REGRESSION from this branch. A leg that
reproduces a baseline GAP is a pre-existing gap and is not this branch's to answer.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_gate():
    path = ROOT / "scripts" / "full_parity_gate.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("full_parity_gate", path)
    mod = importlib.util.module_from_spec(spec)
    # register before exec: the module's @dataclass resolves its own __module__ through sys.modules
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="perf/b2z2_gate/gate_asg.json")
    ap.add_argument("--report", default="perf/b2z2_gate/gate_trunkship.json")
    ap.add_argument("--workdir", default="perf/b2z2_gate/trunkship_work")
    ap.add_argument("--out", default="perf/b2z2_trunk_ship/gate_control.json")
    a = ap.parse_args()

    gate = load_gate()
    base = {l["leg"]: l for l in json.load(open(ROOT / a.baseline))["legs"]}

    report = ROOT / a.report
    if report.exists():
        mode = "final"
        live = {l["leg"]: (l["verdict"], l.get("detail", ""))
                for l in json.load(open(report))["legs"]}
    else:
        mode = "interim"
        wd = ROOT / a.workdir
        live = {}
        for lid, leg in gate.LEGS_BY_ID.items():
            j = wd / (lid + ".json")
            if not j.exists():
                continue
            live[lid] = gate.extract_verdict(leg, json.load(open(j)))

    rows, regressions = [], []
    for lid in sorted(live):
        verdict, detail = live[lid]
        b = base.get(lid)
        if b is None:
            rows.append({"leg": lid, "status": "NOT-IN-BASELINE", "live_verdict": verdict})
            regressions.append(lid)
            continue
        same_verdict = verdict == b["verdict"]
        same_detail = detail.strip() == (b.get("detail") or "").strip()
        if same_verdict and same_detail:
            status = "REPRODUCES"
        elif same_verdict:
            status = "DETAIL-DIFF"
        else:
            status = "VERDICT-DIFF"
        if status != "REPRODUCES":
            regressions.append(lid)
        rows.append({"leg": lid, "status": status,
                     "baseline_verdict": b["verdict"], "live_verdict": verdict,
                     "baseline_detail": b.get("detail", ""), "live_detail": detail})

    missing = sorted(l for l in base if l not in live)
    out = {"mode": mode, "scored": len(live), "baseline_total": len(base),
           "not_yet_scored": missing, "rows": rows, "regressions": regressions}
    (ROOT / a.out).write_text(json.dumps(out, indent=1) + "\n")

    w = max((len(r["leg"]) for r in rows), default=10)
    for r in rows:
        bv = r.get("baseline_verdict", "-")
        print("%-*s  %-14s %s -> %s" % (w, r["leg"], r["status"], bv, r["live_verdict"]))
        if r["status"] != "REPRODUCES" and "baseline_detail" in r:
            print("%-*s    baseline: %s" % (w, "", r["baseline_detail"]))
            print("%-*s    live    : %s" % (w, "", r["live_detail"]))
    print("\nmode=%s  scored %d/%d  not yet scored: %d"
          % (mode, len(live), len(base), len(missing)))
    print("REGRESSIONS: " + (", ".join(regressions) if regressions else "none"))
    return 1 if regressions else 0


if __name__ == "__main__":
    sys.exit(main())
