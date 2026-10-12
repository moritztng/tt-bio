#!/usr/bin/env python3
"""Grade DAVIS arms against a base arm, with a second-seed base arm as the floor.

    python perf/nesso1/davis_arms.py DV_DIR --base base --floor seed2 [ARM ...]

DV_DIR/<arm>/davis_<target>.json is what davis_validate.py writes. Every arm scores the same
compounds in the same order, so the per-ligand |arm - base| on the affinity value (`pred`) and the
binder probability (`prob`) is a parity number; the floor arm's |seed2 - base| is how far the same
model moves with nothing changed but the seed. An arm passes a target when its median and p90
per-ligand deltas sit inside the floor's and its Pearson vs pKd is not below base by more than the
floor arm is. `pred` falls as pKd rises, so the correlation is compared by magnitude.
"""
import argparse
import json
import math
import statistics
from pathlib import Path


def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    sxx = sum((a - mx) ** 2 for a in xs)
    syy = sum((b - my) ** 2 for b in ys)
    return sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else float("nan")


def load(d: Path) -> dict:
    out = {}
    for p in sorted(d.glob("davis_*.json")):
        t = json.loads(p.read_text())
        if "pairs" not in t:  # davis_t<i>.json is the per-index summary, not a target
            continue
        out[t["target_id"]] = {q["record"]: q for q in t["pairs"] if q.get("pred") is not None}
    return out


def p90(xs):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(0.9 * len(xs)))]


def compare(base, arm):
    """Per target: n, median/p90 |delta| on pred and prob, Pearson vs pKd of arm and base."""
    rows = {}
    for tid, b in base.items():
        a = arm.get(tid)
        if not a:
            continue
        recs = [r for r in b if r in a]
        dp = [abs(a[r]["pred"] - b[r]["pred"]) for r in recs]
        dq = [abs(a[r]["prob"] - b[r]["prob"]) for r in recs]
        pk = [b[r]["pkd"] for r in recs]
        rows[tid] = {"n": len(recs),
                     "pred_med": statistics.median(dp), "pred_p90": p90(dp),
                     "prob_med": statistics.median(dq), "prob_p90": p90(dq),
                     "r_arm": pearson([a[r]["pred"] for r in recs], pk),
                     "r_base": pearson([b[r]["pred"] for r in recs], pk)}
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dv", type=Path)
    ap.add_argument("arms", nargs="+")
    ap.add_argument("--base", default="base")
    ap.add_argument("--floor", default="seed2")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    base = load(a.dv / a.base)
    floor = compare(base, load(a.dv / a.floor))
    rep = {"base": a.base, "floor": {a.floor: floor}, "arms": {}}
    for name in a.arms:
        rows = compare(base, load(a.dv / name))
        for tid, r in rows.items():
            f = floor.get(tid)
            if f is None or f["n"] != r["n"]:
                r["pass"] = None
                continue
            r["pass"] = (r["pred_med"] <= f["pred_med"] and r["pred_p90"] <= f["pred_p90"]
                         and abs(r["r_arm"]) >= abs(r["r_base"]) - abs(abs(f["r_arm"]) - abs(f["r_base"])))
        rep["arms"][name] = rows
    hdr = "%-8s %-7s %3s %8s %8s %8s %8s %7s %7s %s"
    print(hdr % ("arm", "target", "n", "pred_med", "pred_p90", "prob_med", "prob_p90", "r_arm", "r_base", "pass"))
    for name, rows in [(a.floor, floor)] + list(rep["arms"].items()):
        for tid, r in rows.items():
            print("%-8s %-7s %3d %8.4f %8.4f %8.4f %8.4f %7.4f %7.4f %s" % (
                name, tid, r["n"], r["pred_med"], r["pred_p90"], r["prob_med"], r["prob_p90"],
                r["r_arm"], r["r_base"], r.get("pass", "floor")))
    if a.json:
        a.json.write_text(json.dumps(rep, indent=1) + "\n")


if __name__ == "__main__":
    main()
