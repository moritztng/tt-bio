#!/usr/bin/env python3
"""Score one of our trajectories against one of upstream's, per parameter per step.

PROTOCOL 5's bar is 1e-06 per parameter per step, the fp32 master's own floor. The reading
is relative L2 of the parameter VALUE, `||ours - theirs|| / ||theirs||`, which is what
`of3t-rebind`'s D107.json reported so the numbers stay comparable.

A16: every reading is published beside its measured zero baseline -- here the arm that never
steps at all, so its parameter stays at its initial value. A reading at or above that carries
no information about the update rule.
"""
import argparse
import json

import numpy as np


def rel(a, b):
    num = float(np.linalg.norm(a - b))
    den = float(np.linalg.norm(b))
    return num / den if den > 0 else float("nan")


def score(ours, theirs, init):
    rows = []
    for o, t in zip(ours["trace"], theirs["trace"]):
        assert o["k"] == t["k"]
        for name in sorted(t["theta"]):
            a = np.asarray(o["theta"][name], np.float64)
            b = np.asarray(t["theta"][name], np.float64)
            rows.append({"k": o["k"], "param": name,
                         "participation": t["participation"][name],
                         "rel": rel(a, b),
                         "zero_baseline": rel(np.asarray(init[name], np.float64), b),
                         "ours_norm": float(np.linalg.norm(a)),
                         "theirs_norm": float(np.linalg.norm(b))})
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("grads")
    ap.add_argument("ours")
    ap.add_argument("theirs")
    ap.add_argument("--bar", type=float, default=1e-06)
    ap.add_argument("--label", default="")
    ap.add_argument("--out")
    a = ap.parse_args()

    data = np.load(a.grads)
    init = {k.split("/", 1)[1]: data[k] for k in data.files if k.startswith("init/")}
    ours, theirs = json.load(open(a.ours)), json.load(open(a.theirs))
    rows = score(ours, theirs, init)

    zero_rows = [r for r in rows if r["participation"] == 0]
    live_rows = [r for r in rows if r["participation"] > 0]
    after = [r for r in rows if r["param"].startswith("conf.")
             and r["k"] > max([r["k"] for r in zero_rows] or [0])]
    worst = max(rows, key=lambda r: r["rel"])
    # A16: the reading is published beside the zero baseline OF THE SAME ROW -- the arm that
    # never steps, so the parameter keeps its initial value. A reading at or above it is a
    # ceiling. The minimum over all rows is not that baseline and would flatter a bad reading
    # taken on a row whose reference had barely moved.
    summary = {
        "label": a.label,
        "bar": a.bar,
        "reference": theirs.get("reference"), "reference_dtype": theirs.get("dtype"),
        "reference_skip_zero_participation": theirs.get("skip_zero_participation"),
        "worst_rel_all": worst["rel"],
        "worst_row": {k: worst[k] for k in ("k", "param", "participation")},
        "worst_row_zero_baseline": worst["zero_baseline"],
        "worst_row_separation_from_zero": (worst["zero_baseline"] / worst["rel"]
                                           if worst["rel"] > 0 else float("inf")),
        "worst_rel_zero_participation": max((r["rel"] for r in zero_rows), default=None),
        "worst_rel_live": max(r["rel"] for r in live_rows),
        "worst_rel_after_zero_step": max((r["rel"] for r in after), default=None),
        "zero_baseline_range": [min(r["zero_baseline"] for r in rows),
                                max(r["zero_baseline"] for r in rows)],
        "over_bar": sorted({r["param"] for r in rows if r["rel"] > a.bar}),
        "n_over_bar": sum(1 for r in rows if r["rel"] > a.bar),
        "n_readings": len(rows),
        "rows": rows,
    }
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=1))
    if a.out:
        json.dump(summary, open(a.out, "w"), indent=1)
