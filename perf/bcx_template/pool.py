#!/usr/bin/env python3
"""Pool the template A/B runs: arm medians, matched pairs, and the same two with host stalls cut.

A round whose `sequence_gradients` exceeds `--stall` seconds is a host stall, not a reading of
this lever: the box is shared, both arms take them, and one 42 s round moves an arm median by
more than the lever does. They are reported, counted and then cut, and every number is given
both ways so the cut is visible rather than assumed.
"""
import argparse
import json
import statistics as st

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
ap.add_argument("--stall", type=float, default=25.0)
a = ap.parse_args()

rows = []
for r in a.runs:
    d = json.load(open(r + "/round_ab.json"))
    for x in d["rounds"]:
        if x["round"] > 2:
            rows.append({**x, "run": r.rstrip("/").rsplit("/", 1)[-1]})

def report(rs, label):
    on = sorted(x["sequence_gradients_s"] for x in rs if x["template_on_device"])
    off = sorted(x["sequence_gradients_s"] for x in rs if not x["template_on_device"])
    pairs = []
    for run in {x["run"] for x in rs}:
        by = {x["round"]: x for x in rs if x["run"] == run}
        for x in by.values():
            if not x["template_on_device"]:
                continue
            nb = [by[r]["sequence_gradients_s"] for r in (x["round"] - 1, x["round"] + 1)
                  if r in by and by[r]["template_on_device"] is False]
            if nb:
                pairs.append(round(sum(nb) / len(nb) - x["sequence_gradients_s"], 3))
    tmpl = [x["device_template_s"] for x in rs if x["template_on_device"]]
    out = {
        "label": label, "n_on": len(on), "n_off": len(off),
        "off_median": round(st.median(off), 3), "on_median": round(st.median(on), 3),
        "ratio_off_over_on": round(st.median(off) / st.median(on), 3),
        "seconds_saved_arm_medians": round(st.median(off) - st.median(on), 3),
        "paired_n": len(pairs), "paired_median_s": round(st.median(pairs), 3),
        "paired_mean_s": round(sum(pairs) / len(pairs), 3),
        "paired_faster_on_card": sum(1 for d in pairs if d > 0),
        "device_template_median_s": round(st.median(tmpl), 4),
        "on_quartiles": [on[len(on) // 4], on[len(on) // 2], on[3 * len(on) // 4]],
        "off_quartiles": [off[len(off) // 4], off[len(off) // 2], off[3 * len(off) // 4]],
    }
    ranked = sorted(rs, key=lambda x: x["sequence_gradients_s"])
    rank_on = sum(i + 1 for i, x in enumerate(ranked) if x["template_on_device"])
    u_on = rank_on - len(on) * (len(on) + 1) / 2
    out["fraction_of_cross_pairs_on_is_faster"] = round(1 - u_on / (len(on) * len(off)), 3)
    return out

# Pooling the RAW seconds across runs is not a reading of this lever: the two runs sat on
# different baselines (one at an off median of 14.9 s, the other at 16.7 s), and mixing them
# moves the pooled median without moving either arm's own. Per-run arm medians and the pooled
# matched pairs are the estimators; the naive pooled median is printed beside them to be seen,
# not used.
per_run = {}
for run in sorted({x["run"] for x in rows}):
    rs = [x for x in rows if x["run"] == run]
    per_run[run] = report(rs, run)

res = {"runs": a.runs, "stall_threshold_s": a.stall, "per_run": per_run,
       "all": report(rows, "every round"),
       "stalls_cut": report([x for x in rows if x["sequence_gradients_s"] <= a.stall],
                            f"rounds under {a.stall} s"),
       "stalls": sorted(round(x["sequence_gradients_s"], 2) for x in rows
                        if x["sequence_gradients_s"] > a.stall)}
print(json.dumps(res, indent=1))
