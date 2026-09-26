#!/usr/bin/env python3
"""Split `perf/bcx_round/analyze.py`'s per-round table by the arm each round ran.

`round_ab.py` tags every round_start with its arm; this joins that to the analysis rows and
reports the medians per arm. Round 1 is dropped (BindCraft 2's jit compile) and so is round 2,
which pays the first ttnn program compile for whichever dtype it was the first to ask for.
"""
import json
import statistics as st
import sys


def med(xs):
    return round(st.median(xs), 4) if xs else None


def main(events_path, analysis_path, out=None, drop=2):
    ev = json.load(open(events_path))["events"]
    arm_of = {e["round"]: e.get("arm") for e in ev if e["kind"] == "round_start"}
    an = json.load(open(analysis_path))
    rows = [dict(r, arm=arm_of.get(r["round"])) for r in an["rounds"]]
    body = [r for r in rows if r["round"] > drop]
    arms = sorted({r["arm"] for r in body if r["arm"]})
    per = {}
    for a in arms:
        rs = [r for r in body if r["arm"] == a]
        per[a] = {
            "n_rounds": len(rs),
            "wall": med([r["wall"] for r in rs]),
            "wall_min": min(r["wall"] for r in rs), "wall_max": max(r["wall"] for r in rs),
            "taped_s": med([r["taped_s"] for r in rs]),
            "taped_n": med([r["taped_n"] for r in rs]),
            "bwd_s": med([r["bwd_s"] for r in rs]),
            "device_s": med([r["taped_s"] + r["bwd_s"] for r in rs]),
            "host_in_sg": med([r["host_in_sg"] for r in rs]),
            "sg": med([r["sg"] for r in rs]),
            "aiclk_med": med([r["aiclk_med"] for r in rs]),
            "aiclk_min": min(r["aiclk_min"] for r in rs if r["aiclk_min"]),
            "load1": med([r["load1"] for r in rs]),
        }
    base = per.get("bf16")
    if base:
        for a in per:
            per[a]["x_vs_bf16"] = {k: round(base[k] / per[a][k], 4)
                                   for k in ("wall", "taped_s", "bwd_s", "device_s")
                                   if per[a][k]}
    res = {"stamp": an["stamp"], "dropped_rounds": drop, "per_arm": per, "rows": rows}
    print(json.dumps(per, indent=1))
    print()
    for r in rows:
        print(" ".join(f"{k}={r[k]}" for k in
                       ("round", "arm", "wall", "taped_n", "taped_s", "bwd_s",
                        "host_in_sg", "aiclk_med", "load1")))
    if out:
        json.dump(res, open(out, "w"), indent=1)
    return res


if __name__ == "__main__":
    main(*sys.argv[1:])
