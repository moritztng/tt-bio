#!/usr/bin/env python3
"""The QUIET leg: CPU-capped and uncapped rounds of one run, side by side.

Rounds are cut and attributed by `perf/bcx_round/analyze.py`, unchanged. This joins each
round to the priority it ran at and to the CPU seconds and run-queue wait its threads
accumulated, then reports the two arms' medians and ranges.
"""
import io
import json
import statistics as st
import sys
import contextlib
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "bcx_round"))
import analyze  # noqa: E402


def med(xs):
    return round(st.median(xs), 3) if xs else None


def main(run_dir):
    ev_path = f"{run_dir}/round_events.json"
    out = f"{run_dir}/round_summary.json"
    with contextlib.redirect_stdout(io.StringIO()):
        analyze.main(ev_path, out)
    rows = json.load(open(out))["rounds"]
    ev = json.load(open(ev_path))["events"]
    cpus = {e["round"]: e["cpus"] for e in ev if e["kind"] == "cpus"}
    full = max(cpus.values()) if cpus else None
    sch = {e["round"]: e for e in ev if e["kind"] == "sched"}
    lc = {}
    for e in ev:
        if e["phase"] == "lower_compile":
            lc[e["round"]] = lc.get(e["round"], 0) + e["dt"]
    for r in rows:
        k = r["round"]
        a, b = sch.get(k), sch.get(k + 1)
        r["cpus"] = cpus.get(k)
        if a and b:
            r["cpu_s"] = round(b["cpu_s"] - a["cpu_s"], 2)
            r["wait_s"] = round(b["wait_s"] - a["wait_s"], 2)
        r["lower_compile_s"] = round(lc.get(k, 0), 3)
        r["host_s"] = round(r["wall"] - r["taped_s"] - r["bwd_s"] - r["primal_s"], 3)
        r["host_share"] = round(r["host_s"] / r["wall"], 4)
    # Out: BindCraft 2's 10 ms nested entry from its compile path, and the round that pays
    # the jit compile. Both are named in bcx-round and neither is a gradient round.
    body = [r for r in rows if r["wall"] > 1.0 and r["lower_compile_s"] < 1.0]
    arms = {}
    for label, sel in (("all_cpus", lambda r: r["cpus"] == full), ("capped", lambda r: r["cpus"] != full)):
        rs = [r for r in body if sel(r)]
        if not rs:
            continue
        arms[label] = {
            "n": len(rs), "cpus": rs[0]["cpus"],
            "wall_med": med([r["wall"] for r in rs]),
            "wall_min": min(r["wall"] for r in rs), "wall_max": max(r["wall"] for r in rs),
            "host_med": med([r["host_s"] for r in rs]),
            "host_min": min(r["host_s"] for r in rs), "host_max": max(r["host_s"] for r in rs),
            "device_med": med([r["taped_s"] + r["bwd_s"] + r["primal_s"] for r in rs]),
            "bwd_med": med([r["bwd_s"] for r in rs]),
            "host_share_med": med([r["host_share"] for r in rs]),
            "cpu_s_med": med([r.get("cpu_s") for r in rs if "cpu_s" in r]),
            "wait_s_med": med([r.get("wait_s") for r in rs if "wait_s" in r]),
            "lower_compile_med": med([r["lower_compile_s"] for r in rs]),
            "aiclk_med": med([r["aiclk_med"] for r in rs if r["aiclk_med"]]),
            "aiclk_min": min(r["aiclk_min"] for r in rs if r["aiclk_min"]),
            "load1_min": min(r["load1"] for r in rs if r["load1"] is not None),
            "load1_max": max(r["load1"] for r in rs if r["load1"] is not None),
        }
    res = {"arms": arms, "rounds": rows}
    json.dump(res, open(f"{run_dir}/quiet.json", "w"), indent=1)
    print(json.dumps(arms, indent=1))
    for r in rows:
        print(" ".join(f"{k}={r.get(k)}" for k in (
            "round", "cpus", "wall", "host_s", "host_share", "taped_s", "bwd_s", "cpu_s",
            "wait_s", "lower_compile_s", "aiclk_med", "aiclk_min", "load1")))


if __name__ == "__main__":
    main(sys.argv[1])
