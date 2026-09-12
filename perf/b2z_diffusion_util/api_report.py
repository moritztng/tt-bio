#!/usr/bin/env python3
"""Same two numbers as `util_report.py`, from the per-program profiler API instead of tracy.

`get_all_programs_perf_data()` returns core_count / num_available_cores / durations per
dispatched program and nothing else -- no op code, no shapes -- so this cannot rank offenders.
It exists to cross the tracy ops report, which can, against an instrument that never builds
the host log. Agreement on the duration-weighted mean is the check that the CSV's region
finding and per-op medians are not inventing the answer.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path


def duration_of(rec, prefer):
    an = rec["analyses"]
    for k in prefer:
        if k in an and an[k]["duration"]:
            return float(an[k]["duration"]), k
    best = max(an.items(), key=lambda kv: kv[1]["duration"] or 0) if an else None
    return (float(best[1]["duration"]), best[0]) if best else (0.0, None)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", type=Path, required=True)
    ap.add_argument("--calls", type=int, required=True,
                    help="how many Diffusion calls the captured programs span")
    ap.add_argument("--calls-per-fold", type=int, default=200)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-cores", type=int, default=2,
                    help="drop the 1-core fence/bookkeeping programs below this")
    a = ap.parse_args()

    d = json.loads(a.api.read_text())
    progs = d["programs"]
    prefer = ("device_kernel_duration", "DEVICE KERNEL DURATION", "kernel_duration")
    keys = set()
    rows = []
    for p in progs:
        dur, k = duration_of(p, prefer)
        keys.add(k)
        if p["core_count"] < a.min_cores:
            continue
        rows.append({"cores": p["core_count"], "avail": p["num_available_cores"],
                     "ns": dur, "runtime_id": p["runtime_id"]})
    avail = {r["avail"] for r in rows}
    if len(avail) != 1:
        raise SystemExit(f"non-uniform num_available_cores: {sorted(avail)}")
    C = avail.pop()
    tot = sum(r["ns"] for r in rows)
    wmean = sum(r["ns"] * r["cores"] / C for r in rows) / tot
    idle = sum(r["ns"] * (C - r["cores"]) / C for r in rows)
    under = [r for r in rows if r["cores"] / C < 0.5]
    per_call = a.calls_per_fold / a.calls / 1e9
    out = {
        "source": str(a.api),
        "duration_keys_seen": sorted(k for k in keys if k),
        "num_available_cores": C,
        "programs_captured": len(progs),
        "programs_counted": len(rows),
        "programs_per_call": len(rows) / a.calls,
        "calls_captured": a.calls,
        "kernel_ms_per_call": tot / 1e6 / a.calls,
        "duration_weighted_mean_core_utilization": wmean,
        "idle_core_fraction_seconds_per_fold_all_ops": idle * per_call,
        "under_grid_ops_per_call": len(under) / a.calls,
        "core_seconds_lost_per_fold_under_grid":
            sum(r["ns"] * (C - r["cores"]) / C for r in under) * per_call,
        "core_count_histogram": dict(sorted(
            {c: sum(1 for r in rows if r["cores"] == c)
             for c in {r["cores"] for r in rows}}.items())),
        "median_kernel_ns": st.median([r["ns"] for r in rows]) if rows else 0,
    }
    a.out.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "core_count_histogram"}, indent=1))
    print("core-count histogram:", json.dumps(out["core_count_histogram"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
