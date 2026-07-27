"""Aggregate a Tracy ops report into a matrix-engine utilization table.

Reads ``ops_perf_results_*.csv`` produced by

    python -m tracy -r --profiler-capture-perf-counters fpu ...

and reports, per op code and in total, how much device kernel time was spent
and what fraction of the matrix engine (FPU) was actually busy while it ran.
``Avg FPU util on full grid (%)`` already folds in grid occupancy, so a value
of 30% means the op used 30% of the chip's matrix-engine issue slots over its
own duration -- the number a "is it compute-saturated?" claim needs.

Usage:
    python summarize_fpu_utilization.py <report.csv> [--top 25] [--json out.json]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


DUR = "DEVICE KERNEL DURATION [ns]"
FPU = "Avg FPU util on full grid (%)"
MATH = "Avg Math util on full grid (%)"
SFPU = "Avg SFPU util on full grid (%)"
DRAM = "DRAM BW UTIL (%)"
CORES = "CORE COUNT"
AVAIL = "AVAILABLE WORKER CORE COUNT"


def num(value: str):
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        out = float(value)
    except ValueError:
        return None
    return None if math.isnan(out) else out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    rows = list(csv.DictReader(args.csv_path.open()))
    agg: dict[str, dict] = {}
    total_ns = 0.0
    fpu_weighted = 0.0
    fpu_ns = 0.0
    for row in rows:
        code = row.get("OP CODE") or "?"
        dur = num(row.get(DUR))
        if dur is None:
            continue
        entry = agg.setdefault(
            code,
            {"count": 0, "ns": 0.0, "fpu_ns": 0.0, "fpu_w": 0.0,
             "sfpu_w": 0.0, "dram_w": 0.0, "dram_ns": 0.0,
             "cores": 0.0, "avail": 0.0, "shapes": {}},
        )
        entry["count"] += 1
        entry["ns"] += dur
        total_ns += dur
        fpu = num(row.get(FPU))
        if fpu is not None:
            entry["fpu_ns"] += dur
            entry["fpu_w"] += dur * fpu
            entry["sfpu_w"] += dur * (num(row.get(SFPU)) or 0.0)
            fpu_ns += dur
            fpu_weighted += dur * fpu
        dram = num(row.get(DRAM))
        if dram is not None:
            entry["dram_ns"] += dur
            entry["dram_w"] += dur * dram
        entry["cores"] += (num(row.get(CORES)) or 0.0) * dur
        entry["avail"] += (num(row.get(AVAIL)) or 0.0) * dur
        shape = "x".join(
            (row.get(f"INPUT_0_{ax}_PAD[LOGICAL]") or "").strip()
            for ax in ("W", "Z", "Y", "X")
        )
        shape2 = "x".join(
            (row.get(f"INPUT_1_{ax}_PAD[LOGICAL]") or "").strip()
            for ax in ("W", "Z", "Y", "X")
        )
        key = f"{shape} @ {shape2}"
        entry["shapes"][key] = entry["shapes"].get(key, 0.0) + dur

    ordered = sorted(agg.items(), key=lambda kv: -kv[1]["ns"])
    label = f" [{args.label}]" if args.label else ""
    print(f"# {args.csv_path.name}{label}")
    print(f"ops={len(rows)} distinct_op_codes={len(agg)} "
          f"total_device_kernel_ms={total_ns / 1e6:.3f}")
    if fpu_ns:
        print(f"TIME-WEIGHTED AVG FPU UTIL (full grid) = "
              f"{fpu_weighted / fpu_ns:.2f}%  "
              f"(counter coverage {fpu_ns / total_ns * 100:.1f}% of device time)")
    print()
    header = (f"{'OP CODE':38s} {'n':>6s} {'ms':>9s} {'%tot':>6s} "
              f"{'FPU%':>7s} {'SFPU%':>7s} {'cores':>6s} {'DRAM%':>7s}")
    print(header)
    print("-" * len(header))
    for code, e in ordered[: args.top]:
        fpu_avg = e["fpu_w"] / e["fpu_ns"] if e["fpu_ns"] else float("nan")
        sfpu_avg = e["sfpu_w"] / e["fpu_ns"] if e["fpu_ns"] else float("nan")
        dram_avg = e["dram_w"] / e["dram_ns"] if e["dram_ns"] else float("nan")
        cores = e["cores"] / e["ns"] if e["ns"] else float("nan")
        print(f"{code[:38]:38s} {e['count']:6d} {e['ns'] / 1e6:9.3f} "
              f"{e['ns'] / total_ns * 100:5.1f}% {fpu_avg:7.2f} {sfpu_avg:7.2f} "
              f"{cores:6.1f} {dram_avg:7.2f}")

    if args.json:
        out = {
            "csv": str(args.csv_path),
            "label": args.label,
            "ops": len(rows),
            "total_device_kernel_ns": total_ns,
            "avg_fpu_util_pct": fpu_weighted / fpu_ns if fpu_ns else None,
            "fpu_counter_coverage_pct": fpu_ns / total_ns * 100 if total_ns else None,
            "by_op": {
                code: {
                    "count": e["count"],
                    "ns": e["ns"],
                    "pct_of_total": e["ns"] / total_ns * 100 if total_ns else None,
                    "avg_fpu_util_pct": e["fpu_w"] / e["fpu_ns"] if e["fpu_ns"] else None,
                    "avg_sfpu_util_pct": e["sfpu_w"] / e["fpu_ns"] if e["fpu_ns"] else None,
                    "avg_dram_bw_util_pct": e["dram_w"] / e["dram_ns"] if e["dram_ns"] else None,
                    "avg_cores": e["cores"] / e["ns"] if e["ns"] else None,
                    "top_shapes": sorted(
                        e["shapes"].items(), key=lambda kv: -kv[1]
                    )[:5],
                }
                for code, e in ordered
            },
        }
        args.json.write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
