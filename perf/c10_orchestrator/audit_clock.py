#!/usr/bin/env python3
"""Audit recorded clocks and historical fit arithmetic without importing device code."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BASELINE_COMMIT = "463da8e9bd326be091a008da7f886c3ecf5ece20"
BASELINE_SOURCE = "perf/b2z2_aiclk_pin/out/mainab_qb2c1.json"
BASELINE_SHA256 = "543cd77b4bed2e0b1c3c603a6c510edbee6b956654d7914fdf0ee65c97337297"


def number(row, key):
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"missing or invalid {key}: {value!r}")
    return value


def validate_baseline(data, target_mhz=1350.0):
    """Validate per-fold aggregates; these records cannot prove continuous clock coverage."""
    timed = [row for row in data["runs"] if row.get("warmup") is False]
    if len(timed) != 16:
        raise ValueError(f"expected 16 timed folds, found {len(timed)}")
    result = {}
    for arm in ("base", "force"):
        rows = [row for row in timed if row.get("arm") == arm]
        if len(rows) != 8:
            raise ValueError(f"expected 8 {arm} folds, found {len(rows)}")
        for row in rows:
            if row.get("foreign_tt") != []:
                raise ValueError(f"fold {row.get('i')} has foreign holders or no holder record")
            low, mean, high = [number(row, key) for key in ("aiclk_min", "aiclk_mean", "aiclk_max")]
            count = number(row, "aiclk_n")
            if not 0 < low <= mean <= high < 3000 or count < 1 or int(count) != count:
                raise ValueError(f"invalid clock aggregates in fold {row.get('i')}")
            if low < 1200:
                raise ValueError(f"fold {row.get('i')} includes a below-1200 MHz clock artifact")
            if number(row, "fold_s") <= 0:
                raise ValueError("fold duration must be positive")
            if arm == "force" and (low, mean, high) != (target_mhz,) * 3:
                raise ValueError(f"fold {row.get('i')} does not record a constant {target_mhz:g} MHz pin")
        result[arm] = {
            "n": len(rows),
            "median_fold_s": statistics.median(row["fold_s"] for row in rows),
            "mean_of_fold_clock_means_mhz": statistics.mean(row["aiclk_mean"] for row in rows),
            "sampled_clock_min_mhz": min(row["aiclk_min"] for row in rows),
            "sampled_clock_max_mhz": max(row["aiclk_max"] for row in rows),
            "clock_sample_count": sum(row["aiclk_n"] for row in rows),
            "foreign_holder_folds": 0,
            "distinct_cif_digests": len({row["cif_sha256"] for row in rows}),
        }
    return result


def fit(rows, target_mhz):
    x = [1 / number(row, "aiclk_mean") for row in rows]
    y = [number(row, "fold_s") for row in rows]
    xm, ym = statistics.mean(x), statistics.mean(y)
    slope = sum((a - xm) * (b - ym) for a, b in zip(x, y)) / sum((a - xm) ** 2 for a in x)
    intercept = ym - slope * xm
    return {
        "n": len(rows),
        "intercept_s": intercept,
        "slope_mhz_s": slope,
        "projected_fold_s_at_target_clock": intercept + slope / target_mhz,
    }


def audit(baseline_path, historical_path, target_mhz, target_s):
    if not math.isfinite(target_mhz) or target_mhz <= 0 or not math.isfinite(target_s) or target_s <= 2.901:
        raise ValueError("target clock must be positive and target seconds must exceed model intercept 2.901")
    baseline = json.loads(baseline_path.read_text())
    historical = json.loads(historical_path.read_text())
    rows = [row for row in historical["runs"] if row.get("warmup") is False]
    if len(rows) != 6:
        raise ValueError("historical fit requires its original six timed folds")
    for row in rows:
        if number(row, "aiclk_mean") <= 0:
            raise ValueError("historical mean clock must be positive")
    ordered = sorted(rows, key=lambda row: row["aiclk_mean"])
    # These are the rounded published model constants, not profiler observations.
    intercept, slope = 2.901, 15355.0
    targets = []
    for fixed in (intercept, 1.0):
        work = (target_s - fixed) * target_mhz
        targets.append({
            "assumed_fixed_s": fixed,
            "projected_work_mcycles": work,
            "projected_work_reduction_mcycles": slope - work,
            "projected_work_reduction_fraction": 1 - work / slope,
        })
    return {
        "scope": "CPU audit of historical artifacts; no new fold or per-op cycle budget",
        "baseline": {
            "reference_git_commit": BASELINE_COMMIT,
            "reference_git_path": BASELINE_SOURCE,
            "matches_reference_bytes": hashlib.sha256(baseline_path.read_bytes()).hexdigest() == BASELINE_SHA256,
            "input_sha256": hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
            "recorded_measured_commit": baseline["env"]["commit"],
            "recorded_started_utc": baseline["env"]["started_utc"],
            "recorded_host": baseline["env"]["host"],
            "recorded_device_node": baseline["env"]["device_node"],
            "arms": validate_baseline(baseline),
            "coverage_limit": "Recorded min/mean/max are during-fold samples, not continuous proof; raw timestamps are absent.",
        },
        "historical_fit": {
            "input_sha256": hashlib.sha256(historical_path.read_bytes()).hexdigest(),
            "recorded_measured_commit": historical["env"]["commit"],
            "observations": [{key: row[key] for key in ("i", "fold_s", "aiclk_mean", "aiclk_max", "aiclk_n", "foreign_tt")} for row in rows],
            "two_clock_extremes": fit([ordered[0], ordered[-1]], target_mhz),
            "all_six_least_squares": fit(rows, target_mhz),
            "leave_one_out": [{"omitted_fold": row["i"], **fit(rows[:i] + rows[i + 1:], target_mhz)} for i, row in enumerate(rows)],
            "interpretation": "Exploratory inverse-clock model. Co-tenancy and host load vary; intercept is not identified host time and slope is not measured device cycles.",
        },
        "conditional_targets": {
            "target_clock_mhz": target_mhz,
            "target_fold_s": target_s,
            "published_model_intercept_s": intercept,
            "published_model_work_mcycles": slope,
            "published_model_projected_fold_s": intercept + slope / target_mhz,
            "scenarios": targets,
            "units": "MHz * seconds = Mcycles dimensionally; these are model-equivalent work units, not counted device cycles.",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=HERE / "pinned_baseline.json")
    parser.add_argument("--historical", type=Path, default=REPO / "perf/b2z2_cell_recheck/out/cell_main_qb2c1_clock.json")
    parser.add_argument("--target-mhz", type=float, default=1350.0)
    parser.add_argument("--target-s", type=float, default=10.0)
    parser.add_argument("--out", type=Path, help="write JSON here instead of stdout")
    args = parser.parse_args()
    try:
        result = audit(args.baseline, args.historical, args.target_mhz, args.target_s)
    except (ValueError, KeyError, OSError) as error:
        parser.exit(1, f"audit failed: {error}\n")
    rendered = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.out:
        args.out.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
