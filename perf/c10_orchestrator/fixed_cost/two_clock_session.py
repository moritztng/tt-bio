#!/usr/bin/env python3
"""The clock-immune fixed cost, measured from an interleaved two-clock session already on disk.

`b2z2-aiclk-default-decision` ran an A/B on commit 0df13ad9 that alternated a forced 800 MHz arm
with an unforced ~1339 MHz arm, same process, same fixture, six non-warmup folds. Nobody ever
solved it for the fixed term. Two clocks and two unknowns is all it takes: the fixed term is the
only part of the fold that does not scale with the clock.

CPU only. Reads committed JSON from the child's branch; opens no device and measures nothing new.
"""
import json
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).parent
TWO_CLOCK = HERE / "inputs" / "mainab_down800_qb2c1.json"
PINNED = HERE / "inputs" / "mainab_qb2c1.json"
CURRENT_512 = 14.8813252375   # origin/wk/c10-bare-baseline bc66f7d6d, pooled median, 1350 MHz


def fit(points):
    """Least squares of t = a + b/f over (seconds, MHz) points."""
    n = len(points)
    sx = sum(1.0 / f for _, f in points)
    sy = sum(t for t, _ in points)
    sxx = sum((1.0 / f) ** 2 for _, f in points)
    sxy = sum(t / f for t, f in points)
    denom = n * sxx - sx * sx
    if denom == 0:
        raise ValueError("every point shares a clock label; that cannot identify a slope")
    b = (n * sxy - sx * sy) / denom
    return (sy - b * sx) / n, b


def accepted(doc):
    return [r for r in doc["runs"] if not r.get("warmup")]


def main():
    two = json.loads(TWO_CLOCK.read_text())
    pin = json.loads(PINNED.read_text())
    runs = accepted(two)
    pts = [(r["fold_s"], r["aiclk_mean"]) for r in runs]
    a, b = fit(pts)
    loo = [fit([p for j, p in enumerate(pts) if j != i]) for i in range(len(pts))]

    pr = accepted(pin)
    forced = [r for r in pr if r["arm"] == "force"]
    base = [r for r in pr if r["arm"] == "base"]
    fm, ff = st.median([r["fold_s"] for r in forced]), st.median([r["aiclk_mean"] for r in forced])
    bm, bf = st.median([r["fold_s"] for r in base]), st.median([r["aiclk_mean"] for r in base])

    work_now = (CURRENT_512 - a) * 1350.0
    out = {
        "scope": "CPU solve of a committed interleaved two-clock A/B. No device, no new timing.",
        "inputs": {
            "two_clock": "origin/wk/b2z2-aiclk-default-decision perf/b2z2_aiclk_pin/out/mainab_down800_qb2c1.json",
            "pinned_cross_check": "origin/wk/b2z2-aiclk-default-decision perf/b2z2_aiclk_pin/out/mainab_qb2c1.json",
            "commit": two["env"]["commit"],
            "fixture": two["env"]["fixture"],
            "steps": two["env"]["steps"],
            "recycles": two["env"]["recycles"],
            "timer": "time.perf_counter around state.predict_one, the same boundary c10-bare-baseline used",
        },
        "measured": {
            "fixed_s": a,
            "work_Mcycles": b,
            "n_folds": len(pts),
            "arm_medians_s": {
                "forced_800MHz": st.median([r["fold_s"] for r in runs if r["arm"] == "force"]),
                "unforced_1339MHz": st.median([r["fold_s"] for r in runs if r["arm"] == "base"]),
            },
            "worst_residual_s": max(abs(t - (a + b / f)) for t, f in pts),
            "leave_one_out_fixed_s": [min(x[0] for x in loo), max(x[0] for x in loo)],
            "leave_one_out_work_Mcycles": [min(x[1] for x in loo), max(x[1] for x in loo)],
            "foreign_holder_in_every_fold": all(r.get("foreign_tt") for r in runs),
        },
        "cross_check_separate_clean_session": {
            "forced_s": fm, "forced_MHz": ff, "base_s": bm, "base_MHz": bf,
            "measured_gap_s": bm - fm,
            "predicted_gap_s": b * (1.0 / bf - 1.0 / ff),
            "gap_error_s": (bm - fm) - b * (1.0 / bf - 1.0 / ff),
            "predicted_forced_s": a + b / ff,
            "forced_error_s": a + b / ff - fm,
            "foreign_holders": sum(len(r.get("foreign_tt", [])) for r in pr),
        },
        "applied_to_the_current_tree": {
            "measured_512_s_at_1350": CURRENT_512,
            "work_Mcycles_if_fixed_term_unchanged": work_now,
            "work_change_vs_0df13ad9_pct": 100.0 * (work_now / b - 1.0),
            "fixed_pct_of_fold": 100.0 * a / CURRENT_512,
            "ten_seconds": {
                "fixed_untouched_cut_pct": 100.0 * (1.0 - (10.0 - a) * 1350.0 / work_now),
                "fixed_at_2s_cut_pct": 100.0 * (1.0 - 8.0 * 1350.0 / work_now),
                "fixed_at_1s_cut_pct": 100.0 * (1.0 - 9.0 * 1350.0 / work_now),
                "fold_s_if_fixed_term_cut_to_1s_alone": CURRENT_512 - (a - 1.0),
            },
        },
        "limits": [
            "Every fold in the two-clock session had one foreign TT holder. It is present in both "
            "arms so the comparison is controlled, but an additive per-fold holder cost inflates "
            "the fixed term.",
            "Commit 0df13ad9, not current main. The current tree's work term is higher; the fixed "
            "term is assumed unchanged and that assumption is exactly what c10-fixed-cost tests.",
            "Two clocks and two parameters leave no degrees of freedom. The evidence that the model "
            "fits is the within-arm scatter and the independent cross-check, not a residual test.",
            "Clock-immune is not the same as host CPU. It includes clock-immune device and dispatch "
            "cost, and only a device-side split can separate them.",
        ],
    }
    json.dump(out, sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
