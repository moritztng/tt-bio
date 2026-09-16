#!/usr/bin/env python3
"""Re-examine the inverse-clock fit that defines the C10 target.

The campaign's objective (work term 15355 -> 9584, a 37.6 % cycle cut) comes from a two-point fit
through folds 1 and 6 of a six-fold sweep. Three of those six folds had a foreign TT holder, all six
carry an arithmetic-mean clock label for a clock that swung up to 1350 MHz inside the fold, and the
fit was never checked against an out-of-sample point. There is one: the eight-fold pinned arm at a
sampled min=max=1350 MHz, and its unpinned sibling at a mean 1339.2 MHz.

CPU only. Reads the banked clock audit; opens no device and measures nothing new.
"""
import json
import sys
from pathlib import Path

AUDIT = Path("/home/moritz/.coworker/state/c10-orchestrator-clock-audit.json")


def fit(points):
    """Least squares of t = a + b/f. Returns (intercept_s, slope_MHz_s)."""
    n = len(points)
    if n < 2:
        raise ValueError("need at least two points")
    sx = sum(1.0 / f for _, f in points)
    sy = sum(t for t, _ in points)
    sxx = sum((1.0 / f) ** 2 for _, f in points)
    sxy = sum(t / f for t, f in points)
    denom = n * sxx - sx * sx
    if denom == 0:
        raise ValueError("clock labels are degenerate; a fit needs spread")
    b = (n * sxy - sx * sy) / denom
    return (sy - b * sx) / n, b


def predict(a, b, f):
    return a + b / f


def main():
    audit = json.loads(AUDIT.read_text())
    obs = audit["historical_fit"]["observations"]
    force = audit["baseline"]["arms"]["force"]
    base = audit["baseline"]["arms"]["base"]
    pinned = (force["median_fold_s"], 1350.0)
    unpinned = (base["median_fold_s"], base["mean_of_fold_clock_means_mhz"])

    everything = [(o["fold_s"], o["aiclk_mean"]) for o in obs]
    clean = [(o["fold_s"], o["aiclk_mean"]) for o in obs if not o["foreign_tt"]]
    cotenanted = [(o["fold_s"], o["aiclk_mean"]) for o in obs if o["foreign_tt"]]

    candidates = {
        "two_point_campaign_fit": [everything[0], everything[5]],
        "six_point_least_squares": everything,
        "clean_folds_plus_pinned": clean + [pinned],
        "cotenanted_folds_plus_pinned": cotenanted + [pinned],
    }

    out = {
        "scope": "CPU re-examination of a banked fit. No device, no fold, no new timing.",
        "inputs": {
            "audit": str(AUDIT),
            "historical_commit": audit["historical_fit"]["recorded_measured_commit"],
            "pinned_commit": audit["baseline"]["recorded_measured_commit"],
        },
        "fits": {},
    }
    for name, pts in candidates.items():
        a, b = fit(pts)
        entry = {
            "n": len(pts),
            "intercept_s": a,
            "slope_MHz_s": b,
            "predicted_at_1350": predict(a, b, 1350.0),
            "error_vs_pinned_s": predict(a, b, 1350.0) - pinned[0],
            "pinned_point_in_fit": pinned in pts,
            "unpinned_arm_out_of_sample": {
                "measured_s": unpinned[0],
                "clock_mhz": unpinned[1],
                "predicted_s": predict(a, b, unpinned[1]),
                "error_s": predict(a, b, unpinned[1]) - unpinned[0],
            },
            "target_10s_at_1350": {},
        }
        for fixed in (None, 1.0, 2.0):
            held = a if fixed is None else fixed
            if held >= 10.0:
                entry["target_10s_at_1350"]["fixed_%s" % ("as_fitted" if fixed is None else fixed)] = \
                    "unreachable: the fixed term alone exceeds 10 s"
                continue
            need = (10.0 - held) * 1350.0
            entry["target_10s_at_1350"]["fixed_%s" % ("as_fitted" if fixed is None else fixed)] = {
                "work_term_MHz_s": need,
                "cycle_cut_pct": 100.0 * (1.0 - need / b),
            }
        out["fits"][name] = entry

    out["clock_label_quality"] = [
        {"fold": o["i"], "fold_s": o["fold_s"], "mean_mhz": o["aiclk_mean"],
         "max_mhz": o["aiclk_max"], "samples": o["aiclk_n"], "cotenanted": bool(o["foreign_tt"])}
        for o in obs
    ] + [
        {"fold": "pinned arm median of %d" % force["n"], "fold_s": force["median_fold_s"],
         "mean_mhz": force["mean_of_fold_clock_means_mhz"], "max_mhz": force["sampled_clock_max_mhz"],
         "samples": force["clock_sample_count"], "cotenanted": False},
        {"fold": "unpinned arm median of %d" % base["n"], "fold_s": base["median_fold_s"],
         "mean_mhz": base["mean_of_fold_clock_means_mhz"], "max_mhz": base["sampled_clock_max_mhz"],
         "samples": base["clock_sample_count"], "cotenanted": False},
    ]
    json.dump(out, sys.stdout, indent=2, sort_keys=True)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
