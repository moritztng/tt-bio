#!/usr/bin/env python3
"""The fold ratio and the floor it has to clear, both built from this session's own draws.

The statistic is the MEDIAN OF PAIRED RATIOS over adjacent (base, ship) folds -- adjacent in
time, and with the order reversed inside each rep, so a drift in the box cancels within the pair
rather than accumulating into the arm that ran second. The null is the same estimator applied to
SAME-ARM adjacent pairs: base against base and ship against ship, which are folds that differ by
nothing except when they ran. Its 95 % band is the floor. A ratio inside that band is a null.

Reusing another row's floor is what `baseline-median-must-recompute-from-draw-log` is about, and
a floor built from a different estimator (per-position, or worst single pair) answers a different
question than the one the headline asks.
"""
import json, random, statistics as st, sys
from pathlib import Path


def pairs(folds, want):
    """Adjacent pairs in run order. `want` is ('base','ship') for the A/B or a same-arm pair."""
    out = []
    for a, b in zip(folds, folds[1:]):
        if (a["arm"], b["arm"]) == want:
            out.append((a, b))
    return out


def ratio_pairs(folds):
    """base/ship, from every adjacent base-ship or ship-base pair. >1 means ship is faster."""
    rs = []
    for a, b in zip(folds, folds[1:]):
        if a["arm"] == b["arm"]:
            continue
        base, ship = (a, b) if a["arm"] == "base" else (b, a)
        rs.append(base["fold_s"] / ship["fold_s"])
    return rs


def same_arm_ratios(folds):
    """The null draws: adjacent same-arm pairs, oriented earlier/later."""
    return [a["fold_s"] / b["fold_s"] for a, b in zip(folds, folds[1:]) if a["arm"] == b["arm"]]


def boot_band(draws, n, iters=20000, seed=0):
    rng = random.Random(seed)
    meds = sorted(st.median(rng.choices(draws, k=n)) for _ in range(iters))
    return meds[int(0.025 * iters)], meds[int(0.975 * iters)]


def main():
    src = Path(sys.argv[1])
    d = json.loads(src.read_text())
    folds = [f for f in d["folds"] if f["tag"] != "cold"]
    ab = ratio_pairs(folds)
    aa = same_arm_ratios(folds)
    assert ab and aa, "need both A/B and same-arm adjacent pairs"
    stat = st.median(ab)
    lo, hi = boot_band(aa, len(ab))
    ab_lo, ab_hi = boot_band(ab, len(ab), seed=1)
    shas = {f["sha256"] for f in folds}
    per_arm = {a: sorted(f["fold_s"] for f in folds if f["arm"] == a) for a in ("base", "ship")}
    res = {
        "n_folds": len(folds), "n_ab_pairs": len(ab), "n_aa_pairs": len(aa),
        "paired_median_ratio": round(stat, 5),
        "paired_ratio_95ci": [round(ab_lo, 5), round(ab_hi, 5)],
        "aa_floor_95": [round(lo, 5), round(hi, 5)],
        "clears_floor": bool(stat > hi or stat < lo),
        "verdict": ("ship faster, outside the floor" if stat > hi else
                    "ship SLOWER, outside the floor" if stat < lo else
                    f"cannot distinguish from zero at n={len(ab)} pairs"),
        "median_s": {a: round(st.median(v), 3) for a, v in per_arm.items()},
        "min_s": {a: round(min(v), 3) for a, v in per_arm.items()},
        "ratio_of_medians": round(st.median(per_arm["base"]) / st.median(per_arm["ship"]), 5),
        "bit_exact_all_folds": len(shas) == 1,
        "distinct_sha256": sorted(shas),
        "plddt": sorted({f["plddt"] for f in folds}),
        "loadavg_1m": {"min": min(f["loadavg_1m"] for f in folds),
                       "max": max(f["loadavg_1m"] for f in folds)},
        "ab_ratios": [round(r, 5) for r in ab],
        "aa_ratios": [round(r, 5) for r in aa],
    }
    d["analysis"] = res
    src.write_text(json.dumps(d, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
