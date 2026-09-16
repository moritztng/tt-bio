#!/usr/bin/env python3
"""What the two measured sizes say about the clock-immune fixed cost.

c10-bare-baseline measured, on the current tree at a pinned and during-sampled 1350 MHz,
14.8813 s at 512 aa and 9.6801 s at 298 aa. Two equations, three unknowns: a shared fixed term F
and one work term per size. That is a one-parameter family, not an answer -- pick the work ratio
r = W512/W298 and F follows. This prints the family so nobody quotes one member of it as measured.

CPU only. Arithmetic on a banked measurement; opens no device and measures nothing.
"""
import json
import sys

T512 = 14.8813252375   # pooled median, 16 accepted folds, origin/wk/c10-bare-baseline bc66f7d6d
T298 = 9.6800812565    # pooled median, 16 accepted folds, same session pair
CLOCK = 1350.0

ANCHORS = {
    1.600: "token axis padded to a multiple of 32: 512/320",
    1.718: "raw token ratio 512/298",
    1.839: "the value implied by the clock-sweep intercept of 3.478 s",
    2.000: "",
    2.500: "",
    2.950: "a pure N^2 pair term",
    5.070: "a pure N^3 triangle term",
}


def family(r):
    """Given the work ratio, return the shared fixed term and both work terms."""
    if r <= 1.0:
        raise ValueError("512 aa cannot do less work than 298 aa")
    w298 = (T512 - T298) * CLOCK / (r - 1.0)
    return {"work_ratio": r, "fixed_s": T298 - w298 / CLOCK,
            "W298_Mcycles": w298, "W512_Mcycles": r * w298}


def demand(entry, goal=10.0, fixed_floor=1.0):
    """What reaching `goal` seconds at 512 aa costs under this member of the family."""
    f, w = entry["fixed_s"], entry["W512_Mcycles"]
    out = {}
    out["cut_with_fixed_untouched_pct"] = None if f >= goal else 100.0 * (1.0 - (goal - f) * CLOCK / w)
    out["cut_with_fixed_at_%.1fs_pct" % fixed_floor] = 100.0 * (1.0 - (goal - fixed_floor) * CLOCK / w)
    out["fixed_cut_alone_reaches_goal"] = out["cut_with_fixed_at_%.1fs_pct" % fixed_floor] <= 0.0
    return out


def main():
    rows = []
    for r, note in sorted(ANCHORS.items()):
        e = family(r)
        e["anchor"] = note
        e["fixed_pct_of_512_fold"] = 100.0 * e["fixed_s"] / T512
        e["target_10s"] = demand(e)
        rows.append(e)
    json.dump({
        "scope": "CPU arithmetic on c10-bare-baseline's pinned medians. No device, no new timing.",
        "inputs": {"T512_s": T512, "T298_s": T298, "clock_MHz": CLOCK,
                   "source": "origin/wk/c10-bare-baseline bc66f7d6d perf/c10_bare_baseline/baseline.json"},
        "identifiability": "Two sizes cannot separate the fixed term from the work exponent. A "
                           "second PINNED CLOCK at one size can, because the fixed term is the only "
                           "part that does not scale with the clock.",
        "family": rows,
    }, sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
