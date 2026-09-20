#!/usr/bin/env python3
"""Is APB's +0.0551 s partly host load rather than lever?

THE SAME QUESTION AS `clock_confound_audit.py`, ON THE CONFOUND THAT IS BIGGER. This campaign has
measured host contention at **+4.25 s and +5.66 s per fold** in two separate incidents. The APB
effect is 0.0551 s. So load is worth roughly a hundred times the effect, and a small systematic
difference in loadavg between the arms would swamp it.

Session 3's scored blocks ran at loadavg 1.33-1.91, which is quiet but NOT constant. The arms
interleave base,on,base inside each block precisely so that drift cancels, and this checks that it
did rather than assuming it.

CALIBRATED FROM THIS SESSION, NOT IMPORTED. The +4.25 s figure came from a different box state and
using it here would be a unit substitution. Instead the sensitivity is fitted from the session's own
BASE folds -- 96 of them, spanning the loadavg range the session actually saw -- by ordinary least
squares of fold_s on loadavg1. Base-only, so the lever cannot leak into the slope that is then used
to correct for the lever's confound.

    python3 perf/c14_land/load_confound_audit.py
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SESSION = REPO / "perf/c14_land/apb3_ab.json"
SCORED_BLOCKS = {0, 1, 2, 3}
EFFECT_S = 0.0551


def ols(xs, ys):
    """(slope, intercept, r) for y = a*x + b."""
    n = len(xs)
    mx, my = st.mean(xs), st.mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx == 0:
        return 0.0, my, 0.0
    a = sxy / sxx
    syy = sum((y - my) ** 2 for y in ys)
    r = sxy / (sxx * syy) ** 0.5 if syy > 0 else 0.0
    return a, my - a * mx, r


def main() -> int:
    d = json.loads(SESSION.read_text())
    rows = []
    for leg in d["blocks"]:
        if leg.get("block") not in SCORED_BLOCKS or leg.get("returncode"):
            continue
        for f in (leg.get("result") or {}).get("folds", []):
            if f.get("loadavg1") is not None and f.get("fold_s") is not None:
                rows.append((leg["arm"], leg["block"], f["loadavg1"], f["fold_s"]))

    by = {}
    for arm, blk, la, fs in rows:
        by.setdefault(arm, []).append((la, fs))
    base, on = by.get("base", []), by.get("on", [])
    if not (base and on):
        print("missing an arm -- cannot audit")
        return 1

    print(f"scored blocks {sorted(SCORED_BLOCKS)}\n")
    print(f"{'arm':6s}{'folds':>7}{'mean load':>12}{'median':>9}{'min':>7}{'max':>7}")
    for name, v in (("base", base), ("on", on)):
        la = [x[0] for x in v]
        print(f"{name:6s}{len(v):>7}{st.mean(la):>12.4f}{st.median(la):>9.3f}"
              f"{min(la):>7.2f}{max(la):>7.2f}")

    # Sensitivity fitted on BASE folds only, so the lever cannot bias the correction.
    slope, _, r = ols([x[0] for x in base], [x[1] for x in base])
    gap = st.mean([x[0] for x in on]) - st.mean([x[0] for x in base])
    # A LOWER load on the `on` arm makes it finish sooner, inflating a positive delta.
    explained = -gap * slope
    print(f"\nload sensitivity, fitted on the 96 base folds : {slope:+.4f} s per unit loadavg "
          f"(r = {r:+.3f})")
    print(f"mean loadavg gap (on - base)                  : {gap:+.4f}")
    print(f"  worth                                       : {explained:+.4f} s of fold time")
    print(f"  measured APB effect                         : {EFFECT_S:+.4f} s")
    print(f"  fraction attributable to load               : {100*explained/EFFECT_S:+.1f} %")

    # The verdict turns on the DIRECTION of the gap, not on the arms being equal. They are not
    # equal here, and a label saying they were would contradict the table printed above it.
    if gap > 0:
        print(f"\nVERDICT: LOAD-ADVERSE, effect survives")
        print(f"  The arms did NOT run at equal load: the `on` arm averaged {gap:+.4f} higher, "
              f"{100*gap/st.mean([x[0] for x in base]):.0f} % above base. A HIGHER load makes a "
              "fold SLOWER, so the load difference works AGAINST the measured speedup rather than "
              "for it. Load cannot have manufactured this effect; if anything the true lever "
              "effect is slightly larger than the measured 0.0551 s.")
    elif abs(explained) > 0.2 * EFFECT_S:
        print(f"\nVERDICT: LOAD-CONFOUNDED")
        print("  the `on` arm ran at lower load by enough to explain a material share of the "
              "effect. Restate or re-measure.")
    else:
        print(f"\nVERDICT: LOAD-CLEAN")
        print("  the `on` arm ran at lower load, but by too little to matter at the fitted "
              "sensitivity.")

    print(f"\n  The fitted slope is NOT load-bearing here and should not be quoted as a "
          f"correction: r = {r:+.3f} over a range of {min(x[0] for x in base):.2f}-"
          f"{max(x[0] for x in base):.2f} means the sensitivity is undetermined on a box this "
          "quiet. The direction argument above needs no slope.")
    print("  CAVEAT on the metric: loadavg1 is a ONE-MINUTE average, and the bracket runs "
          "base,on,base, so an `on` leg's reading partly reflects the base leg that preceded it. "
          "That is a plausible cause of the gap and it does not change the direction argument.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
