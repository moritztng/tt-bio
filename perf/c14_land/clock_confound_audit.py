#!/usr/bin/env python3
"""Is APB's +0.0551 s partly AICLK rather than lever?

WHY THIS IS NOT PARANOIA. The campaign's own clock discipline says the AICLK SETS the fold time on
this part: 800 MHz reads 21.90 s, ~1063 reads 17.34 s, 1350 reads 14.69 s at 512 aa. Between 1063
and 1350 that is 2.65 s over 287 MHz, about **9.2 ms of fold time per MHz**. The measured APB effect
is 0.0551 s. So a mean-AICLK difference of only about **6 MHz** between the arms would account for
the whole of it.

Session 3 was pinned `TT_BIO_AICLK=1350` and sampled during every fold, but the samples read
1268-1350, not a flat 1350 -- the governor droops under load. A pin is a request, not a guarantee,
and "both arms saw the same range" is not the same claim as "both arms saw the same mean".

So this compares mean AICLK per arm over the SCORED blocks only, converts any gap to seconds with
the curve above, and states what fraction of the effect it could explain.

    python3 perf/c14_land/clock_confound_audit.py
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SESSION = REPO / "perf/c14_land/apb3_ab.json"
SCORED_BLOCKS = {0, 1, 2, 3}          # the four complete blocks score_bracket.py kept
MS_PER_MHZ = (17.34 - 14.69) / (1350 - 1063)   # 9.24 ms per MHz, from the brief's own cells
EFFECT_S = 0.0551


def main() -> int:
    d = json.loads(SESSION.read_text())
    per_arm, per_block = {}, {}
    for leg in d["blocks"]:
        if leg.get("block") not in SCORED_BLOCKS or leg.get("returncode"):
            continue
        arm = leg["arm"]
        for f in (leg.get("result") or {}).get("folds", []):
            c = f.get("clock") or {}
            m = c.get("aiclk_mean")
            if m is None:
                continue
            per_arm.setdefault(arm, []).append(m)
            per_block.setdefault(leg["block"], {}).setdefault(arm, []).append(m)

    print(f"scored blocks {sorted(SCORED_BLOCKS)}   {MS_PER_MHZ*1000:.2f} ms of fold time per MHz\n")
    print(f"{'arm':6s}{'folds':>7}{'mean MHz':>11}{'median':>9}{'min':>7}{'max':>7}")
    for arm in sorted(per_arm):
        v = per_arm[arm]
        print(f"{arm:6s}{len(v):>7}{st.mean(v):>11.2f}{st.median(v):>9.1f}{min(v):>7.1f}{max(v):>7.1f}")

    base, on = per_arm.get("base", []), per_arm.get("on", [])
    if not (base and on):
        print("\nmissing an arm -- cannot audit")
        return 1

    gap = st.mean(on) - st.mean(base)
    # A HIGHER clock on the `on` arm makes it finish sooner, which inflates a positive delta.
    explained = gap * MS_PER_MHZ
    print(f"\nmean AICLK gap (on - base) : {gap:+.3f} MHz")
    print(f"  worth, at {MS_PER_MHZ*1000:.2f} ms/MHz : {explained:+.4f} s of fold time")
    print(f"  measured APB effect        : {EFFECT_S:+.4f} s")
    print(f"  fraction attributable to clock : {100*explained/EFFECT_S:+.1f} %")

    print(f"\n{'block':>6}{'base MHz':>11}{'on MHz':>10}{'gap':>9}{'s explained':>13}")
    for b in sorted(per_block):
        a = per_block[b]
        if "base" in a and "on" in a:
            g = st.mean(a["on"]) - st.mean(a["base"])
            print(f"{b:>6}{st.mean(a['base']):>11.2f}{st.mean(a['on']):>10.2f}"
                  f"{g:>+9.3f}{g*MS_PER_MHZ:>+13.4f}")

    verdict = ("CLOCK-CONFOUNDED" if explained > 0.5 * EFFECT_S else
               "CLOCK-CONTRIBUTING" if explained > 0.2 * EFFECT_S else
               "CLOCK-CLEAN")
    print(f"\nVERDICT: {verdict}")
    if verdict == "CLOCK-CLEAN":
        print("  the arms saw statistically the same clock, so the delta is the lever and not the "
              "governor. The effect stands as measured.")
    else:
        print("  a material share of the measured effect is explained by the arms running at "
              "different clocks. The number must be restated or re-measured under a harder pin.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
