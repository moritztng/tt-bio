"""How much a ratio's meaning depends on the clock it was taken at. CPU arithmetic, no device.

The campaign opened on this hypothesis: an unrecorded, probably-throttled clock inflates the
denominator of a wall-time A/B, so a lever that deletes real cycles reads SMALLER than it should and
the byte levers were all under-priced.

Under the campaign's own planning fit that is only half right, and the half it gets wrong is the
byte levers. Run this to see the arithmetic. ``fold_s = F + W/f`` with F = 2.901 s clock-immune and
W = 15355 MHz*s of clock-scaling work.

* A lever that deletes DEVICE CYCLES (a byte or arithmetic lever) shrinks W. Both arms carry the
  same F, so throttling inflates numerator and denominator together. The measured ratio moves the
  wrong way for the hypothesis: it is slightly LARGER at a low clock.
* A lever that deletes CLOCK-IMMUNE SECONDS (host work, dispatch, launch overhead) shrinks F. At a
  low clock F is a smaller share of a longer fold, so its ratio really is suppressed, and by a lot.

So the class the throttled clock under-priced is the host/dispatch/launch class, not the byte class.
"""
from __future__ import annotations

from corpus import BASE_MHZ, BURST_MHZ, FIXED_S, TARGET_WORK_MHZ_S, WORK_MHZ_S, fold_seconds_at


def cycle_lever_ratio(frac_work_deleted: float, f: float) -> float:
    """Wall ratio of a lever that deletes ``frac`` of the clock-scaling work, measured at clock f."""
    w = WORK_MHZ_S
    return (FIXED_S + w / f) / (FIXED_S + w * (1 - frac_work_deleted) / f)


def fixed_lever_ratio(seconds_deleted: float, f: float) -> float:
    """Wall ratio of a lever that deletes ``seconds_deleted`` of the clock-immune term, at clock f."""
    return (FIXED_S + WORK_MHZ_S / f) / (FIXED_S - seconds_deleted + WORK_MHZ_S / f)


def report() -> str:
    L = []
    L.append("clock sensitivity of a wall-time ratio, under fold_s = "
             f"{FIXED_S} + {WORK_MHZ_S:.0f}/AICLK")
    L.append(f"  fold at {BASE_MHZ} MHz: {fold_seconds_at(BASE_MHZ):.3f} s   "
             f"fold at {BURST_MHZ} MHz: {fold_seconds_at(BURST_MHZ):.3f} s")
    L.append(f"  clock-immune share: {100*FIXED_S/fold_seconds_at(BASE_MHZ):.1f} % at "
             f"{BASE_MHZ} MHz, {100*FIXED_S/fold_seconds_at(BURST_MHZ):.1f} % at {BURST_MHZ} MHz")
    L.append("")
    L.append("A. a lever that deletes DEVICE CYCLES (byte / arithmetic class)")
    L.append("   work deleted    ratio @800    ratio @1350    @1350 is")
    for frac in (0.01, 0.02, 0.03, 0.05, 0.10):
        a, b = cycle_lever_ratio(frac, BASE_MHZ), cycle_lever_ratio(frac, BURST_MHZ)
        L.append(f"   {100*frac:6.1f} %       {a:.5f}x      {b:.5f}x      "
                 f"{100*((b-1)/(a-1)-1):+.1f} % of the 800 MHz reading")
    L.append("")
    L.append("B. a lever that deletes CLOCK-IMMUNE SECONDS (host / dispatch / launch class)")
    L.append("   seconds deleted ratio @800    ratio @1350    @1350 is")
    for s in (0.10, 0.25, 0.50, 1.00, 1.90):
        a, b = fixed_lever_ratio(s, BASE_MHZ), fixed_lever_ratio(s, BURST_MHZ)
        L.append(f"   {s:6.2f} s       {a:.5f}x      {b:.5f}x      "
                 f"{100*((b-1)/(a-1)-1):+.1f} % of the 800 MHz reading")
    L.append("")
    L.append("C. converting an unrecorded-clock fold ratio to cycles: the clock band")
    L.append("   dW = (W + F*f) * (1 - 1/r), so the SAME ratio implies more deleted work if it was")
    L.append(f"   taken at burst. The band over [{BASE_MHZ}, {BURST_MHZ}] MHz is a flat "
             f"{100*((WORK_MHZ_S+FIXED_S*BURST_MHZ)/(WORK_MHZ_S+FIXED_S*BASE_MHZ)-1):.1f} % on every ratio.")
    L.append(f"   Target: W must fall {WORK_MHZ_S:.0f} -> {TARGET_WORK_MHZ_S:.0f} MHz*s, "
             f"{100*(1-TARGET_WORK_MHZ_S/WORK_MHZ_S):.1f} %.")
    L.append(f"   Or, holding W, F must fall {FIXED_S} -> "
             f"{10.0 - WORK_MHZ_S/BURST_MHZ:.3f} s, which is below zero: "
             "the fixed term alone cannot reach 10.0 s at burst.")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    print(report(), end="")
