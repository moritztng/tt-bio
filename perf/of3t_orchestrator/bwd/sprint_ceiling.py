#!/usr/bin/env python3
"""What the backward sprint can deliver if EVERY remaining job fully succeeds (a CEILING).

Not a projection. Each job is credited its own best published number and J0 is credited a full
solve of the per-verb cost (2.107 ms -> 0.24 ms, which of3t-bwsurvey called already pessimistic
per verb but which is still a complete win). Kernel savings are repriced per LEDGER R206 because
they scale with the per-verb cost: a saving is (verbs removed) x (cost per verb), so if J0 lands,
J1's 46.4 s becomes 46.4 x 0.24/2.107. Summing them at today's cost double-counts.

Run it; do not quote the prose.
"""
#
# !! STALE INPUTS, 2026-09-25 pass 450. The constants below (466.702 / 456.668 / 356.00 /
# 168922) come from a step banked at 451ed56f4, BEFORE 502ed112e put a host float64 softmax and
# layer norm inside every tape(). of3t-tapedfwd measured a taped forward at 251.66 s against the
# 3.415 s banked, ~65x, and the same scope reaches the backward. The ARITHMETIC in this file is
# still correct and the REASONING it encodes still holds -- a dropped term is a dropped term, and
# substitutes are still substitutes -- but the OUTPUT NUMBERS are against a tree nobody runs.
# Re-run with the re-taken constants once fullstep.py has been run on main. Do not quote the
# figures below until then. See the STALE BASELINE banner in state/of3t/BACKWARD.md.
#
STEP, BWD, VERB_S, N = 466.702, 456.668, 356.00, 168922   # of3t-gpugap PARTITION, arm B rep 2
PER_NOW, PER_J0 = VERB_S / N * 1000, 0.24                 # ms per backward verb
H200 = (7.0, 8.0)                                          # whole training step, crop 384
FLOOR = (8.5, 11.0)                                        # silicon ratio: compute / bandwidth
KERNELS = {"J1 lnbw": 46.4, "J2 softbw": 32.8, "J4 wheelbw": 15.0}   # J3 retired, R208

nonbwd, nonverb = STEP - BWD, BWD - VERB_S
kern_today = sum(KERNELS.values())
kern_after = kern_today * (PER_J0 / PER_NOW)

def vs(s): return f"{s/H200[1]:.1f}-{s/H200[0]:.1f}x"

if __name__ == "__main__":
    print(f"partition  non-backward {nonbwd:.3f} | verb calls {VERB_S:.2f} @ {PER_NOW:.3f} ms "
          f"| NON-VERB backward {nonverb:.3f}\n")
    rows = [("kernels alone, J0 fails", STEP - kern_today),
            ("J0 alone, full solve",    nonbwd + nonverb + PER_J0 / 1000 * N),
            ("J0 + kernels, repriced",  nonbwd + nonverb + PER_J0 / 1000 * N - kern_after)]
    for label, s in rows:
        print(f"  {label:26s} {s:7.2f} s -> {STEP/s:5.2f}x   vs H200 {vs(s)}")

    lo, hi = H200[0] * FLOOR[0], H200[1] * FLOOR[1]
    best = rows[-1][1]
    print(f"\nthe silicon floor is {FLOOR[0]}-{FLOOR[1]}x, so reaching it means a step of "
          f"{lo:.0f}-{hi:.0f} s -- a {STEP/hi:.1f}-{STEP/lo:.1f}x software win.")
    print(f"the whole remaining list delivers {STEP/best:.2f}x at its CEILING, which is "
          f"{100*(STEP/best-1)/(STEP/lo-1):.0f}% of the way there.\n")
    print(f"UNOWNED: the {nonverb:.1f} s of NON-VERB backward is {100*nonverb/STEP:.1f}% of the "
          f"step,\n  larger than J1+J2+J4 combined ({kern_today:.1f} s), and no job addresses it.")
