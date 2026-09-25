#!/usr/bin/env python3
"""The step partition and what a per-verb repricing does to it (LEDGER R205, R206).

Numbers are of3t-bwsurvey's own, from `of3t-gpugap` PARTITION arm B rep 2. This script exists so
the correction is reproducible rather than asserted: run it, not the prose.
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
STEP, BWD, VERB_S, N = 466.702, 456.668, 356.00, 168922
H200 = (7.0, 8.0)
HW_FLOOR = (8.5, 11.0)          # compute / bandwidth silicon ratio, BACKWARD.md section 1

nonbwd  = STEP - BWD            # 10.034 s
nonverb = BWD - VERB_S          # 100.668 s -- the term the "9.1x" headline dropped
per_ms  = VERB_S / N * 1000     # 2.107 ms

def step_at(per_ms_new, drop_nonverb=False):
    verbs = per_ms_new / 1000 * N
    return nonbwd + (0.0 if drop_nonverb else nonverb) + verbs

if __name__ == "__main__":
    print(f"step {STEP}  backward {BWD}  non-backward {nonbwd:.3f}")
    print(f"verb calls {VERB_S} over {N} = {per_ms:.3f} ms/verb")
    print(f"NON-VERB backward = {nonverb:.3f} s   <- dropped by the 51 s / 9.1x headline\n")

    correct = step_at(0.24)
    wrong   = step_at(0.24, drop_nonverb=True)
    print(f"verbs repriced to 0.24 ms:")
    print(f"  correct : {correct:7.2f} s -> {STEP/correct:5.2f}x  ; vs H200 "
          f"{correct/H200[1]:.1f}-{correct/H200[0]:.1f}x")
    print(f"  headline: {wrong:7.2f} s -> {STEP/wrong:5.2f}x  ; vs H200 "
          f"{wrong/H200[1]:.1f}-{wrong/H200[0]:.1f}x")
    print(f"\nsilicon floor is {HW_FLOOR[0]}-{HW_FLOOR[1]}x, so a post-fix gap BELOW it is "
          f"impossible.")
    print(f"  correct  reading {correct/H200[1]:.1f}-{correct/H200[0]:.1f}x : above the floor, OK")
    print(f"  headline reading {wrong/H200[1]:.1f}-{wrong/H200[0]:.1f}x : REFUTED by the floor")

    print("\nR206 -- a kernel saving is (verbs removed) x (cost per verb), so it scales with J0:")
    for p in (per_ms, 0.24):
        print(f"  J1 at {p:.3f} ms/verb: 31104 -> 9072 verbs saves "
              f"{(31104-9072)*p/1000:5.1f} s")
    print("  => J0 and the kernel jobs are SUBSTITUTES. Never sum them.")
