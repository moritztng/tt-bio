# of3t-p10draws pre-registration: the amended instrument at 128 draws

Committed and pushed before any 128-draw artifact exists. Everything in PREREGISTRATION.md and
AMENDMENT-one-process-per-checkpoint.md stands except what is listed here. The verdict is
whatever `gradeD.sh` prints.

## Why

At 32 draws both designs graded UNRESOLVED (`out/VERDICT_G.json`): two-step Delta -0.0097 +-
0.0163 against T 0.0194, twelve-step -0.0290 +- 0.0252 against T 0.0451. The half-width
straddles the bar. The spread of the paired differences (sd 0.047 and 0.073) says about 90 and
80 draws would resolve them if Delta holds, so 128 draws leaves room. The draw count is fixed
here, before scoring, and is not raised or cut after reading any result.

## What changes

1. **Draws.** Each of the six arms is scored alone in a fresh process (`chainG.sh -n 128`) at
   eval seeds 0..127 after the same warm-up and the same seed 20260926 call, and all 128 are
   graded. The draws stay paired across arms. Artifacts: `out/D128_<arm>_a.json`.
2. **The bars are fixed at their 32-draw values**, not recomputed from the 128-draw floor:
   T(two) = 0.019428875906963877, T(twelve) = 0.04507007146777562 (a third of the retraining-seed
   floor F measured at 32 draws). The 128-draw F is reported beside the verdict. The floor rule
   stays: a design whose 128-draw floor is inside its own half-width is UNRESOLVED.
3. **H.** Seeds 0..31 of each 128-draw process replay the exact call sequence of the 32-draw
   processes, so H = the largest |M| difference over those 32 seeds between the new process and
   the two 32-draw processes (G_*_a, G_*_b, on other cards) of the same weights, together with
   the F_* pairs. H enters h as before: h = 1.96 * sd(d_s) / sqrt(128) + H.

## Verdict rule, unchanged

Per design: PASS if |Delta| + h <= T; FAIL if |Delta| - h > T; UNRESOLVED otherwise.
GO if both PASS, NO-GO if twelve FAILS, PARTIAL otherwise.

If a design is still UNRESOLVED, `draws_to_resolve` in the verdict file is the draw count the
measured sd(d_s) says is needed. Running it would be a further pre-registration.

## Command

    perf/of3t_p10instr/gradeD.sh
