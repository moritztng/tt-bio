# of3t-p10draws pre-registration: the two-step design at 256 draws

Committed and pushed before any 256-draw artifact exists. PREREG-128.md stands except for what is
listed here. The verdict is whatever `gradeE.sh` prints.

## Why

At 128 draws (`out/VERDICT_D128.json`, commit 0e3a12abb) the twelve-step design graded PASS and
the two-step design graded UNRESOLVED: Delta -0.0112 +- 0.0104 against T 0.0194, sd(d_s) 0.060,
H 0.0. That spread says 207 draws resolve it if Delta holds. 256 draws leaves room for Delta to
move; the count is fixed here and is not raised or cut after reading any result.

## What changes

1. **Only the two-step design is re-scored.** TF7, B2trunk and I_TFs1 each run alone in a fresh
   process (`chainG.sh -n 256`) at eval seeds 0..255, one arm per card on qb2 cards 1, 2, 3.
   Artifacts: `out/D256_<arm>_a.json`. The twelve-step grade stays the 128-draw PASS; it is not
   re-run, so it cannot be re-drawn into a different answer.
2. **The bar stays** T(two) = 0.019428875906963877.
3. **H.** Seeds 0..127 of each 256-draw process replay the 128-draw call sequence, so H now also
   covers the D256 vs D128 vs G processes of the same weights.

## Verdict rule, unchanged

two: PASS if |Delta| + h <= T; FAIL if |Delta| - h > T; UNRESOLVED otherwise.
Row verdict: GO if two PASSes (twelve already PASS), PARTIAL if two is UNRESOLVED or FAILs
(twelve did not fail, so NO-GO is not reachable). An UNRESOLVED here is reported as is, with its
`draws_to_resolve`; there is no further pre-registration in this row.

## Command

    perf/of3t_p10instr/gradeE.sh
