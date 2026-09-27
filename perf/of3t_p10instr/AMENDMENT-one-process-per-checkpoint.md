# Amendment to PREREGISTRATION.md: one checkpoint per process

Committed before any arm is scored on the amended instrument, and before the one-process runs
`out/AB2.json` and `out/AB12.json` (the original design, produced 11:04Z) have been read. Every
rule of PREREGISTRATION.md stands except the two below. The verdict is whatever `grade.py`
prints on the command at the bottom.

## Why

The original design scores every arm of an A/B in one process. That process is biased by the
order it loads weights in, which was found on non-arm weights before any arm was read:

* `out/N3a.json` scores base, L40G, L40Gs1 in that order and `out/N3b.json` the reverse. The
  start weights read the same in both (M 11.9217), but 7fb8 on L40G reads 3.02 in N3a and 5.87
  in N3b, and 7vus on L40Gs1 reads 2.11 in N3a and 10.73 in N3b. These are means over 32 draws,
  so it is a shift in what is being scored, not a draw landing on the other side of a boundary.
* `out/F_L40G_{a,b}.json` and `out/F_L40Gs1_{a,b}.json` score each of those checkpoints ALONE in
  a fresh process, warmed on its own weights, once on card 1 and once on card 2. The two
  processes agree on all 33 seeds x 4 targets to the last digit (max |a - b| = 0.0) for both
  checkpoints. So a fresh process is repeatable, and card-independent, where a shared one is not.

The within-process leak is not root-caused. The suspect named at 11:00Z, start-weight trimul
in-projections cached in `TriangleMultiplication._gp_cache`, is refuted by the code:
`train_in_projections` (`tt_bio/train/openfold3.py:349`) gives every trimul device leaves at
build, and `_gp_in_chunks` then caches nothing. `evalnoise.py` now records that walk after
scoring (`trimul` in each artifact) so the claim is checked on the device, not read from code.

## The two changes

1. **One checkpoint per process.** Each arm is scored alone in a fresh process by `chainG.sh`,
   after one discarded warm-up call on its own weights, at eval seeds 0..31 (20260926 reported,
   never graded). Each arm is scored twice, in two processes on two different cards (suffix `a`
   and `b`). The grade reads the `a` artifacts. The draws stay paired: every process draws the
   same noise at the same seed.
2. **H is measured on the arms.** H = the largest |M(a) - M(b)| over the six arms and the two F
   checkpoints. It replaces the N3a/N3b term, which measured the shared-process bias this
   amendment removes.

An artifact whose `trimul.host_in_proj` is not 0 is invalid and `grade.py` refuses to grade it.

## Command

    perf/of3t_p10instr/gradeG.sh

which runs `grade.py ab` with the history set to {F_*_a, G_*_a} against {F_*_b, G_*_b}, design
`two` = G_TF7_a, G_B2trunk_a, G_I_TFs1_a and design `twelve` = G_I_D12_a, G_X12b_a,
G_I_D12s1_a, and writes `out/VERDICT_G.json`.

The original one-process design is graded too, on `out/AB2.json` and `out/AB12.json` with the
N3a/N3b history exactly as PREREGISTRATION.md wrote it, and reported beside this verdict as
known-biased. It does not enter the verdict.
