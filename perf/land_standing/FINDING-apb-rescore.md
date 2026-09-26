# APB re-scored at the merge tree: it FIRES, it discharges the grid hard stop — and it moves OpenFold3 by 1.254 A

2026-09-26 09:5xZ, `land-standing`. Gate re-scored at merge tree `317725511` (`origin/main`
`29bc7d00c` confirmed an ancestor) with `TT_BIO_APB_CONCAT_HEADS=1` in the environment, not a
second default flip. This is the re-score the 09-20 branch-tip green could not stand in for.

## It fires — counted, not assumed

Unlike `TT_BIO_TRIATT_FUSED_HIFI`, which served 0 of everything at the gate's targets, APB executes:

    pid 2107690 (openfold3 arm)   served 192    declined    0
    pid 2113076                   served 484    declined 4800
    pid 2115893                   served 484    declined 4800

So the arms below are a verdict ON the lever, not merely a regression check around it. Note the
ratio: **484 of 5284 is 9.2 % served** in the two later processes — the guard is
`_APB_CONCAT_HEADS and not atom_level and self.dtype != ttnn.float32`, so most head re-assembly
calls decline. A perf claim on this lever is a claim about that 9 %.

## The grid hard stop is discharged on THIS tree

`l1-budget`, the arm that catches card-dependent output, returns **one md5 across all three grid
classes** with the flag on:

    l1-budget:native   0 clashes   a2e7f667fb45e7c50f7a55f5336e0819   PASS
    l1-budget:8x8      0 clashes   a2e7f667fb45e7c50f7a55f5336e0819   PASS
    l1-budget:narrow   0 clashes   a2e7f667fb45e7c50f7a55f5336e0819   PASS

This matters because the sibling lever `TT_BIO_MM_SHORT_M_BW` was killed by exactly this arm.

## But OpenFold3 moves, and the lever's accuracy case never covered OpenFold3

Every arm still PASSES its bar. That is not the whole story:

    arm             APB=1 (this run)   baseline (APB off)   delta
    openfold3       2.912 / 0.767      1.658 / 0.902        +1.254 A , TM -0.135
    boltz2          1.878 / 0.864      1.827 / 0.902        +0.051 A , TM -0.038
    rf3             1.240 / 0.958      1.239 / 0.958        unchanged
    opendde         1.397 / 0.940      1.395 / 0.940        unchanged
    protenix-v2     1.997 / 0.867      1.999 / 0.867        unchanged
    esmfold2        1.343 / 0.961      1.343 / 0.961        unchanged
    esmfold2-fast   1.708 / 0.918      1.708 / 0.918        unchanged

**Why this is attributable to the lever and not to drift**, checked rather than asserted:

- the baseline appears **twice independently** — the dividing-k gate (174 s) and the fused-HiFi
  gate (138 s) both report openfold3 `1.658 / 0.902` and boltz2 `1.827 / 0.902`, identical to
  three decimals, so these arms are deterministic;
- `git diff --stat 48bf63c2a HEAD -- tt_bio/` is **empty** — `tt_bio` is byte-identical between the
  fused-HiFi gate tree and this one, so the only variable is the env var.

OpenFold3 keeps **0.588 A** of headroom against its 3.5 A bar where it had 1.842 A, and **0.067**
of TM headroom above the 0.70 floor where it had 0.202. It passes. It passes with a third of the
margin it had.

**And the lever's original accuracy evidence does not cover this model.** `c14-land-tail` cleared it
on Boltz-2 (0.2244 A all-atom vs a 0.35 A bar) and OpenDDE (0.1337 A) — the two arms that move
least here. OpenFold3, where it moves most, was not in that case.

## What I am NOT concluding yet

A single gate target at a single seed is not a seed floor. 1.254 A could still sit inside
OpenFold3's own seed spread on this target — this row's own dividing-k work measured a 1.974757 A
seed floor on a different fixture, which is larger than this delta. **So this is not yet "the lever
fails accuracy"; it is "the lever visibly changes the model its case never tested, and the case has
to be extended before it can land default-on."**

The control that settles attribution is an APB-off arm at THIS tree in the same session. The two
baselines above are that control in substance, but they ran at a different hour, so an interleaved
one is owed before any verdict.
