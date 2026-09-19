# Does TT_BIO_APB_CONCAT_HEADS make the output depend on the core grid?

Answer: **no**, and it is discharged on the fixture where the same question caught region T.

## Why it had to be asked

Region T (`TT_BIO_TRIATT_B8`) was fold-measured at +0.2020 s, was accuracy-clear at 0.37848 A
against a 0.60 A bar, and was still refused as a shipped default because its output moved with
the core grid. APB is in the same risk class: it keeps the padded head lanes, so `proj_g` and
`proj_o` run at `n_heads * padded_head_dim` = 1024 instead of 768. The added lanes are exactly
zero and contribute nothing, but the matmul's K changes, the blocking changes with it, and the
blocking is derived from the live grid, so the order the real 768 lanes are summed in can move
with the grid.

## The first attempt was blind, and its own control says so

Six boltz-2 folds on `cdk2x2_298`, arms base and on, grids native / 8x8 / 9x9
(`apb_grid_independence.json`): one digest per arm across all three grids, the two arms
differing from each other. That looks like a discharge and is not one.

Two checks say why. `TT_BIO_FORCE_GRID` does take effect -- `COMPUTE_GRID_MAIN` reads (8,8) and
(9,9) while the physical grid stays (11,10) -- so the legs really did run on three grids. But the
**positive control fails to fire**: region T, the configuration known to be grid-dependent, folds
this same fixture to `ab216735afcbdedc428d2a24` at native AND at 8x8. A detector that cannot see
the known positive cannot certify a negative. Boltz-2 at 298 aa is not a sensitive fixture for
this question.

## The fixture that does fire

`release_gate.py --model l1-budget` -- protenix-v2 on `affinity_fkg.yaml`, folded at native, at a
forced 8x8, and at a capped trimul width. This is the arm that split region T into two md5s and
is how it was refused. Same tree (`e837d4042`), same card, flag flipped between the two runs and
nothing else:

    TT_BIO_APB_CONCAT_HEADS=0   native / 8x8 / narrow   c3073854d423570ae48cb8ce35ccb27e   GATE PASS
    TT_BIO_APB_CONCAT_HEADS=1   native / 8x8 / narrow   a2e7f667fb45e7c50f7a55f5336e0819   GATE PASS

**One md5 across three grids in each arm, and the two arms differ.** The arms differing is what
makes the result non-vacuous: it proves the flag reaches protenix-v2 and changes its answer, so
the three-grid agreement is a property of the flag and not of a code path that never ran.

The off-arm md5 also matches what `c14-stack-land` recorded for this arm on a different tree
(`3adbca531`), which is a free cross-tree agreement on the control.

## Two conclusions, and the second is an owed precondition rather than a result

1. **APB is grid-independent.** The hard stop that refused region T does not apply to it. What is
   left between this flag and a shipped default is the seconds question and nothing else on this
   axis.
2. **APB reaches protenix-v2, not only boltz-2.** Its accuracy is discharged on boltz-2 only
   (0.224352 A all-atom on `cdk2x2_298` against a 0.35 A PASS bar, 0.000000 A A/A control). The
   gate's protenix-v2 accuracy floor is 6.0 A against a measured 3.87 A, and the silu regression
   Moritz refused in September was ~1.1 A of CA-RMSD on that model, which would have passed that
   floor. So a green gate is not an accuracy reading. **Flipping this default owes one paired
   protenix-v2 Angstrom reading, flag on against flag off, same seed, against that model's own
   seed-scatter floor.**
