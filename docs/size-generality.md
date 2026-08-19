# Size generality

Every model here is tuned at one sequence length and used at all of them. This page records what
the fleet measured across the ladder, and the rule and the gate arm that keep the two from drifting
apart again.

## The rule

**A perf lever may not land default-ON on the strength of one sequence length. Any threshold
constant carries a validity range, stated where the constant is defined.**

The same rule covers the other calibration axis: a resource figure measured on one board type
(per-core L1, DRAM budget) is a figure for that board type, and shipping it everywhere is the same
mistake with a different variable. Both are one constant calibrated at one point and applied at
every point. Where a constant's range is not known, say so at the definition rather than leaving
the reader to assume it was checked.

`scripts/release_gate.py --model size-ladder` enforces the size half of this, and it is in the
default arm set, so a release runs it whether or not anyone remembers to. It folds each structure
model at 256, 512, 640 and 768 aa, counts which perf levers actually fire at each rung by effect,
and fails when the fired set or the runtime scaling exponent moved away from
`docs/size_ladder_baseline.json`.

Two properties are worth knowing before you read a red run:

**It is a change detector, not a purity check.** Some levers are legitimately dark at some sizes,
and the baseline ships today's dark set with a one-line reason for each. The arm fails when the
answer changes and nobody said it should, in either direction. A lever that *starts* firing at a
size it was never measured at is also a failure, because that is what a threshold quietly widening
looks like.

**Re-recording is a human action that costs four sizes.** `--size-ladder-record` re-measures every
rung, and the baseline stores each lever's resolved value at each one. So flipping a default to ON
fails the arm until someone re-records, and re-recording measures four sequence lengths. The rule
enforces itself instead of relying on a reviewer noticing.

Baselines are per board type. The L1 budgets scale to the part's measured per-core L1, so two
boards can legitimately fire different levers at the same sequence length. A board with no recorded
baseline is a loud failure, never a silent skip.

## Why the ladder includes 640

256, 512, 768 and 1024 all have a padded length that the SDPA chunk size divides, so they all sit
on the lattice the fused triangle-attention kernel is served on. That kernel was silently declining
at padded 448, 576, 640, 704, 832, 896 and 960 while 256/512/768/1024 were served. A ladder built
only from multiples of 256 holds "padded length divides the chunk size" constant at every rung and
cannot see that class of defect at all. 640 is the off-lattice control, and it is the rung the arm's
own red-condition proof fires at.

640 is a lever rung only, not a timing rung. At the measured 6.5 % run-to-run noise floor, a
3-sigma exponent band over 512 to 640 is +-1.24 and over 640 to 768 is +-1.51, both at or past the
size of the cliff worth catching, so an exponent gate there would be a coin flip. The exponent is
checked over 256 to 512 (+-0.50) and 512 to 768 (+-0.68).

## What the 2026-08-19 sweep found

Five models, re-measured after 703 commits had landed with every perf decision screened at 512 aa
only. Wall times are not comparable across models (different fixtures, recycle counts and hosts);
the exponent and the lever columns are the point.

| model | rung | wall s | exponent vs prev | levers dark | status |
|---|---:|---:|---|---|---|
| boltz-2 | 128 | 8.85 | — | K1, K1 tail, K2, E6 | K2 dark at the small end since at least 08-13, on `memory_config`, not the gate this campaign is named after |
| boltz-2 | 512 | 26.55 | N^1.58 over the pair track | none | the tuned anchor |
| boltz-2 | 768 | — | N^3.48 (256 to 768, gate arm) | K2 half-dark, 560 of 1120 calls declined | the cliff is still on main six days later; the transpose and q-chunk gates have since closed |
| protenix-v2 | 128 | 12.32 | — | E6, K1, K1 tail, K2 | honest dark: the E6 window excludes 128 and the L1 leg measures a real loss there |
| protenix-v2 | 256 | 22.74 | N^0.89 | E6 not even offered | unexplained by the window; live lead |
| protenix-v2 | 512 | 55.06 | N^1.28 | none | **E6 serves 2416 calls here and zero at every other rung — a single-size lever, found by counter** |
| protenix-v2 | 768 | — | in progress | E6, K2 | the pair output is DRAM and the window admits the size, so something downstream refuses |
| openfold3 | all | — | in progress | — | no rung measured yet this pass; 1024 aa OOMs on allocation count, not size |
| opendde | 128 | 14.50 | — | — | 08-13 reference ladder |
| opendde | 256 | 28.93 | N^1.00 | transition big-chunk fires | |
| opendde | 512 | 88.76 | N^1.62 | transition big-chunk dark | only the current transpose headroom admits L1 here |
| opendde | 768 | 267.50 | N^2.72 | K2 dark, refiner q-split dark | the refiner track runs ~1.945x the token count, so it leaves the q-split cap a rung before the main track does |
| opendde | 1024 | 705.50 | N^3.37 | K2 dark, refiner q-split dark | the q-split cap was raised to 1024 on boltz-2 numbers alone and has never been folded here |
| esmfold2 | all | — | in progress | — | |

Rows marked in progress are owed by the three measurement tasks running alongside this one; the
gate baseline in `docs/size_ladder_baseline.json` is the machine-readable version of the same
matrix and is complete for every model it lists.

Three findings generalise beyond their own model.

**Single-size levers are real and they are found by counting, not by arguing.** protenix-v2's E6
channel move serves every call at 512 aa and none at 128, 256, 768 or 1024. Nothing about the
config says so; the counter does.

**The dark end is not only the large end.** Seven levers that serve every call at 512 aa serve zero
at 128 aa on boltz-2, and four of them decline explicitly rather than never being reached. The
512-aa tuning window is bounded on both sides and only the upper bound had ever been written down.

**A lever's validity range widened on one model's numbers applies to every model.** The SDPA
q-split cap went from 768 to 1024 padded tokens on boltz-2 measurements, and it ships default-ON to
models whose refiner track crosses that cap a whole rung earlier than their main track does. That
is this page's rule broken inside the codebase the rule is for, which is why the rule is now a gate
arm and not a paragraph.

## Running it

```
TT_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" python3 scripts/release_gate.py --model size-ladder
```

Roughly 20 to 30 minutes for five models. Folds are single-sequence at 6 sampling steps: enough to
resolve every guard, cheap enough that nobody skips the arm for cost. The price of hermetic folds is
that a cliff living only in the MSA path is invisible here.

After an intentional size-affecting change, re-record and write a one-line reason for every newly
dark lever:

```
TT_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" python3 scripts/release_gate.py --model size-ladder \
    --size-ladder-record
```

A dark lever with no reason is a failure, not a pass by silence. Per-rung census artifacts land in
`perf/sizegate/baseline/` and are the first thing to diff when the arm goes red.
