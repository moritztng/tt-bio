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
and fails when the fired set, the clause a guard declines on, or the runtime scaling exponent moved
away from `docs/size_ladder_baseline.json`. The clause matters on its own: a guard that starts
refusing for a different reason has changed behaviour without changing either the fired count or the
wall time, so nothing else in the arm can see it.

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

Baselines are per board type and per core grid. The L1 budgets scale to the part's measured per-core
L1, and some guards are sized against the grid, so two cards can legitimately fire different levers
at the same sequence length. Board type alone does not pin the grid, because harvesting means one
board type presents several. A card with no recorded baseline, or one whose grid differs from the
baseline's, is a loud failure telling you to re-record rather than a silent skip or a false drift
report.

## Why the ladder includes 640

256, 512, 768 and 1024 all have a padded length that the SDPA chunk size divides, so they all sit
on the lattice the fused triangle-attention kernel is served on. That kernel was silently declining
at padded 448, 576, 640, 704, 832, 896 and 960 while 256/512/768/1024 were served. A ladder built
only from multiples of 256 holds "padded length divides the chunk size" constant at every rung and
cannot see that class of defect at all. 640 is the off-lattice control, and it is the rung the arm's
own red-condition proof fires at.

640 is a lever rung only, not a timing rung. Run-to-run noise is measured per model when the
baseline is recorded, and it ranges from 0.7 % to 7.1 % across the five models. At a 6.5 % floor a 3-sigma exponent band
over 512 to 640 is +-1.24 and over 640 to 768 is +-1.51, both at or past the size of the cliff worth
catching, so an exponent gate on either half would be a coin flip, and splitting 512 to 768 would
also destroy the one interval that is gateable. The exponent is checked over 256 to 512 (+-0.50 on
boltz-2) and 512 to 768 (+-0.54). A model too noisy for a meaningful band has its exponent recorded
as skipped, with the measured noise as the reason, rather than getting a gate that cries wolf.

## The rung is a padded token count, not a residue count

The ladder's rungs are residue counts, but every size-conditioned gate in the engine keys on the
**padded token** count, and the two only agree for a bare protein. A token is a residue or a
ligand heavy atom, and the total is padded up to a multiple of `PAIRFORMER_PAD_MULTIPLE` (64), so
adding a 20-atom ligand at 640 aa gives 660 tokens and lands on padded 704, a rung no apo protein
on the 64-aa lattice ever reaches.

That is how Boltz-2 shipped unable to fold anything at padded 704. Apo 640 aa pads to 640 and
folds, apo 768 aa pads to 768 and folds, and the one rung between them is only reachable with a
ligand on the lattice or with an off-lattice residue count (641 to 704 aa apo dies identically).
Both a protein-only ladder and a ligand ladder pass every rung on their own; only the two axes
together reach it.

So when reading a size result, convert to padded tokens first. A ligand ladder at the same
residue rungs is a different set of shapes, not a repeat of the apo one, and that is the point of
running it.

## The token axis is padded too, and that one is a correctness fix

Protenix-v2, OpenDDE and OpenDDE-abag pad the trunk token axis up to a multiple of 64 as well,
masked, and slice back on exit. Without it an unaligned token count hands the triangle attention a
ragged key axis, and both the stock and the fused attention read those padded columns as if they
held real data: relative error against the aligned answer is 0.914 ragged, 0.038 padded. So the
padding is not a perf lever, it is the difference between a right and a wrong number, and it is on
by default.

The padding itself does not change the answer. Two different poison fill values give bit-identical
trunk fingerprints, so what the masked columns contain cannot reach the output.

Cost follows the same rule as the rung arithmetic above. When the token count is already a multiple
of 64 the pad is 0, both padding sites early-out, and this costs exactly nothing. When it is not,
you pay for the rounded-up width on work that scales as the square of it. At 298 tokens, which
rounds up to 320, that is 4.8 % on Protenix-v2 and 6.0 % on OpenDDE.
`TT_BIO_PROTENIX_TOKEN_BUCKET=0` restores the old ragged path for an A/B,
and `TT_BIO_PROTENIX_TOKEN_PAD_MULTIPLE` overrides the multiple.

Note what this does to the ladder. Every rung is a multiple of 64, so the pad is 0 at all four and
the size-ladder arm cannot see this lever at any of them. An off-lattice rung
(`RELEASE_GATE_SIZE_RUNGS=298`) is the only way to price it, which is the same blindness this page
describes one level up: a ladder built only from multiples of the thing you are testing tests
nothing.

## What the 2026-08-19 sweep found

The gate's own baseline, recorded on one p150a at a 13x10 grid, current main, single-sequence folds at
6 sampling steps. Wall times are comparable down a column but not across models. `k` is the log-log
runtime exponent between rungs.

| model | 256 aa | 512 aa | 640 aa | 768 aa | k 256→512 | k 512→768 | noise at 512 |
|---|---:|---:|---:|---:|---:|---:|---:|
| boltz-2 | 6.9 s | 19.6 s | 28.0 s | 37.3 s | 1.51 | 1.59 | 7.1 % |
| esmfold2 | 19.6 s | 47.7 s | 67.9 s | 109.6 s | 1.28 | 2.05 | 0.9 % |
| protenix-v2 | 15.8 s | 44.0 s | 68.8 s | 104.9 s | 1.48 | 2.14 | 0.7 % |
| openfold3 | 11.3 s | 32.3 s | 54.6 s | 84.6 s | 1.51 | 2.38 | 2.8 % |
| opendde | 25.2 s | 71.4 s | 120.0 s | 191.8 s | 1.50 | 2.44 | 0.8 % |

Every model scales between N^1.3 and N^2.5. **Nothing shows the N^3.6 cliff** the 2026-08-13 sweep
recorded over 512→768, and boltz-2, where that cliff was measured, is now the flattest model in the
table at N^1.59. An earlier reading in this campaign did reproduce N^3.48, but it came from a card
later found to be running folds about 2x slow. Warm, on a freshly reset card, the cliff is not there.

What the lever census found at each rung, per model:

| model | levers dark | worth knowing |
|---|---|---|
| boltz-2 | TriMul F1 at every rung; 5 more at 256 | K2 fires at 256 through 768 on this card |
| esmfold2 | both matmul-config guards at 768 only, all 25823 calls | the pair track switches to row-blocked execution at 768 and there is no tuned matmul block for the shapes it then presents. Neither guard is reached at all below 768, so no smaller size could have shown it |
| protenix-v2 | K2 half-dark at 768, 1208 of 2416 calls | E6 serves 2416 calls at 512 and 640 and 4512 at 768, and is dark only at 256 |
| openfold3 | TriMul F1 at every rung; E6 never offered at any rung | the declined matmul-config count rises 440 → 1288 at 640, the row-blocked path again |
| opendde | TriMul F1 at every rung; **the SDPA q-chunk overflow set is non-empty at 640 and 768** | one shape overflows its per-core buffer budget and silently takes the slow path. The set is empty at 256 and 512. This is the third gate of the 2026-08-13 above-640 defect, closed on boltz-2 and still open here |

Four findings generalise beyond their own model.

**A default-ON lever can be inert on most of the models it ships to.** The TriMul F1 tail fusion
declines 100 % of its calls at every rung on boltz-2, openfold3 and opendde — three of the five. It
allow-lists a single matmul block shape, and those three models' triangle-multiplication tails
present a different one, which is a property of the hidden channel count and not of sequence length.
It serves on protenix-v2, so the kernel works. Nothing is visibly broken, no config says so, and only
a counter finds it.

**The dark end is not only the large end.** Seven levers that serve every call at 512 aa serve zero at
128 aa on boltz-2, and four decline explicitly rather than never being reached. Most models have their
largest dark set at 256, not at 768.

**Whether a lever is single-size is itself a per-card question.** protenix-v2's E6 channel move was
reported as serving only at 512 aa; on this card it serves at 512, 640 and 768. Both measurements are
real. What travels is the method, not the verdict, which is why the baseline is keyed by grid and
re-recorded per card rather than asserted once.

**A model can carry a defect that its siblings have already fixed.** The SDPA q-chunk overflow closed
on boltz-2 and is still open on opendde at 640 and 768. Fixing a size-conditioned gate on the model
where it was found says nothing about the other four.

## Running it

```
TT_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" python3 scripts/release_gate.py --model size-ladder
```

About an hour for five models, measured at 3444 s on a p150a. Folds are single-sequence at 6 sampling steps: enough to
resolve every guard, cheap enough that nobody skips the arm for cost. The price of hermetic folds is
that a cliff living only in the MSA path is invisible here.

After an intentional size-affecting change, re-record and write a one-line reason for every newly
dark lever:

```
TT_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" python3 scripts/release_gate.py --model size-ladder \
    --size-ladder-record
```

A dark lever with no reason is a failure, not a pass by silence. Recording pre-fills each one with
the clause the guard actually declined on, so the reason you write is a confirmation rather than an
archaeology exercise. Per-rung census artifacts land in
`perf/sizegate/baseline/` and are the first thing to diff when the arm goes red.
