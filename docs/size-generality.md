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
away from `docs/size_ladder_baseline.json`. Nesso-1 rides the same rungs through `tt-bio affinity`
instead of `predict`, because it returns a scalar rather than a structure and `predict` cannot fold
it. The clause matters on its own: a guard that starts
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
| nesso1 | 6.5 s | 10.2 s | 13.0 s | 17.7 s | 0.64 | 1.36 | 0.9 % |

Nesso-1 was recorded later, on a different p150a at the same 13x10 grid; its row is the only one
not from the 08-19 sweep, and the baseline records that per model. Its exponents are low because
its pocket crop pins the token count after the first trunk pass: only one of six passes runs at
full N, so the fold is mostly crop-sized work and the full-N pass takes a larger share as N grows.
That is why k rises from 0.64 to 1.36 rather than staying flat.

The five structure models scale between N^1.3 and N^2.5. **Nothing shows the N^3.6 cliff** the 2026-08-13 sweep
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
| nesso1 | K2 at every rung, all 768 calls; TriMul F1 and the minimal-matmul guard at every rung; **the SDPA q-chunk overflow set is non-empty at 640 and 768** | K2 is dark because `affinity=True` adds a per-row pair-mask slice, which makes the triangle bias `[S, h, S, S]` instead of `[1, h, S, S]`: the kernel exists to read one batch-broadcast mask per head, so it is inapplicable to this path rather than mis-tuned. Two more size-conditioned gates appear above 512: the transpose headroom gate answers DRAM for 96 of 576 calls at 768, and the pair projections' L1 destination is refused for 16.6 % of calls from 512 up |

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
where it was found says nothing about the other four. Nesso-1's leg found the same overflow at the
same two rungs, on a sixth model, the first time it ran.

**An arm can be blind to a code path, not only to a size.** The five structure legs all fold the same
apo fixture, so until Nesso-1 joined, no rung of this arm exercised an affinity pairformer for any
model — four of those five have no affinity module to exercise. K2, which is 100 % dark on that path,
read as fully served at every rung. A ladder covers the sizes you list; it covers only the code the
fixture reaches, and that is a separate thing to check.

## Running it

```
TT_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" python3 scripts/release_gate.py --model size-ladder
```

About an hour for six models: 3444 s for the five structure legs plus 313 s for nesso1, on a p150a.
Structure folds are single-sequence at 6 sampling steps, enough to resolve every guard and cheap
enough that nobody skips the arm for cost; nesso1 runs `tt-bio affinity` at every shipped default.
The price of hermetic folds is that a cliff living only in the MSA path is invisible here.

Nesso-1's leg needs the checkpoint's 413 MB `ccd.pkl`, which is never committed. Point `NESSO_CACHE`
at the HuggingFace cache holding it if it is not in the default one.

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

## Adding one lever without re-recording

Instrumenting a guard that already shipped adds a lever to the census and changes no behaviour, but
the arm still reports it missing for every model at every rung. Re-recording to admit it costs every
fold twice over and replaces a good timing baseline with one measured on whatever host was free:

```
TT_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" python3 scripts/release_gate.py --model size-ladder \
    --size-ladder-record-lever PAIR_PROJ_L1_OUT
```

One fold per (model, rung), and the splice is refused unless every other lever in the census still
matches the baseline exactly, by the same comparator the check uses. If anything else moved it says
so and tells you to re-record, so nothing can be laundered through this mode. The spliced rows carry
their own `levers_added` stamp, because they were measured on a different host at a different commit
than the timings beside them. Use it only for a counter-only change; anything that touches a
threshold or a default needs the four sizes.
