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
model at 256, 512, 640, 768, 896 and 1024 aa, plus any top rung a model reaches on its own
(rf3 folds 1095 aa, so it also gets 1088), counts which perf levers actually fire at each rung
by effect, and fails when the fired set, the clause a guard declines on, or the runtime scaling
exponent moved away from `docs/size_ladder_baseline.json` and the fragments beside it. A model
whose guard refuses a rung records that refusal, and a later run that folds the rung the baseline
says it refuses is a failure, because the ceiling moved. Nesso-1 rides the same rungs through `tt-bio affinity`
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

**Re-recording is a human action that costs six sizes.** `--size-ladder-record` re-measures every
rung, and the baseline stores each lever's resolved value at each one. So flipping a default to ON
fails the arm until someone re-records, and re-recording measures six sequence lengths. The rule
enforces itself instead of relying on a reviewer noticing.

Baselines are per board type and per core grid. The L1 budgets scale to the part's measured per-core
L1, and some guards are sized against the grid, so two cards can legitimately fire different levers
at the same sequence length. Board type alone does not pin the grid, because harvesting means one
board type presents several. A card with no recorded baseline, or one whose grid differs from the
baseline's, is a loud failure telling you to re-record rather than a silent skip or a false drift
report.

## Why the ladder reaches 1024

1024 used to be left off entirely because OpenFold3 OOMs there on allocation count, and an arm that
is red on arrival for one model is an arm someone switches off. A different model paid for that:
boltz-2 has served 1024-residue jobs in production while the largest size anything measured was
768, so its scaling across the top third of its supported range was unwatched.

The ladder now runs to 1024 for every model, and a model that cannot fold a rung records that rung
as a refusal instead of holding everyone else's ladder down. That refusal is information the arm
carries, not a reason to leave the rung out. Every rung is a multiple of 32, the token-axis
bucketing the fused kernels are served on.

A model whose guard reaches past 1024 gets its own top rung in `SIZE_LADDER_EXTRA_RUNGS` rather
than widening the shared set. rf3 folds 1095 aa, so it measures 1088 too; widening the shared
ladder instead would put an 1088 refusal cell on eight models and make check mode demand a
baseline row nobody has recorded. `--size-ladder-rungs` filters each model's own ladder, so a
resume pass naming 1088 measures rf3 there and is a no-op for everyone else.

## Where the baseline lives

`docs/size_ladder_baseline.json` plus `docs/size_ladder_baseline.d/<model>.json`, read as one. Pass
`--size-ladder-fragment` and a record writes only its own model's fragment, dropping that model's
rows from the monolith so the two cannot disagree. The reason is merge mechanics rather than taste:
six models recorded in parallel on six branches all create the same new card key in the same file,
so they conflict on a file none of them disagree about. Fragments never collide, and a model with
no fragment is served from the monolith exactly as before.

Recording a card type for the first time inherits each dark lever's exemption reason from the
newest other card that has one, tagged `[carried from <card>]`. Only the judgement half carries;
the counts and the decline clause are re-measured from the entry being written. Without that, a
new card writes TODO on every dark lever it has and the check cannot pass until a human retypes
judgements the file already holds one card block away.

## Why the ladder includes 640

256, 512, 768 and 1024 all have a padded length that the SDPA chunk size divides, so they all sit
on the lattice the fused triangle-attention kernel is served on. That kernel was silently declining
at padded 448, 576, 640, 704, 832, 896 and 960 while 256/512/768/1024 were served. A ladder built
only from multiples of 256 holds "padded length divides the chunk size" constant at every rung and
cannot see that class of defect at all. 640 is the off-lattice control, and it is the rung the arm's
own red-condition proof fires at.

The Wormhole ladder then caught the real thing at 896 and 1088. `TRIATT_PERSISTENT_MASK` serves
all 1088 of rf3's calls at 768 and at 1024 and none at all at 896 or 1088. The controlling
quantity is not the size: once L1 refuses every chunk wider than the production pick of 256, the
fused path survives only if 256 divides the padded length, and 1024 refuses more configs than 896
while still serving. Thirteen of the fifteen tile-aligned lengths from 640 to 1088 have a
non-dividing fallback, on a path shared by rf3, boltz-2, protenix-v2, openfold3 and opendde.

`TT_BIO_TRIATT_NARROW_Q_FALLBACK` offers a dividing chunk below the production pick before one
that pads. It is off, and the fold A/B that would justify turning it on did not. At 896 aa the
flag does exactly what it was written to do, restoring the fused kernel to all 1088 calls; at
1088 aa it does not restore it at all, so something other than the padding mask blocks that rung.
Neither arm produced a usable wall: the control arm alone read 267 s and 327 s on the same size
and the same code, which is the contention floor of a host serving 23 production workers, and a
lever worth a few per cent cannot be measured through it. Restoring a lever is not the same as
recovering time, and this pair of rungs has yet to show it recovers any.

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
ligand heavy atom, and the total is padded up to a multiple of `token_axis.TOKEN_BUCKET` (32), so
adding a 20-atom ligand at 640 aa gives 660 tokens and lands on padded 672, a rung no apo protein
on the ladder's own lattice ever reaches.

The lattice is what matters here, not the number. Boltz-2 once shipped unable to fold anything at
padded 704, back when the multiple was 64: apo 640 aa padded to 640 and folded, apo 768 aa padded
to 768 and folded, and the one rung between them was reachable only with a ligand or with an
off-lattice residue count (641 to 704 aa apo died identically). A protein-only ladder and a ligand
ladder each passed every rung on their own; only the two axes together reached it. Moving the
multiple to 32 halves the lattice spacing and so doubles the number of reachable rungs the ladder
does not visit. It does not remove the blind spot, it makes it finer.

So when reading a size result, convert to padded tokens first. A ligand ladder at the same
residue rungs is a different set of shapes, not a repeat of the apo one, and that is the point of
running it.

## The token axis is padded too, and that one is a correctness fix

Every shipped model pads the trunk token axis up to a multiple of 32, masked, and slices back on
exit. Without it an unaligned token count hands the triangle attention a ragged key axis, and both
the stock and the fused attention read those padded columns as if they held real data: relative
error against the aligned answer is 0.914 ragged, 0.038 padded. So the padding is not a perf lever,
it is the difference between a right and a wrong number, and it is on by default.

One mechanism does it for all of them, `token_axis.bucketed_pairformer`, and the per-model facts
live as data in the `TOKEN_AXIS` table beside it rather than as copies of the code. The fused
attention also masks its own ragged tail now (`TT_BIO_SDPA_RAGGED_PAD`, default on), which is a
belt-and-braces guard rather than a duplicate: bucketing removes raggedness at the source, the guard
keeps the primitive safe for anything that still reaches it ragged.

The padding itself does not change the answer. Two different poison fill values give bit-identical
trunk fingerprints, so what the masked columns contain cannot reach the output.

Cost follows the padded pair area, so it does follow the rung arithmetic above once you convert to
padded tokens. When the token count is already a multiple of 32 the pad is 0, every padding site
early-outs, and this costs exactly nothing. When it is not, you pay for the columns you added,
quadratically: 298 tokens round to 320, a pair area 1.15x larger, and that costs 4.8 % on
Protenix-v2. OpenDDE pays it twice at the same input, trunk 298 to 320 and refiner 580 to 640, and
costs 6.0 %. Both of those figures are unchanged by the move to 32, because 298 rounds to 320 either
way.

**Why the multiple is 32 and not 64**, measured on Boltz-2 with the arm order alternated:

| length | 64 against 32 | why |
|---|---|---|
| 76 aa | 64 is **-20.1 %** | 32 pads to 96, 64 pads to 128, 2.37x the triangle work |
| 20 aa | 64 is **+4.18 %** | both are one tile of real work; a smoke-size fold is dispatch-bound |

The 76 aa reading is the one that decides it, because 20 aa is a smoke fixture and nothing runs it in
production. Note the sign flip: the tile-count argument that a narrower pad can never be slower is
true of tiles and false of wall clock, since the two widths do not select the same kernel. Both ends
were measured on the same model for that reason, and a size ladder is what separates them.

`TT_BIO_TOKEN_BUCKET=0` restores the old ragged path for an A/B, fleet-wide, and
`TT_BIO_TOKEN_BUCKET_MULTIPLE` re-runs every model at another width in one variable. The older
`TT_BIO_PROTENIX_TOKEN_BUCKET` and `TT_BIO_PROTENIX_TOKEN_PAD_MULTIPLE` still work for that family
and are ANDed with the global.

There used to be an exception here, and it bit the off-lattice rung this page recommends.
`TT_BIO_TOKEN_BUCKET=0` failed on Boltz-2 at any token count that is not already a multiple of 32:
the diffusion path sized its atom axis as `padded_seq * 14`, so with the bucket off 298 tokens
asked for 4172 atoms, the 32-atom attention window could not partition that, and the fold died on
`Invalid arguments to reshape` before the first sampling step. Confirmed on Blackhole at 298 aa,
2026-09-11, and closed the same day by the lever below, which is now on by default. Turning
`TT_BIO_ATOM_AXIS_BUCKET=0` back off together with `TT_BIO_TOKEN_BUCKET=0` reopens it.

`TT_BIO_ATOM_AXIS_BUCKET` sizes the atom axis on the real atom count, `ceil(N/448)*448`, instead of
assuming every token is a tryptophan, which is what the 14 above is. It is a multiple of 32 for any
composition, so it closes that crash by construction, and at 512 aa it drops the axis from 7168
atoms to 4480 and the atom transformer from 224 attention windows to 140. **On by default since
2026-09-11**, worth 1.0470x on the 512 aa fold. The shorter axis changes the contraction length of
the one matmul that reduces over atoms, so the answer is byte-identical at some sizes (298 aa, CIF
sha256 `71653ff72cbf01b0` either way) and reassociated at others (512 aa). Measurements, the
accuracy control and the domain-split reading of the 512 aa difference are in
`perf/b2x-integrate/`; `TT_BIO_ATOM_AXIS_BUCKET=0` restores the token-derived pad for an A/B.

Note what this does to the ladder. Every rung is a multiple of 64 and therefore of 32 too, so the
residue axis pads to 0 at all four and the arm cannot price this lever on Protenix-v2. It still sees
it on OpenDDE, whose refiner runs a second token axis at roughly twice the residue count minus one
per glycine, and that lands on a multiple of 32 only for a sequence with the right glycine count. An off-lattice rung
(`RELEASE_GATE_SIZE_RUNGS=298`) is the only way to price it on the residue axis, which is the same
blindness this page describes one level up: a ladder built only from multiples of the thing you are
testing tests nothing.

## The MSA row axis is padded too, and that one is pure cost

Boltz-2, BoltzGen and Nesso-1 pad the alignment's row axis the way they pad the token axis, and for
the same reason: one shape means one compiled program. The row axis uses a single 1024 step, so a
fold whose alignment is shallower than 1024 rows runs the MSA track at 1024 anyway. The perf
fixture's alignment has 35 rows. Every `--single_sequence` fold has one.

The padded rows are inert. They sit behind a row mask that zeroes the outer product's `a`
projection, so they cannot reach a real row, and the only thing they cost is time. Measured on one
Wormhole chip at 512 tokens, one `MSALayer` call is `139.6 ms + 0.0996 ms per padded row`, so the
1024 step spends 96 ms per call, 1.5 s per fold, on rows that are not there.

`TT_BIO_MSA_DEPTH_LADDER=1` drops the padding to the smallest power-of-two rung that holds the real
alignment, floored at 64 because the row axis tiles at 32. Above 1024 the single step is already
the finer rule and the ladder defers to it, so turning the flag on can only remove padding, never
add it. It is **off by default**: the reduction then runs over a shorter axis, which moves the last
bf16 bit, and the flag stays behind the release gate until that has a Blackhole accuracy re-measure.

Read the win against your alignment's true depth, not against the fixture's:

| true rows | 1 | 35 | 256 | 1024 | 2048 or deeper |
|---|---|---|---|---|---|
| padding removed per call | 960 | 960 | 768 | 0 | 0 |
| worth | ~1.5 s/fold | ~1.5 s/fold | ~1.2 s/fold | nothing | nothing |

A production fold against an MSA server usually lands in the right-hand columns and gains nothing.
This is a win on shallow alignments and single-sequence folds, not a fold-time improvement the
service can claim in general.

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

## First Wormhole baseline: boltz-2 to 1024

`docs/size_ladder_baseline.d/boltz2.json` (2026-09-07, `tt-galaxy-wh l`, 8x9 grid) is the first
size-ladder baseline recorded on Wormhole, and the first for boltz-2 above 768: 256/512/640/768/
896/1024, median of 3 after a discarded warm-up. 640→768→896→1024 reads a clean k = 1.99/2.03/2.18
— a quadratic in tokens with no lever going dark anywhere in the band the platform's 1024 ceiling
opens up but no ladder had ever measured.

One open anomaly, recorded rather than silently smoothed over: the 512 rung sits ~23% above what
its own 640-1024 curve predicts (55.5 s measured vs 45.1 s extrapolated), inside the same rung
whose measured run-to-run sigma is 12%. The leading mechanistic suspect — `TRANSPOSE_L1_HEADROOM`
(1.25), the only lever whose census differs at 512 vs the rest of the ladder — was A/B screened by
forcing the tensor off its L1 route (`TT_BIO_TRANSPOSE_L1_HEADROOM=8.0`) and **refuted**: the
forced/DRAM arm was 10.9% slower, not faster, at the median. The screen's own within-arm spread
(14.3% across three identical folds) is close to the size of the anomaly, so it reads as host
contention on a shared measurement box rather than a real regression. Not re-recorded on a quiet
host yet; that is the one follow-up this baseline leaves open.

## Running it

```
TT_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" python3 scripts/release_gate.py --model size-ladder
```

About an hour for six models: 3444 s for the five structure legs plus 313 s for nesso1, on a p150a.
Structure folds are single-sequence at 6 sampling steps, enough to resolve every guard and cheap
enough that nobody skips the arm for cost; nesso1 runs `tt-bio affinity` at every shipped default.
The price of hermetic folds is that a cliff living only in the MSA path is invisible here.

Nesso-1's leg needs the checkpoint's 413 MB `ccd.pkl`, which is never committed. It is looked for
under `--cache`, `NESSO_CACHE`, `HF_HOME` and the default cache in that order, and the arm checks for
it once before the first fold rather than failing twelve times after twelve model loads.

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
    --size-ladder-record-lever FLAG[,FLAG...]
```

One fold per (model, rung), and the splice is refused unless every other lever in the census still
matches the baseline exactly, by the same comparator the check uses. If anything else moved it says
so and tells you to re-record, so nothing can be laundered through this mode. The spliced rows carry
their own `levers_added` stamp, because they were measured on a different host at a different commit
than the timings beside them. Use it only for a counter-only change; anything that touches a
threshold or a default needs the four sizes.

Two things it settled the first time it ran. A single cold fold reproduced the warm-recorded
baseline's census exactly at all 24 (model, rung) pairs, which the mode assumed and nobody had
measured. And the refusal earned its keep: it caught that the baseline predates the commit which
taught the census to record WHY a guard declined, so three levers read as having changed their
decline clause with served and declined identical. An absent clause means not measured, not "no
clause". **The rule, alongside the L1-budget leg's: an instrument change that widens what the
baseline compares re-records in the same commit.**

## ESMFold2 on Wormhole, 256 to 1024, and the lever the old ladder top was hiding

`docs/size_ladder_baseline.d/esmfold2.json` and `esmfold2-fast.json` (2026-09-08,
`tt-galaxy-wh l`, 11x10/8x9 grid) hold both served ESMFold2 checkpoints at all six rungs.

| aa | 256 | 512 | 640 | 768 | 896 | 1024 |
|---|---:|---:|---:|---:|---:|---:|
| `esmfold2` s | 39.2 | 90.7 | 133.6 | 192.4 | 252.6 | 327.8 |
| `esmfold2-fast` s | 23.6 | 53.2 | 76.4 | 109.9 | 152.5 | 184.3 |

The exponent reads 1.210 -> 1.736 -> 2.000 -> 1.766 -> 1.952 across the five intervals, which is
one workload changing mix rather than several regimes: the ESMC language model is near-linear in L
and dominates at the bottom of the ladder, the pair work is not and takes over. Nothing here is a
fast path falling off, and the 896 -> 1024 number is not a cliff: the top rungs are close together,
so `ln(n2/n1)` is small and the 3-sigma band on that interval is +-1.13 at the measured 3.5 % noise
floor. `esmfold2-fast` reads 1.418 over the same interval.

**What the two new rungs found is in the census, not the timing, and it has since been fixed.**
Recorded before the fix, `PAIR_FFN_FUSED_RESIDUAL` and `PAIR_FFN_FILL_ASSEMBLY` served every call
at 512, 640 and 768 aa and none at 896 and 1024, on both checkpoints, while `PAIR_FFN_L1_SLICE`
kept serving. That was `_row_blocked`'s L1 retry ladder (`tt_bio/esmc.py`) dropping G then F and
keeping C-in, because the block is `rows` rows of `[1, rows, L, C]` at a fixed `rows = 32` and
stops fitting L1 somewhere between 768 and 896 on a 72-core grid. It was invisible while the ladder
stopped at 768, which is the last rung where both levers fire, and the platform had been serving
1024 for a while. A count is not a timing: it has no error bar, so the census carries this finding
at sizes where the exponent cannot.

The retry ladder now halves the block before it gives up a lever, and drops G, then F, then C-in
only once the block cannot shrink past `PAIR_FFN_ROW_BLOCK_MIN`. The block not fitting is a size
problem, and G and F are destinations, not sizes. Both affected rungs settle at `rows = 16` after
a single halving and both levers serve every call again. It is worth 5.9 % at 896 aa, measured as
an alternating six-fold A/B against the unpatched engine on one card: 252.3 s against 268.2 s
median, with no overlap between the arms. At 1024 aa the same halving is inside the noise. The
change is bit-exact, confirmed by an identical CIF sha256 from both engines at both rungs, and it
cannot reach the four other models that share `SwiGLUFFN`: `PAIR_FFN_L1_SLICE`, which every
`_row_blocked` call bumps, reads 0 served and 0 declined for every one of them at every rung on
every card type, so they never enter the row-blocked path at all.

The old ladder's exponents were distorted by this. Before the fix the curve read
1.199 -> 1.685 -> 2.113 -> 2.276 -> 1.296, a spike into 896 and a collapse out of it, and both were
the same inflated 896 cell rather than anything about the workload.

Two more levers are dark for dtype rather than size, both because `tt_bio/main.py` forces `--fast`
for ESMFold2 on Wormhole (ESMC-6B in normal precision needs ~12.8 GB against a ~12 GB chip):
`TRIMUL_IN_PROJ_DUAL_NOC` declines 100 % of calls at every rung on `dtype`, and
`PAIR_PROJ_MINIMAL_MATMUL` declines from 512 up on `dtype:(8,32)` — where the 8 is `k_tiles`, which
is exactly what that lever wants, so ESMFold2 is the one model whose pair projection fits it and it
is refused only on precision. Widening that guard is a parity decision, not a tuning one.

`esmfold2-fast` stays in `SIZE_LADDER_EXEMPT`, and now on a measurement: its dark set is identical
to `esmfold2`'s at all six rungs, with call counts about half (24 trunk blocks against 48). Its
fragment is recorded anyway, as the evidence for the exemption rather than an argument for it.
