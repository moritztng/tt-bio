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

**The pair projections' L1 destination is a size gate on four of six models.** `PAIR_PROJ_L1_OUT`
was found on Nesso-1 at 532 tokens and had neither a counter nor a kill switch, so nothing could say
it had happened. Counted, served of offered:

| model | 256 aa | 512 aa | 640 aa | 768 aa |
|---|---:|---:|---:|---:|
| boltz-2 | 1512/1512 | 1512/1512 | 393/1512 | 0/1120 |
| esmfold2 | 0/0 | 17216/17216 | 21520/21520 | 1/25824 |
| protenix-v2 | 1244/1244 | 480/480 | 480/480 | 480/480 |
| openfold3 | 1184/1184 | 1184/1184 | 337/1184 | 48/896 |
| opendde | 1324/3436 | 480/2600 | 480/2600 | 480/2592 |
| nesso1 | 1152/1152 | 961/1152 | 961/1152 | 960/1152 |

It is essentially 100 % dark at 768 aa on boltz-2 and esmfold2, and only protenix-v2 keeps it at
every rung. Two mechanisms, and the baseline's reason names which one per rung: `no_config`, where
the config builder finds no viable L1-destination program config for the shape, and
`memoised`/`l1_clash`, where the allocator refuses once and the refusal sticks for that operand
class. On Nesso-1 the result is bit-identical with the leg on and off, so this costs time and not
accuracy. Recorded, not blessed.

**An arm can be blind to a code path, not only to a size.** The five structure legs all fold the same
apo fixture, so until Nesso-1 joined, no rung of this arm exercised an affinity pairformer for any
model — four of those five have no affinity module to exercise. K2, which is 100 % dark on that path,
read as fully served at every rung. A ladder covers the sizes you list; it covers only the code the
fixture reaches, and that is a separate thing to check.

## The affinity path, and the two models that have one

Every rung of this arm folded apo protein until 2026-08-21. The shared fixture is CDK2 with no
ligand, so no fold could enter an affinity module at any size, on any model. Nothing said so: a
lever that is never reached is counted the same way as a lever that is fully served, which is the
exact failure this page exists to catch, on a code-path axis instead of a size axis.

Nesso-1 closed it for itself and measured the cost of the blind spot. `TRIATT_PERSISTENT_MASK`
serves 0 of 2304 calls with `affinity=True` at every rung, because the per-row pair-mask slice makes
the triangle bias `[S, h, S, S]` instead of batch-broadcast `[1, h, S, S]` and the fused kernel
declines by construction. Read off the apo fixture the same lever looks fully served.

`boltz2-affinity` is the other half. It folds a protein+ligand ladder through
`predict --model boltz2`, which runs the structure trunk and then the affinity module, and it is a
second leg rather than a swapped fixture: the finding is the per-lever difference between the two
paths, so both rows have to exist.

There is no third leg. Boltz-2 and Nesso-1 are the only shipped models with an affinity head.
ESMFold2, Protenix-v2 and OpenFold3 have none, and neither does OpenDDE, which ships co-folding
only. `tests/test_size_ladder_gate.py` asserts that by finding the affinity-head class in the
source rather than trusting a hand-kept list, so a model that grows one has to bring a leg with it.

What the apo fixture still hides for those four is the **ligand**, not the affinity module. None of
their rungs presents one, and OpenDDE's shipped path is protein plus ligand.
`perf/sizegate/inputs/holo/` is that input: the affinity ladder with the affinity property removed.

**Read a difference against the holo control, not against the apo row.** The ligand raises the token
count (256 aa featurizes to 276 tokens, the ligand being tokenised per heavy atom), so an
apo-vs-affinity lever change has two candidate causes, and one of them is the size effect this arm
already measures. Holo is the same protein, the same ligand and the same token count with no
affinity property, so apo→holo isolates the ligand and holo→affinity isolates the module.

The leg carries no exponent gate, and the reason is measured. Affinity runs on a pocket crop, so its
cost does not scale with the rung: at 256 aa on qb1 card 0 `affinity_runtime_s` is 172.1 s against a
`structure_runtime_s` of 18.5 s. A size-independent term that large flattens `k` to about 0.2 where
the structure half alone reads about 1.5, far inside the ±0.50 tolerance floor, so the band could not
fail on any cliff the structure half could produce. That is the same call the arm already makes for a
model too noisy to gate: record it as skipped with the numbers, rather than ship an unfalsifiable
band. The apo `boltz-2` row gates the structure trunk at these four rungs already.

## Boltz-2 cannot fold a ligand at 640 aa

The first thing the ligand ladder found is not a dark lever, it is a crash. Boltz-2 dies at trunk
0/4, about 6 s in, on a p150a at a 13x10 grid:

```
Statically allocated circular buffers in program 36 clash with L1 buffers on core range
[(x=0,y=0) - (x=12,y=9)]. L1 buffer allocated at 342016 and static circular buffer region
ends at 356864
```

Measured across both ligand fixtures, and the apo rung as the control:

| model | fixture | 256 | 512 | 640 | 768 |
|---|---|---|---|---|---|
| boltz-2 | apo (protein only) | ok | ok | **ok, 43.9 s** | ok |
| boltz-2 | holo (protein + ligand) | ok, 16.6 s | ok, 52.9 s | **L1 clash** | ok, 78.3 s |
| boltz-2 | holo + affinity | ok, 190.6 s | ok | **L1 clash** | not reached |
| opendde | holo (protein + ligand) | ok, 138.1 s | | **ok** | |

Both ligand fixtures fail at the same two addresses, so it is the ligand and not the affinity
module: the structure-only holo fold has no affinity property and crashes identically. The apo fold
at the same 640 aa is fine, and so is the ligand fold at 768. It is the combination.

It is also Boltz-2's own. OpenDDE folds the same 640 aa ligand fixture to completion, so the
defect is in the Boltz-2 trunk rather than in the ligand path or the featurizer both models
share. OpenDDE's trunk is the Protenix-v2 family; Boltz-2's is not.

That combination is the point. 640 is the arm's off-lattice rung, the one size in the ladder whose
padded length the SDPA chunk size does not divide, added because 256/512/768/1024 all sit on the
lattice the fused kernel is served on. A ladder of protein-only folds passes every rung. A ladder of
ligand folds on the lattice passes every rung. Only the two together find this, which is the same
argument that put 640 in the ladder, now paying out on a second axis.

The error is the issue #11 signature that the L1-budget leg exists for, but on a p150a at 13x10
rather than a p300c at 11x10, and reached through the token count a ligand adds rather than through a
part's core count. Not fixed here: this arm records, and a fix is a perf/L1 change that has to be
screened on its own.

## The census used to under-count on a loaded host

Seven of the levers keep no `*_STATS` counter of their own and are counted by monkeypatching
their helper: `ADALN_S_HOIST`, `QKV_MM_CONFIG`, `TRANSPOSE_L1_RESIDENT`, `B2_ADALN_S_MEMO`,
`B2_BIAS_SLICE_HOIST`, `PAIR_PROJ_MINIMAL_MATMUL`, `PAIR_TRANSPOSE_VIA_ROW_MAJOR`. A
monkeypatch counts only the calls made after it lands, and the install used to be reached
only from the census's 3-second dump thread. Every call between `tt_bio.tenstorrent` becoming
importable and the first tick was therefore uncounted, and whether that mattered depended on
how busy the machine was.

Those seven levers, and only those, are exposed. The `*_STATS` levers count inside the
shipped code and cannot be raced. That is what makes the signature recognisable: the lost set
is exactly the `wrap` rows of `LEVERS`, never a subset and never anything else.

Measured on the boltz2-affinity fold at 256 aa, same fixture and same commit throughout, with
the load as the only variable:

| run | load on the box | calls counted |
|---|---|---:|
| idle, twice (manual and record mode) | none | 11446 |
| pricing sweep, twice | 3 concurrent folds | 7456 |
| controlled, before the fix | 32 busy loops | 7456 |
| controlled, after the fix | 32 busy loops | **11446** |

The 3990-call gap is those seven levers reading exactly `0/0`, while six of the seven are
served on the apo fold. A census taken under contention therefore reports levers going dark,
and nothing in the artifact distinguishes that from a lever that really went dark.

The install is now driven by an import hook rather than by a clock, so it lands as soon as the
modules are ready (`_wrap_on_import` in `scripts/lever_census.py`). It does not change what a
quiet host measures: the fixed census reads under load the same 11446 the idle runs read, so
the fix removes a failure mode rather than moving a number. What it cannot tell you is whether
every existing baseline was recorded quiet. Any that was not has those seven levers recorded
low, and the arm will now say so, which is the point.

One trap the fix had to avoid, and `tests/test_lever_census_wraps.py` pins it: a module is in
`sys.modules` from the moment its execution *starts*, so an import hook can fire while
`tt_bio.tenstorrent` is half-built. The install used to set its did-this-already flag before
the first attribute access, which under an import hook would claim the flag, throw, get
swallowed, and leave all seven counters dead for the whole process. Deterministically wrong
instead of load-dependently wrong. It now checks that every name it rebinds exists before
claiming anything, and a half-built module is retried rather than burned.

**Do not fan census folds across a host's idle cards** to save wall clock. That is the standing
practice for independent single-card measurements and it is right for a timing, but a census is
a different measurement: the folds contend for CPU, not just for cards. The fix makes this
survivable rather than fatal, and serial is still the way to record a baseline you intend to
gate against.

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
    --size-ladder-record-lever PAIR_PROJ_L1_OUT
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
