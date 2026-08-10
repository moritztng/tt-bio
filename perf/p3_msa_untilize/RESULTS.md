# X3 — `protenix-trunk--p3-msa-untilize`, Phase 3

Protenix-v2, trunk only, 298 aa. Everything below was **measured on qb1 card 2 at ttnn 0.67.4**,
which is a campaign absolute, on branch `wk/protenix-trunk--p3-msa-untilize`. Host load at
measurement time: `uptime` 1.12 / 1.12 / 1.10 on 32 cores, cards 0 and 1 carrying `p3-sdpa` and
`p3-narrow-write`, card 3 idle. Nothing merges to main.

**Headline: Q14 is closed and the answer is that the lever does not exist on qb1.** The op P5 priced
at 1425.1 ms/fold on qb2 costs **43.7 ms/fold** here. T5's pc figure is the one that reproduces. What
does exist on this card is the same defect at a *different shape*, and it is a ttnn single-core
fallback, proven by forcing the fallback on a shape that is otherwise fast.

**Delivered: 22.0 ms/fold, both arms `torch.equal` True at the stage output.**

---

## Predictions (before measuring)

Committed and pushed as `perf/p3_msa_untilize/PREDICTIONS.md` in `e5cdc39d`, before the first
`get_device()` in this pass. Reproduced here in full, scored in the results below.

Two things I settled by inspection and recorded first, so they could not be retro-fitted:

- **The brief's third candidate is dead on inspection.** The untilized tensor is `z = (rows*C, D*J)`
  with `C = D = 32` and `J` the token count, so the last dimension is `32 x 298 = 9536` **whatever
  the MSA depth is**. Depth sets `S`, the contraction length, which is the inner dimension of the
  producing matmul and never appears in the untilize's shape. "The MSA depth reached sets the
  last-dim width" cannot be the explanation.
- **pc card 0, qb1 and qb2 are the same silicon.** `tt-smi -ls` reports Blackhole `p150a` on both
  hosts, so a Wormhole-vs-Blackhole split in the untilize implementation is not available either.

**D1.1 — it reproduces.** `to_layout(TILE -> ROW_MAJOR)` on `(9536, 9536)` bf16 lands at
**20000-45000 us/call**, within 40 % of P5's qb2 figure (35627 us). **Wrong if under 5000 us.** Confidence stated
as moderate, not high.

**D1.2 — `trunk_msa` lands near qb2's 3492.9, not near pc's 1979.3**, i.e. above 3000 ms/fold.
**Wrong if below 2400.**

**D1.3 — the width sweep shows a cliff, not a ramp**: a jump of more than 5x between two adjacent
widths. **Wrong if us/call rises smoothly**, which would put the mechanism on per-transaction NOC
issue rather than on P5's circular-buffer account.

**D1.4 — the same-bytes control is fast.** `(298, 1024, 298)` untilizes in **900-1600 us**.

**D2.1** Chunking the untilize into 32 blocks of 10 tile-columns runs the chain **at least 5x**
faster. **Wrong if under 2x.**
**D2.2** The chunked arm's `(d, c)` ordering compensated by permuting `o_weight` is **not**
guaranteed bit-exact; I predict `torch.equal` **False** on that arm, because it permutes the K index
of the consumer's reduction. The arm that restores the exact `(c, d)` order is the one I predict
**True**. This prediction exists because P5's A1 arm died exactly here, `torch.equal` False at max
abs **0.814**, and re-running A1 or the rank-3-first variant would be a wasted pass.
**D2.3** Removing the untilize is not available: a TILE reshape that changes the last dim has to
relayout, and P5 already measured the one reshape that does not (4.4 us, untilize unchanged).

**D3.1 (C3)** saves **15-35 ms/fold** against P5's 26.4 on qb2, `torch.equal` True at `trunk_msa`'s
output. **D3.2 (C5)** saves **10-20 ms/fold** against P5's 14.8, `torch.equal` True at
`trunk_template`'s output. Order assumed for the `p3-narrow-write` overlap: **the hoist lands first.**

**D4.1** C4's remaining fixed term prices **under 30 ms/fold** in my slice and I expect to close C4
with an evidenced ceiling rather than a lever.

### Scored

| # | predicted | measured | verdict |
|---|---|---|---|
| D1.1 | 20000-45000 us | **1004.2 us** | **WRONG, and this is the finding.** The defect is not on this card at this shape |
| D1.2 | `trunk_msa` > 3000 ms/fold | **1852.4 ms/fold** | **WRONG.** qb1 sits with pc (1979.3), not with qb2 (3492.9) |
| D1.3 | a cliff, > 5x between adjacent widths | 723.8 us at 240 tile-cols, **24999.7 us at 248** — 34.5x | **RIGHT** |
| D1.4 | 900-1600 us | **1318.1 us** | **RIGHT** |
| D2.1 | >= 5x from chunking | **11.9x**, on the shape where the fallback fires | RIGHT where the fallback fires; void at 298 aa, where there is nothing to fix |
| D2.2 | the reordering arm not bit-exact | not built — the arm it would fix does not fire at 298 aa | not reached |
| D3.1 | 15-35 ms/fold | **8.4 ms/fold** | **WRONG, a miss.** Called as one, reason below |
| D3.2 | 10-20 ms/fold | **13.6 ms/fold** | **RIGHT** |
| D4.1 | under 30 ms/fold, closed on a ceiling | **1.8 ms/fold ceiling**, unreachable | RIGHT |

Three of nine predictions lost, and the two that lost hardest are the two the leg existed to test.

---

## Roofs, measured on this card

**I did not inherit a roof.** Every figure below was streamed on **qb1 card 2** in this pass by
`perf/p3_msa_untilize/p3_untilize_probe.py`, one process, `ttnn.synchronize_device` on both sides of
every timed region. `compute_with_storage_grid_size()` returns **13x10**; `CORE_GRID_MAIN` is 11x10.

| roof | qb1 card 2, this pass | method |
|---|---:|---|
| DRAM -> DRAM copy, 63 MB | **399.1 GB/s** | DRAM -> DRAM clone (ladder 282.2 at 8 MB, 390.2 at 32 MB) |
| DRAM read, 63 MB | **379.5 GB/s** | DRAM -> L1 clone |
| DRAM write, 8 MB | **197.5 GB/s** | L1 -> DRAM clone |
| single-core untilize | **10.1 GB/s** | `ttnn.untilize(use_multicore=False)`, size-independent |

**I deliberately did not measure a compute roof this pass, and here is why that is not a gap.** Every
op this leg delivers against sits on the **memory side at arithmetic intensity 0**: the untilize, the
relayout chain and the PWA weight slices compute nothing at all, and C5 removes calls rather than
making one faster. There is therefore no compute denominator any row here needs, and the qb1 machine
balance (B1's 337.9 FLOP/byte at the square-default roof) is quoted only to place the ops, not used
as a denominator by any figure below. The one place arithmetic does enter, C4's transitions, is
closed on a fixed-term fit rather than on a % of a roof.

---

## What changed, and the A/B that measured it

### Deliverable 1 — Q14, settled on this card

**The measurement that settles it, and it does not depend on attributing anything to a single op.**
`trunk_msa` and `trunk_template` timed with a device sync on both sides, inside a real 298 aa fold
of `examples/prot300.yaml` (`n_msa=35`), three folds per arm alternating base/arm in one session:

| stage wall | pc card 0 (T5, 0.68.0) | qb2 card 1 (P5, 0.68.0) | **qb1 card 2, this pass, 0.67.4** |
|---|---:|---:|---:|
| `trunk_msa` | 1979.3 ms/fold | 3492.9 ms/fold | **1852.4 ms/fold** |
| `trunk_template` | 719.0 ms/fold | 820.1 ms/fold | **690.9 ms/fold** |

**qb1 is with pc.** `trunk_msa` here is 6.4 % below T5's pc stage-wall figure, and 47.0 % below P5's qb2 stage-wall
figure. The pc and qb2 numbers are both ratios taken at 0.68.0; this
column is the qb1 absolute that replaces them for ranking.

**The op itself, in the live fold, counted not derived.** `ttnn.to_layout` was wrapped so that only
the calls untilizing a >= 4096-wide 2-D TILE tensor are synced and timed; everything else in the fold
passes through untouched. **40 calls/fold, counted** (4 MSA blocks x 10 recycling cycles), median
**1091.6 us/call**, shape `(9536, 9536)` = **43.7 ms/fold** at the charter's x10 stage conversion,
which the counted `Trunk._template` = **10 calls/fold** confirms independently.

**Probe against production, side by side, with the gap named.** The standalone probe on the same
shape in a separate process is **1004.2 us/call**; the live fold is **1091.6**, i.e. the in-fold cost
is **8.7 % higher**, measured against the probe figure. That gap is the right sign and the right size for a
DRAM path shared with the rest of the stage, and it means the probe under-reports rather than
over-reports. Against P5's 35627.3 us/call on qb2 the qb1 in-fold figure is **32.6x smaller**.

**So C1 is dead on qb1.** 1425.1 ms/fold is not available here; the whole op costs 43.7, and at
1091.6 us for 173.4 MB moved twice it runs at **333.0 GB/s, 83.4 % of this card's measured
399.1 GB/s copy roof**. There is no headroom to take.

### Why the two cards differ, named

**Not the wheel, and not the MSA depth.** pc runs 0.68.0 and is fast; qb2 runs 0.68.0 and is slow, so
the wheel cannot be the variable. Depth cannot be it either, for the reason recorded before I opened
the device: the untilized width is `32 x tokens`, and depth never enters the shape.

**It is a ttnn op-selection fallback whose trigger shape differs per chip, and the mechanism is
measured rather than argued.** `ttnn.untilize` has a multi-core path and a single-core path. Forcing
the single-core path on the shape that is otherwise **fast** reproduces the pathology exactly:

| shape | `to_layout` | `untilize(use_multicore=True)` | `untilize(use_multicore=False)` |
|---|---:|---:|---:|
| 298 x 298 tiles (the 298 aa fold's own shape) | **1008.1 us / 360.8 GB/s** | 998.8 us / 364.2 GB/s | **36070.6 us / 10.1 GB/s** |
| 256 x 256 tiles | **26659.4 us / 10.1 GB/s** | 26690.5 us / 10.1 GB/s | 26687.8 us / 10.1 GB/s |

**10.1 GB/s is the single-core untilize kernel, and it is size-independent** — the same rate appears
at 64, 128, 173 and 256 MB. So the slow rows everywhere in this record are **1 of 130 cores**, and
the fast rows are the multi-core path at 83-92 % of the measured copy roof, which by charter §4.5
cannot be core-starved. On the bad shape, asking for `use_multicore=True` explicitly does not help:
the op refuses the multi-core path and falls back silently.

**This replaces P5's mechanism hypothesis rather than confirming it.** P5 proposed a circular-buffer
serialisation that degrades as the tile row stops fitting, and back-solved 10.2 GB/s to "~4
core-equivalents of 110". It is not a degradation and it is not four cores: it is a binary path
selection and one core. P5's kill test (sweep the width, look for a cliff) was the right test and it
does show a cliff — but the cliff is not monotonic in width, which a CB-capacity account requires.

**The map on qb1 card 2, at ~fixed total bytes (88804 tiles), is an island of badness, not a
threshold:**

| last-dim tile-cols | 128 | 192 | 224 | 240 | **248** | **254** | **256** | **257** | **258** | 264 | 288 | **298** | 320 | 384 | 512 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| us/call | 498 | 498 | 627 | 724 | **25000** | **26238** | **26680** | **96180** | **27055** | 771 | 976 | **998** | 1120 | 1583 | 2928 |

and it is not a function of the last dim alone — holding the last dim at the bad 256 tile-columns and
varying the row count gives fast at 64 and 298 tile-rows (241.1 and 918.6 us) and slow at 128, 256,
346 and 512 (13319, 26630, 35976, 53243 us). **The predicate is a joint function of the two tile
counts and it lives inside tt-metal's untilize op selection, which the wheel does not ship as
source.** I am naming it as a ttnn defect with the fallback identified and the trigger mapped, and
stopping there: chasing the exact predicate into a source build is a different leg.

**Overlap: serial, and the evidence is the arithmetic.** 40 counted untilize calls at 1091.6 us is
43.7 ms of a 1852.4 ms `trunk_msa` stage wall — 2.36 % of that stage wall — and the synced per-call
sum of the ops I timed lands inside the stage wall rather than beside it, so the stage is nearer
`compute + comm` than `max(compute, comm)`. Nothing in this stage hides one op's DRAM traffic behind
a neighbour's compute. Caveat I am carrying rather than burying: the syncs around the untilize were
active in **both** arms, so the 1852.4 ms stage wall is an upper bound on the unsynced one. It is
nowhere near 3492.9 either way.

### Deliverable 2 — the fix, and why it is not shipped

**At 298 aa there is nothing to fix.** The op runs at 83.4 % of the measured copy roof. Deliverable 2
is void on this card, which is the outcome the brief named as a successful one.

**But the fallback does fire on shapes a real fold reaches, and there is a bit-exact mitigation.** A
Protenix target of **248-258 tokens** on qb1 card 2 makes `z` 248-258 tiles square and lands inside
the island: at T=256 the untilize costs 26.7 ms/call x 40 calls/fold = **1.07 s/fold** for one op.
Row-blocking the untilize walks it out of the fallback:

| T=256 tokens, 256 x 256 tiles, 128.0 MB | us/call | GB/s | vs whole | `torch.equal` |
|---|---:|---:|---:|---|
| whole (what ships today) | 26644.9 | 10.1 | 1.00x | — |
| row blocks of 32 tiles | 2259.9 | 118.8 | **11.79x** | **True** |
| row blocks of 64 tiles | 2235.8 | 120.1 | **11.92x** | **True** |
| row blocks of 128 tiles | 28034.3 | 9.6 | 0.95x | True |

**Bit-exact, `torch.equal` True on every arm** — it is a slice-and-concat of disjoint row bands, so
no arithmetic exists to change. **And it must not be applied unconditionally**: the same blocking on
the 298 x 298 shape costs **3.1x** (997.3 -> 3119.3 us), because it forces the copy through a
concat that the multi-core path does not need.

**Recommendation: record it, do not ship it.** A conditional guard needs the fallback predicate, and
I have the map for one chip at one wheel, not the rule. The charter's §1 is explicit that a finding
outside the 298 aa scope gets recorded rather than chased. It is written up here and in the merge
recommendation as a flagged follow-up.

### Deliverable 3 — the two banked levers, delivered on qb1

Both are production changes on `wk/protenix-trunk--p3-msa-untilize` (`c7f3eca5`). The A/B is
`perf/p3_msa_untilize/p3_stage_ab.py`: **one session, one card, arms alternating fold by fold**
(base, arm, base, arm, ...), so host drift hits both equally, and the **baseline arm is the
unmodified pre-change body restored in-session** rather than a remembered number.

**C5 — hoist the loop-invariant template z projection** (`tt_bio/protenix.py`, `Trunk._template`).
The projection `self._lin(zn, "template_embedder.linear_no_bias_z.weight")` sat inside
`for t in range(nt)` on a loop-invariant operand and a loop-invariant weight, so it ran nt times and
nt-1 results were thrown away.

| `trunk_template` | fold 0 | fold 1 | fold 2 | median |
|---|---:|---:|---:|---:|
| baseline | 690.87 | 690.77 | 690.85 | **690.9 ms/fold** |
| hoisted | 677.25 | 677.26 | 678.39 | **677.3 ms/fold** |

**Delivered 13.6 ms/fold**, 1.97 % of the baseline stage wall (690.9 ms/fold). The three raw walls span
0.17 % of their own median wall on each arm, so the separation is far outside the noise.

**C3 — cache the PairWeightedAveraging per-head weight views** (`tt_bio/tenstorrent.py`). The eight
per-head slices of `proj_z`, `proj_m`, `proj_g` and `proj_o` were re-cut on every call; they are now
cut once in `__init__`.

| `trunk_msa` | fold 0 | fold 1 | fold 2 | median |
|---|---:|---:|---:|---:|
| baseline | 1852.39 | 1850.98 | 1854.01 | **1852.4 ms/fold** |
| cached | 1843.00 | 1844.79 | 1843.95 | **1844.0 ms/fold** |

**Delivered 8.4 ms/fold**, 0.45 % of the baseline stage wall (1852.4 ms/fold). **This is a miss against
my own prediction of 15-35 and I am calling it one.** The reason is that the unaligned-slice penalty
is much smaller on this card at these weights: on qb1 an unaligned slice of a 4096x4096 source costs
**227.59 us against 28.93 us aligned, 7.9x**, where P5 measured **202.42 against 7.49, 27.0x** on
qb2. The absolute unaligned cost is nearly the same on the two cards; qb1's *aligned* path is 3.9x
slower, so the recoverable difference is smaller. The production weights are far smaller than that
probe source, and 8.4 ms/fold over 30 counted `PairWeightedAveraging` calls is 280 us/call across 32
slices, ~11 us saved per unaligned slice. P5's 26.4 was a qb2 ratio; **8.4 is the qb1 absolute that
replaces it.**

**Counted call structure, in a live fold, not derived:** `PairWeightedAveraging.__call__` **30/fold**
(3 PWA-carrying MSA blocks x 10 cycles, so 240 slice-sites per weight per fold, confirming P5's
x240), `Trunk._template` **10/fold** (confirming the charter's x10 stage conversion),
`Trunk._lin` **46/fold** with the hoist in place.

**The end-to-end fold wall does not resolve either of these, and I am saying so rather than quoting
it as if it did.** Base folds 29.603 / 29.534 / 29.678 s, arm folds 29.704 / 29.552 / 29.579 s. The
arm's median is 24 ms lower, the same sign and size as the 22.0 ms/fold the two stage walls give, but
the base arm's own spread is 144 ms, so a 22 ms change is six times below what a 29.6 s wall
resolves in three folds. **The delivered figures are the stage walls, which are the production path
and are synced on both sides; the fold wall is reported as not resolving them.**

### Deliverable 4 — C4, closed

Row sweep at the MSA transition's own shape, `(35*mult, 320, 128) @ (128, 512)`, on this card:

| mult | `layer_norm` us | `linear` us | `linear` + `activation="silu"` us |
|---:|---:|---:|---:|
| 1x | 37.33 | 105.14 | 178.18 |
| 2x | 59.14 | 179.76 | 298.44 |
| 4x | 62.01 | 332.95 | 541.05 |

**P5's partial kill is confirmed on qb1: these are not launch-floor bound.** 2x rows costs 1.58x
(`layer_norm`) and 1.71x (`linear`), against the 1.46-1.62x P5 measured. Fitting the fixed term:
`linear`'s slope over 2x->4x is 76.60 us per 1x-unit, so its fixed term is **28.54 us, 27.1 % of that 1x call wall (105.14 us)** (P5 got 27.9 % of its own 1x call wall, on a different card — the *share* replicates
even though the absolutes do not). `layer_norm`'s slope over 1x->2x is 21.81, giving a fixed term of
**15.52 us, 41.6 % of that 1x call wall (37.33 us)**.

**The ceiling, and it closes C4.** The MSA stack's own transitions run 40 calls/fold (4 MSA blocks x
10 cycles). `(15.52 + 28.54) us x 40 = **1.76 ms/fold**` is the entire recoverable fixed term in my
slice, and it is recoverable only by batching calls that are sequential links in a residual chain,
which is not available. **C4 is CLOSED on an evidenced ceiling of 1.8 ms/fold, not partially.** For
completeness, `Transition.__call__` was counted at **1520/fold** fold-wide, but that count spans
`pf_stack`, the template stack and the diffusion path, which are other teams' slices; only the 40
MSA-stack calls are mine and only they are priced here.

### The `p3-narrow-write` overlap, reconciled not summed

**Order assumed: my hoist lands first**, as predicted before measuring. C5 takes
`tt_bio/protenix.py:306` from **40 calls/fold to 10** (nt=4 templates x 10 recycling cycles, of which
10 remain), so `p3-narrow-write`'s program-config lever on that site is priced against 10 calls, not
40: their 9.6 ms/fold becomes **~2.4**. The pair is worth **13.6 + 2.4 = ~16.0 ms/fold combined, not
24.4**, and the two figures must not be added at their standalone values. If their config lands
first instead, it takes the larger share and C5's credit falls proportionally; either way the pair is
~16-17, and STATUS's ~17 stands within the difference between P5's qb2 14.8 and my qb1 13.6.

**Not double-counted with x524.** My stage figures convert **x10**, one stage call per recycling
cycle, and cover the stage's own ops only. The 40 `trunk_msa` `PairformerLayer` executions the
attention and trimul legs already price at x524 are not in any number above.

---

## Delivered ms/fold

| lever | delivered, qb1 card 2, 0.67.4 | measured how | parity |
|---|---:|---|---|
| **C5** — hoist the template z projection | **13.6 ms/fold** | `trunk_template` stage wall, live 298 aa fold, A/B in one session | `torch.equal` **True**, max abs 0.0 |
| **C3** — cache the PWA per-head weight views | **8.4 ms/fold** | `trunk_msa` stage wall, same session, same folds | `torch.equal` **True**, max abs 0.0 |
| **total delivered** | **22.0 ms/fold** | | both bit-exact |
| C1 — the OPM untilize | **0**, and the lever does not exist here | op is 43.7 ms/fold in-fold at 83.4 % of the copy roof | — |
| C4 — the transition fixed term | **0**, closed on a 1.8 ms/fold ceiling | row-sweep fit at the stage's own shape | — |
| the untilize fallback mitigation | **not shipped** — 11.9x where it fires, 0.32x where it does not | probe, `torch.equal` True | bit-exact, unconditional application is a regression |

Against the ranking X3 inherited: rank 2 was **179.2 or 1425.1 ms/fold, unreconciled**. It is now
**43.7 ms/fold on qb1, with no recoverable headroom**, and rank 2 should be struck. Ranks 9 and 10
were 26.4 and ~17 as qb2 ratios; they are now **8.4** and **~16.0** as qb1 absolutes.

---

## Parity

Measured at the fold's own 298 aa shape, on the **stage output tensor**, baseline vs arm, captured on
the same fold index in the same session:

| stage | `torch.equal` | max abs | mean abs | shape |
|---|---|---:|---:|---|
| `trunk_msa` (C3) | **True** | 0.0 | 0.0 | the stage's own 298 aa output |
| `trunk_template` (C5) | **True** | 0.0 | 0.0 | the stage's own 298 aa output |

C3's parity is the one that needed measuring rather than asserting: P5 could only argue it was
bit-identical because it had no second value to compare. Running the pre-change body in the same
session gave the second value, and `torch.equal` returned True.

The untilize row-blocking arm is also **`torch.equal` True** against the unblocked untilize at every
block size tried, at both 256 x 256 and 298 x 298 tiles — it is a slice-and-concat of disjoint row
bands, and the measurement confirms the argument rather than replacing it.

**No arm here reorders a reduction.** P5's A1 arm did, and returned `torch.equal` False at max abs
**0.814**; I did not rebuild A1 or the rank-3-first variant, and the reordering arm my predictions
flagged as probably-not-exact was never reached because the defect it would fix does not fire at
298 aa on this card.

---

## Merge recommendation

**Recommend merging C5 and C3 — 22.0 ms/fold, both `torch.equal` True at the stage output — subject
to Moritz's explicit OK.** Branch `wk/protenix-trunk--p3-msa-untilize`, commit `c7f3eca5`. Nothing
has been merged.

- **Parity class: bit-exact.** Measured, not argued, at the fold's own shape. Both changes are
  structural: C5 removes 30 evaluations of a function of loop-invariant arguments, C3 hands the same
  device tensor to every call instead of re-deriving it. Neither touches an accumulation order.
- **Release-gate status: not gated.** No dependency change, no accuracy risk, and no OOM risk — C3
  holds 32 additional per-head views per `PairWeightedAveraging` instance, ~16 kB each at 298 aa,
  which is why it was safe to cut them at construction in the first place.
- **Merge order matters for the ledger, not for correctness.** If `p3-narrow-write`'s config on
  `protenix.py:306` merges too, the pair is ~16.0 ms/fold combined, not 13.6 + 9.6.

**Flagged, not proposed for merge: the untilize single-core fallback.** A 248-258 token Protenix
target on qb1 card 2 pays ~1.07 s/fold on one op, and a bit-exact 11.9x row-blocking mitigation
exists but regresses the 298 aa path 3.1x if applied blind. This wants either the tt-metal predicate
(a source build, a different leg) or an upstream ttnn bug report. Recording it, not chasing it.

---

## Corrections to the inherited record

1. **Q14 is closed and neither figure was wrong; the org was comparing two different cards' op
   selection.** `to_layout:3059` costs **1091.6 us/call in a live fold on qb1 card 2 at 0.67.4,
   43.7 ms/fold at 40 counted calls**. T5's pc figure is the one that reproduces. P5's 1425.1 is real
   on qb2 card 1 and is a **qb2 ratio that does not transfer**; C1 should be struck from the Phase-3
   ranking. The independent cross-check agrees: `trunk_msa` is **1852.4 ms/fold** here against T5's
   1979.3 on pc and P5's 3492.9 on qb2.
2. **The variable is neither the wheel nor the MSA depth, and the mechanism is a ttnn single-core
   fallback.** pc runs 0.68.0 and is fast, so the wheel is out; the untilized width is `32 x tokens`
   and depth never enters the shape, so depth is out. Forcing `use_multicore=False` on the shape that
   is otherwise fast reproduces the pathology exactly — **36070.6 us at 10.1 GB/s, against 998.8 us
   at 364.2 GB/s multi-core**. The slow path is **1 of 130 cores**, and on the trigger shapes the op
   refuses the multi-core path even when asked for it.
3. **P5's circular-buffer account is superseded.** The cost is not a degradation that scales with
   last-dim width and it is not "~4 core-equivalents"; it is a binary path selection and one core.
   The trigger is also not the last dim alone: on qb1, holding the last dim at 256 tile-columns gives
   fast at 64 and 298 tile-rows and slow at 128, 256, 346 and 512.
4. **The brief's "MSA depth sets the last-dim width" is wrong** and was recorded as wrong before the
   device was opened. Width is `32 x token count`; depth is the producing matmul's contraction length
   and never appears in the untilize's shape.
5. **C3 is 8.4 ms/fold on qb1, not 26.4.** The unaligned-slice penalty is 7.9x here (227.59 vs
   28.93 us at identical output bytes) against 27.0x on qb2, because qb1's *aligned* slice path is
   3.9x slower while the unaligned absolute is nearly identical. P5's mechanism — the start offset,
   not the size — replicates exactly; only its magnitude is a card property.
6. **C4's fixed-term *share* replicates across cards even though its absolutes do not:** 27.1 % of
   the 1x `linear` call here against P5's 27.9 % of its own. The ceiling it implies in my slice is
   **1.8 ms/fold** and it is unreachable, so C4 is closed rather than left partial.
7. **T5's site line numbers are 25 low against `origin/main` and its `@1695`/`@1701` are the
   triangle-attention qkv/gate projections, not trimul** — P5's correction, inherited and not
   re-derived. Likewise T5's "no `activation=` anywhere in either stage": `Transition` fc1 passes
   `activation="silu"`, and on this card it costs **178.18 us against 105.14 for the identical
   matmul, 1.69x** at the MSA transition's 1x shape, against P5's 4.11x on qb2. Recorded; the
   `silu` premium itself is T3's row, not mine.
