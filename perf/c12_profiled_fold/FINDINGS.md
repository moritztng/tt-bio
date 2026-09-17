# The fold's unit census is exact. Its per-op device time needs the whole board, and the board is split.

`ws:c12-profiled-fold`, pass 1, 2026-09-17. qb2 card 3 (board 410D), `eab845ad1`, shipped
`ttnn==0.68.0` wheel, AICLK forced to 1350 MHz through tt-kmd's ARC queue and sampled at 1 kHz
during every timed leg. No production code changed.

## VERDICT: PARTIAL. 81.5 % of the fold is now measured per op. Three campaign figures do not survive it.

## 1. A profiled fold is not the instrument, and the arithmetic says so before the first run

The fold dispatches 465,664 ttnn calls. tt-metal's device profiler costs 48 bytes of DRAM per
dispatched program per RISC per core and ~350 kB of host CSV per program, so a whole-fold capture is
on the order of 10^5 GB of CSV against qb2's 3.1 TB free. `--enable-sum-profiling` does not change
it: the sum budget buys two extra optional markers per program
(`profiler_state_manager.cpp:39`) and still allocates a per-program DRAM slot. Above 1000 programs
without `--op-support-count` the failure is total, not graceful.

So the instrument is one *instance* of each repeating unit, grabbed out of a live fold and replayed
fenced under the profiler, weighted by the fold's own calls per unit. Grabbing the whole unit is
what removes both of the census's biases in one move: the unit's own code allocates its own
intermediates and picks its own memory and program configs, so there is no DRAM-operand overprice
and no `@grid110` arm the fold does not run. The precursor trick is `b2z-kernel-cycle-census`'s.

## 2. BLOCKER: device profiling on qb2 needs the whole board, and C12's card pinning splits it

Measured on this host today, not recalled. A tt-metal **source** build refuses a single P300 chip:

```
TT_VISIBLE_DEVICES=3 ... python3 -c "import ttnn; ttnn.open_device(device_id=0)"
  -> TT_FATAL in tt::Cluster::generate_cluster_descriptor()
```

One chip of a two-chip board makes the cluster type `CUSTOM`, which then demands a mesh graph
descriptor (`tt_cluster.cpp:198,273`). Legal subsets on qb2 are `{0,1}`, `{2,3}` and all four. The
shipped `ttnn==0.68.0` wheel predates the assert and opens a single card, but it has the device
profiler compiled out (`ENABLE_TRACY=OFF` is not the issue — the wheel simply does not carry it).
The two Tracy-enabled source builds on this host, `/home/ttuser/tt-metal-b2z` and
`-k10`, both carry the assert.

The pair opens and folds correctly: verified this pass with `TT_VISIBLE_DEVICES=3,2`, which
reproduced the `b2z` instrument proof (2048^3 bf16 matmul, 0.1520 ms back to back against that
row's 0.1537 ms) and leased both chips under this worker's identity. Then card 2 was taken by
`c12-unfused-silu-bh` and `tt_bio`'s `CardSetLease` correctly refused the open:
`physical card 2 ... is in use by worker:c12-unfused-silu-bh (pid 964065)`.

**Every other qb2 C12 row is pinned to card 2 and this row to card 3, so the board is never whole.**
The pinning was introduced to separate C12 from the release gate on board 4103, which it does. It
also makes device profiling structurally impossible for as long as any card-2 row is running. This
row needs board 410D as a **pair**, not a card; it does not need it for long (one unit leg is a
short precursor plus a few fenced reps, minutes, not a fold sweep).

## 3. SPINE: the fold's unit census, exact and instrument-independent

One session, card 3, clock held at exactly 1350 MHz on every sample of every leg. Legs interleave
plain and bracketed folds inside one session (A,B,A) so a constant co-tenant cancels.

| class | calls per fold |
|---|---|
| AdaLN | 12000 |
| AttentionPairBias | 6264 |
| DiffusionTransformerLayer | 6000 |
| ConditionedTransitionBlock | 6000 |
| Transition | 960 |
| DiffusionTransformer | 600 |
| TriangleMultiplication | 560 |
| TriangleAttention | 560 |
| **PairformerLayer** | **280** |
| DiffusionModule / Diffusion | 200 / 200 |
| MSALayer | 16 |
| PairWeightedAveraging / OuterProductMean | 16 / 16 |
| Pairformer / MSA | 5 / 4 |
| PairAssemblyDevice / RelPosGather | 2 / 2 |
| PairformerModule / PairConditioningDevice / ConfidenceHeadsDevice | 1 / 1 / 1 |

`PairformerLayer` = 280 reproduces `b2z-kernel-cycle-census`'s independently derived 280 calls per
fold exactly. That is the cross-check that this census is the fold's and not the harness's.

Roots, i.e. the classes with no instrumented ancestor, and their unsynced inclusive host wall in the
bracketed leg:

| root | calls | incl_s |
|---|---|---|
| Pairformer | 4 | 7.7063 |
| DiffusionModule | 200 | 5.0294 |
| MSA | 4 | 0.6723 |
| PairformerModule | 1 | 0.2609 |
| PairConditioningDevice | 1 | 0.1479 |
| PairAssemblyDevice | 2 | 0.0805 |
| ConfidenceHeadsDevice | 1 | 0.0231 |
| **sum** | | **13.9205** |

against a 20.4148 s bracketed fold: **68.2 % of the fold is inside one of these seven classes.**
The bracket itself costs **1.165x** (20.4148 s against a 17.5213 s plain median), which is 7x this
session's own A/A floor of 0.419 s, so the bracket's cost is real and its seconds carry it. The
call counts do not: they are integers and the instrument cannot inflate them.

## 4. The unsynced host tree cannot split a block, and the evidence is in its own skew

`Pairformer/PairformerLayer/Transition`: 512 calls, 5.7904 s inclusive, **mean 11.309 ms against a
median of 0.7414 ms — 15.3x.** A ttnn call is an async enqueue, so when the dispatch queue fills the
calling thread spins, and which sibling call absorbs that wait is arbitrary. The block's own
inclusive time is sound (`PairformerLayer` median 30.32 ms against mean 30.09 ms, 0.8 % apart, and
`b2z` measured the same block at 36.4994 ms synced wall in a 24.7 s-fold session), but the split
*within* a block is an artifact of dispatch-queue timing and no amount of averaging fixes it.

**This is why the per-op answer needs the device profiler and not a cheaper host instrument.** It
also retires the cheapest fallback: a per-op synced host timer would serialise a block that is
98.9 % inside a kernel back to back, and price the serialisation instead of the op.

## 5. SCREEN, not a number of record: co-tenancy costs this fold 2.64 s at an identical clock

The plain arm read **17.3118 s and 17.7308 s, median 17.5213 s**, with AICLK reading exactly 1350
MHz on all 15,017 and 15,396 samples taken inside those two intervals, zero read errors. The
campaign's fold of record is 14.881 s at the same pinned clock on a quiet box. Board 4103 was
running the v0.9.0 release gate on cards 0 and 1 and card 2 was running a sibling C12 row
throughout.

So **+2.64 s, +17.7 %, at a clock that did not move.** This is the mechanism the brief named: a
co-tenant steals host cores, inflating host time without moving it with the clock, so a two-clock
control cannot see it. It is a screen and it is labelled one. It is also a measured upper bound on
how much of this fold is host-contendable, which is a number Axis A did not have.

## 6. Method correction: a sample gap is the sampler, not the clock

The first reduction discarded two of three legs on the pinned helper's rule that no sample gap may
exceed 10 ms. Both discarded legs had min = max = 1350 MHz with zero read errors; the worst gaps
were 10.96 ms and 17.66 ms out of 15,017 and 17,847 samples. That is a 1 kHz sysfs sampler being
descheduled on a box carrying a release gate, not the clock leaving the target. The gate now scores
the two claims separately — the clock must read the target on **every** sample with no read error
(unchanged, load-bearing), and the sampler must have covered the interval (no gap over 60 ms, at
most 0.5 % of the interval unobserved). Both discarded legs cover 99.99 %+ of their interval.

## 7. Owed

1. **PER-OP.** Needs board 410D as a pair. `runs/pfl1` (PairformerLayer, 280 calls/fold) and
   `runs/dtl1` (DiffusionTransformerLayer, 6000 calls/fold) together cover the two roots that are
   79 % of the root sum; add `MSALayer`, `PairConditioningDevice`, `PairAssemblyDevice`,
   `ConfidenceHeadsDevice` and `Diffusion`'s own exclusive body for ~full coverage.
2. **RESIDUAL.** Follows from 1 against a quiet-box fold, not against the 17.5213 s screen.
3. **REPLAY-BIAS.** Follows from 1. The class split and its `align_quality` are implemented and
   untested against a real ops report.
4. **PERTURBATION.** The unit-level profiled-vs-bare wall is measured per leg by the harness
   (`synced_wall_ms_per_call` with the profiler on and off). The whole-fold profiled-vs-unprofiled
   ratio is not measurable at all, for the reason in section 1, and the row should say so rather
   than quote a proxy.

---

# Pass 2, same day, once the board came free: 81.5 % of the fold, measured per op

Board 410D as a pair (cards 3+2) under this worker's own lease, Tracy source build
`/home/ttuser/tt-metal-b2z`, AICLK held at 1350 MHz and sampled through every profiled region.
Card 2 freed between the sibling row's sweep processes and the legs below took it for minutes each.

Every leg carries two controls. The **fence** window must contain exactly 2 x 3
`UnaryDeviceOperation` rows, and the **rep control** splits that window into `reps` chunks and
requires identical op-code histograms. Both are reported per leg, and the fence control earned its
place at once: a shape-only fence test matched the block's own 1x1x32x32 `BinaryNg` clusters and
produced a 410-row window that is not divisible by 3 reps. The rep control caught it.

## PairformerLayer, 264 calls per fold, 137 device programs per call

Selected by `transform_s=True` so the 16 MSA-internal layers cannot contaminate it. Rep control OK:
30.3745 / 30.8238 / 30.5286 ms, spread 1.472 %.

| device op | ms/call | % | programs |
|---|---|---|---|
| GenericOpDeviceOperation | **12.0252** | **39.3** | 14 |
| MatmulDeviceOperation | 6.8362 | 22.4 | 46 |
| BinaryNgDeviceOperation | 5.5979 | 18.3 | 31 |
| LayerNormDeviceOperation | 3.6782 | 12.0 | 20 |
| TransposeDeviceOperation | 1.4693 | 4.8 | 5 |
| Slice, Permute, Concat, Softmax, NlpCreateHeads, ReshapeView, Copy | 0.9688 | 3.2 | 21 |
| **device total** | **30.5756** | | **137** |

Bare synced wall 30.7345 ms, so **99.5 % of the block is inside a kernel**, reproducing
`b2z-kernel-cycle-census`'s 98.9 % on the same block at a different commit and clock.

## Diffusion, 200 calls per fold, 1096 device programs per call

Rep control OK: 20.29 / 20.2961 / 20.3209 ms, spread 0.152 %.

| device op | ms/call | % | programs |
|---|---|---|---|
| MatmulDeviceOperation | **9.0148** | **44.4** | 381 |
| BinaryNgDeviceOperation | 3.7573 | 18.5 | 339 |
| SDPAOperation | 2.1659 | 10.7 | 30 |
| LayerNormDeviceOperation | 1.7690 | 8.7 | 114 |
| NlpCreateHeadsDeviceOperation | 1.4713 | 7.2 | 30 |
| ReshapeView, Pad, Slice, Transpose, Concat, Copy, Tilize, Untilize, NLPConcatHeads | 2.1240 | 10.5 | 195 |
| **device total** | **20.3023** | | **1096** |

**Zero `GenericOpDeviceOperation` programs.** The fused generic path does not fire anywhere on the
diffusion side; attention there lands on `SDPAOperation`.

## COMPOSED: 12.1324 s of device in two units, 81.5 % of the 14.881 s fold

```
PairformerLayer    264 x 30.5756 ms =  8.0720 s
Diffusion          200 x 20.3023 ms =  4.0605 s
                                      -------
                                      12.1324 s   81.5 % of 14.881 s
remainder                              2.7486 s
```

That 2.7486 s has to hold **everything else in the fold**: the MSA module, the 16 MSA-internal
pairformer layers, pair conditioning, pair assembly, the confidence head, the sampler's own
per-step body, featurisation, the CIF write, and every second of exposed host time.

### Consequence 1: F = 3.983 s is not exposed host work, and this settles it

`c10-fixed-cost` measured F = 3.9830 s +/- 0.1181 as the term that does not move with AICLK, and its
own UNCOUNTED section refused to call it removable CPU work. It cannot be: **3.983 s exceeds the
fold's entire non-Pairformer, non-Diffusion remainder of 2.7486 s by 1.234 s**, and that remainder
still has real device work in it. Whatever F is, it is not 3.983 s of host code waiting to be
deleted. Axis A does not have 2.983 s of free host work to spend.

### Consequence 2: the census overprices every class it priced, in the same direction

Compared at the device-op level, because that is where the comparison is sound: the census's classes
do not partition the same way the device ops do (`linear` and `matmul` are both
`MatmulDeviceOperation`; `multiply_`, `add_`, `multiply` and `add` are all `BinaryNgDeviceOperation`),
so each census group is summed before comparing. Measured seconds are **lower bounds** -- they cover
81.5 % of the fold, so the missing units can only push them up.

| device op | measured s | census s | signed | ratio | census group |
|---|---|---|---|---|---|
| MatmulDeviceOperation | 3.6077 | 6.1394 | **-2.5317** | 0.588x | linear 4.6604 + matmul 1.479 |
| BinaryNgDeviceOperation | 2.2293 | 2.8642 | -0.6349 | 0.778x | multiply_ 1.751 + add_ 0.8473 + multiply 0.1256 + add 0.1403 |
| LayerNormDeviceOperation | 1.3248 | 1.5330 | -0.2082 | 0.864x | layer_norm 1.357 + layer_norm_w 0.176 |

**Which replay arm does the fold resemble? Neither.** The census published `linear` at 4.6604 s from
its `@grid110` arm against 10.7316 s bare, and the fold is **below even the faster arm**. On 81.5 %
of the fold the Matmul group measures 3.6077 s against a 6.1394 s price. Axis B is sized off
`linear` = 4.6604 s, so every prediction quoted against it -- silu +0.270 s, K-block +0.3423 s,
cond-hoist +0.1445 s -- is quoted against a denominator that is at least 1.7x too large, before the
missing 18.5 % of coverage is added back.

In cycles at 1350 MHz: the Matmul group's 2.5317 s of overprice is **3417.8 Mcycles**, BinaryNg's
857.1 Mcycles, LayerNorm's 281.1 Mcycles.

### Consequence 3: generic_op's 4.0841 s "floor" is beaten by 0.7170 s, so it is not a floor

The call census closes exactly. 14 `generic_op` programs per PairformerLayer x 280 PairformerLayer
calls per fold = **3,920**, which is the campaign's own generic_op call count to the call. At
12.0252 ms per PairformerLayer that is

```
280 x 12.0252 ms = 3.3671 s   measured device
                   4.0841 s   the traffic floor the campaign carries
                  -0.7170 s   0.824x
```

A measured device time **below** its own traffic floor is a contradiction: nothing runs faster than
the bytes it must move. So either the 0.9127 TB byte count or the rate behind it is wrong, and
`generic_op`'s 4.0841 s may not be quoted as a floor until one of them is re-derived. (The 3.3671 s
assumes the 16 MSA-internal layers cost the same per call as the 264 measured ones; they carry the
same `z` shape. Measuring them is owed.)

With generic_op at 3.3671 s rather than 4.0841 s, the arithmetic the brief called impossible --
4.084 + 3.983 = 8.07 s inside 4.344 s of unaccounted fold -- has one of its two terms cut and the
other reclassified. It still does not fit, and now the reason is clear: the census's priced classes
were themselves overpriced by ~3.4 s, so there was never 4.344 s of unaccounted fold to fit into.

## PERTURBATION: per unit, and the whole-fold number is not measurable

| unit | bare wall | profiled wall | ratio |
|---|---|---|---|
| PairformerLayer | 30.7345 ms | 31.3153 ms | **1.0189x** |
| Diffusion | 20.6703 ms | 36.7973 ms | **1.780x** |
| DiffusionTransformerLayer | 0.7936 ms | 1.5109 ms | **1.904x** |

The overhead tracks programs-per-second, not seconds: a 137-program block absorbs it, a
41-program layer does not. **The device kernel sums are not affected in kind** -- the profiler times
kernels, not the gaps between them, and the rep control puts the per-rep spread at 1.472 % / 0.152 %
/ 1.014 %. What is disqualified is the *wall* of a program-starved unit under the profiler, and any
A/B taken under it.

The whole-fold profiled-vs-unprofiled ratio the brief asks for **cannot be measured at all** at
465,664 calls, for the reason in section 1, and this row will not quote a proxy for it.

## Method: grab the LARGEST unit that fits the budget

`DiffusionTransformerLayer` is quoted above for perturbation only, not as in-fold device time, and
the reason is a bias this harness introduced itself. Thirty DTL calls run inside one `Diffusion`
call, and:

```
30 x 0.7242 ms = 21.7260 ms   standalone DTL replay
                 20.3023 ms   the whole Diffusion call that contains all thirty
                              -> 1.070x, and the part cannot exceed the whole
30 x 41        = 1230         programs
                 1096         programs in the whole Diffusion call
                              -> 134 extra programs in the standalone replay
```

The standalone replay does not just run slower, it runs a **different graph**: DTL alone shows 3
`PadDeviceOperation` and 4 `CopyDeviceOperation` programs per call that the in-fold path does not
need. `ttnn.clone` of the grabbed operands does not preserve their in-fold memory config, so a unit
small enough to be dominated by its own boundary inherits exactly the DRAM-operand bias this row
exists to remove. A 1096-program unit does not. **So: grab the largest unit the profiler budget
allows, and treat any unit whose program count is comparable to its operand count as suspect.**

## Still owed after pass 2

1. Coverage from 81.5 % to ~100 %: `MSALayer` (16), the 16 MSA-internal `PairformerLayer` calls
   (which also pins generic_op's remaining 224 calls), `PairConditioningDevice`,
   `PairAssemblyDevice`, `ConfidenceHeadsDevice`, and the sampler body `DiffusionModule` runs
   outside `Diffusion` (1.2575 s of host-inclusive wall over 200 calls).
2. The python-class split. `align_quality` is 0.3187 / 0.3780, so the per-census-class table is not
   quotable and everything above is grouped by device op instead. The recorded call sequence misses
   shapes for `generic_op` (its tensors arrive in a list) and for every site that passes operands as
   keyword arguments. Fixing the recorder is a small change and it is what turns the group table
   into the per-class table the brief asks for.
3. The 14.881 s denominator on a quiet box. Everything above divides by the campaign's figure, not
   by a fold this row measured; this row's own fold read 17.5213 s co-tenanted.
