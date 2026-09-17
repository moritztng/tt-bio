# The fold's unit census is exact. Its per-op device time needs the whole board, and the board is split.

`ws:c12-profiled-fold`, pass 1, 2026-09-17. qb2 card 3 (board 410D), `eab845ad1`, shipped
`ttnn==0.68.0` wheel, AICLK forced to 1350 MHz through tt-kmd's ARC queue and sampled at 1 kHz
during every timed leg. No production code changed.

## VERDICT: PARTIAL. The denominator is measured; the per-op split is blocked on a card, not on a method.

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
