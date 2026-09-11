# The Boltz-2 512 aa fold is not dispatch-bound. It is 81.7 % device, and the device deficit is inside the ops.

`ws:b2x-op-cost-curve`, P9 of `ws:boltz2-2x-orchestrator`. One card: **qb2 physical card 0, one
Blackhole processor of a p300c, 11x10 compute grid, ttnn 0.68.0, commit 6f150345**. Every timed
run under benchlock on an idle box (loadavg 0.00-0.18 at acquisition). No model code changed.

## The short version

| question | answer |
|---|---|
| is the fold dispatch-bound? | **No.** 19.362 s of a 23.710 s fold is device time, measured by replaying each phase from a ttnn trace with the host removed. |
| why did it look dispatch-bound? | tt-metal **burns the calling thread's CPU while it waits for dispatch-queue room**. `time.thread_time()` counts that wait as host work. An identical 2048-cube matmul reads 6.2 us of "host CPU" per call when 1000 are in flight and **144.7 us** when 12 000 are. |
| how much of the 21.846 s of main-thread CPU can a dispatch lever reach? | **1.136 s, 5 %.** A perfect dispatch lever is 1.050x. Delete every byte of serial host work as well and it is 1.225x. |
| is `bioir-dispatch-graph` wrong? | No. Its 1.112x ceiling was right and it generalises: 1.05x. Its 32.526 ms/step device figure reproduces here at **32.518 ms**, on a different card class and a different ttnn. |
| so where is the fold? | 12.527 s trunk device, 6.504 s sampler device, 0.331 s confidence device, 3.212 s serial host. |
| is op count the device-side lever? | **No.** The per-op device fixed cost is **6.36 us**, and 428 of them is 2.7 ms of a 41.4 ms pairformer block. |
| what is left, then? | The block costs **41.415 ms** against a cost model of **17.7 ms** from its bytes at the measured roof plus its op count at the measured fixed cost. **23.7 ms/block, 6.3 s of the fold, is neither.** Per sub-unit the shortfall is 1.6-1.9x on the three byte-heavy ones and **3.21x** on the pair-track transition. That is the campaign's remaining quantity. |

## 1. Per-phase host CPU and per-phase device floor

`perf/b2x_op_cost/host_phase_cost.py` brackets every `tt_bio.tenstorrent` module for wall and
main-thread CPU with the device synced only at the top of the tree, so a phase's wall is
comparable with a plain fold's. `device_floor.py` then captures one settled call of
`PairformerLayer`, `MSALayer` and the diffusion step `Diffusion` as a ttnn trace and replays it
back-to-back: the host pays 0.96-1.17 us per replay, so what is left is device time.

Plain benchlocked folds: **23.707 s** and **23.625 s** (A/A floor 0.082 s, 0.35 %), main-thread
CPU 18.897 / 18.801 s. The bracketed fold is 23.710 s, so the instrument costs nothing.

| phase | wall s | main-thread CPU s | CPU % of wall | device floor s | % device | exposed host s |
|---|---|---|---|---|---|---|
| TrunkModule | 12.882 | 12.653 | 98.2 | **12.527** | 97.2 | 0.355 |
| DiffusionModule, 200 steps | 7.151 | 2.859 | 40.0 | **6.504** | 90.9 | 0.647 |
| PairformerModule, confidence | 0.465 | 0.242 | 52.2 | **0.331** | 71.3 | 0.134 |
| residual, outside every class | 3.212 | 3.205 | 99.8 | 0 | 0 | **3.212** |
| **fold** | **23.710** | 18.960 | 80.0 | **19.362** | **81.7** | **4.348** |

Device floors, each a median of 5 bursts of 16 replays:

| region | device ms/call | calls/fold | device s/fold | replay host cost |
|---|---|---|---|---|
| PairformerLayer | **41.4152** | 280 (256 trunk + 16 in MSALayer + 8 confidence) | 11.596 | 0.96 us/call |
| MSALayer (includes its nested PairformerLayer) | **120.2894** | 16 | 1.925 | 1.17 us/call |
| Diffusion, one sampling step | **32.5179** | 200 | 6.504 | 1.06 us/call |

Trunk floor = 256 x 41.4152 + 16 x 120.2894 = 12.527 s against a 12.882 s wall. The 16
PairformerLayer calls nested inside MSALayer are inside the MSALayer figure and are not counted
twice; the census confirms the split, 433 ttnn calls per trunk block against 378 per MSA-nested
block.

Two independent checks on the floors. `MSALayer` at 120.289 ms sits against
`b2x-baseline-attrib`'s synced 122.781 ms/call, 2.0 % apart. `Diffusion` at 32.518 ms against
`bioir-dispatch-graph`'s 32.526 ms on qb1 card 3 (p150a, ttnn 0.67.4) is **0.03 % apart** on a
different card class, a different ttnn and a different harness.

**What a dispatch lever can reach.** 1.136 s inside the three device phases (0.355 + 0.647 +
0.134): 23.710 -> 22.574 s, **1.050x**. The other 3.212 s of exposed host is the residual, which
issues **79 ttnn calls in the whole fold** and moves no device bytes: the sampler's between-step
EDM arithmetic (2.74 s, on record at `d0183cc0`), featurisation and the CIF write. It is serial
host time, not dispatch, and no op-count change touches it. Both together: 19.362 s, **1.225x**.

## 2. Why the fold looked dispatch-bound

`b2x-baseline-attrib` measured 21.846 s of main-thread CPU in a 26.037 s fold and concluded the
fold is dispatch-bound. The measurement reproduces (18.960 s in 23.710 s on a quiet box, 80.0 %).
The conclusion does not, because **`thread_time()` on this stack counts dispatch-queue wait as
host CPU**.

The control the first pass needed. Issue back-to-back 2048-cube bf16 matmuls, one identical op,
and vary only how many:

| reps | apparent host us/call | device us/call | CPU / issue wall | drain s |
|---|---|---|---|---|
| 100 | **6.48** | 152.13 | 0.998 | 0.015 |
| 400 | **6.32** | 156.62 | 1.000 | 0.060 |
| 1 000 | **6.22** | 163.96 | 1.000 | 0.158 |
| 2 000 | **37.53** | 166.07 | 1.000 | 0.257 |
| 5 000 | **114.67** | 166.10 | 1.000 | 0.257 |
| 12 000 | **144.70** | 166.11 | 1.000 | 0.257 |

The op never changes. Its real host issue cost is 6.2-6.5 us, visible while the queue has room.
The drain saturates at 0.257 s, so the queue holds about 1 550 of these matmuls; past that the
calling thread waits, and it waits by burning CPU, so the apparent host cost climbs 23x. The
first pass's control used 400 reps of a 4096-cube matmul: issue wall 0.003 s, drain 0.375 s. It
never filled the queue and so proved only that a push is cheap.

The same thing on the code in question. One real pairformer block, two ways:

| how it is issued | apparent host CPU ms/block | wall ms/block |
|---|---|---|
| device synced after every call (queue empty at entry) | **4.231** | 41.494 |
| 16 back-to-back, no syncs (queue fills) | **28.018** | 41.300 |
| inside the fold, 256 back-to-back (queue permanently full) | **~41.7** | 41.4 |

Same ops, same device work, same wall. The apparent host cost moves 10x with nothing but queue
occupancy. And injecting 10 us of real host CPU into every ttnn call moved the trunk's wall by
**-0.014 s** and its CPU by -0.016 s: the trunk absorbed 1.3 s of injected work without noticing,
because it was already spending that time waiting.

`b2x-baseline-attrib` refuted spin-on-a-full-queue three ways. Each of the three misses:

* *"the device is at 37 % of one roof and 10 % of the other, so there is nothing to back up
  against."* Roof fraction is not occupancy. What fills a queue is device **time**, not device
  **efficiency**; the matmul above backs up while running at whatever fraction of roof it runs at.
* *"`ttnn.deallocate`, host-only, costs 0.6 us across 138 213 calls."* `deallocate` enqueues no
  device work, so it never waits for queue room. This census reads it at 0.81 us. It measures the
  empty-queue path and says nothing about the full one.
* *"the expensive calls are the ones that build a program."* Those are exactly the calls with the
  most device work behind them, so they are where the host catches up. This census reproduces the
  pattern (`ttnn.linear` 62.36 us/call against their 64 us) and the queue control shows one
  unchanging matmul reading 6.2 us or 144.7 us depending only on how many are in flight.

The two campaign figures were never in conflict. 88.3 % device-busy and 80-84 % main-thread CPU
are both right and sum past 100 % because they overlap. What was wrong was reading the second as
host work.

## 3. The op-cost curve, which is the question this task was opened for

With the host removed, a per-op device cost curve is finally measurable: each point is a ttnn
trace holding R repetitions of one op, captured once and replayed, so the host pays ~1 us for the
whole trace. `ttnn.add` on two tiled bf16 DRAM operands into a preallocated output, 3 x size
moved, 11x10 grid, median of 5 replays, 12 sizes from 64 KB to 134 MB per operand.

| MB/operand | MB moved | us/op | GB/s | % of 429.9 GB/s |
|---|---|---|---|---|
| 0.066 | 0.197 | **6.360** | 30.9 | 7.2 |
| 0.131 | 0.393 | **6.365** | 61.8 | 14.4 |
| 0.262 | 0.786 | **6.380** | 123.3 | 28.7 |
| 0.524 | 1.573 | **6.465** | 243.3 | 56.6 |
| 1.049 | 3.146 | 9.933 | 316.7 | 73.7 |
| 2.097 | 6.292 | 15.704 | 400.6 | 93.2 |
| 4.194 | 12.583 | 30.108 | 417.9 | 97.2 |
| 8.389 | 25.166 | 58.041 | 433.6 | 100.9 |
| 16.777 | 50.332 | 114.694 | 438.8 | 102.1 |
| 33.554 | 100.663 | 226.463 | 444.5 | 103.4 |
| 67.109 | 201.327 | 452.439 | **445.0** | 103.5 |
| 134.218 | 402.653 | 908.724 | 443.1 | 103.1 |

Three numbers, read off directly rather than fitted:

* **t_fixed = 6.36 us.** The first four points are flat to 1.6 % across an 8x range of bytes. An
  op that moves 0.2 MB and an op that moves 1.6 MB cost the same. That flat floor is the per-op
  device cost, and it needs no least-squares: below ~2.8 MB moved, size does not matter.
* **BW_eff = 445.0 GB/s**, the asymptote, reached at 25 MB moved and held to 134 MB. Note it is
  **3.5 % above the 429.9 GB/s "streaming roof"** the whole campaign prices against, so that roof
  is slightly low and every "% of roof" in the campaign is correspondingly flattered.
* **the knee is at 2.1 MB/operand, 6.3 MB moved**, where the curve first reaches 90 % of the
  asymptote. Consistent with the crossover the two parameters predict: 6.36 us x 445 GB/s =
  2.83 MB moved.

A two-parameter fit is reported in the JSON for completeness and is worse: over sizes <= 8 MB it
gives t_fixed 4.666 us and BW_eff 513 GB/s, above the measured asymptote, because a narrow fit
window trades the intercept against the slope. Use 6.36 us and 445 GB/s.

**The original brief's falsifier fires.** It said: if the curve at 15.5 MB is above 85 % of the
big-stream roof, op size is not the issue and the pairformer's 37 % is caused by something inside
the block. At 16.8 MB/operand the curve is at **102.1 %** of the 429.9 roof and **98.6 %** of the
measured 445. It also said: if `t_fixed` comes out under 2 us, op count is not worth attacking.
6.36 us is not under 2, but 428 of them is **2.72 ms of a 41.415 ms block, 6.6 %**, so the
practical conclusion is the same. Op count is not the lever, at either end.

## 4. The cost model against a real block, and the 23.7 ms it does not explain

The check the brief asked for. One pairformer block: 428 ops, 6.651 GB (the corrected
buffer-address count from `b2x-diffusion-layer-bytes`), measured device time 41.4152 ms.

    bytes at the measured roof   6.651 GB / 445.0 GB/s   = 14.95 ms
    op count at the measured floor   428 x 6.36 us       =  2.72 ms
    model (upper bound: the two do not add for large ops) = 17.67 ms
    measured                                              = 41.42 ms
    residual                                              = 23.75 ms, 57 %

The model explains **42.7 %** of the block. The same instrument on each sub-unit, capturing and
replaying one settled call of the shipped `TriangleMultiplication`, `TriangleAttention`,
`Transition` (the pair-track one, selected by its input volume) and `AttentionPairBias`
(`subunit_floor_512_qb2c0.json`), against the corrected per-sub-unit byte counts from
`b2x-baseline-attrib`'s A1 table:

| sub-unit | calls/block | device ms/call | device ms/block | MB/block | ops/block | model ms | measured / model |
|---|---|---|---|---|---|---|---|
| TriangleMultiplication | 2 | 9.3975 | 18.795 | 4229.4 | 62 | 9.89 | **1.90x** |
| TriangleAttention | 2 | 4.3910 | 8.782 | 2231.8 | 54 | 5.35 | **1.64x** |
| Transition, pair track | 1 | 8.7050 | 8.705 | 474.2 | 258 | 2.71 | **3.21x** |
| AttentionPairBias | 1 | 1.0160 | 1.016 | 151.4 | 32 | 0.54 | **1.88x** |
| everything else in the block | | | 4.117 | | | | |
| **block** | | | **41.415** | 6651 | 428 | **17.67** | **2.34x** |

The four sub-units plus 4.117 ms of block-level layer norms, adds and the single-track transition
sum to the independently measured 41.4152 ms, so the decomposition closes on one instrument and
one card.

Read it two ways. The three byte-heavy sub-units sit at **1.6-1.9x** their model, tightly
clustered, which is a per-op efficiency factor and not an op-count or op-size effect. The
pair-track transition sits at **3.21x** and is the outlier, carrying 21 % of the block's time for
7 % of its bytes. `b2x-baseline-attrib` pointed at that sub-unit and it was right to; the
mechanism is different, because its 258 ops at the measured 6.36 us floor are **1.64 ms of its
8.705 ms, 19 %**. Halving its op count is worth at most 0.8 ms/block, 0.22 s/fold. The other
6.0 ms/block is neither its bytes, nor its op count, nor its host.

One caution on that JSON: `Transition` and `AttentionPairBias` each serve several shapes in this
model (single track, MSA, diffusion), so the per-fold products the script prints for those two
rows, 960 and 6264 calls, are meaningless. Only the per-call figure is, and only at the shape it
was captured on. `TriangleMultiplication` and `TriangleAttention` are 2 per pairformer block and
560 per fold exactly, so those products hold.

Two candidates survive, and they are distinguishable:

1. **The counted bytes are a lower bound.** The corrected counter charges each consuming op its
   operands once. A `ttnn.matmul` that blocks over K re-reads an operand once per block, and a
   fused `generic_op` that spills re-reads too. If real DRAM traffic is ~2.2x the count, the block
   is at ~90 % of the bandwidth roof and **the bandwidth axis is closed, not open at 2.4x**. This
   is the possibility the campaign most needs to rule in or out, because bulletin 2 reopened that
   axis on the 40.9 % figure.
2. **Per-core inefficiency at the op's own working-set size.** 6.651 GB over 110 cores is 60 MB
   per core per block; the deficit would then be L1 blocking, bank conflicts and compute that does
   not overlap the loads. `ttnn.add` is the wrong probe for that: it is one pass, perfectly
   coalesced, no reuse.

**The experiment that separates them** is the curve of section 3 re-run on the ops the block
actually uses: a trimul-shaped `ttnn.matmul` and a `ttnn.generic_op` triangle kernel, in
isolation, with the same trace-replay instrument. If an isolated matmul at the block's shapes also
lands at ~45 % of its byte model, the deficit is intrinsic to the op and candidate 1 is the
answer; if it lands near 100 %, the deficit is created by the block's op sequence and candidate 2
is. One card, one device open, no fold needed.

## 5. What this does to the campaign

* **The 17:58 repoint was based on an artifact.** "The fold is dispatch-bound" is not true; op
  count reduction as a dispatch lever is worth 1.05x, and `bioir-dispatch-graph`'s 1.112x ceiling
  was right all along. `ttnn.linear` at 109 887 calls and 7.025 s of apparent host CPU is not
  7 s of recoverable time; it is mostly the trunk waiting for its own triangle kernels.
* **The trunk is not host-bound, it is 97.2 % device at about half its cost model.** That is where
  the fold is: 12.527 s of the 23.710 s.
* **`b2x-baseline-attrib`'s pair-transition lever survives with a different mechanism.** Its
  7.993 ms/block at 13.8 % of roof is device time, not dispatch, so bringing it to the trimul's
  efficiency is still worth ~1.5 s of the fold. It has to be achieved by making each op efficient,
  not merely by issuing fewer of them, because 258 ops at 6.36 us is 1.6 ms of the 7.993.
* **The 429.9 GB/s roof is 3.5 % low.** Use 445.0 GB/s.
* **Nothing here is release-gated.** No model code changed; the only new files are under
  `perf/b2x_op_cost/`.

## 6. Files and how to reproduce

* `host_phase_cost.py` -> `host_phase_512_qb2c0.json`, `TABLES.md`. Per-phase wall and
  main-thread CPU, the per-(phase, op) census, the host-CPU injection slope, the pairformer block
  replay. One device open, benchlock.
* `device_floor.py` -> `device_floor_512_qb2c0.json`. The queue control, the block two ways, the
  three per-phase device floors.
* `op_cost_curve.py` -> `op_cost_curve_512_qb2c0.json`. The size curve, the fits, the block model.
* `device_floor.py --phases grab,floor --grab TriangleMultiplication,TriangleAttention,Transition,AttentionPairBias`
  -> `subunit_floor_512_qb2c0.json`. The per-sub-unit device floors of section 4.
* `phase_table.py <json> [out.md]` regenerates `TABLES.md`.

Each is `TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:b2x-op-cost-curve
benchlock.sh b2x-op-cost-curve -- python perf/b2x_op_cost/<script>.py --out <json>` from the
repo root with `PYTHONPATH` at the checkout. `ttnn.clone` has no `output_tensor` parameter, so the
curve's copy arm is `add` only; the error is recorded in the JSON.

## PARITY

Bit-exact, and measured rather than asserted. No model code changed. Both fold runs wrote

    4f3995a69be5d6106f2b6a2aca57b29f8bb84810bf751e39a0258956b09d15e5  cdk2x2_512.cif

which is byte-identical to `b2x-baseline-attrib`'s current-main reference `4f3995a69be5d610`,
taken on card 1. So this task's card, its 1 GiB trace region and its instruments change nothing
about the structure. No `cdk2x2_298` control is owed: a bit-identical arm has nothing to control
against.

The microbenchmarks are `ttnn.add` and `ttnn.matmul` on random tensors. The three trace-replay
legs capture and replay **unmodified shipped** `PairformerLayer.__call__`, `MSALayer.__call__`
and `Diffusion.__call__` with their own arguments, cloned from a live fold. Every fold in every
run is the published protocol (`cdk2x2_512.yaml` + its 35-row a3m, 3 recycles, 200 sampling
steps, 1 sample, seed 0, templates off) through the production `_WorkerState.predict_one`.

One deliberate perturbation, stated: every run opens the device with a 1 GiB ttnn trace region,
which the trace-replay legs need. It costs nothing measurable. The plain benchlocked folds read
23.707 and 23.625 s against `b2x-baseline-attrib`'s 23.841 s median without the region, a 0.7 %
difference against an A/A floor of 0.35 % and a cross-card offset of the same order.
