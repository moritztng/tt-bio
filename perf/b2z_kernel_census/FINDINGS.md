# The pairformer block is inside a kernel 98.9 % of its own wall. The gap is 1.1 %.

`ws:b2z-kernel-cycle-census`, pass 1. One card: **qb2 physical card 0, one Blackhole processor of
a p300c, 11x10 grid**, tt-metal built from source at tag **v0.68.0** (`1452925b`, the commit the
shipped `ttnn==0.68.0` wheel is cut from) with `ENABLE_TRACY=ON`, at
`/home/ttuser/tt-metal-b2z`. No model code changed.

## GAP-FRACTION: 1.1 %

One `PairformerLayer` at 512 aa, three profiled repetitions fenced inside one process:

| | ms | % of span |
|---|---|---|
| span, first core start to last core finish | **36.2456** | 100 |
| inside a kernel | **35.8397** | **98.9** |
| in no kernel at all (dispatch, launch, barriers) | **0.4059** | **1.1** |

The block dispatches **272 device programs** (428 is the ttnn *call* count; the rest are views,
deallocates and other calls that dispatch nothing). The median inter-op gap is **0.5 us** and the
whole-block total is 0.41 ms. The profiler's own `OP TO OP LATENCY` column sums to 0.149 ms, so
both readings of (c) agree that it is a rounding error on this block.

**The 6.36 us/op `t_fixed` from `b2x-op-cost-curve` is not a device-side inter-op gap.** It is
host-side issue cost, and under back-to-back dispatch the device never sees it. Any lever whose
mechanism is "fewer, larger programs" — fusion for dispatch-count reasons, megakernels, trace
replay — has **0.41 ms per block, 1.1 %,** to win. That is the campaign's fewer-and-larger-ops
axis, priced.

## The instrument, proved before it was trusted

`ttnn.matmul` 2048^3 bf16 HiFi2 on the same card, 20 reps:

| | ms |
|---|---|
| bare back-to-back wall, no profiler | 0.1537 |
| profiled back-to-back wall | 0.1610 |
| profiler `DEVICE KERNEL DURATION`, median | **0.1475** |
| profiler `DEVICE FW DURATION`, median | 0.1497 |

Reported kernel time is **4.1 %** under the bare back-to-back wall and FW duration is **2.6 %**
under it, both inside the 10 % bar. The same check on the real block: profiled synced wall
36.4994 ms against a device span of 36.2456 ms, **0.7 % apart**.

Profiling cost the 2048-cube matmul **+4.7 %** of wall and the block **+0.7 %**. Small, and the
census is a ranking, not an A/B.

## Baseline correction: this block is 36.50 ms, not 41.4152 ms

`b2x-op-cost-curve` measured 41.4152 ms/call at commit `6f150345`. That was before
`b2x-integrate` merged the two default-on Boltz-2 diffusion levers and
`TT_BIO_TRIMUL_MASK_AFTER_MOVE`. On current main the same grabbed block reads **36.4994 ms**
synced wall, **1.135x** faster. Anyone still pricing a pairformer lever off 41.4152 ms is
pricing it against a block that no longer exists. At 280 calls/fold the block is 10.22 s, not
11.60 s.

## Where the 36.2 ms is

| op code | programs | kernel ms | % of span | TRISC1 (math thread) ms | dataflow ms |
|---|---|---|---|---|---|
| GenericOpDeviceOperation | 16 | **13.2973** | **36.7** | 13.1108 | 13.2973 |
| MatmulDeviceOperation | 113 | **9.2118** | **25.4** | 8.3509 | 9.2106 |
| BinaryNgDeviceOperation | 52 | 5.8538 | 16.2 | 5.8176 | 5.8537 |
| LayerNormDeviceOperation | 41 | 3.6511 | 10.1 | 3.6183 | 3.6508 |
| TransposeDeviceOperation | 8 | 2.1663 | 6.0 | 0.6953 | 2.1663 |
| ReshapeViewDeviceOperation | 3 | 0.5112 | 1.4 | 0 | 0.5112 |
| SliceDeviceOperation | 33 | 0.3885 | 1.1 | 0 | 0.3885 |
| ConcatDeviceOperation | 1 | 0.3476 | 1.0 | 0 | 0.3476 |
| PermuteDeviceOperation | 3 | 0.2727 | 0.8 | 0.2532 | 0.2727 |
| SoftmaxDeviceOperation | 1 | 0.1173 | 0.3 | 0.1161 | 0.1173 |
| NlpCreateHeadsDeviceOperation | 1 | 0.0221 | 0.1 | 0 | 0.0221 |

Sixteen `GenericOpDeviceOperation` programs — the fused trimul and SDPA kernels — are **36.7 %**
of the block on their own, the ten largest of them 0.86-1.31 ms each. 113 matmuls are another
25.4 %.

Per-RISC residency over the span: BRISC 98.7 %, TRISC2 88.3 %, TRISC1 88.2 %, TRISC0 87.4 %,
NCRISC 84.4 %. Core launch skew is **1.2-1.9 us** per op, so cores start together and the span is
not core skew either.

Grid occupancy, weighted by kernel time, is **103.8 of 110 cores**. 163 of the 272 programs run
on the full 110, 96 run on 86, and only 13 run on 16-80 cores. **20.5 % of the span** is on a
partial grid, the biggest single piece being the 96 programs on 86 cores.

## What this does to the campaign

The 23.7 ms the campaign is hunting is not in the gaps between ops and it is not core skew and it
is not partial-grid idling beyond 20 %. It is **inside the kernels**, with all five RISCs resident
and the math thread present for 88 % of the span. Which of "retiring math" and "stalled on data"
that 88 % is, is the one question this pass has not yet answered, and it is the question that
picks the campaign's next lever.

## Owed, next pass

1. **(a) vs (b).** TRISC1 resident is not TRISC1 busy — an `ttnn.add` keeps the math thread
   resident for 226 of 227 us while doing nothing. Split it with the FPU/INSTRN hardware counters
   (`--profiler-capture-perf-counters=fpu,instrn`, budget ~21x the marker count) and with FLOPs
   per op derived from shapes. The report CSV currently has empty `INPUT_*` shape columns even
   under `--no-op-info-cache`; find the column that carries them or read shapes from the harness.
2. **The diffusion step.** `--phase step` is written and untested: its precursor needs the trunk,
   so it truncates the pairformer stacks and runs recycling 1 before grabbing.
3. **The CIF sha256** of a fold run against the profiler-enabled build. Not run this pass.

---

# The diffusion step is the opposite block: 1066 tiny programs and the device starves between them

Same card, same build, same harness, `--phase step`. The precursor for a settled `Diffusion` call
needs the trunk to have run, so it runs at recycling 1 with the pairformer stacks truncated to two
blocks and is aborted at the grab. Shapes are untouched; only the values entering the step differ.

| | ms | % of span |
|---|---|---|
| span, eager dispatch | 45.5048 | 100 |
| inside a kernel | 22.0152 | 48.4 |
| in no kernel at all | 23.4895 | **51.6** |

1066 dispatched programs, mean kernel time **20.7 us**, mean gap **22.1 us**. Per-RISC residency is
half what the pairformer block shows: BRISC 48.3 %, NCRISC 35.6 %, TRISC1 33.4 %. Grid occupancy
weighted by kernel time is **83.3 of 110 cores**, and 124 programs run on 16 cores.

By op code: `Matmul` 387 programs = 9.07 ms = 19.9 %; `BinaryNg` 339 = 3.59 ms = 7.9 %;
`SDPAOperation` 30 = 2.50 ms = 5.5 %; `LayerNorm` 114 = 1.74 ms = 3.8 %; `Permute` 12 = 1.47 ms =
3.2 %; `NlpCreateHeads` 30 = 1.47 ms = 3.2 %; `ReshapeView` 36 = 1.36 ms = 3.0 %.

## Read this gap fraction carefully — it is an eager number, not a traced one

The census dispatches eagerly, so its "gap" is device idle from **any** cause, host dispatch
included. That did not matter for the pairformer block: its ops average 133 us, the host stays far
ahead, and the eager span is 0.7 % under the profiled wall. It matters here. The eager span is
**45.5 ms against the 32.5179 ms trace-replay floor** `b2x-op-cost-curve` measured for this same
step, so **about 13 ms of the 23.5 ms gap is host dispatch that trace replay already removes** —
and `bioir-dispatch-graph` already priced trace on the fold at 1.013x, so that 13 ms is not money
on the table, it is money already counted and rejected.

What survives is an upper bound: under trace, the diffusion step's device-side gap is **at most
~10.5 ms of a 32.5 ms step, ~32 %**, and possibly much less. Pinning it needs
`--device-trace-profiler` against the trace-replay harness, which is the first thing the next pass
should do. Until then the honest statement is:

* **PairformerLayer, GAP-FRACTION 1.1 %** — solid. Eager already equals the span, so the traced
  figure can only be smaller. Fewer-and-larger-programs is worth 1.1 % here, full stop.
* **Diffusion step, GAP-FRACTION 51.6 % eager, <=32 % traced** — a real result with a bound on it,
  not a measured traced number. Do not quote 51.6 % as a device figure.

The mechanism is the same one either way: the diffusion step's programs are **6.4x smaller** than
the pairformer block's (20.7 us against 133 us) and run on **83 of 110 cores** against 103.8. Two
blocks of the same fold sit on opposite sides of the dispatch knee, which is why one number for
"the fold" was never going to fit both.

---

# The corrected sub-unit floor: four classes, 10.161 s/fold, not 22.443 s

Run on qb2 card 0 at 512 aa on the **wheel** stack (ttnn 0.68.0, not the source build), because that
is the stack the table it corrects was taken on. `perf/b2x_op_cost/device_floor.py` with the
orchestrator's fix, commit `a0735332`, output `perf/b2z_kernel_census/subunit_floor_corrected_512_qb2c0.json`
and `subunit_apb_512_qb2c0.json`. The b2x artifact is untouched.

| class | shape priced | ms/call | calls priced | of all calls | s/fold | b2x s/fold | delta |
|---|---|---|---|---|---|---|---|
| TriangleMultiplication | 1x512x512x128 | 6.8577 | 560 | 560 | **3.8403** | 5.2626 | −1.42 |
| TriangleAttention | 1x512x512x128 | 4.3797 | 560 | 560 | **2.4526** | 2.4590 | −0.01 |
| Transition | 1x512x512x128 | 8.7075 | 296 | 960 | **2.5774** | 8.3568 | **−5.78** |
| AttentionPairBias | 1x512x768 | 0.2688 | 4800 | 6264 | **1.2902** | 6.3642 | **−5.07** |
| | | | | | **10.161** | **22.443** | **−12.28** |

Against containers worth 20.025 s, so the corrected four fit, with 9.86 s of container time in glue
these four classes do not name. The amendment's check passes.

The 12.28 s has **two different causes** and they must not be merged:

* **10.85 s is the counting bug**, Transition and AttentionPairBias.
* **1.42 s is main moving forward.** TriangleMultiplication is 9.3975 ms/call at commit `072da10f`
  and **6.8577 ms/call** at `a0735332`, same shape, same 560 calls, **1.37x**. TriangleAttention
  did not move (4.391 → 4.3797). This is the same effect that took the pairformer block from
  41.4152 ms to 36.4994 ms, and it is now visible in two independent places.

## The shape histogram, which is the evidence

| class | first-arg shape | calls/fold |
|---|---|---|
| TriangleMultiplication | 1x512x512x128 | 560 |
| TriangleAttention | 1x512x512x128 | 560 |
| Transition | 1x512x768 | 400 |
| | 1x512x512x128 | 280 |
| | 1x512x384 | 264 |
| | 1x1024x512x64 | 16 |
| AttentionPairBias | 1x512x768 | 4800 |
| | 1x140x32x128 | 1200 |
| | 1x512x384 | 264 |

TriangleMultiplication and TriangleAttention are single-shape, so they were never at risk. The other
two are not, and the histogram shows exactly how badly: only **280 of 960** Transitions and **0 of
6264** AttentionPairBias calls are on the pair tensor.

## The fix as shipped dropped AttentionPairBias instead of correcting it

The amendment's `device_floor.py` filters AttentionPairBias with the same `_vol(args[0]) > 4_000_000`
predicate it uses for Transition. **AttentionPairBias never takes the pair tensor as its first
argument** — z enters as a bias. Its widest first arg at 512 aa is 1x512x768 = 393 216 elements,
an order of magnitude under the threshold, so the predicate matches nothing, the class is never
grabbed, and it vanishes from the table with no error. Silent omission, not a wrong number.

Fixed here by matching the token track on rank and channel width (3-D first arg, last dim > 512),
which selects the 4800 diffusion-transformer calls. That call is **0.2688 ms**, not the 1.016 ms the
b2x table used — the b2x grab landed on one of the 264 *pairformer* calls at 1x512x384, which is
**3.8x more expensive per call** than the diffusion one because it carries the full 512x512x128 pair
bias while the diffusion transformer's bias is precomputed once per step. So the old row was wrong
in both factors at once, cost and count, and they compounded in the same direction.

Two shapes are still unpriced and the total above excludes them: 664 Transitions (400 at 1x512x768,
264 at 1x512x384) and 1464 AttentionPairBias (1200 at 1x140x32x128, 264 at 1x512x384). 16 of the 296
Transitions priced are 1x1024x512x64, the same element count as the pair shape but a different
layout, so 0.139 s of the Transition row is an upper bound rather than a measurement.

## The census does not have this defect, and here is the proof

The amendment asks whether the kernel census inherits the same inflation. It cannot: it reads one
row per **dispatched device program** out of the profiler and prices each at its own measured
cycles. Nothing is a representative times a count. The spread inside a single op code shows how far
wrong the representative approach would have been here:

| op code | programs | min µs | median µs | max µs | max/min |
|---|---|---|---|---|---|
| TransposeDeviceOperation | 8 | 2.9 | 174.3 | 730.5 | **250x** |
| MatmulDeviceOperation | 113 | 7.5 | 35.7 | 816.8 | **109x** |
| BinaryNgDeviceOperation | 52 | 4.8 | 26.0 | 509.5 | 106x |
| LayerNormDeviceOperation | 41 | 11.3 | 23.4 | 541.1 | 48x |
| GenericOpDeviceOperation | 16 | 409.4 | 913.2 | 1311.4 | 3.2x |

A per-class representative cost in this block is not off by tens of percent. It is off by up to
**250x within one op code**, in either direction depending on which call the grab happened to land
on.

Containment check, which is the other thing that makes the corrected table believable: one
PairformerLayer holds 2 TriangleMultiplication + 2 TriangleAttention + 1 pair Transition + 1
AttentionPairBias = 2(6.8577) + 2(4.3797) + 8.7075 + ~1.0 = **32.2 ms** of the block's measured
**36.4994 ms**, 88 %. The old numbers do not fit in the block at all.

---

# (a) vs (b): the math thread is starved of input, 6 parts in 7

`--enable-sum-profiling` turns on two compute-thread stall accumulators the default profiler leaves
blank: `DEVICE COMPUTE CB WAIT FRONT` (math thread blocked on input tiles) and `CB RESERVE BACK`
(blocked on room to write output). That measures (b) directly instead of inferring it from a roof.
Same block, same card, third independent capture: GAP-FRACTION **0.9 %** against 1.1 %, synced wall
36.8693 ms against 36.4994, so the census repeats.

Per-core-averaged compute-thread budget over the 36.3438 ms block:

| | ms | % of span | % of TRISC1 |
|---|---|---|---|
| TRISC1 resident | 32.1443 | 88.4 | 100 |
| CB wait-front, blocked on input | **18.3366** | 50.5 | **57.0** |
| CB reserve-back, blocked on output | 3.1066 | 8.5 | 9.7 |
| not stalled on a CB | **10.7010** | 29.4 | **33.3** |

**The math thread is resident for 88 % of the block and stalled on a circular buffer for two thirds
of that.** Input stalls beat output stalls **5.9:1**, so the compute engine is starved, not backed
up. That is the answer the pass was for: the lever is feeding the math engine (layout, sharding,
CB depth, data reuse, NoC), not more FLOPs and not fewer programs.

**The error bar, stated.** The accumulators are summed over cores and the report's `CORE COUNT` is
the only available divisor, but that normalisation is not exact: across 232 ops with a compute
kernel the ratio `(cb_wait + cb_reserve) / cores / TRISC1` has median **0.686** and **50 of 232 ops
exceed 1.0** (max 1.97), which is impossible and means the divisor is wrong for those ops. Two op
codes come out with negative "busy" time on this normalisation (BinaryNg −8.7 %, Transpose −9.4 %)
and those two readings should be taken as "essentially all stall", not as numbers. So: the
**direction is unambiguous and the ordering is safe; the 57 / 9.7 / 33.3 split is ±, not exact.**
Nailing it needs the FPU/INSTRN hardware counters, which cost ~21x the marker budget.

Per op code, per-core-averaged (ms per block):

| op code | n | kernel | TRISC1 | CB wait | CB reserve | not stalled |
|---|---|---|---|---|---|---|
| GenericOp (fused trimul/TriAtt) | 16 | 13.4429 | 13.2567 | 7.4981 | 0.4879 | 39.8 % |
| Matmul | 113 | 9.2213 | 8.3761 | 3.3172 | 0.2248 | 57.7 % |
| BinaryNg | 52 | 5.8944 | 5.8583 | 4.5969 | 1.7685 | ~0 |
| LayerNorm | 41 | 3.6217 | 3.5884 | 2.1667 | 0.1220 | 36.2 % |
| Transpose | 8 | 2.1650 | 0.6949 | 0.4737 | 0.2864 | ~0 |

`ReshapeView`, `Slice` and `Concat` emit no TRISC1 row at all: pure data movement, 1.25 ms/block.

## Getting the profiler to emit these at all

`--enable-sum-profiling` drops to tt-metal's legacy Python post-processor, and on pandas 3 that path
dies with `TypeError: Invalid value for dtype 'str'` at `tools/tracy/process_device_log.py:143` — it
writes an int column in place over a pyarrow-backed string column. Two `df.iloc[:, n] = ...` lines
become `df.isetitem(n, ...)`, which replaces the column wholesale and is allowed to change dtype.
Patched in the local build at `/home/ttuser/tt-metal-b2z`; worth upstreaming.
`--process-logs-only` is not a way around it: it appends `.logs` to a path that already ends in
`.logs` and then fails looking for a capture file that was never there.

# Top 5 offenders by deficit seconds per fold

Kernel time per block times calls per fold, all of it category **(b)** — nothing in this block is
(c) and nothing is compute-bound.

| # | offender | s/fold | (a)(b)(c) | what it actually is | who should own it |
|---|---|---|---|---|---|
| 1 | GenericOp, fused trimul + TriAtt, 16/block | **3.76** | (b) | 60 % of its math-thread time is an input-side CB stall. The fused kernel is already the biggest win in the block and it is still starved. | the trimul/TriAtt kernel workstream — CB depth and operand staging inside the existing kernel, not a new fusion |
| 2 | Matmul, 113/block | **2.58** | (b) | 42 % input stall; per-program cost spans 7.5-816.8 µs, so this is 113 different problems | `b2z-op-knob-sweep` — per-matmul memory config and sharding, priced per program not per class |
| 3 | BinaryNg, 52/block | **1.65** | (b) | essentially **all** stall, no useful math. 52 elementwise programs that exist only to read and rewrite a tensor | fusion into the producer kernel — this is the cleanest delete in the block |
| 4 | LayerNorm, 41/block | **1.01** | (b) | 36 % not-stalled, 60 % input stall | same fusion axis as 3 |
| 5 | Transpose, 8/block | **0.61** | (b) | TRISC1 is 0.69 ms of a 2.17 ms kernel: this is a reblock, not a computation | layout — pick a layout that does not need it, rather than making it faster |

Those five are **9.61 s of the pairformer track's 10.22 s**. The diffusion step's own top three are
Matmul 1.81 s/fold, BinaryNg 0.72 s and SDPA 0.50 s.

**What this closes.** The falsifier in the brief said: if kernels are busy > 85 % of the block with
high per-core math utilization and a small gap fraction, the roofs are wrong and the only axis left
is fewer-and-larger ops. Half of that fired and half did not. Kernels *are* busy 98.9 % of the block
and the gap fraction *is* small — but per-core math utilization is **low**, a third at most. So the
deficit is neither the roofs being wrong nor op granularity. It is that the math engine spends most
of its time waiting for tiles to arrive, and that is a memory-layout and data-staging problem.
