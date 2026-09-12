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
