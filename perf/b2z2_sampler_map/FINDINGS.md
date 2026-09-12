# The 200-step diffusion sampler, mapped

`b2z2-sampler-ceiling-map`, 2026-09-12. ARCH: **BH** throughout (qb2 card 0, p300c, 11x10 = 110
worker cores, ttnn 0.68.0). No card held, no fold run, no device opened: every number is
re-derived from artifacts wave 1 already committed.

Reproduce with `python3 step_map.py` then `python3 ceiling.py`. Inputs are copied into `src/`
from `wk/b2z-kernel-cycle-census` and `wk/b2z-diffusion-utilization`.

Ratios are quoted against the **live published cell, 20.079 s** (`site/data/perf-512aa.json`,
`models[0].cells.p150a.s_per_fold`, committed at `fc7fed56`). 2x = **10.040 s**.

## The census is not stale, and that is worth saying first

CONTEXT `§2-CORRECTION` marks the whole 41.4152 / 32.5179 / 120.2894 decomposition stale. For the
trunk that is right. For the sampler it is not: both diffusion censuses were taken at commits
(`57b3a806`, `9db1ad88`) that already contain `fc7fed56`, the merge that turned both diffusion
levers on. The two runs execute the same 1066-program sequence byte for byte, which is the check
that the step did not move between them. Everything below describes the step main runs today.

## One step

| term | ms | basis |
|---|---|---|
| step wall | **26.400** | MEASURED, the page's own cell: 5.280 s / 200 at `fc7fed56` |
| device kernel time | **22.015** | MEASURED, 1066 programs, `b2z-kernel-cycle-census` |
| exposed dispatch gap | 4.385 | DERIVED, wall - kernel, 16.6 % of the wall |
| grid under-fill | 5.341 | MEASURED, per-program core counts, 24.3 % of kernel |
| ops that do no arithmetic at all | 3.369 | MEASURED, TRISC1 never resident, 15.3 % of kernel |

### Per-op, one step

| op | calls | ms/step | us/call | % kernel | mean cores | min | TRISC1 % | s/fold |
|---|---|---|---|---|---|---|---|---|
| Matmul | 387 | 9.068 | 23.4 | 41.19 | 70.0 | 8 | 74.3 | 1.814 |
| BinaryNg | 339 | 3.591 | 10.6 | 16.31 | 110.0 | 110 | 84.1 | 0.718 |
| SDPA | 30 | 2.504 | 83.5 | 11.38 | 110.0 | 110 | 98.5 | 0.501 |
| LayerNorm | 114 | 1.736 | 15.2 | 7.89 | 26.6 | 1 | 92.3 | 0.347 |
| Permute | 12 | 1.470 | 122.5 | 6.68 | 110.0 | 110 | 86.8 | 0.294 |
| NlpCreateHeads | 30 | 1.467 | 48.9 | 6.66 | 34.8 | 16 | **0.0** | 0.293 |
| ReshapeView | 36 | 1.359 | 37.8 | 6.17 | 110.0 | 110 | **0.0** | 0.272 |
| Transpose | 51 | 0.275 | 5.4 | 1.25 | 110.0 | 110 | 38.6 | 0.055 |
| Slice | 30 | 0.227 | 7.6 | 1.03 | 110.0 | 110 | **0.0** | 0.045 |
| Pad | 6 | 0.145 | 24.1 | 0.66 | 110.0 | 110 | **0.0** | 0.029 |
| Copy | 24 | 0.107 | 4.4 | 0.48 | 110.0 | 110 | **0.0** | 0.021 |
| NLPConcatHeads | 6 | 0.064 | 10.7 | 0.29 | 110.0 | 110 | **0.0** | 0.013 |
| UnaryNg | 1 | 0.002 | 1.8 | 0.01 | 110.0 | 110 | 80.0 | 0.000 |

The `s/fold` column reproduces `b2z-kernel-cycle-census`'s published top three — Matmul 1.81,
BinaryNg 0.72, SDPA 0.50 — to three decimals, from a different script. That is the check that
this pipeline is reading the traces the same way the row that produced them did.

**Top three hold 68.9 % of kernel time.** The step is far more concentrated than the trunk, whose
top five hold 94 % but whose largest single class is 37 %.

## The stall identity, applied to the step

`--enable-sum-profiling` has never been run on a diffusion step. `cb_wait_front` and
`cb_reserve_back` are blank in all three committed step CSVs, so the block's inner split
(wait_in / wait_out / compute) cannot be reproduced for the sampler from anything on disk. What
the default profiler does give is the outer bracket, per op and per RISC:

| | Pairformer block | diffusion step |
|---|---|---|
| span | 36.344 ms | 26.400 ms (production wall) |
| outside any kernel | 0.348 ms, 1.0 % | 4.385 ms, **16.6 %** |
| inside a kernel, math thread NOT resident | 3.852 ms, 10.6 % | 6.804 ms, **25.8 %** |
| math thread resident | 32.144 ms, **88.4 %** | 15.211 ms, **57.6 %** |
| of which movement (wait_in + wait_out) | 21.443 ms, 59.0 % of span | **UNMEASURED** |

**The step loses 42.4 % of its wall before the CB-stall question is even asked; the block loses
11.6 %.** So the campaign's floor statement — "57.0 % of the math thread's resident time is spent
waiting for input tiles" — is a statement about the trunk that has never been tested on the
sampler, and the sampler's dominant term is not inside that bracket at all. Two of the step's
three biggest losses are measurable today and neither is a CB stall.

## The 75.7 % occupancy: a shape artifact, and mostly NOT a lever

Reproduced exactly: duration-weighted occupancy **75.739 %** (published 75.74), idle core-fraction
**1.0682 s/fold** (published 1.068), share on the `[1,1,512,D]` token axis **89.16 %** (published
89.2). The core histogram is bimodal, not a tail: 554 programs run on all 110 cores (10.32 ms),
220 on 64 (4.59 ms), 124 on 16 (2.61 ms). The narrow programs are short (21 us mean on the
16-core group) and the long ones are wide (SDPA 83.5 us and Permute 122.5 us, both on 110 cores).
Not a serial tail. Not a dependency stall.

All 5.341 ms of it sits in exactly three op codes: Matmul 3.099, LayerNorm 1.370,
NlpCreateHeads 0.873. Every other op in the step already runs on the full grid.

### And the largest third of it is not recoverable

`b2z-diffusion-utilization` priced "matmul grid height on a 16-tile row axis" at **0.5042 s/fold,
the largest piece**. It is worth **0.0054 s/fold**.

ttnn's 2D matmul costs `ceil(M/gy) * ceil(N/gx) * K` tile-blocks per core. The step's dominant
matmul is M = 512 rows = 16 tiles, N = 768 = 24 tiles. On an 11x10 grid `gy <= 10`, so
`ceil(16/gy) = 2` for every `gy` in 8..10; and `gx <= 11`, so `ceil(24/gx) = 3` for every `gx` in
8..11. **Every grid from 8x8 = 64 cores to 10x11 = 110 cores does 6 tile-blocks per core.**
Re-spreading that matmul onto 110 cores leaves the per-core work unchanged and buys nothing. The
64-core choice is ttnn making the right call, not a defect.

Checked over all 387 matmuls in the step (`step_map.grid_optimality`): **386 are already on an
optimal grid**, and the one that is not costs 0.121 ms/step. **0.31 %** of matmul time is
recoverable. Padding the token axis 512 -> 640 does not help either, for the same reason: 20 tiles
over `gy = 10` is still `per_core_M = 2`.

What survives is LayerNorm (row-parallel over 16 tile-rows, and the programs log
`DistributedLayerNormStage::NOT_DISTRIBUTED`, so a width-sharded variant genuinely reaches more
cores) and NlpCreateHeads, which is better deleted than re-spread. Corrected grid lever:
**0.434 s/fold**, not 1.068.

## The dispatch gap, cross-validated

The Pairformer block is the control for launch latency: its host is fully ahead (132 us/op against
~20 us of issue), so its span-minus-kernel of 0.3476 ms over 272 programs is the device's own
program-to-program latency, **1.278 us/program**. The step's gap is 4.113 us/program. The
2.835 us/program excess is host, and over 1066 programs x 200 steps that is **0.605 s/fold**
against the **0.647 s** `b2z-host-residual-kill` measured independently on card 2. Two unrelated
instruments, 6.5 % apart.

So trace is credited with the host excess only. The 1.278 us/program device floor is carried
through every row below, with an optimistic column beside it that hides that too.

## The SDPA head-plumbing tax

30 SDPA calls cost 2.504 ms. Arranging their operands — NlpCreateHeads 1.467, Reshape 1.359,
Slice 0.227, Transpose 0.275, Concat 0.064 — costs **3.393 ms, 1.35x the attention it feeds**,
and 153 of the step's 1066 programs. Five of those six op classes have TRISC1 residency of
exactly zero: they are programs that exist only to move bytes.

## What the sampler has to deliver, and whether it can

Trunk at its absolute movement-free floor (4.689 s, `b2z2-redteam-ceiling`'s row 4 with the
trunk's own 0.355 s of host held), `rest` held at its measured 2.1805 s:

    2x = 10.040 s.  trunk floor 4.689 + rest 2.180 = 6.869 s
    => the sampler's whole budget is 3.170 s = 15.85 ms/step, a 1.658x on the sampler.

| regime | programs | ms/step | sampler | fold | vs cell | 2x gap | zero-launch |
|---|---|---|---|---|---|---|---|
| today, untraced | 1066 | 26.400 | 5.280 s | 12.149 s | 1.6527x | +2.110 | 1.7813x |
| + trace (already built, default off) | 1066 | 23.378 | 4.676 s | 11.545 s | 1.7392x | +1.505 | 1.7813x |
| + distributed LayerNorm | 1066 | 21.208 | 4.242 s | 11.111 s | 1.8071x | +1.071 | 1.8526x |
| + fuse the SDPA head plumbing | 913 | 18.454 | 3.691 s | 10.560 s | 1.9014x | +0.521 | 1.9444x |
| + fuse half of BinaryNg | 744 | 16.442 | 3.288 s | **10.158 s** | **1.9767x** | **+0.118** | **2.0144x** |

The `today` row lands at 12.149 s against `b2z2-redteam-ceiling`'s 12.124 s for the same
treatment; the 0.025 s is that this row uses the page's own sampler (5.280 s) where the review
used the integration arm's (5.2547 s).

**2x sits exactly on the boundary.** Every sampler lever that exists, stacked, with the trunk
simultaneously at a movement-free floor nobody knows how to reach, lands at **1.9767x** carrying
the measured device launch latency and **2.0144x** if a trace hides it too. The margin is
**0.118 s on a 10.040 s target: 1.2 %.** Any single lever under-delivering by 15 % kills it.

## Levers, ranked by s/fold per unit of build effort

Effort is a relative integer: 1 = flip an existing default and measure it, 5 = swap one op for a
variant tt-metal already ships, 8 = write and validate a fused kernel.

| s/fold | % of step | effort | lever |
|---|---|---|---|
| **0.604** | 11.4 | **1** | **trace the 200-step loop — `--diffusion_trace`, ALREADY BUILT, defaults False** |
| 0.512 | 9.7 | 8 | fuse the SDPA head plumbing (153 programs, 1.35x the SDPA it feeds) |
| 0.262 | 5.0 | 5 | distributed / width-sharded LayerNorm (100 calls/step on 16 of 110 cores) |
| 0.359 | 6.8 | 8 | fuse BinaryNg into its producers (339 calls/step, half credited) |
| 0.005 | 0.1 | 5 | ~~re-spread the diffusion matmuls~~ REFUTED, priced for the record |

Falsifiers are carried per lever in `ceiling.json`.

## The one experiment this row could not run

`--enable-sum-profiling` on a diffusion step, so the sampler gets the same `wait_in / wait_out /
compute` split the block has. It needs a card; `b2z2-diffusion-loop-attack` holds one. Without it,
the 15.211 ms of resident math-thread time is a single undifferentiated block and nobody can say
whether the sampler is movement-floored the way the trunk is.
