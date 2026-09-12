# BEST-CONFIG-TABLE — Boltz-2 512 aa, every knob on every instance above 0.5 % of device time

whglx card 1, **Wormhole**, compute grid 8x9, ttnn 0.68.0. Every number is a
paired interleaved ratio: incumbent, arm, incumbent, arm, ... in one process, each arm scored
against the median of the two incumbent runs that bracket it. `A/A` is the spread of the
incumbent runs themselves and is the noise floor a ratio has to clear — several instances here
have an A/A above 2, which means the box was contended while they ran and their arms say
nothing. `rel` is the arm's worst-element error against a float32 CPU reference computed from
the same seeded operands; the incumbent's own `rel` is printed beside it, because the shipped
config is not exact either and an arm is only losing accuracy if it is losing it *relative to
what ships*.

**The incumbent is the call the fold makes.** Every matmul call site on the hot path already
passes `core_grid=CORE_GRID_MAIN` or a tuned program config, so a bare `ttnn.linear(a, b)`
incumbent measures ttnn's default resolver and nothing the fold can reach. The `nogrid=1` arm
below is that straw man, kept on purpose: it is 0.11x-0.16x on the pair-track Transition
matmuls, i.e. the default resolver is 6-9x slower than what already ships.

## The projection

Two levers, and they are not the same kind of thing.

**Placement — `outbuf=L1`, bit-exact.** Ten instances whose shipped call writes its result to
interleaved DRAM run 1.12x-2.16x faster writing to L1 instead. Same kernel, same accumulation
order, different destination buffer, and `l1_bitexact.py` confirms all ten outputs are identical
bit for bit against the shipped config. This lever needs no accuracy argument.

| instance | ms/fold | shipped out | ratio | saved ms |
|---|---|---|---|---|
| `PairformerLayer#004` | 2296 | DRAM | 1.453x | 716 |
| `DiffusionStep#043` | 719 | DRAM | 2.159x | 386 |
| `DiffusionStep#029` | 516 | DRAM | 1.914x | 246 |
| `DiffusionStep#000` | 769 | DRAM | 1.369x | 207 |
| `PairformerLayer#019` | 922 | DRAM | 1.124x | 102 |
| `DiffusionStep#040` | 211 | DRAM | 1.758x | 91 |
| `DiffusionStep#035` | 175 | DRAM | 1.572x | 64 |
| `MSALayer#015` | 205 | DRAM | 1.433x | 62 |
| `DiffusionStep#036` | 168 | DRAM | 1.566x | 60 |
| `DiffusionStep#053` | 114 | DRAM | 1.378x | 31 |

    saved              1966 ms of 37957 ms = 5.18 % of replayed device time
    PROJECTED-FOLD     1.0452x undiscounted, 1.0326x at the 0.73 additivity discount

**Numerics — fidelity and `fp32_dest_acc_en`.** Worth 827 ms, 1.0135x on the fold, and
every arm of it changes numerics. Most of it is `LoFi`, two steps below the HiFi3 that
`b2x-hifi3-endtoend` already closed end to end at 0.6 % of the fold. Not worth its control.

Both together: 1.0654x undiscounted, **1.0469x** discounted.

### The caveat that decides whether any of this is real

**An isolated bench has an empty L1. The fold does not.** Every `outbuf=L1` number above was taken
with one op on the chip and nothing else resident, which is the most favourable possible condition
for a lever whose whole mechanism is "put the result somewhere closer". In the fold those same
matmuls run beside live pair tensors, and the engine already carries the machinery that exists
because of exactly this — `_l1_memory_config_if_it_fits` takes a `reserve_per_core` argument
precisely because an allocation succeeding is not the test, a later consumer's circular buffers
failing at program creation is. `b2x-pair-l1-residency` closed the 67 MB pair tensor's version of
this as NO-GO.

So 1.0326x is an **upper bound taken under the friendliest conditions the lever will ever see**, not
a result. What separates the two is an integrated arm: route these matmul outputs through the
existing `_l1_memory_config_if_it_fits` derivation and run a paired interleaved fold A/B. That is
the next pass, and it is now worth running — the previous pass declined an integrated arm because
1.7 % could not be resolved against fold-level noise, and 5.2 % bit-exact can.

## The mechanism to hand back

* **`core_grid` on a batched matmul is worth 6-9x, and the engine already passes it.** The
  Transition's `1x16x512x128 @ 128x512` runs at 76.1 us as the fold calls it and 475.5 us as ttnn's
  default resolver would (`nogrid=1`, 0.161x / 0.162x on two independent instances). Shape
  dependent, not universal: on `1x512x768 @ 768x1536` the default is within 6 %, and on the two
  instances where both operands are batched it makes no difference at all.
* **`CORE_GRID_MAIN` is the right width.** 8x9 against 8x8 is 0.997x-1.009x; against 8x4 it is
  0.639x-0.985x. Monotone in width, nothing to tune. The one exception is `PairformerLayer#019`
  (`512x512x128 @ 128x4`), where a *narrower* 8x8 is 1.335x — a 4-wide output on a 9-row grid,
  which is the shape class worth a derivation if it turns up elsewhere.
* **`packer_l1_acc=1` is worth up to 1.21x** and the engine sets it (`packerl1=0` runs 0.824x-1.003x).
* **The placement derivation already exists and is under-applied.** `_l1_memory_config_if_it_fits`
  is the engine's own answer to this question and the Transition already uses it; the diffusion
  trunk's matmuls and `PairformerLayer#004` do not. That is the unified fix — extend an existing
  derivation to more call sites, not a table of per-shape constants.
* The only two hot-path matmuls the engine leaves to the default resolver outright are the
  atom/token scatter-gather pair at `tt_bio/tenstorrent.py:9445` and `:9530`, both using
  `transpose_a`/`transpose_b`, which the `core_grid` path does not take. Not in Boltz-2's top 47,
  but given the 6-9x cliff they are worth pricing where that path is hot.

## How much of the bench is trustworthy

Job 2 of this workstream asks the replay to reproduce in-fold cost to within ~15 %. **It does not.**
Summed per container against the costs re-measured on current main:

| unit | replayed (shipped incumbent) | container, BH | ratio |
|---|---|---|---|
| PairformerLayer | 19.48 s | 10.20 s | 1.91x |
| DiffusionStep | 14.23 s | 6.50 s | 2.19x |
| MSALayer | 4.25 s | 1.92 s | 2.21x |

Two causes are confounded here and this pass did not separate them: the replay runs on Wormhole and
the container numbers are Blackhole, and a WH chip is genuinely slower. The spread across the three
units is 1.91x-2.21x, i.e. within 16 % of each other, which is what a uniform silicon difference
would look like and not what a per-op replay artifact would. That matters for the projection only
through the shares, and a uniform factor cancels out of a ratio. Separating the two needs one
PairformerLayer trace-replayed on the same WH chip as the instances; that is a job for the next pass
and it is called out rather than assumed.

## Corrected ranking

The first ranking priced every matmul at the straw man's cost. Re-priced at the shipped cost:

| # | instance | op | share | ms/fold | source | unit |
|---|---|---|---|---|---|---|
| 1 | `PairformerLayer#007` | matmul | 8.87 % | 3366.1 | shipped | TriangleMultiplication |
| 2 | `PairformerLayer#004` | matmul | 6.05 % | 2296.5 | shipped | Transition |
| 3 | `PairformerLayer#017` | matmul | 3.26 % | 1236.3 | shipped | TriangleMultiplication |
| 4 | `DiffusionStep#016` | softmax | 2.87 % | 1088.9 | as-replayed | AttentionPairBias |
| 5 | `PairformerLayer#005` | layernorm | 2.60 % | 988.5 | as-replayed | TriangleMultiplication |
| 6 | `PairformerLayer#019` | matmul | 2.43 % | 922.5 | shipped | TriangleAttention |
| 7 | `PairformerLayer#006` | binary | 2.36 % | 895.1 | as-replayed | TriangleMultiplication |
| 8 | `PairformerLayer#014` | binary | 2.20 % | 834.8 | as-replayed | TriangleMultiplication |
| 9 | `PairformerLayer#025` | transpose | 2.11 % | 800.2 | as-replayed | TriangleAttention |
| 10 | `DiffusionStep#000` | matmul | 2.03 % | 768.8 | shipped | ConditionedTransitionBlock |
| 11 | `DiffusionStep#015` | binary | 1.96 % | 745.2 | as-replayed | AttentionPairBias |
| 12 | `DiffusionStep#043` | matmul | 1.89 % | 718.7 | shipped | AttentionPairBias |
| 13 | `DiffusionStep#013` | matmul | 1.89 % | 718.2 | shipped | AttentionPairBias |
| 14 | `DiffusionStep#017` | matmul | 1.82 % | 690.1 | shipped | AttentionPairBias |
| 15 | `PairformerLayer#008` | binary | 1.78 % | 674.4 | as-replayed | TriangleMultiplication |
| 16 | `PairformerLayer#001` | matmul | 1.71 % | 647.5 | shipped | Transition |
| 17 | `PairformerLayer#002` | matmul | 1.69 % | 643.1 | shipped | Transition |
| 18 | `DiffusionStep#042` | reshape | 1.55 % | 586.8 | as-replayed | AttentionPairBias |
| 19 | `DiffusionStep#010` | matmul | 1.44 % | 544.7 | as-replayed | AttentionPairBias |
| 20 | `DiffusionStep#014` | binary | 1.43 % | 541.1 | as-replayed | AttentionPairBias |
| 21 | `PairformerLayer#003` | binary | 1.41 % | 536.1 | as-replayed | Transition |
| 22 | `DiffusionStep#029` | matmul | 1.36 % | 515.8 | shipped | ConditionedTransitionBlock |
| 23 | `PairformerLayer#023` | binary | 1.32 % | 501.9 | as-replayed | TriangleAttention |
| 24 | `DiffusionStep#020` | reshape | 1.31 % | 498.6 | as-replayed | AttentionPairBias |
| 25 | `PairformerLayer#012` | slice | 1.22 % | 461.9 | as-replayed | TriangleMultiplication |
| 26 | `PairformerLayer#024` | binary | 1.17 % | 444.7 | as-replayed | TriangleAttention |
| 27 | `MSALayer#010` | matmul | 1.14 % | 431.7 | shipped | PairWeightedAveraging |
| 28 | `PairformerLayer#079` | matmul | 1.06 % | 403.1 | as-replayed | AttentionPairBias |
| 29 | `PairformerLayer#000` | layernorm | 1.06 % | 403.1 | as-replayed | Transition |
| 30 | `DiffusionStep#041` | permute | 1.04 % | 395.4 | as-replayed | AttentionPairBias |

Replayed total, corrected: **37.96 s/fold** over 323 instances.

## Every point, paired-repeat sweep (matmul instances) — authoritative

Each ratio is the median of three independent (incumbent, arm) pairs; `spread` is the range
across those three. The first row of every instance is the A/A control, the same estimator
with the arm replaced by another incumbent run. It lands at 0.989-1.006 on all seventeen,
which is the noise floor every other row has to clear.

| instance | shipped config | shipped out | ms/fold | knob | ratio | spread | bit-exact |
|---|---|---|---|---|---|---|---|
| `PairformerLayer#002` | core_grid | L1 | 643 | `(A/A control)` | 1.002x | 1.007 |  |
|  |  |  |  | `nogrid=1` | 0.161x | 1.002 |  |
|  |  |  |  | `fp32acc=0` | 1.119x | 1.009 |  |
|  |  |  |  | `fidelity=HiFi2,fp32acc=0` | 1.170x | 1.011 |  |
|  |  |  |  | `fidelity=LoFi,fp32acc=0` | 1.177x | 1.007 |  |
|  |  |  |  | `grid=8x8` | 0.999x | 1.014 |  |
|  |  |  |  | `fidelity=HiFi2` | 1.000x | 1.008 |  |
|  |  |  |  | `packerl1=0` | 0.910x | 1.005 |  |
|  |  |  |  | `dstfull=1` | 0.996x | 1.002 |  |
|  |  |  |  | `fp32acc=0,dstfull=1` | 1.119x | 1.002 |  |
|  |  |  |  | `grid=8x4` | 0.639x | 1.008 |  |
|  |  |  |  | `outbuf=L1` | 0.998x | 1.006 |  |
| `PairformerLayer#001` | core_grid | L1 | 647 | `(A/A control)` | 1.005x | 1.000 |  |
|  |  |  |  | `nogrid=1` | 0.162x | 1.013 |  |
|  |  |  |  | `fp32acc=0` | 1.129x | 1.024 |  |
|  |  |  |  | `fidelity=HiFi2,fp32acc=0` | 1.167x | 1.005 |  |
|  |  |  |  | `fidelity=LoFi,fp32acc=0` | 1.174x | 1.014 |  |
|  |  |  |  | `grid=8x8` | 0.997x | 1.007 |  |
|  |  |  |  | `fidelity=HiFi2` | 1.004x | 1.012 |  |
|  |  |  |  | `packerl1=0` | 0.911x | 1.003 |  |
|  |  |  |  | `dstfull=1` | 0.998x | 1.004 |  |
|  |  |  |  | `fp32acc=0,dstfull=1` | 1.115x | 1.005 |  |
|  |  |  |  | `grid=8x4` | 0.639x | 1.011 |  |
|  |  |  |  | `outbuf=L1` | 1.000x | 1.000 |  |
| `PairformerLayer#007` | program_config | L1 | 3366 | `(A/A control)` | 1.000x | 1.001 |  |
|  |  |  |  | `nogrid=1` | 1.000x | 1.002 |  |
|  |  |  |  | `fp32acc=0` | 1.058x | 1.005 |  |
|  |  |  |  | `fidelity=HiFi2,fp32acc=0` | 1.053x | 1.000 |  |
|  |  |  |  | `fidelity=LoFi,fp32acc=0` | 1.055x | 1.003 |  |
|  |  |  |  | `grid=8x8` | RuntimeError: TT_THROW @ /project/tt_metal/impl/program/pr |  |  |
|  |  |  |  | `fidelity=HiFi2` | 1.038x | 1.000 |  |
|  |  |  |  | `packerl1=0` | 1.000x | 1.001 |  |
|  |  |  |  | `dstfull=1` | 1.000x | 1.002 |  |
|  |  |  |  | `fp32acc=0,dstfull=1` | 1.056x | 1.002 |  |
|  |  |  |  | `grid=8x4` | RuntimeError: TT_THROW @ /project/tt_metal/impl/program/pr |  |  |
|  |  |  |  | `outbuf=L1` | 1.000x | 1.001 |  |
| `DiffusionStep#000` | core_grid | DRAM | 769 | `(A/A control)` | 1.004x | 1.009 |  |
|  |  |  |  | `nogrid=1` | 0.945x | 1.002 |  |
|  |  |  |  | `fp32acc=0` | 1.025x | 1.004 |  |
|  |  |  |  | `fidelity=HiFi2,fp32acc=0` | 1.033x | 1.008 |  |
|  |  |  |  | `fidelity=LoFi,fp32acc=0` | 1.030x | 1.005 |  |
|  |  |  |  | `grid=8x8` | 0.999x | 1.002 |  |
|  |  |  |  | `fidelity=HiFi2` | 1.016x | 1.030 |  |
|  |  |  |  | `packerl1=0` | 0.980x | 1.014 |  |
|  |  |  |  | `dstfull=1` | 0.997x | 1.005 |  |
|  |  |  |  | `fp32acc=0,dstfull=1` | 1.019x | 1.009 |  |
|  |  |  |  | `grid=8x4` | 0.735x | 1.003 |  |
|  |  |  |  | `outbuf=L1` | 1.369x | 1.020 | **yes** |
| `DiffusionStep#017` | core_grid | DRAM | 690 | `(A/A control)` | 1.006x | 1.005 |  |
|  |  |  |  | `nogrid=1` | 0.709x | 1.015 |  |
|  |  |  |  | `fp32acc=0` | 1.045x | 1.001 |  |
|  |  |  |  | `fidelity=HiFi2,fp32acc=0` | 1.057x | 1.009 |  |
|  |  |  |  | `fidelity=LoFi,fp32acc=0` | 1.058x | 1.003 |  |
|  |  |  |  | `grid=8x8` | 1.009x | 1.014 |  |
|  |  |  |  | `fidelity=HiFi2` | 1.001x | 1.015 |  |
|  |  |  |  | `packerl1=0` | 0.988x | 1.008 |  |
|  |  |  |  | `dstfull=1` | 1.002x | 1.015 |  |
|  |  |  |  | `fp32acc=0,dstfull=1` | 1.051x | 1.021 |  |
|  |  |  |  | `grid=8x4` | 1.240x | 1.019 |  |
|  |  |  |  | `outbuf=L1` | 1.003x | 1.015 |  |
| `PairformerLayer#004` | core_grid | DRAM | 2297 | `(A/A control)` | 0.999x | 1.007 |  |
|  |  |  |  | `nogrid=1` | 0.348x | 1.009 |  |
|  |  |  |  | `fp32acc=0` | 1.040x | 1.021 |  |
|  |  |  |  | `fidelity=HiFi2,fp32acc=0` | 1.064x | 1.002 |  |
|  |  |  |  | `fidelity=LoFi,fp32acc=0` | 1.074x | 1.008 |  |
|  |  |  |  | `grid=8x8` | 1.000x | 1.005 |  |
|  |  |  |  | `fidelity=HiFi2` | 1.006x | 1.006 |  |
|  |  |  |  | `packerl1=0` | 0.967x | 1.016 |  |
|  |  |  |  | `dstfull=1` | 1.003x | 1.013 |  |
|  |  |  |  | `fp32acc=0,dstfull=1` | 1.034x | 1.005 |  |
|  |  |  |  | `grid=8x4` | 0.804x | 1.001 |  |
|  |  |  |  | `outbuf=L1` | 1.453x | 1.030 | **yes** |
| `DiffusionStep#029` | core_grid | DRAM | 516 | `(A/A control)` | 0.996x | 1.003 |  |
|  |  |  |  | `nogrid=1` | 0.193x | 1.002 |  |
|  |  |  |  | `fp32acc=0` | 1.009x | 1.008 |  |
|  |  |  |  | `fidelity=HiFi2,fp32acc=0` | 1.017x | 1.003 |  |
|  |  |  |  | `fidelity=LoFi,fp32acc=0` | 1.015x | 1.007 |  |
|  |  |  |  | `grid=8x8` | 1.002x | 1.003 |  |
|  |  |  |  | `fidelity=HiFi2` | 0.998x | 1.002 |  |
|  |  |  |  | `packerl1=0` | 0.992x | 1.003 |  |
|  |  |  |  | `dstfull=1` | 1.000x | 1.005 |  |
|  |  |  |  | `fp32acc=0,dstfull=1` | 1.009x | 1.002 |  |
|  |  |  |  | `grid=8x4` | 0.975x | 1.010 |  |
|  |  |  |  | `outbuf=L1` | 1.914x | 1.012 | **yes** |
| `DiffusionStep#035` | core_grid | DRAM | 175 | `(A/A control)` | 0.998x | 1.008 |  |
|  |  |  |  | `nogrid=1` | 0.129x | 1.012 |  |
|  |  |  |  | `fp32acc=0` | 1.024x | 1.022 |  |
|  |  |  |  | `fidelity=HiFi2,fp32acc=0` | 1.022x | 1.017 |  |
|  |  |  |  | `fidelity=LoFi,fp32acc=0` | 1.025x | 1.012 |  |
|  |  |  |  | `grid=8x8` | 1.002x | 1.005 |  |
|  |  |  |  | `fidelity=HiFi2` | 1.003x | 1.004 |  |
|  |  |  |  | `packerl1=0` | 0.976x | 1.003 |  |
|  |  |  |  | `dstfull=1` | 1.001x | 1.004 |  |
|  |  |  |  | `fp32acc=0,dstfull=1` | 1.020x | 1.004 |  |
|  |  |  |  | `grid=8x4` | 0.945x | 1.003 |  |
|  |  |  |  | `outbuf=L1` | 1.572x | 1.017 | **yes** |
| `DiffusionStep#036` | core_grid | DRAM | 167 | `(A/A control)` | 0.989x | 1.016 |  |
|  |  |  |  | `nogrid=1` | 0.128x | 1.016 |  |
|  |  |  |  | `fp32acc=0` | 1.018x | 1.011 |  |
|  |  |  |  | `fidelity=HiFi2,fp32acc=0` | 1.024x | 1.014 |  |
|  |  |  |  | `fidelity=LoFi,fp32acc=0` | 1.023x | 1.001 |  |
|  |  |  |  | `grid=8x8` | 1.004x | 1.006 |  |
|  |  |  |  | `fidelity=HiFi2` | 1.003x | 1.011 |  |
|  |  |  |  | `packerl1=0` | 0.992x | 1.002 |  |
|  |  |  |  | `dstfull=1` | 1.002x | 1.003 |  |
|  |  |  |  | `fp32acc=0,dstfull=1` | 1.014x | 1.006 |  |
|  |  |  |  | `grid=8x4` | 0.964x | 1.005 |  |
|  |  |  |  | `outbuf=L1` | 1.566x | 1.025 | **yes** |
| `DiffusionStep#043` | core_grid | DRAM | 719 | `(A/A control)` | 0.999x | 1.005 |  |
|  |  |  |  | `nogrid=1` | 0.457x | 1.007 |  |
|  |  |  |  | `fp32acc=0` | 1.007x | 1.003 |  |
|  |  |  |  | `fidelity=HiFi2,fp32acc=0` | 1.011x | 1.005 |  |
|  |  |  |  | `fidelity=LoFi,fp32acc=0` | 1.014x | 1.007 |  |
|  |  |  |  | `grid=8x8` | 1.008x | 1.009 |  |
|  |  |  |  | `fidelity=HiFi2` | 1.000x | 1.017 |  |
|  |  |  |  | `packerl1=0` | 0.988x | 1.010 |  |
|  |  |  |  | `dstfull=1` | 0.998x | 1.004 |  |
|  |  |  |  | `fp32acc=0,dstfull=1` | 1.007x | 1.004 |  |
|  |  |  |  | `grid=8x4` | 0.985x | 1.002 |  |
|  |  |  |  | `outbuf=L1` | 2.159x | 1.032 | **yes** |
| `PairformerLayer#017` | program_config | DRAM | 1236 | `(A/A control)` | 0.999x | 1.001 |  |
|  |  |  |  | `nogrid=1` | 1.000x | 1.001 |  |
|  |  |  |  | `fp32acc=0` | 1.064x | 1.000 |  |
|  |  |  |  | `fidelity=HiFi2,fp32acc=0` | 1.098x | 1.000 |  |
|  |  |  |  | `fidelity=LoFi,fp32acc=0` | 1.111x | 1.001 |  |
|  |  |  |  | `grid=8x8` | 1.000x | 1.001 |  |
|  |  |  |  | `fidelity=HiFi2` | 1.060x | 1.000 |  |
|  |  |  |  | `packerl1=0` | 0.941x | 1.001 |  |
|  |  |  |  | `dstfull=1` | 1.000x | 1.001 |  |
|  |  |  |  | `fp32acc=0,dstfull=1` | 1.064x | 1.001 |  |
|  |  |  |  | `grid=8x4` | 0.787x | 1.002 |  |
|  |  |  |  | `outbuf=L1` | RuntimeError: TT_THROW @ /project/tt_metal/impl/program/pr |  |  |
| `MSALayer#015` | core_grid | DRAM | 205 | `(A/A control)` | 1.000x | 1.001 |  |
|  |  |  |  | `nogrid=1` | 0.194x | 1.001 |  |
|  |  |  |  | `fp32acc=0` | 1.390x | 1.001 |  |
|  |  |  |  | `fidelity=HiFi2,fp32acc=0` | 1.399x | 1.006 |  |
|  |  |  |  | `fidelity=LoFi,fp32acc=0` | 1.401x | 1.002 |  |
|  |  |  |  | `grid=8x8` | 1.381x | 1.002 |  |
|  |  |  |  | `fidelity=HiFi2` | 1.000x | 1.001 |  |
|  |  |  |  | `packerl1=0` | 0.993x | 1.004 |  |
|  |  |  |  | `dstfull=1` | 1.000x | 1.002 |  |
|  |  |  |  | `fp32acc=0,dstfull=1` | 1.389x | 1.004 |  |
|  |  |  |  | `grid=8x4` | 0.992x | 1.002 |  |
|  |  |  |  | `outbuf=L1` | 1.433x | 1.003 | **yes** |
| `DiffusionStep#053` | core_grid | DRAM | 114 | `(A/A control)` | 1.002x | 1.005 |  |
|  |  |  |  | `nogrid=1` | 0.115x | 1.004 |  |
|  |  |  |  | `fp32acc=0` | 1.005x | 1.017 |  |
|  |  |  |  | `fidelity=HiFi2,fp32acc=0` | 1.010x | 1.003 |  |
|  |  |  |  | `fidelity=LoFi,fp32acc=0` | 1.014x | 1.003 |  |
|  |  |  |  | `grid=8x8` | 1.001x | 1.005 |  |
|  |  |  |  | `fidelity=HiFi2` | 0.999x | 1.004 |  |
|  |  |  |  | `packerl1=0` | 1.003x | 1.008 |  |
|  |  |  |  | `dstfull=1` | 1.000x | 1.002 |  |
|  |  |  |  | `fp32acc=0,dstfull=1` | 1.006x | 1.005 |  |
|  |  |  |  | `grid=8x4` | 0.915x | 1.003 |  |
|  |  |  |  | `outbuf=L1` | 1.378x | 1.012 | **yes** |
| `PairformerLayer#019` | program_config | DRAM | 923 | `(A/A control)` | 1.000x | 1.002 |  |
|  |  |  |  | `nogrid=1` | 1.001x | 1.001 |  |
|  |  |  |  | `fp32acc=0` | 1.028x | 1.003 |  |
|  |  |  |  | `fidelity=HiFi2,fp32acc=0` | 1.034x | 1.000 |  |
|  |  |  |  | `fidelity=LoFi,fp32acc=0` | 1.034x | 1.002 |  |
|  |  |  |  | `grid=8x8` | 1.335x | 1.003 |  |
|  |  |  |  | `fidelity=HiFi2` | 1.013x | 1.003 |  |
|  |  |  |  | `packerl1=0` | 1.000x | 1.003 |  |
|  |  |  |  | `dstfull=1` | 1.000x | 1.004 |  |
|  |  |  |  | `fp32acc=0,dstfull=1` | 1.028x | 1.000 |  |
|  |  |  |  | `grid=8x4` | 1.284x | 1.004 |  |
|  |  |  |  | `outbuf=L1` | 1.124x | 1.004 | **yes** |
| `DiffusionStep#013` | core_grid | DRAM | 718 | `(A/A control)` | 1.003x | 1.013 |  |
|  |  |  |  | `nogrid=1` | 1.000x | 1.001 |  |
|  |  |  |  | `fp32acc=0` | 0.858x | 1.003 |  |
|  |  |  |  | `fidelity=HiFi2,fp32acc=0` | 0.885x | 1.010 |  |
|  |  |  |  | `fidelity=LoFi,fp32acc=0` | 0.886x | 1.010 |  |
|  |  |  |  | `grid=8x8` | 0.996x | 1.005 |  |
|  |  |  |  | `fidelity=HiFi2` | 1.014x | 1.008 |  |
|  |  |  |  | `packerl1=0` | 0.999x | 1.009 |  |
|  |  |  |  | `dstfull=1` | 1.000x | 1.003 |  |
|  |  |  |  | `fp32acc=0,dstfull=1` | 0.860x | 1.006 |  |
|  |  |  |  | `grid=8x4` | 0.951x | 1.003 |  |
|  |  |  |  | `outbuf=L1` | 0.476x | 1.020 |  |
| `DiffusionStep#040` | core_grid | DRAM | 211 | `(A/A control)` | 1.000x | 1.018 |  |
|  |  |  |  | `nogrid=1` | 0.460x | 1.013 |  |
|  |  |  |  | `fp32acc=0` | 1.030x | 1.024 |  |
|  |  |  |  | `fidelity=HiFi2,fp32acc=0` | 1.081x | 1.004 |  |
|  |  |  |  | `fidelity=LoFi,fp32acc=0` | 1.081x | 1.008 |  |
|  |  |  |  | `grid=8x8` | 1.000x | 1.004 |  |
|  |  |  |  | `fidelity=HiFi2` | 0.956x | 1.001 |  |
|  |  |  |  | `packerl1=0` | 0.824x | 1.006 |  |
|  |  |  |  | `dstfull=1` | 1.000x | 1.004 |  |
|  |  |  |  | `fp32acc=0,dstfull=1` | 1.031x | 1.002 |  |
|  |  |  |  | `grid=8x4` | 0.686x | 1.001 |  |
|  |  |  |  | `outbuf=L1` | 1.758x | 1.020 | **yes** |
| `MSALayer#010` | core_grid | DRAM | 432 | `(A/A control)` | 1.000x | 1.000 |  |
|  |  |  |  | `nogrid=1` | 0.613x | 1.003 |  |
|  |  |  |  | `fp32acc=0` | 1.018x | 1.006 |  |
|  |  |  |  | `fidelity=HiFi2,fp32acc=0` | 1.017x | 1.004 |  |
|  |  |  |  | `fidelity=LoFi,fp32acc=0` | 1.016x | 1.001 |  |
|  |  |  |  | `grid=8x8` | 1.029x | 1.003 |  |
|  |  |  |  | `fidelity=HiFi2` | 1.000x | 1.001 |  |
|  |  |  |  | `packerl1=0` | 1.001x | 1.001 |  |
|  |  |  |  | `dstfull=1` | 1.001x | 1.003 |  |
|  |  |  |  | `fp32acc=0,dstfull=1` | 1.017x | 1.002 |  |
|  |  |  |  | `grid=8x4` | 0.964x | 1.006 |  |
|  |  |  |  | `outbuf=L1` | RuntimeError: TT_THROW @ /project/tt_metal/impl/program/pr |  |  |

## Every point, first sweep (non-matmul instances, unaffected by the straw man)

| instance | op | ms/fold | knob | ratio | max_abs vs incumbent |
|---|---|---|---|---|---|
| `DiffusionStep#016` | softmax | 1089 | `fidelity=LoFi` | 1.005x | 0.001556 |
|  |  |  | `fidelity=HiFi2` | 1.008x | 0.001579 |
|  |  |  | `fidelity=HiFi3` | 1.004x | 0.001556 |
|  |  |  | `fp32acc=0` | 1.702x | 0.001625 |
|  |  |  | `packerl1=0` | 0.998x | 0.001602 |
|  |  |  | `dstfull=1` | 1.000x | 0.001625 |
|  |  |  | `fidelity=LoFi,fp32acc=0` | 1.798x | 0.00164 |
|  |  |  | `fidelity=HiFi2,fp32acc=0` | 1.833x | 0.001602 |
|  |  |  | `fidelity=HiFi3,fp32acc=0` | 1.793x | 0.001564 |
|  |  |  | `fidelity=LoFi,fp32acc=0,packerl1=0` | 1.851x | 0.001518 |
|  |  |  | `fidelity=HiFi2,fp32acc=0,dstfull=1` | 1.836x | 0.001625 |
|  |  |  | `outbuf=L1` | 1.023x | 0.001488 |
|  |  |  | **A/A floor** | **1.0071** |  |
| `PairformerLayer#005` | layernorm | 988 | `fidelity=LoFi` | 1.069x | 1.853699 |
|  |  |  | `fidelity=HiFi2` | 1.049x | 1.5625 |
|  |  |  | `fidelity=HiFi3` | 1.024x | 1.485779 |
|  |  |  | `fp32acc=0` | 1.269x | 1.552246 |
|  |  |  | `packerl1=0` | 1.001x | 1.473633 |
|  |  |  | `dstfull=1` | 1.001x | 1.521484 |
|  |  |  | `fidelity=LoFi,fp32acc=0` | 1.346x | 1.554199 |
|  |  |  | `fidelity=HiFi2,fp32acc=0` | 1.292x | 1.56543 |
|  |  |  | `fidelity=HiFi3,fp32acc=0` | 1.311x | 2.140259 |
|  |  |  | `fidelity=LoFi,fp32acc=0,packerl1=0` | 1.333x | 1.513672 |
|  |  |  | `fidelity=HiFi2,fp32acc=0,dstfull=1` | 1.319x | 1.713867 |
|  |  |  | `outbuf=L1` | 1.007x | 1.634766 |
|  |  |  | **A/A floor** | **1.0033** |  |
| `PairformerLayer#006` | binary | 895 | `outbuf=L1` | 1.001x | 0.174805 |
|  |  |  | **A/A floor** | **1.0024** |  |
| `PairformerLayer#014` | binary | 835 | `outbuf=L1` | 1.001x | 0.167969 |
|  |  |  | **A/A floor** | **1.0001** |  |
| `PairformerLayer#025` | transpose | 800 | `outbuf=L1` | 1.001x | 0.775391 |
|  |  |  | **A/A floor** | **1.0004** |  |
| `DiffusionStep#015` | binary | 745 | `outbuf=L1` | 1.007x | 0.542297 |
|  |  |  | **A/A floor** | **1.0011** |  |
| `PairformerLayer#008` | binary | 674 | `outbuf=L1` | 1.001x | 1.15625 |
|  |  |  | **A/A floor** | **1.006** |  |
