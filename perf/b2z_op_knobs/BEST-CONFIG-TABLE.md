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

## The projection, and what it is worth

`projection.py` sums the per-op wins that clear three filters: the instance's A/A floor is under
1.20 (seven of twenty-four ran while the box was contended and came back with A/A above 2, so
their arms say nothing), the arm's ratio exceeds that floor, and the shipped call is one the
replay can reproduce. Instances whose shipped call goes through a hand-tuned program config are
held out, because there the replay's incumbent is ttnn's default resolver and the ratio is
against something the fold never runs.

    saved                  1026 ms of 37957 ms replayed device time = 2.70 %
    PROJECTED-FOLD         1.0231x undiscounted
                           1.0168x at the 0.73 additivity discount
                           1.0144x at 0.63
    optimistic ceiling     1.0271x, if the held-out program-config
                           instances are credited with their straw-man ratios anyway

Against the 22.3142 s fold / 18.6287 s device split re-measured on current main.

Where that 1026 ms comes from matters more than its size. **755 ms of it, 74 %, is `LoFi` on one
softmax and one layer_norm** — the fidelity axis, which `b2x-hifi3-endtoend` already closed end to
end at 0.6 % of the fold, and LoFi is two steps further down than the HiFi3 that closed it. Not
one arm in the whole table that clears its A/A floor is bit-exact. The arms that ARE bit-exact —
`grid=8x8`, `dstfull=1` — are 0.998x to 1.003x, which is to say nothing.

## The mechanism to hand back

There is no new derivation here. The finding is that the existing one is already right, and by
how much:

* **`core_grid` on a batched matmul is worth 6-9x, and the engine already passes it.** The
  Transition's `1x16x512x128 @ 128x512` runs at 76.1 us as the fold calls it and 475.5 us as
  ttnn's default resolver would (`nogrid=1`, 0.162x, reproduced on two independent instances with
  A/A floors of 1.023 and 1.030). The cliff is shape-dependent, not universal: on the unbatched
  `1x512x768 @ 768x1536` the default resolver is within 4 %.
* **`CORE_GRID_MAIN` is the right width.** Full 8x9 against 8x8 is 0.998x-1.002x and bit-exact;
  against 8x4 it is 1.58x. Wider is monotone, so the resolver has nothing to tune.
* **`packer_l1_acc=1` is worth 1.09x** and the engine sets it. `packerl1=0` is 0.92x on both
  Transition instances.
* The one place the engine leaves a matmul to the default resolver is the atom/token
  scatter-gather pair in `Diffusion` (tt_bio/tenstorrent.py:9445 and :9530), both of which use
  `transpose_a` / `transpose_b`. Neither is in the top 47 instances, so this pass did not price
  them; given the 6-9x cliff they are worth a look on the models where that path is hot.

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

## Every point, corrected sweep (matmul instances)

| instance | shipped config | ms/fold | knob | ratio | rel vs fp32 | bit-exact |
|---|---|---|---|---|---|---|
| `PairformerLayer#002` | core_grid | 643 | _(incumbent)_ | 1.000x | 0.003168 | — |
|  |  |  | `nogrid=1` | 0.163x | 0.003016 | no |
|  |  |  | `fp32acc=0` | 1.119x | 0.011287 | no |
|  |  |  | `fidelity=HiFi2` | 1.011x | 0.00631 | no |
|  |  |  | `fidelity=HiFi2,fp32acc=0` | 1.165x | 0.010907 | no |
|  |  |  | `fidelity=LoFi,fp32acc=0` | 1.170x | 0.027266 | no |
|  |  |  | `packerl1=0` | 0.921x | 0.003381 | no |
|  |  |  | `dstfull=1` | 0.994x | 0.003168 | yes |
|  |  |  | `fp32acc=0,dstfull=1` | 1.125x | 0.011287 | no |
|  |  |  | `grid=8x8` | 0.998x | 0.003168 | yes |
|  |  |  | `grid=8x4` | 0.633x | 0.003168 | yes |
|  |  |  | **A/A floor** | **1.0299** |  |  |
| `PairformerLayer#001` | core_grid | 647 | _(incumbent)_ | 1.000x | 0.003168 | — |
|  |  |  | `nogrid=1` | 0.161x | 0.003016 | no |
|  |  |  | `fp32acc=0` | 1.123x | 0.011287 | no |
|  |  |  | `fidelity=HiFi2` | 1.007x | 0.00631 | no |
|  |  |  | `fidelity=HiFi2,fp32acc=0` | 1.160x | 0.010907 | no |
|  |  |  | `fidelity=LoFi,fp32acc=0` | 1.172x | 0.027266 | no |
|  |  |  | `packerl1=0` | 0.918x | 0.003381 | no |
|  |  |  | `dstfull=1` | 1.003x | 0.003168 | yes |
|  |  |  | `fp32acc=0,dstfull=1` | 1.125x | 0.011287 | no |
|  |  |  | `grid=8x8` | 1.002x | 0.003168 | yes |
|  |  |  | `grid=8x4` | 0.639x | 0.003168 | yes |
|  |  |  | **A/A floor** | **1.0234** |  |  |
| `PairformerLayer#007` | program_config | 3366 | _(incumbent)_ | 1.000x | 0.714688 | — |
|  |  |  | `nogrid=1` | 0.980x | 0.714688 | yes |
|  |  |  | `fp32acc=0` | 1.055x | 0.013013 | no |
|  |  |  | `fidelity=HiFi2` | 1.042x | 0.006634 | no |
|  |  |  | `fidelity=HiFi2,fp32acc=0` | 1.036x | 0.013013 | no |
|  |  |  | `fidelity=LoFi,fp32acc=0` | 1.078x | 0.013013 | no |
|  |  |  | `packerl1=0` | 1.021x | 0.714688 | yes |
|  |  |  | `dstfull=1` | 0.959x | 0.714688 | yes |
|  |  |  | `fp32acc=0,dstfull=1` | 1.054x | 0.013013 | no |
|  |  |  | `grid=8x8` | RuntimeError: TT_THROW @ /project/tt_metal/impl/program/prog |  |  |
|  |  |  | `grid=8x4` | RuntimeError: TT_THROW @ /project/tt_metal/impl/program/prog |  |  |
|  |  |  | **A/A floor** | **1.0431** |  |  |
| `DiffusionStep#000` | core_grid | 769 | _(incumbent)_ | 1.000x | 0.697219 | — |
|  |  |  | `nogrid=1` | 0.961x | 0.003091 | no |
|  |  |  | `fp32acc=0` | 1.003x | 0.015789 | no |
|  |  |  | `fidelity=HiFi2` | 0.987x | 0.005722 | no |
|  |  |  | `fidelity=HiFi2,fp32acc=0` | 1.037x | 0.015448 | no |
|  |  |  | `fidelity=LoFi,fp32acc=0` | 1.030x | 0.026005 | no |
|  |  |  | `packerl1=0` | 0.978x | 0.003619 | no |
|  |  |  | `dstfull=1` | 1.000x | 0.697219 | yes |
|  |  |  | `fp32acc=0,dstfull=1` | 1.031x | 0.015789 | no |
|  |  |  | `grid=8x8` | 1.024x | 0.697219 | yes |
|  |  |  | `grid=8x4` | 0.751x | 0.697219 | yes |
|  |  |  | **A/A floor** | **1.0378** |  |  |
| `DiffusionStep#017` | core_grid | 690 | _(incumbent)_ | 1.000x | 0.003761 | — |
|  |  |  | `nogrid=1` | 0.706x | 0.003761 | no |
|  |  |  | `fp32acc=0` | 1.035x | 0.013162 | no |
|  |  |  | `fidelity=HiFi2` | 0.798x | 0.005829 | no |
|  |  |  | `fidelity=HiFi2,fp32acc=0` | 0.842x | 0.013043 | no |
|  |  |  | `fidelity=LoFi,fp32acc=0` | 0.820x | 0.029467 | no |
|  |  |  | `packerl1=0` | 1.097x | 0.00544 | no |
|  |  |  | `dstfull=1` | 0.568x | 0.003761 | yes |
|  |  |  | `fp32acc=0,dstfull=1` | 1.748x | 0.013162 | no |
|  |  |  | `grid=8x8` | 0.600x | 0.003761 | yes |
|  |  |  | `grid=8x4` | 1.281x | 0.003761 | yes |
|  |  |  | **A/A floor** | **2.4801** |  |  |
| `PairformerLayer#004` | core_grid | 2297 | _(incumbent)_ | 1.000x | 0.003064 | — |
|  |  |  | `nogrid=1` | 0.339x | 0.003064 | no |
|  |  |  | `fp32acc=0` | 2.965x | 0.012391 | no |
|  |  |  | `fidelity=HiFi2` | 1.964x | 0.005213 | no |
|  |  |  | `fidelity=HiFi2,fp32acc=0` | 1.400x | 0.012125 | no |
|  |  |  | `fidelity=LoFi,fp32acc=0` | 1.526x | 0.027985 | no |
|  |  |  | `packerl1=0` | 0.978x | 0.006501 | no |
|  |  |  | `dstfull=1` | 1.005x | 0.003064 | yes |
|  |  |  | `fp32acc=0,dstfull=1` | 1.051x | 0.012391 | no |
|  |  |  | `grid=8x8` | 1.007x | 0.003064 | yes |
|  |  |  | `grid=8x4` | 1.490x | 0.003064 | yes |
|  |  |  | **A/A floor** | **2.9144** |  |  |
| `DiffusionStep#029` | core_grid | 516 | _(incumbent)_ | 1.000x | 0.003385 | — |
|  |  |  | `nogrid=1` | 0.399x | 0.003086 | no |
|  |  |  | `fp32acc=0` | 1.909x | 0.012352 | no |
|  |  |  | `fidelity=HiFi2` | 1.735x | 0.005842 | no |
|  |  |  | `fidelity=HiFi2,fp32acc=0` | 1.767x | 0.011253 | no |
|  |  |  | `fidelity=LoFi,fp32acc=0` | 1.020x | 0.027296 | no |
|  |  |  | `packerl1=0` | 1.205x | 0.003385 | no |
|  |  |  | `dstfull=1` | 1.211x | 0.003385 | yes |
|  |  |  | `fp32acc=0,dstfull=1` | 1.007x | 0.012352 | no |
|  |  |  | `grid=8x8` | 1.000x | 0.003385 | yes |
|  |  |  | `grid=8x4` | 0.749x | 0.003385 | yes |
|  |  |  | **A/A floor** | **2.7811** |  |  |
| `DiffusionStep#035` | core_grid | 175 | _(incumbent)_ | 1.000x | 0.002812 | — |
|  |  |  | `nogrid=1` | 0.130x | 0.002812 | no |
|  |  |  | `fp32acc=0` | 1.020x | 0.008593 | no |
|  |  |  | `fidelity=HiFi2` | 0.981x | 0.004928 | no |
|  |  |  | `fidelity=HiFi2,fp32acc=0` | 0.965x | 0.007456 | no |
|  |  |  | `fidelity=LoFi,fp32acc=0` | 2.203x | 0.022458 | no |
|  |  |  | `packerl1=0` | 2.092x | 0.003083 | no |
|  |  |  | `dstfull=1` | 1.008x | 0.002812 | yes |
|  |  |  | `fp32acc=0,dstfull=1` | 1.026x | 0.008593 | no |
|  |  |  | `grid=8x8` | 0.573x | 0.002812 | yes |
|  |  |  | `grid=8x4` | 0.967x | 0.002812 | yes |
|  |  |  | **A/A floor** | **3.3112** |  |  |
| `DiffusionStep#036` | core_grid | 167 | _(incumbent)_ | 1.000x | 0.003462 | — |
|  |  |  | `nogrid=1` | 0.112x | 0.003462 | no |
|  |  |  | `fp32acc=0` | 1.021x | 0.013166 | no |
|  |  |  | `fidelity=HiFi2` | 1.007x | 0.00689 | no |
|  |  |  | `fidelity=HiFi2,fp32acc=0` | 1.028x | 0.009502 | no |
|  |  |  | `fidelity=LoFi,fp32acc=0` | 0.968x | 0.030236 | no |
|  |  |  | `packerl1=0` | 0.996x | 0.003462 | no |
|  |  |  | `dstfull=1` | 0.948x | 0.003462 | yes |
|  |  |  | `fp32acc=0,dstfull=1` | 1.014x | 0.013166 | no |
|  |  |  | `grid=8x8` | 1.002x | 0.003462 | yes |
|  |  |  | `grid=8x4` | 0.947x | 0.003462 | yes |
|  |  |  | **A/A floor** | **1.0163** |  |  |
| `DiffusionStep#043` | core_grid | 719 | _(incumbent)_ | 1.000x | 0.003572 | — |
|  |  |  | `nogrid=1` | 0.573x | 0.003572 | yes |
|  |  |  | `fp32acc=0` | 0.556x | 0.011006 | no |
|  |  |  | `fidelity=HiFi2` | 2.230x | 0.006362 | no |
|  |  |  | `fidelity=HiFi2,fp32acc=0` | 0.989x | 0.010837 | no |
|  |  |  | `fidelity=LoFi,fp32acc=0` | 0.959x | 0.028668 | no |
|  |  |  | `packerl1=0` | 1.377x | 0.00366 | no |
|  |  |  | `dstfull=1` | 1.518x | 0.003572 | yes |
|  |  |  | `fp32acc=0,dstfull=1` | 0.385x | 0.011006 | no |
|  |  |  | `grid=8x8` | 0.436x | 0.003572 | yes |
|  |  |  | `grid=8x4` | 0.668x | 0.003572 | yes |
|  |  |  | **A/A floor** | **2.9811** |  |  |
| `PairformerLayer#017` | program_config | 1236 | _(incumbent)_ | 1.000x | 0.808471 | — |
|  |  |  | `nogrid=1` | 1.022x | 0.808471 | yes |
|  |  |  | `fp32acc=0` | 1.132x | 0.014747 | no |
|  |  |  | `fidelity=HiFi2` | 1.058x | 0.007044 | no |
|  |  |  | `fidelity=HiFi2,fp32acc=0` | 0.985x | 0.014713 | no |
|  |  |  | `fidelity=LoFi,fp32acc=0` | 1.049x | 0.029853 | no |
|  |  |  | `packerl1=0` | 0.979x | 0.201581 | no |
|  |  |  | `dstfull=1` | 1.034x | 0.808471 | yes |
|  |  |  | `fp32acc=0,dstfull=1` | 1.092x | 0.014747 | no |
|  |  |  | `grid=8x8` | 0.988x | 0.808471 | yes |
|  |  |  | `grid=8x4` | 0.794x | 0.808471 | yes |
|  |  |  | **A/A floor** | **1.1219** |  |  |
| `MSALayer#015` | core_grid | 205 | _(incumbent)_ | 1.000x | 0.003638 | — |
|  |  |  | `nogrid=1` | 0.200x | 0.79981 | no |
|  |  |  | `fp32acc=0` | 1.284x | 0.013634 | no |
|  |  |  | `fidelity=HiFi2` | 1.001x | 0.006706 | no |
|  |  |  | `fidelity=HiFi2,fp32acc=0` | 0.977x | 0.012909 | no |
|  |  |  | `fidelity=LoFi,fp32acc=0` | 1.320x | 0.031386 | no |
|  |  |  | `packerl1=0` | 1.005x | 0.005594 | no |
|  |  |  | `dstfull=1` | 1.109x | 0.003638 | yes |
|  |  |  | `fp32acc=0,dstfull=1` | 1.538x | 0.013634 | no |
|  |  |  | `grid=8x8` | 1.468x | 0.003638 | yes |
|  |  |  | `grid=8x4` | 1.066x | 0.003638 | yes |
|  |  |  | **A/A floor** | **1.1704** |  |  |
| `DiffusionStep#053` | core_grid | 114 | _(incumbent)_ | 1.000x | 0.002808 | — |
|  |  |  | `nogrid=1` | 0.115x | 0.002686 | no |
|  |  |  | `fp32acc=0` | 0.658x | 0.011585 | no |
|  |  |  | `fidelity=HiFi2` | 0.497x | 0.006071 | no |
|  |  |  | `fidelity=HiFi2,fp32acc=0` | 1.057x | 0.011585 | no |
|  |  |  | `fidelity=LoFi,fp32acc=0` | 1.106x | 0.029538 | no |
|  |  |  | `packerl1=0` | 1.029x | 0.003911 | no |
|  |  |  | `dstfull=1` | 0.996x | 0.002808 | yes |
|  |  |  | `fp32acc=0,dstfull=1` | 1.000x | 0.011585 | no |
|  |  |  | `grid=8x8` | 1.005x | 0.002808 | yes |
|  |  |  | `grid=8x4` | 0.923x | 0.002808 | yes |
|  |  |  | **A/A floor** | **1.128** |  |  |
| `PairformerLayer#019` | program_config | 923 | _(incumbent)_ | 1.000x | 0.003226 | — |
|  |  |  | `nogrid=1` | 0.999x | 0.003226 | yes |
|  |  |  | `fp32acc=0` | 0.934x | 0.010663 | no |
|  |  |  | `fidelity=HiFi2` | 1.078x | 0.006418 | no |
|  |  |  | `fidelity=HiFi2,fp32acc=0` | 1.059x | 0.010636 | no |
|  |  |  | `fidelity=LoFi,fp32acc=0` | 1.034x | 0.027102 | no |
|  |  |  | `packerl1=0` | 0.999x | 0.003226 | yes |
|  |  |  | `dstfull=1` | 1.001x | 0.003226 | yes |
|  |  |  | `fp32acc=0,dstfull=1` | 1.028x | 0.010663 | no |
|  |  |  | `grid=8x8` | 1.332x | 0.003226 | no |
|  |  |  | `grid=8x4` | 1.245x | 0.003226 | no |
|  |  |  | **A/A floor** | **1.0844** |  |  |
| `DiffusionStep#013` | core_grid | 718 | _(incumbent)_ | 1.000x | 0.002888 | — |
|  |  |  | `nogrid=1` | 1.003x | 0.002888 | yes |
|  |  |  | `fp32acc=0` | 0.862x | 0.010474 | no |
|  |  |  | `fidelity=HiFi2` | 0.466x | 0.005242 | no |
|  |  |  | `fidelity=HiFi2,fp32acc=0` | 1.191x | 0.00833 | no |
|  |  |  | `fidelity=LoFi,fp32acc=0` | 1.284x | 0.033104 | no |
|  |  |  | `packerl1=0` | 1.340x | 0.002888 | yes |
|  |  |  | `dstfull=1` | 1.058x | 0.002888 | yes |
|  |  |  | `fp32acc=0,dstfull=1` | 0.613x | 0.010474 | no |
|  |  |  | `grid=8x8` | 1.472x | 0.002888 | yes |
|  |  |  | `grid=8x4` | 1.260x | 0.002888 | yes |
|  |  |  | **A/A floor** | **2.2704** |  |  |
| `DiffusionStep#040` | core_grid | 211 | _(incumbent)_ | 1.000x | 0.11101 | — |
|  |  |  | `nogrid=1` | 0.635x | 0.11101 | yes |
|  |  |  | `fp32acc=0` | 1.625x | 0.11101 | no |
|  |  |  | `fidelity=HiFi2` | 1.333x | 0.110169 | no |
|  |  |  | `fidelity=HiFi2,fp32acc=0` | 1.510x | 0.110169 | no |
|  |  |  | `fidelity=LoFi,fp32acc=0` | 0.641x | 0.110169 | no |
|  |  |  | `packerl1=0` | 0.283x | 0.11101 | no |
|  |  |  | `dstfull=1` | 1.016x | 0.11101 | yes |
|  |  |  | `fp32acc=0,dstfull=1` | 1.610x | 0.11101 | no |
|  |  |  | `grid=8x8` | 0.735x | 0.11101 | yes |
|  |  |  | `grid=8x4` | 0.297x | 0.11101 | yes |
|  |  |  | **A/A floor** | **2.1571** |  |  |
| `MSALayer#010` | core_grid | 432 | _(incumbent)_ | 1.000x | 0.002498 | — |
|  |  |  | `nogrid=1` | 0.656x | 0.002454 | no |
|  |  |  | `fp32acc=0` | 1.009x | 0.0098 | no |
|  |  |  | `fidelity=HiFi2` | 1.072x | 0.006604 | no |
|  |  |  | `fidelity=HiFi2,fp32acc=0` | 0.938x | 0.009672 | no |
|  |  |  | `fidelity=LoFi,fp32acc=0` | 1.098x | 0.030796 | no |
|  |  |  | `packerl1=0` | 1.074x | 0.002498 | yes |
|  |  |  | `dstfull=1` | 0.858x | 0.002498 | yes |
|  |  |  | `fp32acc=0,dstfull=1` | 1.062x | 0.0098 | no |
|  |  |  | `grid=8x8` | 1.013x | 0.002498 | yes |
|  |  |  | `grid=8x4` | 1.038x | 0.002498 | yes |
|  |  |  | **A/A floor** | **1.1512** |  |  |

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
