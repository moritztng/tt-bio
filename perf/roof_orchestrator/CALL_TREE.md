# The fold's call tree, and the 2.671 s nobody owns

`roof-budget`'s table is 26 flat rows that mix top-level units, intermediate units and leaves.
`DiffusionModule`, `Diffusion` and `DiffusionTransformer` carry 317.2 / 317.2 / 290.0 GFLOP a call,
which is what nesting looks like, so **any ranking that sums rows double-counts unless the tree is
known**. The summary sums exactly three rows; nothing said which of the other 23 may be added to
what.

`tree.py` asserts the parent/child edges from call-count ratios and time containment, then checks
them. Output: `tree_512_qb2c2.txt`. No new measurement — this reads the committed table only.

## The check

Every parent's children sum to less than the parent, and the leftover is reported rather than
hidden. Two independent closures:

- **Leaves + unattributed = 10.506 s against a cell-above-floor of 10.406 s, a 1.0 % gap** — and the
  gap has the right sign and size, being the at-roof share of the unattributed traffic.
- The three top-level units sum to 24.644 s against an independently bracketed 24.731 s session
  fold, 0.35 % unaccounted.

One row spans two parents and is the only one that does: `Transition|1x512x512x128` is 280 calls =
the trunk pairformer's 264 plus the MSA-side pairformer's 16, split 264/280 and 16/280 here.

## The tree

    PairformerLayer|1x512x384,1x512x512x128      264  13.708 s/fold
      TriangleMultiplication|...,1x512x512       528   5.175
      TriangleAttention|...,1x1x1x512            528   3.929
      Transition|1x512x512x128         (264/280)       3.397
      AttentionPairBias|1x512x384,...            264   0.458
      Transition|1x512x384                       264   0.052
    DiffusionModule|                             200   8.226
      Diffusion|1x4480x3,1                       200   8.068
        DiffusionTransformer|1x512x768           200   5.439
          DiffusionTransformerLayer|1x512x768   4800   4.021
            ConditionedTransitionBlock          4800   1.454
            AttentionPairBias                   4800   1.336
            AdaLN                               9600   1.131
        DiffusionTransformer|1x140x32x128        400   1.318
          DiffusionTransformerLayer|1x140       1200   1.276
            AttentionPairBias|1x140             1200   0.665
            ConditionedTransitionBlock|1x140    1200   0.356
            AdaLN|1x140                         2400   0.190
        Transition|1x512x768                     400   0.053
    MSALayer|1x512x512x128,1x1024x512x64          16   2.710
      OuterProductMean|1x1024x512x64              16   0.891
      PairformerLayer|1x512x512x128               16   0.817
        TriangleMultiplication|1x512x512x128      32   0.300
        TriangleAttention|1x512x512x128           32   0.271
        Transition|1x512x512x128       (16/280)         0.206
      PairWeightedAveraging|1x1024x512x64         16   0.734
      Transition|1x1024x512x64                    16   0.236

## The 17 leaves, the only rows rankable without double-counting

Seconds above the binding roof, at the 17.340 s cell. Roofs measured: 424.7 GB/s stream,
104.93 TFLOP/s dense bf16, machine balance 247.1 FLOP/byte.

| leaf | s above roof | % of its binding roof |
|---|---|---|
| `Transition\|1x512x512x128` | **2.251** | 7.6 % compute |
| `TriangleMultiplication\|...,1x512x512` | **1.458** | 41.9 % bandwidth |
| `TriangleAttention\|...,1x1x1x512` | **1.284** | 37.4 % bandwidth |
| `AdaLN\|1x512x768` | 0.453 | 30.1 % |
| `AttentionPairBias\|1x512x768` | 0.440 | 37.2 % |
| `ConditionedTransitionBlock\|1x512x768` | 0.426 | 40.8 % |
| `OuterProductMean\|1x1024x512x64` | 0.349 | 30.9 % |
| `PairWeightedAveraging\|1x1024x512x64` | 0.244 | 36.8 % |
| `AttentionPairBias\|1x512x384` | 0.227 | 20.6 % |
| `Transition\|1x1024x512x64` | 0.143 | 9.7 % |
| `AttentionPairBias\|1x140x32x128` | 0.134 | 50.0 % |
| `ConditionedTransitionBlock\|1x140` | 0.113 | 38.5 % |
| `TriangleAttention\|1x512x512x128` | 0.101 | 32.8 % |
| `TriangleMultiplication\|1x512x512x128` | 0.089 | 40.4 % |
| `AdaLN\|1x140x32x128` | 0.068 | 34.3 % |
| `Transition\|1x512x384` | 0.032 | 8.8 % compute |
| `Transition\|1x512x768` | 0.023 | 26.0 % compute |
| **leaf total** | **7.835** | |

## The finding: 2.671 s, a quarter of the prize, is in no op class at all

    leaves                     7.835 s   75.3 %
    unattributed in parents    2.671 s   24.7 %
    cell above floor          10.406 s

Per parent, at the cell — time inside a unit that none of its measured children explains:

| parent | unattributed | what is missing |
|---|---|---|
| `DiffusionTransformer\|1x512x768` | **0.994 s** | everything in the token DiT outside its 24 layers |
| `Diffusion\|1x4480x3,1` | **0.882 s** | everything in the denoiser outside both transformers |
| `PairformerLayer\|1x512x384,...` | **0.489 s** | outside trimul / triatt / Transition / AttentionPairBias |
| `DiffusionModule\|` | 0.111 s | |
| `DiffusionTransformerLayer\|1x512x768` | 0.070 s | outside AdaLN / APB / CTB |
| `DiffusionTransformerLayer\|1x140` | 0.046 s | |
| `DiffusionTransformer\|1x140` | 0.029 s | |
| `PairformerLayer\|1x512x512x128` | 0.028 s | |
| `MSALayer\|...` | 0.022 s | |

By path: **diffusion 2.132 s, pairformer 0.517 s, MSA 0.022 s.**

**This is coverage, not error.** The units are module-level marks, so an op that sits in a parent but
in no marked submodule lands in the residual and is invisible to the table. The fold is 98.9 %
in-kernel with a 1.1 % dispatch gap, so this is device work doing something, not host time. The
campaign's mandate — every op class at its binding roof or a named structural reason — **cannot be
satisfied while a quarter of the prize has no op class**, and the two largest single items in it
(0.994 s and 0.882 s, both in the token diffusion path) are each larger than every leaf except the
top three.

## What this changes about the queue

The five Phase A rows dispatched in pass 2 sit against leaves: `Transition|1x512x512x128` (2.251 s,
`roof-pair-transition`), the MSA depth-axis units (`roof-msa-ladder`), `TriangleAttention`'s
projection (`roof-qkv-sdpa-build`) and `AttentionPairBias|1x512x768` (`roof-dit-sdpa`). Between them
they cover roughly 4.5 s of the 7.835 s leaf total and **none of the 2.671 s**. Naming the residual
is now the largest single unowned item in the campaign and is dispatched as `roof-residual-census`.
