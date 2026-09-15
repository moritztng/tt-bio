# The 2.671 s in no op class: named, priced, and put on its roof

One instrument, `roof-budget`'s. FLOPs from `exec_flops.py`, bytes from `real_traffic.py` deduped on buffer address, both on the same 26 captures taken on qb2 card 2 at `f072ae02f`; times from the same process's bracketed fold, 17.270 s, scaled by 1.0000 to the 17.270 s cell of record. Roofs measured in that session: **424.7 GB/s** stream, **104.93 TFLOP/s** dense bf16, machine balance 247.1 FLOP/byte.

Every row is one unit's OWN work: the top-level ttnn ops it issues outside every marked child. That tiles the fold, so there is no unattributed column left to report -- the residual is a row.

| unit | | calls | own ops/call | MB/call | GFLOP/call | FLOP/byte | roof | s/fold | at cell | **above roof** | % of roof | at cell, exact sums |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `Transition\|1x512x512x128` | leaf | 280 | 258 | 340.0 | 103.1 | 303.26 | compute | 2.212 | 2.212 | **1.937** | 12.4 % |  |
| `TriangleAttention\|1x512x512x128,1x1x1x512` | leaf | 528 | 26 | 1183.0 | 113.9 | 96.24 | bandwidth | 2.755 | 2.755 | **1.284** | 53.4 % |  |
| `TriangleMultiplication\|1x512x512x128,1x512x512` | leaf | 528 | 23 | 1745.6 | 86.0 | 49.25 | bandwidth | 3.356 | 3.356 | **1.186** | 64.7 % |  |
| `AttentionPairBias\|1x512x768,1x16x512x512` | leaf | 4800 | 17 | 44.0 | 3.6 | 82.43 | bandwidth | 1.308 | 1.308 | **0.811** | 38.0 % |  |
| `AdaLN\|1x512x768,1x512x768` | leaf | 9600 | 9 | 15.0 | 1.2 | 80.37 | bandwidth | 1.115 | 1.115 | **0.775** | 30.5 % |  |
| `DiffusionTransformerLayer\|1x512x768,1x512x768` | parent | 4800 | 4 | 10.7 | 0.6 | 56.74 | bandwidth | 0.603 | 0.603 | **0.483** | 20.0 % | 0.357 |
| `ConditionedTransitionBlock\|1x512x768,1x512x768` | parent | 4800 | 12 | 37.4 | 5.4 | 145.32 | bandwidth | 0.862 | 0.862 | **0.439** | 49.1 % | 0.825 |
| `OuterProductMean\|1x1024x512x64,1024x1x1` | leaf | 16 | 24 | 7319.3 | 622.8 | 85.09 | bandwidth | 0.656 | 0.656 | **0.380** | 42.0 % |  |
| `AttentionPairBias\|1x140x32x128,140x4x32x128` | leaf | 1200 | 41 | 117.9 | 1.6 | 13.75 | bandwidth | 0.620 | 0.620 | **0.287** | 53.7 % |  |
| `PairWeightedAveraging\|1x1024x512x64,1x512x512x128` | leaf | 16 | 135 | 7181.7 | 191.2 | 26.62 | bandwidth | 0.467 | 0.467 | **0.196** | 57.9 % |  |
| `AttentionPairBias\|1x512x384,1x512x512x128` | leaf | 264 | 31 | 151.4 | 3.6 | 23.97 | bandwidth | 0.284 | 0.284 | **0.190** | 33.1 % |  |
| `ConditionedTransitionBlock\|1x140x32x128,1x140x32x128` | parent | 1200 | 12 | 40.5 | 1.3 | 32.6 | bandwidth | 0.253 | 0.253 | **0.139** | 45.2 % | 0.195 |
| `Transition\|1x1024x512x64` | leaf | 16 | 514 | 606.2 | 51.6 | 85.08 | bandwidth | 0.153 | 0.153 | **0.130** | 14.9 % |  |
| `Diffusion\|1x4480x3,1` | parent | 200 | 21 | 40.7 | 1.3 | 32.75 | bandwidth | 0.124 | 0.124 | **0.105** | 15.4 % | 0.118 |
| `AdaLN\|1x140x32x128,1x140x32x128` | leaf | 2400 | 11 | 11.6 | 0.3 | 25.52 | bandwidth | 0.166 | 0.166 | **0.101** | 39.3 % |  |
| `DiffusionModule\|` | parent | 200 | 2 | 9.9 | 0.0 | 0.0 | bandwidth | 0.097 | 0.097 | **0.092** | 4.8 % | 0.131 |
| `DiffusionTransformerLayer\|1x140x32x128,1x140x32x128` | parent | 1200 | 4 | 14.0 | 0.1 | 10.63 | bandwidth | 0.131 | 0.131 | **0.091** | 30.1 % | 0.107 |
| `TriangleAttention\|1x512x512x128` | leaf | 32 | 25 | 1178.8 | 113.8 | 96.58 | bandwidth | 0.168 | 0.168 | **0.079** | 52.9 % |  |
| `TriangleMultiplication\|1x512x512x128` | leaf | 32 | 20 | 1610.8 | 86.0 | 53.37 | bandwidth | 0.191 | 0.191 | **0.070** | 63.5 % |  |
| `DiffusionTransformer\|1x512x768,1x512x768` | parent | 200 | 0 | 0.0 | 0.0 | 0.0 | bandwidth | 0.067 | 0.067 | **0.067** | 0.0 % | 0.030 |
| `Transition\|1x512x768` | leaf | 400 | 8 | 9.5 | 3.6 | 380.08 | compute | 0.049 | 0.049 | **0.035** | 28.2 % |  |
| `Transition\|1x512x384` | leaf | 264 | 8 | 4.8 | 1.8 | 380.08 | compute | 0.033 | 0.033 | **0.028** | 13.8 % |  |
| `DiffusionTransformer\|1x140x32x128,1x140x32x128` | parent | 400 | 0 | 0.0 | 0.0 | 0.0 | bandwidth | 0.018 | 0.018 | **0.018** | 0.0 % | 0.009 |
| `MSALayer\|1x512x512x128,1x1024x512x64` | parent | 16 | 3 | 806.4 | 0.0 | 0.0 | bandwidth | 0.026 | 0.026 | **-0.004** | 114.6 % | 0.017 |
| `PairformerLayer\|1x512x512x128` | parent | 16 | 10 | 672.1 | 0.0 | 0.0 | bandwidth | 0.015 | 0.015 | **-0.011** | 173.1 % | 0.003 |
| `PairformerLayer\|1x512x384,1x512x512x128` | parent | 264 | 16 | 679.0 | 0.0 | 0.0 | bandwidth | 0.195 | 0.195 | **-0.227** | 216.6 % | 0.465 |

The rows close on the fold: they sum to 15.924 s at the cell against 15.924 s for the three top-level units, a 0.00 % gap. The control is the leaves: all 15 leaf rows reproduce `roof-budget`'s own bytes and FLOPs for the same unit to the last digit, so the only new thing here is the parent/child partition.

**2.391 s of the cell is glue** -- a unit's own ops rather than a child's. 0.085 s of that runs NO device op at all.

## The glue rows, and what is in them

- `DiffusionTransformerLayer|1x512x768,1x512x768` -- 0.603 s at the cell, 10.7 MB and 0.6 GFLOP a call: 2x `add`, 1x `linear`, 1x `multiply`.
- `ConditionedTransitionBlock|1x512x768,1x512x768` -- 0.862 s at the cell, 37.4 MB and 5.4 GFLOP a call: 5x `linear`, 3x `multiply_`.
- `ConditionedTransitionBlock|1x140x32x128,1x140x32x128` -- 0.253 s at the cell, 40.5 MB and 1.3 GFLOP a call: 3x `multiply_`, 5x `linear`.
- `Diffusion|1x4480x3,1` -- 0.124 s at the cell, 40.7 MB and 1.3 GFLOP a call: 5x `add`, 1x `matmul`, 3x `linear`, 2x `layer_norm`, 2x `permute`.
- `DiffusionModule|` -- 0.097 s at the cell, 9.9 MB and 0.0 GFLOP a call: 2x `from_torch`.
- `DiffusionTransformerLayer|1x140x32x128,1x140x32x128` -- 0.131 s at the cell, 14.0 MB and 0.1 GFLOP a call: 2x `add`, 1x `multiply`, 1x `linear`.
- `DiffusionTransformer|1x512x768,1x512x768` -- **no op at all**. 0.067 s at the cell with an empty capture between its children.
- `DiffusionTransformer|1x140x32x128,1x140x32x128` -- **no op at all**. 0.018 s at the cell with an empty capture between its children.
- `MSALayer|1x512x512x128,1x1024x512x64` -- 0.026 s at the cell, 806.4 MB and 0.0 GFLOP a call: 3x `add_`.
- `PairformerLayer|1x512x512x128` -- 0.015 s at the cell, 672.1 MB and 0.0 GFLOP a call: 5x `add_`.
- `PairformerLayer|1x512x384,1x512x512x128` -- 0.195 s at the cell, 679.0 MB and 0.0 GFLOP a call: 7x `add_`, 1x `layer_norm`.

## What this changes

CALL_TREE's two largest unowned items, `DiffusionTransformer|1x512x768` at 0.994 s and `Diffusion|1x4480x3,1` at 0.882 s, are the two rows where three independent things all say the same: the capture holds no op or almost none (0 and 21 ops a call), the ops that are there account for 0.0 % and 2.2 % of the binding roof, and the two timing estimators disagree by 12.6x and 7.1x. Neither is an op class and neither carries a lever.

What is real is smaller and already fast. The trunk `PairformerLayer` runs 7 residual `add_` and 1 `layer_norm` a block, 679.0 MB, and at 216.6 % of the 424.7 GB/s stream roof it is the highest-utilisation row in the fold: -0.227 s above roof out of 0.195 s. The `ConditionedTransitionBlock` pair of rows, 1.115 s at the cell between them, are new only because CALL_TREE ranked the block as a leaf while also charging every one of its AdaLN calls to the layer above it.

## Owed

The timing column is two estimators that bracket the answer rather than one that measures it, because the committed fold records a median and a total per unit and nothing between them. The glue is 2.391 s by one and 2.257 s by the other. Closing that needs the attrib fold re-taken with per-call times kept, which is one 25 s fold. It does not change which rows are device work: that comes from the captures, and they are the committed ones.

Three rows rest on a child frame that ran to the end of its capture, so their op lists are a floor: a child that closes late takes the parent's next ops with it, never the reverse. Bounded, by comparing each such region against the same child's own capture:

- `MSALayer|1x512x512x128,1x1024x512x64` -- the open frame is `PairformerLayer`, 356 ops, and that unit's own capture is 356 ops. Nothing was swallowed; its 3 `add_` is exact.
- `DiffusionTransformer|1x140x32x128` -- three layer regions of 66 ops, 198 in total, against 198 in the unit's own capture. Its zero own ops is exact.
- `Diffusion|1x4480x3,1` -- the atom encoder region is 211 ops, which is 79 + 66 + 66: three layers, the first on the longer `AttentionPairBias` path that the standalone layer capture also took. Nothing swallowed there. The decoder region is 202, four ops past 198, and those four sit at the end of the capture. So this row's glue is 21 to 25 ops and 40.7 to about 45 MB a call, 2.2 to 2.4 % of its roof either way.
