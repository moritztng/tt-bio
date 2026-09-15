# Boltz-2 512 aa at f072ae02f: the roofline budget

qb2 card 2 (p300c, 11x10 grid, AICLK 800 MHz), ttnn 0.68.0, one process, one session. FLOPs and bytes come off the same 26 captures; times come off the same process's bracketed fold, per unit by the `trimmed` rule. Session fold 17.270 s at loadavg 2.22, 1.34, 0.59; the benchlocked cell of record is 17.270 s, so seconds are also given scaled by 1.0000.

Roofs measured in this session with dispatch amortised: dense bf16 HiFi4 **104.93 TFLOP/s**, starved 8192^2 add **424.7 GB/s**. Machine balance 247.1 FLOP/byte.

| unit | calls | ms/call | GFLOP/call | MB/call | FLOP/byte | binding roof | % of it | s/fold | **s above roof** | at the cell |
|---|---|---|---|---|---|---|---|---|---|---|
| `PairformerLayer|1x512x384,1x512x512x128` | 264 | 32.987 | 508.2 | 6758.3 | 75.19 | bandwidth | 48.2 % | 8.709 | **4.508** | 4.508 |
| `DiffusionModule|` | 200 | 27.223 | 317.2 | 4035.8 | 78.59 | bandwidth | 34.9 % | 5.445 | **3.544** | 3.544 |
| `Diffusion|1x4480x3,1` | 200 | 26.707 | 317.2 | 4035.5 | 78.6 | bandwidth | 35.6 % | 5.341 | **3.441** | 3.441 |
| `DiffusionTransformer|1x512x768,1x512x768` | 200 | 19.840 | 290.0 | 2894.1 | 100.2 | bandwidth | 34.3 % | 3.968 | **2.605** | 2.605 |
| `DiffusionTransformerLayer|1x512x768,1x512x768` | 4800 | 0.811 | 12.1 | 121.3 | 99.58 | bandwidth | 35.2 % | 3.894 | **2.523** | 2.523 |
| `Transition|1x512x512x128` | 280 | 7.939 | 103.1 | 340.0 | 303.26 | compute | 12.4 % | 2.223 | **1.948** | 1.948 |
| `TriangleMultiplication|1x512x512x128,1x512x512` | 528 | 6.348 | 86.0 | 1745.6 | 49.25 | bandwidth | 64.7 % | 3.352 | **1.182** | 1.182 |
| `MSALayer|1x512x512x128,1x1024x512x64` | 16 | 114.145 | 1368.3 | 22100.2 | 61.91 | bandwidth | 45.6 % | 1.826 | **0.994** | 0.994 |
| `TriangleAttention|1x512x512x128,1x1x1x512` | 528 | 4.574 | 113.9 | 1183.0 | 96.24 | bandwidth | 60.9 % | 2.415 | **0.945** | 0.945 |
| `ConditionedTransitionBlock|1x512x768,1x512x768` | 4800 | 0.296 | 6.6 | 52.4 | 126.7 | bandwidth | 41.7 % | 1.422 | **0.829** | 0.829 |
| `AttentionPairBias|1x512x768,1x16x512x512` | 4800 | 0.272 | 3.6 | 44.0 | 82.43 | bandwidth | 38.1 % | 1.305 | **0.808** | 0.808 |
| `AdaLN|1x512x768,1x512x768` | 9600 | 0.116 | 1.2 | 15.0 | 80.37 | bandwidth | 30.5 % | 1.116 | **0.776** | 0.776 |
| `DiffusionTransformer|1x140x32x128,1x140x32x128` | 400 | 3.019 | 8.8 | 534.6 | 16.53 | bandwidth | 41.7 % | 1.207 | **0.704** | 0.704 |
| `DiffusionTransformerLayer|1x140x32x128,1x140x32x128` | 1200 | 0.990 | 3.7 | 186.0 | 19.78 | bandwidth | 44.3 % | 1.187 | **0.662** | 0.662 |
| `OuterProductMean|1x1024x512x64,1024x1x1` | 16 | 42.764 | 622.8 | 7319.3 | 85.09 | bandwidth | 40.3 % | 0.684 | **0.408** | 0.408 |
| `AttentionPairBias|1x140x32x128,140x4x32x128` | 1200 | 0.523 | 1.6 | 117.9 | 13.75 | bandwidth | 53.1 % | 0.628 | **0.295** | 0.295 |
| `PairformerLayer|1x512x512x128` | 16 | 31.701 | 502.7 | 6389.0 | 78.69 | bandwidth | 47.5 % | 0.507 | **0.267** | 0.267 |
| `ConditionedTransitionBlock|1x140x32x128,1x140x32x128` | 1200 | 0.283 | 1.6 | 48.6 | 33.28 | bandwidth | 40.4 % | 0.34 | **0.203** | 0.203 |
| `PairWeightedAveraging|1x1024x512x64,1x512x512x128` | 16 | 29.422 | 191.2 | 7181.7 | 26.62 | bandwidth | 57.5 % | 0.471 | **0.2** | 0.2 |
| `AttentionPairBias|1x512x384,1x512x512x128` | 264 | 1.076 | 3.6 | 151.4 | 23.97 | bandwidth | 33.1 % | 0.284 | **0.19** | 0.19 |
| `Transition|1x1024x512x64` | 16 | 10.424 | 51.6 | 606.2 | 85.08 | bandwidth | 13.7 % | 0.167 | **0.144** | 0.144 |
| `AdaLN|1x140x32x128,1x140x32x128` | 2400 | 0.071 | 0.3 | 11.6 | 25.52 | bandwidth | 38.3 % | 0.171 | **0.105** | 0.105 |
| `TriangleMultiplication|1x512x512x128` | 32 | 5.996 | 86.0 | 1610.8 | 53.37 | bandwidth | 63.3 % | 0.192 | **0.071** | 0.071 |
| `TriangleAttention|1x512x512x128` | 32 | 4.676 | 113.8 | 1178.8 | 96.58 | bandwidth | 59.4 % | 0.15 | **0.061** | 0.061 |
| `Transition|1x512x768` | 400 | 0.124 | 3.6 | 9.5 | 380.08 | compute | 27.8 % | 0.05 | **0.036** | 0.036 |
| `Transition|1x512x384` | 264 | 0.129 | 1.8 | 4.8 | 380.08 | compute | 13.4 % | 0.034 | **0.03** | 0.03 |

## The three headline numbers

- **219.49 TFLOP executed** over the three disjoint top-level units. Tile padding accounts for 0.15 % of it.
- **2.9449 TB moved**, deduped on buffer address, of which 0.0011 TB is the counter's terminal-output charge (an upper bound on its overcount).
- **floor 6.934 s, set by bandwidth.** Every unit but the pair Transition has an arithmetic intensity under the 247 FLOP/byte machine balance, so the traffic term binds: 2.9449 TB at 424.7 GB/s.

The arithmetic side, three ways, two of them fictions:

| priced at | s | why |
|---|---|---|
| the best measured dense-cube HiFi4 rate, 104.93 TFLOP/s | 2.092 | no op in this fold is a dense cube |
| each shape as a standalone ttnn.matmul | 43.393 | prices the fold's fused kernels as separate DRAM round trips, so it lands above the fold itself |
| the traffic those same shapes move | 6.934 | the one that binds |

## What is padding and what is work

The MSA axis is padded to 1024 rows. `MSA_PAD_MULTIPLE = 1024` in `tt_bio/tenstorrent.py`, and this fixture has 35 MSA rows, so the MSA block executes 1368.3 GFLOP per call where FlopCounterMode counts 595.2 logical. The three units inside it whose shapes carry the depth axis move 15107.2 MB of the block's 22100.2 MB (68.4 %) and 1.322 s of the 17.270 s cell. A finer MSA ladder is the lever; how much of that is recoverable is a measurement someone else has to take.

Tile padding is not where the FLOPs go: 0.15 %. The atom axis moved the other way since `flops_bytes_512.json` was written, 4480 atoms at the tip against 7168 there, so the atom transformer term is 1.28x its logical count rather than the up to 4x that file self-declares.

