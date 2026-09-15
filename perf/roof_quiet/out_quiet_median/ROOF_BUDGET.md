# Boltz-2 512 aa at f072ae02f: the roofline budget

qb2 card 2 (p300c, 11x10 grid, AICLK 800 MHz), ttnn 0.68.0, one process, one session. FLOPs and bytes come off the same 26 captures; times come off the same process's bracketed fold, per unit by the `median` rule. Session fold 17.270 s at loadavg 2.22, 1.34, 0.59; the benchlocked cell of record is 17.270 s, so seconds are also given scaled by 1.0000.

Roofs measured in this session with dispatch amortised: dense bf16 HiFi4 **104.93 TFLOP/s**, starved 8192^2 add **424.7 GB/s**. Machine balance 247.1 FLOP/byte.

| unit | calls | ms/call | GFLOP/call | MB/call | FLOP/byte | binding roof | % of it | s/fold | **s above roof** | at the cell |
|---|---|---|---|---|---|---|---|---|---|---|
| `PairformerLayer|1x512x384,1x512x512x128` | 264 | 32.990 | 508.2 | 6758.3 | 75.19 | bandwidth | 48.2 % | 8.709 | **4.508** | 4.508 |
| `DiffusionModule|` | 200 | 27.066 | 317.2 | 4035.8 | 78.59 | bandwidth | 35.1 % | 5.413 | **3.513** | 3.513 |
| `Diffusion|1x4480x3,1` | 200 | 26.582 | 317.2 | 4035.5 | 78.6 | bandwidth | 35.7 % | 5.316 | **3.416** | 3.416 |
| `DiffusionTransformer|1x512x768,1x512x768` | 200 | 19.773 | 290.0 | 2894.1 | 100.2 | bandwidth | 34.5 % | 3.955 | **2.592** | 2.592 |
| `DiffusionTransformerLayer|1x512x768,1x512x768` | 4800 | 0.810 | 12.1 | 121.3 | 99.58 | bandwidth | 35.3 % | 3.888 | **2.517** | 2.517 |
| `Transition|1x512x512x128` | 280 | 7.900 | 103.1 | 340.0 | 303.26 | compute | 12.4 % | 2.212 | **1.937** | 1.937 |
| `TriangleAttention|1x512x512x128,1x1x1x512` | 528 | 5.217 | 113.9 | 1183.0 | 96.24 | bandwidth | 53.4 % | 2.755 | **1.284** | 1.284 |
| `TriangleMultiplication|1x512x512x128,1x512x512` | 528 | 6.357 | 86.0 | 1745.6 | 49.25 | bandwidth | 64.7 % | 3.356 | **1.186** | 1.186 |
| `MSALayer|1x512x512x128,1x1024x512x64` | 16 | 112.625 | 1368.3 | 22100.2 | 61.91 | bandwidth | 46.2 % | 1.802 | **0.969** | 0.969 |
| `ConditionedTransitionBlock|1x512x768,1x512x768` | 4800 | 0.296 | 6.6 | 52.4 | 126.7 | bandwidth | 41.8 % | 1.419 | **0.827** | 0.827 |
| `AttentionPairBias|1x512x768,1x16x512x512` | 4800 | 0.273 | 3.6 | 44.0 | 82.43 | bandwidth | 38.0 % | 1.308 | **0.811** | 0.811 |
| `AdaLN|1x512x768,1x512x768` | 9600 | 0.116 | 1.2 | 15.0 | 80.37 | bandwidth | 30.5 % | 1.115 | **0.775** | 0.775 |
| `DiffusionTransformer|1x140x32x128,1x140x32x128` | 400 | 2.971 | 8.8 | 534.6 | 16.53 | bandwidth | 42.4 % | 1.188 | **0.685** | 0.685 |
| `DiffusionTransformerLayer|1x140x32x128,1x140x32x128` | 1200 | 0.975 | 3.7 | 186.0 | 19.78 | bandwidth | 44.9 % | 1.17 | **0.644** | 0.644 |
| `OuterProductMean|1x1024x512x64,1024x1x1` | 16 | 41.025 | 622.8 | 7319.3 | 85.09 | bandwidth | 42.0 % | 0.656 | **0.381** | 0.381 |
| `AttentionPairBias|1x140x32x128,140x4x32x128` | 1200 | 0.517 | 1.6 | 117.9 | 13.75 | bandwidth | 53.7 % | 0.62 | **0.287** | 0.287 |
| `PairformerLayer|1x512x512x128` | 16 | 31.235 | 502.7 | 6389.0 | 78.69 | bandwidth | 48.2 % | 0.5 | **0.259** | 0.259 |
| `ConditionedTransitionBlock|1x140x32x128,1x140x32x128` | 1200 | 0.280 | 1.6 | 48.6 | 33.28 | bandwidth | 40.8 % | 0.336 | **0.199** | 0.199 |
| `PairWeightedAveraging|1x1024x512x64,1x512x512x128` | 16 | 29.170 | 191.2 | 7181.7 | 26.62 | bandwidth | 58.0 % | 0.467 | **0.196** | 0.196 |
| `AttentionPairBias|1x512x384,1x512x512x128` | 264 | 1.076 | 3.6 | 151.4 | 23.97 | bandwidth | 33.1 % | 0.284 | **0.19** | 0.19 |
| `Transition|1x1024x512x64` | 16 | 9.538 | 51.6 | 606.2 | 85.08 | bandwidth | 15.0 % | 0.153 | **0.13** | 0.13 |
| `AdaLN|1x140x32x128,1x140x32x128` | 2400 | 0.069 | 0.3 | 11.6 | 25.52 | bandwidth | 39.4 % | 0.166 | **0.1** | 0.1 |
| `TriangleAttention|1x512x512x128` | 32 | 5.239 | 113.8 | 1178.8 | 96.58 | bandwidth | 53.0 % | 0.168 | **0.079** | 0.079 |
| `TriangleMultiplication|1x512x512x128` | 32 | 5.979 | 86.0 | 1610.8 | 53.37 | bandwidth | 63.4 % | 0.191 | **0.07** | 0.07 |
| `Transition|1x512x768` | 400 | 0.122 | 3.6 | 9.5 | 380.08 | compute | 28.2 % | 0.049 | **0.035** | 0.035 |
| `Transition|1x512x384` | 264 | 0.127 | 1.8 | 4.8 | 380.08 | compute | 13.6 % | 0.033 | **0.029** | 0.029 |

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

The MSA axis is padded to 1024 rows. `MSA_PAD_MULTIPLE = 1024` in `tt_bio/tenstorrent.py`, and this fixture has 35 MSA rows, so the MSA block executes 1368.3 GFLOP per call where FlopCounterMode counts 595.2 logical. The three units inside it whose shapes carry the depth axis move 15107.2 MB of the block's 22100.2 MB (68.4 %) and 1.276 s of the 17.270 s cell. A finer MSA ladder is the lever; how much of that is recoverable is a measurement someone else has to take.

Tile padding is not where the FLOPs go: 0.15 %. The atom axis moved the other way since `flops_bytes_512.json` was written, 4480 atoms at the tip against 7168 there, so the atom transformer term is 1.28x its logical count rather than the up to 4x that file self-declares.

