# Boltz-2 512 aa at f072ae02f: the roofline budget

qb2 card 2 (p300c, 11x10 grid, AICLK 800 MHz), ttnn 0.68.0, one process, one session. FLOPs and bytes come off the same 26 captures; times come off the same process's bracketed fold. Session fold 24.731 s at loadavg 28.11, 28.60, 27.72; the benchlocked cell of record is 17.340 s, so seconds are also given scaled by 0.7011.

Roofs measured in this session with dispatch amortised: dense bf16 HiFi4 **104.93 TFLOP/s**, starved 8192^2 add **424.7 GB/s**. Machine balance 247.1 FLOP/byte.

| unit | calls | ms/call | GFLOP/call | MB/call | FLOP/byte | binding roof | % of it | s/fold | **s above roof** | at the cell |
|---|---|---|---|---|---|---|---|---|---|---|
| `PairformerLayer|1x512x384,1x512x512x128` | 264 | 51.924 | 508.2 | 6758.3 | 75.19 | bandwidth | 30.6 % | 13.708 | **9.507** | 6.666 |
| `DiffusionModule|` | 200 | 41.128 | 317.2 | 4035.8 | 78.59 | bandwidth | 23.1 % | 8.226 | **6.325** | 4.435 |
| `Diffusion|1x4480x3,1` | 200 | 40.340 | 317.2 | 4035.5 | 78.6 | bandwidth | 23.6 % | 8.068 | **6.168** | 4.324 |
| `DiffusionTransformer|1x512x768,1x512x768` | 200 | 27.194 | 290.0 | 2894.1 | 100.2 | bandwidth | 25.1 % | 5.439 | **4.076** | 2.858 |
| `Transition|1x512x512x128` | 280 | 12.869 | 103.1 | 340.0 | 303.26 | compute | 7.6 % | 3.603 | **3.328** | 2.333 |
| `TriangleMultiplication|1x512x512x128,1x512x512` | 528 | 9.801 | 86.0 | 1745.6 | 49.25 | bandwidth | 41.9 % | 5.175 | **3.005** | 2.107 |
| `DiffusionTransformerLayer|1x512x768,1x512x768` | 4800 | 0.838 | 12.1 | 121.3 | 99.58 | bandwidth | 34.1 % | 4.021 | **2.649** | 1.858 |
| `TriangleAttention|1x512x512x128,1x1x1x512` | 528 | 7.441 | 113.9 | 1183.0 | 96.24 | bandwidth | 37.4 % | 3.929 | **2.458** | 1.723 |
| `MSALayer|1x512x512x128,1x1024x512x64` | 16 | 169.362 | 1368.3 | 22100.2 | 61.91 | bandwidth | 30.7 % | 2.71 | **1.877** | 1.316 |
| `ConditionedTransitionBlock|1x512x768,1x512x768` | 4800 | 0.303 | 6.6 | 52.4 | 126.7 | bandwidth | 40.8 % | 1.454 | **0.861** | 0.604 |
| `AttentionPairBias|1x512x768,1x16x512x512` | 4800 | 0.278 | 3.6 | 44.0 | 82.43 | bandwidth | 37.2 % | 1.336 | **0.839** | 0.588 |
| `DiffusionTransformer|1x140x32x128,1x140x32x128` | 400 | 3.295 | 8.8 | 534.6 | 16.53 | bandwidth | 38.2 % | 1.318 | **0.815** | 0.571 |
| `AdaLN|1x512x768,1x512x768` | 9600 | 0.118 | 1.2 | 15.0 | 80.37 | bandwidth | 30.1 % | 1.131 | **0.791** | 0.555 |
| `DiffusionTransformerLayer|1x140x32x128,1x140x32x128` | 1200 | 1.063 | 3.7 | 186.0 | 19.78 | bandwidth | 41.2 % | 1.276 | **0.75** | 0.526 |
| `OuterProductMean|1x1024x512x64,1024x1x1` | 16 | 55.719 | 622.8 | 7319.3 | 85.09 | bandwidth | 30.9 % | 0.891 | **0.616** | 0.432 |
| `PairformerLayer|1x512x512x128` | 16 | 51.033 | 502.7 | 6389.0 | 78.69 | bandwidth | 29.5 % | 0.817 | **0.576** | 0.404 |
| `PairWeightedAveraging|1x1024x512x64,1x512x512x128` | 16 | 45.890 | 191.2 | 7181.7 | 26.62 | bandwidth | 36.8 % | 0.734 | **0.464** | 0.325 |
| `AttentionPairBias|1x512x384,1x512x512x128` | 264 | 1.733 | 3.6 | 151.4 | 23.97 | bandwidth | 20.6 % | 0.458 | **0.363** | 0.255 |
| `AttentionPairBias|1x140x32x128,140x4x32x128` | 1200 | 0.555 | 1.6 | 117.9 | 13.75 | bandwidth | 50.0 % | 0.665 | **0.332** | 0.233 |
| `ConditionedTransitionBlock|1x140x32x128,1x140x32x128` | 1200 | 0.297 | 1.6 | 48.6 | 33.28 | bandwidth | 38.5 % | 0.356 | **0.219** | 0.154 |
| `Transition|1x1024x512x64` | 16 | 14.779 | 51.6 | 606.2 | 85.08 | bandwidth | 9.7 % | 0.236 | **0.214** | 0.15 |
| `TriangleAttention|1x512x512x128` | 32 | 8.466 | 113.8 | 1178.8 | 96.58 | bandwidth | 32.8 % | 0.271 | **0.182** | 0.128 |
| `TriangleMultiplication|1x512x512x128` | 32 | 9.381 | 86.0 | 1610.8 | 53.37 | bandwidth | 40.4 % | 0.3 | **0.179** | 0.125 |
| `AdaLN|1x140x32x128,1x140x32x128` | 2400 | 0.079 | 0.3 | 11.6 | 25.52 | bandwidth | 34.3 % | 0.19 | **0.125** | 0.088 |
| `Transition|1x512x384` | 264 | 0.196 | 1.8 | 4.8 | 380.08 | compute | 8.8 % | 0.052 | **0.047** | 0.033 |
| `Transition|1x512x768` | 400 | 0.133 | 3.6 | 9.5 | 380.08 | compute | 26.0 % | 0.053 | **0.039** | 0.028 |

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

The MSA axis is padded to 1024 rows. `MSA_PAD_MULTIPLE = 1024` in `tt_bio/tenstorrent.py`, and this fixture has 35 MSA rows, so the MSA block executes 1368.3 GFLOP per call where FlopCounterMode counts 595.2 logical. The three units inside it whose shapes carry the depth axis move 15107.2 MB of the block's 22100.2 MB (68.4 %) and 1.305 s of the 17.340 s cell. A finer MSA ladder is the lever; how much of that is recoverable is a measurement someone else has to take.

Tile padding is not where the FLOPs go: 0.15 %. The atom axis moved the other way since `flops_bytes_512.json` was written, 4480 atoms at the tip against 7168 there, so the atom transformer term is 1.28x its logical count rather than the up to 4x that file self-declares.

