# Boltz-2 512 aa at f072ae02f: the roofline budget

qb2 card 2 (p300c, 11x10 grid, AICLK 800 MHz), ttnn 0.68.0, one process, one session. FLOPs and bytes come off the same 26 captures; times come off the same process's bracketed fold, per unit by the `percall` rule. Session fold 17.270 s at loadavg 2.22, 1.34, 0.59; the benchlocked cell of record is 17.270 s, so seconds are also given scaled by 1.0000.

Roofs measured in this session with dispatch amortised: dense bf16 HiFi4 **104.93 TFLOP/s**, starved 8192^2 add **424.7 GB/s**. Machine balance 247.1 FLOP/byte.

| unit | calls | ms/call | GFLOP/call | MB/call | FLOP/byte | binding roof | % of it | s/fold | **s above roof** | at the cell |
|---|---|---|---|---|---|---|---|---|---|---|
| `DiffusionModule|` | 200 | 64.008 | 317.2 | 4035.8 | 78.59 | bandwidth | 14.8 % | 12.802 | **10.901** | 10.901 |
| `Diffusion|1x4480x3,1` | 200 | 62.190 | 317.2 | 4035.5 | 78.6 | bandwidth | 15.3 % | 12.438 | **10.538** | 10.538 |
| `DiffusionTransformerLayer|1x512x768,1x512x768` | 4800 | 2.404 | 12.1 | 121.3 | 99.58 | bandwidth | 11.9 % | 11.54 | **10.168** | 10.168 |
| `MSALayer|1x512x512x128,1x1024x512x64` | 16 | 561.837 | 1368.3 | 22100.2 | 61.91 | bandwidth | 9.3 % | 8.989 | **8.157** | 8.157 |
| `DiffusionTransformer|1x512x768,1x512x768` | 200 | 46.071 | 290.0 | 2894.1 | 100.2 | bandwidth | 14.8 % | 9.214 | **7.851** | 7.851 |
| `PairformerLayer|1x512x384,1x512x512x128` | 264 | 42.516 | 508.2 | 6758.3 | 75.19 | bandwidth | 37.4 % | 11.224 | **7.023** | 7.023 |
| `PairformerLayer|1x512x512x128` | 16 | 390.927 | 502.7 | 6389.0 | 78.69 | bandwidth | 3.8 % | 6.255 | **6.014** | 6.014 |
| `TriangleMultiplication|1x512x512x128` | 32 | 149.621 | 86.0 | 1610.8 | 53.37 | bandwidth | 2.5 % | 4.788 | **4.667** | 4.667 |
| `ConditionedTransitionBlock|1x512x768,1x512x768` | 4800 | 0.959 | 6.6 | 52.4 | 126.7 | bandwidth | 12.9 % | 4.604 | **4.012** | 4.012 |
| `AdaLN|1x512x768,1x512x768` | 9600 | 0.406 | 1.2 | 15.0 | 80.37 | bandwidth | 8.7 % | 3.894 | **3.554** | 3.554 |
| `Transition|1x512x512x128` | 280 | 13.421 | 103.1 | 340.0 | 303.26 | compute | 7.3 % | 3.758 | **3.483** | 3.483 |
| `AttentionPairBias|1x512x768,1x16x512x512` | 4800 | 0.817 | 3.6 | 44.0 | 82.43 | bandwidth | 12.7 % | 3.923 | **3.426** | 3.426 |
| `TriangleMultiplication|1x512x512x128,1x512x512` | 528 | 10.474 | 86.0 | 1745.6 | 49.25 | bandwidth | 39.2 % | 5.53 | **3.36** | 3.36 |
| `DiffusionTransformerLayer|1x140x32x128,1x140x32x128` | 1200 | 2.855 | 3.7 | 186.0 | 19.78 | bandwidth | 15.3 % | 3.426 | **2.9** | 2.9 |
| `DiffusionTransformer|1x140x32x128,1x140x32x128` | 400 | 8.024 | 8.8 | 534.6 | 16.53 | bandwidth | 15.7 % | 3.21 | **2.706** | 2.706 |
| `TriangleAttention|1x512x512x128` | 32 | 80.551 | 113.8 | 1178.8 | 96.58 | bandwidth | 3.4 % | 2.578 | **2.489** | 2.489 |
| `TriangleAttention|1x512x512x128,1x1x1x512` | 528 | 6.769 | 113.9 | 1183.0 | 96.24 | bandwidth | 41.2 % | 3.574 | **2.103** | 2.103 |
| `AttentionPairBias|1x140x32x128,140x4x32x128` | 1200 | 1.684 | 1.6 | 117.9 | 13.75 | bandwidth | 16.5 % | 2.021 | **1.688** | 1.688 |
| `Transition|1x1024x512x64` | 16 | 67.374 | 51.6 | 606.2 | 85.08 | bandwidth | 2.1 % | 1.078 | **1.055** | 1.055 |
| `ConditionedTransitionBlock|1x140x32x128,1x140x32x128` | 1200 | 0.780 | 1.6 | 48.6 | 33.28 | bandwidth | 14.6 % | 0.936 | **0.799** | 0.799 |
| `PairWeightedAveraging|1x1024x512x64,1x512x512x128` | 16 | 49.026 | 191.2 | 7181.7 | 26.62 | bandwidth | 34.5 % | 0.784 | **0.514** | 0.514 |
| `AdaLN|1x140x32x128,1x140x32x128` | 2400 | 0.233 | 0.3 | 11.6 | 25.52 | bandwidth | 11.7 % | 0.558 | **0.493** | 0.493 |
| `OuterProductMean|1x1024x512x64,1024x1x1` | 16 | 41.807 | 622.8 | 7319.3 | 85.09 | bandwidth | 41.2 % | 0.669 | **0.393** | 0.393 |
| `AttentionPairBias|1x512x384,1x512x512x128` | 264 | 1.306 | 3.6 | 151.4 | 23.97 | bandwidth | 27.3 % | 0.345 | **0.251** | 0.251 |
| `Transition|1x512x768` | 400 | 0.258 | 3.6 | 9.5 | 380.08 | compute | 13.4 % | 0.103 | **0.089** | 0.089 |
| `Transition|1x512x384` | 264 | 0.189 | 1.8 | 4.8 | 380.08 | compute | 9.1 % | 0.05 | **0.045** | 0.045 |

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

The MSA axis is padded to 1024 rows. `MSA_PAD_MULTIPLE = 1024` in `tt_bio/tenstorrent.py`, and this fixture has 35 MSA rows, so the MSA block executes 1368.3 GFLOP per call where FlopCounterMode counts 595.2 logical. The three units inside it whose shapes carry the depth axis move 15107.2 MB of the block's 22100.2 MB (68.4 %) and 2.531 s of the 17.270 s cell. A finer MSA ladder is the lever; how much of that is recoverable is a measurement someone else has to take.

Tile padding is not where the FLOPs go: 0.15 %. The atom axis moved the other way since `flops_bytes_512.json` was written, 4480 atoms at the tip against 7168 there, so the atom transformer term is 1.28x its logical count rather than the up to 4x that file self-declares.

