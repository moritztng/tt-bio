# RFD4 Device-Only Step Ceiling — p1_def_serving
Source CSV: `/home/ttuser/r4t/ceiling/p1_def/ops.csv` | warm steps `[2, 3]` (n=2) | worker grid `110` Tensix (p300c, 1350 MHz) | peaks: HiFi4 `152.1` TFLOP/s, DRAM `512` GB/s.
Price basis (PERF/SIZING): BH Galaxy `$160,000 / 32` vs DGX B200 `$562,154 / 8` = `14.05385x`; B200 depth 2 = `0.175` s/step.

## Step-to-Step Stability (DEVICE KERNEL DURATION, ms)
| phase | step 1 | step 2 | step 3 | warm mean |
|---|---:|---:|---:|---:|
| `fwd` | 0.14 | 0.15 | 0.15 | 0.15 |
| `cand_pair` | 9.55 | 9.55 | 9.55 | 9.55 |
| `cand_trunk` | 1063.46 | 1061.94 | 1062.89 | 1062.42 |
| `win_pair` | 2.35 | 2.35 | 2.35 | 2.35 |
| `win_trunk` | 342.73 | 343.07 | 342.89 | 342.98 |
| `heads_bwd` | 0.00 | 0.00 | 0.00 | 0.00 |
| `tape_bwd` | 429.31 | 428.86 | 429.36 | 429.11 |
| `fold_bwd` | 14.71 | 14.70 | 14.69 | 14.69 |
| `opt` | 0.00 | 0.00 | 0.00 | 0.00 |
| `rebuild` | 4.10 | 4.09 | 4.10 | 4.09 |
| `digest` | 0.00 | 0.00 | 0.00 | 0.00 |
| **compute total (excl. host-IO fold_bwd/rebuild)** | 1847.54 | 1845.93 | 1847.19 | **1846.56** |

## Measured Device Time & Achieved Roofline by Op Class (compute phases, warm mean)
| class | meas ms | % card | DRAM GB | TFLOP | ach GB/s | % 512 GB/s | ach TF/s | % 152.1 TF/s | per-op roof ms | roof/meas | ops |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `move` | 638.72 | 34.6% | 123.7 | 0.00 | 193.6 | 37.8% | 0.0 | 0.0% | 241.55 | 37.8% | 3380 |
| `matmul` | 585.93 | 31.7% | 117.4 | 23.28 | 200.3 | 39.1% | 39.7 | 26.1% | 229.88 | 39.2% | 1933 |
| `eltwise` | 412.90 | 22.4% | 156.9 | 0.00 | 380.1 | 74.2% | 0.0 | 0.0% | 306.51 | 74.2% | 3735 |
| `norm` | 126.65 | 6.9% | 40.1 | 0.00 | 316.3 | 61.8% | 0.0 | 0.0% | 78.25 | 61.8% | 502 |
| `generic_fused` | 53.01 | 2.9% | 19.3 | 0.00 | 363.5 | 71.0% | 0.0 | 0.0% | 37.64 | 71.0% | 108 |
| `reduce` | 25.94 | 1.4% | 3.8 | 0.00 | 146.4 | 28.6% | 0.0 | 0.0% | 7.42 | 28.6% | 430 |
| `softmax` | 3.42 | 0.2% | 0.9 | 0.00 | 257.9 | 50.4% | 0.0 | 0.0% | 1.72 | 50.4% | 56 |
| **TOTAL** | **1846.56** | **100.0%** | **462.0** | **23.28** | **250.2** | **48.9%** | **12.6** | **8.3%** | **902.96** | **48.9%** | |

Measured op-class efficiencies used to price unported terms:
- Matmul (`matmul` class): `39.2%` of per-op roofline (`39.7` TFLOP/s overall; `MinimalMatmul` = `61.6` TFLOP/s; fwd matmuls = `47.3%`).
- DRAM streaming (`BinaryNg`/`Typecast`/`Concat`/`Unary`): `382.6` GB/s = `74.7%` of 512 GB/s (`87.0%` of 440 GB/s measured roof).
- All non-matmul DRAM ops (incl. `Permute`/`Transpose`/`LayerNorm`/`Reduce`): `269.4` GB/s = `52.6%` of 512 GB/s.

## Phase x Op-Class Breakdown (warm mean)
| phase | class | meas ms | DRAM GB | ach GB/s | TFLOP | ach TF/s | per-op roof ms | roof/meas | ops |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `cand_pair` | `generic_fused` | 9.06 | 0.9 | 100.2 | 0.00 | 0.0 | 1.77 | 19.6% | 1 |
| `cand_pair` | `move` | 0.49 | 0.2 | 414.5 | 0.00 | 0.0 | 0.39 | 81.0% | 1 |
| `cand_trunk` | `move` | 410.00 | 64.9 | 158.3 | 0.00 | 0.0 | 126.80 | 30.9% | 597 |
| `cand_trunk` | `matmul` | 317.24 | 77.1 | 243.0 | 15.50 | 48.9 | 151.14 | 47.6% | 634 |
| `cand_trunk` | `eltwise` | 232.25 | 89.6 | 385.7 | 0.00 | 0.0 | 174.94 | 75.3% | 542 |
| `cand_trunk` | `norm` | 100.46 | 32.0 | 318.1 | 0.00 | 0.0 | 62.41 | 62.1% | 244 |
| `cand_trunk` | `softmax` | 2.46 | 0.7 | 286.6 | 0.00 | 0.0 | 1.38 | 56.0% | 28 |
| `win_pair` | `generic_fused` | 2.35 | 0.3 | 128.6 | 0.00 | 0.0 | 0.59 | 25.1% | 1 |
| `win_trunk` | `move` | 145.91 | 33.7 | 230.9 | 0.00 | 0.0 | 65.79 | 45.1% | 694 |
| `win_trunk` | `eltwise` | 85.49 | 32.8 | 383.5 | 0.00 | 0.0 | 64.03 | 74.9% | 647 |
| `win_trunk` | `matmul` | 84.43 | 19.9 | 235.9 | 3.88 | 45.9 | 38.89 | 46.1% | 662 |
| `win_trunk` | `norm` | 26.19 | 8.1 | 309.6 | 0.00 | 0.0 | 15.84 | 60.5% | 258 |
| `win_trunk` | `softmax` | 0.96 | 0.2 | 184.1 | 0.00 | 0.0 | 0.34 | 36.0% | 28 |
| `tape_bwd` | `matmul` | 184.26 | 20.4 | 110.7 | 3.90 | 21.2 | 39.84 | 21.6% | 637 |
| `tape_bwd` | `eltwise` | 95.16 | 34.6 | 363.4 | 0.00 | 0.0 | 67.54 | 71.0% | 2546 |
| `tape_bwd` | `move` | 82.17 | 24.8 | 302.0 | 0.00 | 0.0 | 48.47 | 59.0% | 2083 |
| `tape_bwd` | `generic_fused` | 41.59 | 18.1 | 434.2 | 0.00 | 0.0 | 35.27 | 84.8% | 106 |
| `tape_bwd` | `reduce` | 25.94 | 3.8 | 146.4 | 0.00 | 0.0 | 7.42 | 28.6% | 430 |

## Matmul Kernel Breakdown by Phase
| phase | op | meas ms | DRAM GB | TFLOP | ach TF/s | % HiFi4 | roof ms | roof/meas | ops |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `cand_trunk` | `MinimalMatmulDeviceOperation` | 198.37 | 48.9 | 12.27 | 61.8 | 40.7% | 95.51 | 48.2% | 84 |
| `tape_bwd` | `MatmulDeviceOperation` | 184.26 | 20.4 | 3.90 | 21.2 | 13.9% | 39.84 | 21.6% | 637 |
| `cand_trunk` | `MatmulDeviceOperation` | 118.87 | 28.2 | 3.23 | 27.2 | 17.9% | 55.63 | 46.8% | 550 |
| `win_trunk` | `MinimalMatmulDeviceOperation` | 50.60 | 12.3 | 3.07 | 60.6 | 39.9% | 23.95 | 47.3% | 84 |
| `win_trunk` | `MatmulDeviceOperation` | 33.83 | 7.7 | 0.81 | 24.0 | 15.8% | 14.94 | 44.2% | 578 |

## Top 12 Device Op Codes (compute phases, warm mean)
| op code | meas ms | % card | DRAM GB | ach GB/s | TFLOP | ach TF/s | roof ms | roof/meas | ops |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `BinaryNgDeviceOperation` | 407.98 | 22.1% | 155.2 | 380.4 | 0.00 | 0.0 | 303.13 | 74.3% | 3470 |
| `MatmulDeviceOperation` | 336.96 | 18.2% | 56.2 | 166.9 | 7.95 | 23.6 | 110.42 | 32.8% | 1765 |
| `MinimalMatmulDeviceOperation` | 248.97 | 13.5% | 61.2 | 245.7 | 15.33 | 61.6 | 119.46 | 48.0% | 168 |
| `PermuteDeviceOperation` | 239.40 | 13.0% | 19.3 | 80.8 | 0.00 | 0.0 | 37.79 | 15.8% | 144 |
| `TransposeDeviceOperation` | 173.83 | 9.4% | 20.8 | 119.9 | 0.00 | 0.0 | 40.70 | 23.4% | 184 |
| `LayerNormDeviceOperation` | 126.65 | 6.9% | 40.1 | 316.3 | 0.00 | 0.0 | 78.25 | 61.8% | 502 |
| `SliceDeviceOperation` | 107.27 | 5.8% | 40.7 | 379.6 | 0.00 | 0.0 | 79.53 | 74.1% | 1579 |
| `TypecastDeviceOperation` | 90.21 | 4.9% | 35.5 | 393.7 | 0.00 | 0.0 | 69.36 | 76.9% | 946 |
| `GenericOpDeviceOperation` | 53.01 | 2.9% | 19.3 | 363.5 | 0.00 | 0.0 | 37.64 | 71.0% | 108 |
| `ReduceDeviceOperation` | 25.02 | 1.4% | 3.4 | 137.7 | 0.00 | 0.0 | 6.73 | 26.9% | 423 |
| `ConcatDeviceOperation` | 12.01 | 0.7% | 4.7 | 387.4 | 0.00 | 0.0 | 9.09 | 75.7% | 118 |
| `TilizeWithValPaddingDeviceOperation` | 5.20 | 0.3% | 0.2 | 45.6 | 0.00 | 0.0 | 0.46 | 8.9% | 75 |

## Per-Term Device-Only Ceiling (where this goes, NOT where it is)
| term | ms/step (ZeRO-1 1/32) | ms/step (1-chip 1/1) | % of ZeRO-1 ceiling | basis |
|---|---:|---:|---:|---|
| cand_pair (`PairOnCard` K=4, measured device) | 9.55 | 9.55 | 0.5% | measured device |
| cand_trunk (K=4 depth=2 grad-off, measured device) | 1062.42 | 1062.42 | 56.1% | measured device |
| win_pair (P2 `PairOnCard.taped` fwd, measured device) | 2.35 | 2.35 | 0.1% | measured device |
| win_trunk (K=1 depth=2 taped, measured device) | 342.98 | 342.98 | 18.1% | measured device |
| tape_bwd (trunk + P2 pair bwd, measured device) | 429.11 | 429.11 | 22.6% | measured device |
| fwd misc (masks, measured device) | 0.15 | 0.15 | 0.0% | measured device |
| pair_const C1 (proj + K=4 scales, roof / eff) | 26.54 | 26.54 | 1.4% | roof / measured class eff |
| glue_heads fwd+bwd (roof / eff) | 1.03 | 1.03 | 0.1% | roof / measured class eff |
| glue_cond_init fwd+bwd (roof / eff) | 2.15 | 2.15 | 0.1% | roof / measured class eff |
| fold_vjp (`DeviceFolds` + fp32 grad accum, roof / eff) | 7.95 | 7.95 | 0.4% | roof / measured class eff |
| split_rebuild (`DeviceFolds` bf16 + caches, roof / eff) | 5.69 | 5.69 | 0.3% | roof / measured class eff |
| opt Muon NS fp32 (282 mats, 5 iters, ZeRO-1 1/32, roof / eff) | 4.40 | 140.76 | 0.2% | roof / measured class eff |
| opt elementwise fp32 (Muon+Adam+EMA, ZeRO-1 1/32, roof / eff) | 0.47 | 14.98 | 0.0% | roof / measured class eff |
| host batch feed (overlapped) | 0.00 | 0.00 | 0.0% | roof / measured class eff |
| **CEILING (device only, host overlapped)** | **1894.8 ms (1.8948 s)** | **2045.6 ms (2.0456 s)** | **100.0%** | |

- **ZeRO-1 (1/32) per chip (DP eff 1.00)**: `1.8948 s/step` -> **`1.30x`** per $ vs B200 `0.175 s` (`1.17x` at 0.90 DP eff, `1.10x` at 0.85 DP eff).
- **Unsharded 1-chip (1/1) per chip**: `2.0456 s/step` -> **`1.20x`** per $ vs B200 `0.175 s`.
- Note: measured on qb2 p300c (`110` worker cores); BH Galaxy has `120` Tensix cores (`+9.1%` compute roof, same `512 GB/s` DRAM).

SUMMARY label=p1_def_serving meas_ms=1846.56 unp32_ms=48.22 unp1c_ms=199.09 ceil32_s=1.8948 ratio32=1.30 ratio32_dp85=1.10 ceil1c_s=2.0456 ratio1c=1.20
