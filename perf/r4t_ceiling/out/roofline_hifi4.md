# RFD4 Step 1 Analytic FLOP/Byte Count and Blackhole Roofline
Batch `batch00.npz` | crop 256 | K=4 | depth=2 | attn_fp32=on | trained params=178,448,304 (282 Muon=175,994,112, 276 Adam=2,454,192)

## Hardware Peaks (cited from knowledgebase/knowledge/)
| Hardware | Worker cores | LoFi TFLOP/s | HiFi2 TFLOP/s | HiFi4 TFLOP/s | Split-bf16 fp32 TFLOP/s | DRAM GB/s | HiFi4 ridge (FLOP/B) | Source |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| BH p300c (qb2 measured) | 110 | 608.3 | 304.1 | 152.1 | 50.7 | 512 | 297 | machines.md:150, claims.md:60-62, ops.csv grid=110 |
| BH Galaxy (32-chip target) | 120 | 663.6 | 331.8 | 165.9 | 55.3 | 512 | 324 | claims.md:233 (120 Tensix/chip, 512 GB/s) |
| BH p150a (physical 140c) | 140 | 774.1 | 387.1 | 193.5 | 64.5 | 512 | 378 | machines.md:150 (140 Tensix, 774 FP8, 512 GB/s) |
| BH p300c measured ttnn roof | 110 | - | - | 115.69 | 38.56 | 440.0 | 263 | state/of3t/BACKWARD.md:243 |
| BH p150a measured ttnn roof | 130 | 161.14 | 140.35 | 100.55 | 33.52 | 435.2 | 231 | PLAYBOOKS.md:443, state/rfd4/ARCH.md:107-111 |

## Model Forward + Backward (p300c 110c HiFi4 = 152.1 TFLOP/s, 512 GB/s)
Note: aten byte totals are UNFUSED (every PyTorch intermediate hits DRAM); L1 kernel fusion (K14 serving kernels) beats unfused DRAM time, approaching the compute + irreducible-DRAM floor.

| phase | class | size | GFLOP | unfused GB | calls | t_compute ms | t_dram_unfused ms | unfused roof ms |
|---|---|---|---:|---:|---:|---:|---:|---:|
| cand_k | eltwise | pair | 45.3 | 203.92 | 1620 | 0.30 | 398.29 | 398.29 |
| cand_k | eltwise | single | 0.1 | 0.29 | 638 | 0.00 | 0.56 | 0.56 |
| cand_k | matmul | pair | 15608.3 | 79.05 | 568 | 102.64 | 154.39 | 154.39 |
| cand_k | matmul | single | 0.0 | 0.00 | 3 | 0.00 | 0.00 | 0.00 |
| cand_k | move | pair | 4.0 | 16.14 | 201 | 0.03 | 31.52 | 31.52 |
| cand_k | move | single | 0.0 | 0.06 | 104 | 0.00 | 0.12 | 0.12 |
| cand_k | norm | pair | 6.2 | 24.55 | 85 | 0.04 | 47.94 | 47.94 |
| cand_k | norm | single | 0.0 | 0.03 | 235 | 0.00 | 0.07 | 0.07 |
| cand_k | reduce | pair | 0.0 | 6.69 | 177 | 0.00 | 13.06 | 13.06 |
| cand_k | reduce | single | 0.0 | 0.04 | 92 | 0.00 | 0.09 | 0.09 |
| cand_k | softmax | pair | 0.1 | 0.35 | 28 | 0.00 | 0.69 | 0.69 |
| win_bwd | eltwise | pair | 9.8 | 51.41 | 419 | 0.06 | 100.41 | 100.41 |
| win_bwd | eltwise | single | 0.2 | 1.14 | 1760 | 0.00 | 2.23 | 2.23 |
| win_bwd | matmul | pair | 3884.4 | 21.54 | 415 | 25.54 | 42.07 | 42.07 |
| win_bwd | matmul | single | 18.2 | 0.17 | 191 | 0.12 | 0.33 | 0.33 |
| win_bwd | move | pair | 0.5 | 2.09 | 70 | 0.00 | 4.08 | 4.08 |
| win_bwd | move | single | 0.0 | 0.03 | 44 | 0.00 | 0.06 | 0.06 |
| win_bwd | norm | pair | 0.8 | 4.59 | 35 | 0.01 | 8.96 | 8.96 |
| win_bwd | norm | single | 0.0 | 0.01 | 7 | 0.00 | 0.02 | 0.02 |
| win_bwd | reduce | pair | 0.0 | 3.49 | 76 | 0.00 | 6.82 | 6.82 |
| win_bwd | reduce | single | 0.0 | 0.09 | 263 | 0.00 | 0.17 | 0.17 |
| win_bwd | softmax | pair | 0.0 | 0.07 | 14 | 0.00 | 0.13 | 0.13 |
| win_fwd | eltwise | pair | 11.1 | 50.01 | 510 | 0.07 | 97.67 | 97.67 |
| win_fwd | eltwise | single | 0.2 | 1.04 | 1756 | 0.00 | 2.04 | 2.04 |
| win_fwd | matmul | pair | 3884.0 | 20.83 | 400 | 25.54 | 40.69 | 40.69 |
| win_fwd | matmul | single | 18.0 | 0.17 | 185 | 0.12 | 0.32 | 0.32 |
| win_fwd | move | pair | 0.1 | 0.21 | 57 | 0.00 | 0.41 | 0.41 |
| win_fwd | move | single | 0.0 | 0.05 | 122 | 0.00 | 0.10 | 0.10 |
| win_fwd | norm | pair | 1.5 | 6.13 | 70 | 0.01 | 11.96 | 11.96 |
| win_fwd | norm | single | 0.0 | 0.02 | 250 | 0.00 | 0.04 | 0.04 |
| win_fwd | reduce | pair | 0.0 | 1.61 | 32 | 0.00 | 3.15 | 3.15 |
| win_fwd | reduce | single | 0.0 | 0.07 | 241 | 0.00 | 0.13 | 0.13 |
| win_fwd | softmax | pair | 0.0 | 0.09 | 28 | 0.00 | 0.17 | 0.17 |
| cand_k | attn_fp32_extra | +2B/el | 0.0 | 5.46 | 24blks | 0.00 | 10.67 | 10.67 |
| win_fwd | attn_fp32_extra | +2B/el | 0.0 | 1.37 | 24blks | 0.00 | 2.67 | 2.67 |
| win_bwd | attn_fp32_extra | +2B/el | 0.0 | 1.63 | 24blks | 0.00 | 3.19 | 3.19 |

## On-Card System Terms (Fold VJP, Split Rebuild, Optimizer)
| term | GFLOP (HiFi4-eq) | DRAM GB | t_compute ms | t_dram ms | roof 1-chip ms | roof ZeRO-1 1/32 ms |
|---|---:|---:|---:|---:|---:|---:|
| fold_vjp (DeviceFolds + fp32 grad accum) | 0.0 | 2.14 | 0.00 | 4.18 | 4.18 | 4.18 |
| split_rebuild (DeviceFolds bf16 + caches) | 0.0 | 1.53 | 0.00 | 2.99 | 2.99 | 2.99 |
| opt_muon_ns (282 mats, 5 iters, split-bf16 fp32) | 8397.5 | 16.64 | 55.22 | 32.50 | 55.22 | 1.73 |
| opt_elementwise (282 Muon + 276 Adam + EMA + clip) | 5.4 | 5.73 | 0.04 | 11.19 | 11.19 | 0.35 |

## Isolated Host-Side Submodules to Move on Card (included inside model totals above; priced separately in Step 3)
| component | matmul GFLOP | matmul GB | mem GB | t_compute ms | t_dram ms | roof ms |
|---|---:|---:|---:|---:|---:|---:|
| glue_heads (fwd K+1 + bwd) | 5.43 | 0.09 | 0.16 | 0.04 | 0.48 | 0.48 |
| glue_cond_init (fwd K+1 x d + bwd) | 11.69 | 0.19 | 0.33 | 0.08 | 1.01 | 1.01 |
| pair_const_c1 (C1 proj + K scales) | 53.15 | 0.94 | 8.36 | 0.35 | 18.16 | 18.16 |

## Analytic Totals
- `cand_k`: 15.664 TFLOP, 336.6 GB unfused | compute floor 103.0 ms | unfused DRAM roof 657.4 ms
- `win_fwd`: 3.915 TFLOP, 81.6 GB unfused | compute floor 25.7 ms | unfused DRAM roof 159.4 ms
- `win_bwd`: 3.914 TFLOP, 86.3 GB unfused | compute floor 25.7 ms | unfused DRAM roof 168.5 ms
- System terms (fold_vjp + rebuild + opt): **73.6 ms** (1-chip unsharded) / **9.3 ms** (ZeRO-1 1/32)
- **Compute-floor step (fully fused model + system)**: **228.1 ms** (1-chip, 10.78x/$) / **163.7 ms** (ZeRO-1 1/32, 15.02x/$)
- **Unfused-aten roof step (zero fusion + system)**: **1058.8 ms** (1-chip, 2.32x/$) / **994.5 ms** (ZeRO-1 1/32, 2.47x/$)
