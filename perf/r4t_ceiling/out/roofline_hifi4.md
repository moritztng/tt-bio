# batch batch00.npz depth 2 K 4 params 435622465 (opt sees 178448304); peak 152.1 TFLOP/s (HiFi4) 512 GB/s; ridge 297 flop/B
| phase | class | size | GFLOP | GB | calls | t_compute ms | t_dram ms | roofline ms |
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
| optimizer | muon_ns_fp32 | matrix | 8530.3 | 15.71 | 0 | 56.10 | 30.69 | 56.10 |
| optimizer | elementwise_fp32 | param | 5.4 | 7.85 | 0 | 0.04 | 15.34 | 15.34 |
TOTAL cand_k 646.7 ms
TOTAL win_bwd 165.3 ms
TOTAL win_fwd 156.7 ms
TOTAL optimizer 71.4 ms
TOTAL step 1040.1 ms
