roofs: matmul 100.6 TFLOP/s, copy 388 GB/s, eltwise 402 GB/s, layernorm 326 GB/s, softmax 182 GB/s, reduction 142 GB/s, permute 71 GB/s

evo: wall 29.1 ms, 280 calls, enqueue 11.8 ms, synced sum 39.7 ms, loadavg 72.4
| # | op | in shapes | calls | ms | share | cum | class | vs own roof |
|---|---|---|---|---|---|---|---|---|
| 1 | to_memory_config | [81, 4, 256, 256]BFL [81, 4, 256, | 12 | 2.26 | 5.7 % | 5.7 % | layout | 159 %: above the DRAM roof, operand L1-resident or in place |
| 2 | typecast | [1, 256, 256, 128]BFL [1, 256, 256, | 12 | 2.11 | 5.3 % | 11.0 % | layout | 88 % |
| 3 | add_ | [81, 4, 256, 256]FLO|[1, 4, 256, 256]FLO [81, 4, 256, | 6 | 1.83 | 4.6 % | 15.6 % | eltwise | 160 %: above the DRAM roof, operand L1-resident or in place |
| 4 | add_ | [1, 256, 256, 128]FLO|[1, 256, 256, 128]FLO [1, 256, 256, | 6 | 1.75 | 4.4 % | 20.0 % | eltwise | 99 % |
| 5 | matmul | [81, 4, 256, 32]BFL|[81, 4, 32, 256]BFL [81, 4, 256, | 6 | 1.29 | 3.2 % | 23.3 % | matmul | 73 % |
| 6 | typecast | [81, 4, 256, 256]BFL [81, 4, 256, | 6 | 1.28 | 3.2 % | 26.5 % | layout | 208 %: above the DRAM roof, operand L1-resident or in place |
| 7 | permute | [1, 256, 256, 64]BFL [1, 64, 256, | 8 | 1.26 | 3.2 % | 29.7 % | layout | 32 % |
| 8 | slice | [256, 4, 256, 32]BFL [81, 4, 256, | 18 | 1.24 | 3.1 % | 32.8 % | layout | 133 %: above the DRAM roof, operand L1-resident or in place |
| 9 | matmul | [81, 4, 256, 256]BFL|[81, 4, 256, 32]BFL [81, 4, 256, | 6 | 1.15 | 2.9 % | 35.7 % | matmul | 80 % |
| 10 | typecast | [81, 4, 256, 256]FLO [81, 4, 256, | 6 | 1.13 | 2.9 % | 38.5 % | layout | 258 %: above the DRAM roof, operand L1-resident or in place |
| 11 | softmax_in_place | [81, 4, 256, 256]FLO [81, 4, 256, | 6 | 1.05 | 2.6 % | 41.2 % | softmax | 739 %: above the DRAM roof, operand L1-resident or in place |
| 12 | layer_norm | [1, 256, 256, 128]BFL|[128]BFL|[128]BFL [1, 256, 256, | 5 | 1.01 | 2.6 % | 43.7 % | layernorm | 69 % |
| 13 | linear | [65536, 1024]BFL|[1024, 128]BFL|[128]BFL [65536, | 1 | 0.97 | 2.4 % | 46.2 % | matmul | 41 % |
| 14 | typecast | [1, 256, 256, 128]FLO [1, 256, 256, | 6 | 0.94 | 2.4 % | 48.5 % | layout | 99 % |
| 15 | chunk | [1, 256, 256, 256]BFL [1, 256, 256, 64]|[1, 256, 256, 64]|[1 | 4 | 0.93 | 2.3 % | 50.9 % | layout | 127 %: above the DRAM roof, operand L1-resident or in place |
| 16 | multiply_ | [1, 256, 256, 64]BFL|[1, 256, 256, 64]BFL [1, 256, 256, | 8 | 0.90 | 2.3 % | 53.1 % | eltwise | 73 % |
| 17 | reshape | [8192, 8192]BFL [256, 1024, | 1 | 0.86 | 2.2 % | 55.3 % | layout | 86 % |
| 18 | to_layout | [256, 1024, 256]BFL [256, 1024, | 1 | 0.83 | 2.1 % | 57.4 % | layout | 89 % |
| 19 | to_layout | [8192, 8192]BFL [8192, | 1 | 0.83 | 2.1 % | 59.5 % | layout | 93 % |
| 20 | permute | [256, 1024, 256]BFL [256, 256, | 1 | 0.75 | 1.9 % | 61.4 % | layout | 98 % |
| 21 | matmul | [1, 64, 256, 256]BFL|[1, 64, 256, 256]BFL [1, 64, 256, | 4 | 0.72 | 1.8 % | 63.2 % | matmul | 44 % |
| 22 | matmul | [8192, 1]BFL|[8192, 1]BFL [8192, | 1 | 0.69 | 1.7 % | 64.9 % | matmul | 54 % |
| 23 | linear | [1, 256, 256, 512]BFL|[512, 128]BFL|[128]BFL [1, 256, 256, | 1 | 0.68 | 1.7 % | 66.7 % | matmul | 32 % |
| 24 | generic_op |  [256, 4, 256, | 2 | 0.64 | 1.6 % | 68.3 % | eltwise | 21 % |
| 25 | linear | [1, 256, 256, 128]BFL|[128, 128]BFL|[128]BFL [1, 256, 256, | 4 | 0.62 | 1.6 % | 69.8 % | matmul | 69 % |
| 26 | linear | [1, 256, 256, 128]BFL|[128, 512]BFL|[512]BFL [1, 256, 256, | 1 | 0.60 | 1.5 % | 71.3 % | matmul | 38 % |
| 27 | permute | [1, 64, 256, 256]BFL [1, 256, 256, | 4 | 0.57 | 1.4 % | 72.8 % | layout | 38 % |
| 28 | add_ | [256, 256, 128]BFL|[128]BFL [256, 256, | 2 | 0.56 | 1.4 % | 74.2 % | eltwise | 41 % |
| 29 | layer_norm | [256, 256, 128]BFL|[128]BFL|[128]BFL [256, 256, | 3 | 0.52 | 1.3 % | 75.5 % | layernorm | 95 % |
| 30 | permute | [256, 256, 128]BFL [256, 256, | 2 | 0.48 | 1.2 % | 76.7 % | layout | 44 % |
| 31 | permute | [81, 4, 256, 32]BFL [81, 4, 32, | 6 | 0.48 | 1.2 % | 77.9 % | layout | 62 % |
| 32 | multiply_ | [256, 256, 128]BFL|[256, 256, 128]BFL [256, 256, | 2 | 0.40 | 1.0 % | 78.9 % | eltwise | sync latency, no kernel time resolved |
| 33 | multiply_ | [1, 256, 256, 128]BFL|[1, 256, 256, 128]BFL [1, 256, 256, | 2 | 0.38 | 1.0 % | 79.9 % | eltwise | 86 % |
| 34 | linear | [256, 256, 128]BFL|[128, 128]BFL|[128]BFL [256, 256, | 2 | 0.33 | 0.8 % | 80.7 % | matmul | 110 %: above the DRAM roof, operand L1-resident or in place |
| 35 | concat |  [1, 256, 256, | 2 | 0.33 | 0.8 % | 81.5 % | layout | 41 % |
| 36 | add_ | [13, 4, 256, 256]FLO|[1, 4, 256, 256]FLO [13, 4, 256, | 2 | 0.32 | 0.8 % | 82.3 % | eltwise | 71 % |
| 37 | concat |  [256, 4, 256, | 2 | 0.31 | 0.8 % | 83.1 % | layout | 46 % |
| 38 | clone | [1, 256, 256, 64]BFL [1, 256, 256, | 4 | 0.29 | 0.7 % | 83.8 % | layout | 86 % |
| 39 | softmax_in_place | [13, 4, 256, 256]FLO [13, 4, 256, | 2 | 0.28 | 0.7 % | 84.5 % | softmax | 185 %: above the DRAM roof, operand L1-resident or in place |
| 40 | linear | [256, 256, 128]BFL|[128, 4]BFL [256, 256, | 2 | 0.27 | 0.7 % | 85.2 % | matmul | 47 % |
| 41 | linear | [256, 1, 256]BFL|[256, 256]BFL|[256]BFL [256, 1, | 2 | 0.27 | 0.7 % | 85.9 % | matmul | 1 % |
| 42 | slice | [256, 4, 256, 32]BFL [13, 4, 256, | 6 | 0.24 | 0.6 % | 86.5 % | layout | 216 %: above the DRAM roof, operand L1-resident or in place |
| 43 | layer_norm | [1, 256, 256]BFL|[256]BFL|[256]BFL [1, 256, | 2 | 0.24 | 0.6 % | 87.1 % | layernorm | sync latency, no kernel time resolved |
| 44 | typecast | [13, 4, 256, 256]BFL [13, 4, 256, | 2 | 0.21 | 0.5 % | 87.6 % | layout | 68 % |
| 45 | typecast | [13, 4, 256, 256]FLO [13, 4, 256, | 2 | 0.18 | 0.5 % | 88.1 % | layout | 87 % |
| 46 | linear | [256, 1, 256]BFL|[256, 768]BFL [256, 1, | 1 | 0.18 | 0.5 % | 88.6 % | matmul | 2 % |
| 47 | squeeze | [256, 1, 256, 128]BFL [256, 256, | 2 | 0.18 | 0.5 % | 89.0 % | layout | 110 %: above the DRAM roof, operand L1-resident or in place |
| 48 | linear | [256, 256, 128]BFL|[128, 8]BFL [256, 256, | 1 | 0.16 | 0.4 % | 89.4 % | matmul | 74 % |
| 49 | matmul | [13, 4, 256, 32]BFL|[13, 4, 32, 256]BFL [13, 4, 256, | 2 | 0.16 | 0.4 % | 89.8 % | matmul | 45 % |
| 50 | softmax | [256, 8, 1, 1]BFL [256, 8, 1, | 1 | 0.15 | 0.4 % | 90.2 % | softmax | 0 % |

extra: wall 21.9 ms, 207 calls, enqueue 9.2 ms, synced sum 34.0 ms, loadavg 71.7
| # | op | in shapes | calls | ms | share | cum | class | vs own roof |
|---|---|---|---|---|---|---|---|---|
| 1 | to_memory_config | [81, 4, 256, 256]BFL [81, 4, 256, | 12 | 2.37 | 7.0 % | 7.0 % | layout | 149 %: above the DRAM roof, operand L1-resident or in place |
| 2 | typecast | [1, 256, 256, 128]BFL [1, 256, 256, | 11 | 2.30 | 6.8 % | 13.7 % | layout | 89 % |
| 3 | add_ | [81, 4, 256, 256]FLO|[1, 4, 256, 256]FLO [81, 4, 256, | 6 | 1.88 | 5.5 % | 19.2 % | eltwise | 159 %: above the DRAM roof, operand L1-resident or in place |
| 4 | add_ | [1, 256, 256, 128]FLO|[1, 256, 256, 128]FLO [1, 256, 256, | 5 | 1.58 | 4.7 % | 23.9 % | eltwise | 94 % |
| 5 | permute | [1, 256, 256, 64]BFL [1, 64, 256, | 8 | 1.39 | 4.1 % | 28.0 % | layout | 29 % |
| 6 | matmul | [81, 4, 256, 32]BFL|[81, 4, 32, 256]BFL [81, 4, 256, | 6 | 1.36 | 4.0 % | 32.0 % | matmul | 67 % |
| 7 | typecast | [81, 4, 256, 256]BFL [81, 4, 256, | 6 | 1.33 | 3.9 % | 35.9 % | layout | 196 %: above the DRAM roof, operand L1-resident or in place |
| 8 | slice | [256, 4, 256, 32]BFL [81, 4, 256, | 18 | 1.28 | 3.8 % | 39.7 % | layout | 124 %: above the DRAM roof, operand L1-resident or in place |
| 9 | matmul | [81, 4, 256, 256]BFL|[81, 4, 256, 32]BFL [81, 4, 256, | 6 | 1.23 | 3.6 % | 43.3 % | matmul | 74 % |
| 10 | typecast | [81, 4, 256, 256]FLO [81, 4, 256, | 6 | 1.18 | 3.5 % | 46.8 % | layout | 231 %: above the DRAM roof, operand L1-resident or in place |
| 11 | multiply_ | [1, 256, 256, 64]BFL|[1, 256, 256, 64]BFL [1, 256, 256, | 8 | 1.14 | 3.3 % | 50.1 % | eltwise | 61 % |
| 12 | softmax_in_place | [81, 4, 256, 256]FLO [81, 4, 256, | 6 | 1.11 | 3.3 % | 53.4 % | softmax | 682 %: above the DRAM roof, operand L1-resident or in place |
| 13 | typecast | [1, 256, 256, 128]FLO [1, 256, 256, | 6 | 1.08 | 3.2 % | 56.5 % | layout | 95 % |
| 14 | layer_norm | [1, 256, 256, 128]BFL|[128]BFL|[128]BFL [1, 256, 256, | 5 | 1.07 | 3.1 % | 59.7 % | layernorm | 75 % |
| 15 | chunk | [1, 256, 256, 256]BFL [1, 256, 256, 64]|[1, 256, 256, 64]|[1 | 4 | 1.03 | 3.0 % | 62.7 % | layout | 140 %: above the DRAM roof, operand L1-resident or in place |
| 16 | add_ | [1, 256, 256, 128]FLO|[1, 1, 1, 128]FLO [1, 256, 256, | 1 | 1.01 | 3.0 % | 65.7 % | eltwise | 21 % |
| 17 | matmul | [1, 64, 256, 256]BFL|[1, 64, 256, 256]BFL [1, 64, 256, | 4 | 0.88 | 2.6 % | 68.3 % | matmul | 39 % |
| 18 | linear | [1, 256, 256, 512]BFL|[512, 128]BFL|[128]BFL [1, 256, 256, | 1 | 0.77 | 2.3 % | 70.5 % | matmul | 29 % |
| 19 | generic_op |  [256, 4, 256, | 2 | 0.72 | 2.1 % | 72.6 % | eltwise | 19 % |
| 20 | linear | [1, 256, 256, 128]BFL|[128, 128]BFL|[128]BFL [1, 256, 256, | 4 | 0.66 | 1.9 % | 74.6 % | matmul | 65 % |
| 21 | permute | [1, 64, 256, 256]BFL [1, 256, 256, | 4 | 0.64 | 1.9 % | 76.4 % | layout | 32 % |
| 22 | linear | [1, 256, 256, 128]BFL|[128, 512]BFL|[512]BFL [1, 256, 256, | 1 | 0.61 | 1.8 % | 78.2 % | matmul | 38 % |
| 23 | add_ | [256, 256, 128]BFL|[128]BFL [256, 256, | 2 | 0.55 | 1.6 % | 79.9 % | eltwise | 43 % |
| 24 | permute | [81, 4, 256, 32]BFL [81, 4, 32, | 6 | 0.53 | 1.6 % | 81.4 % | layout | 49 % |
| 25 | permute | [256, 256, 128]BFL [256, 256, | 2 | 0.52 | 1.5 % | 83.0 % | layout | 42 % |
| 26 | multiply_ | [256, 256, 128]BFL|[256, 256, 128]BFL [256, 256, | 2 | 0.44 | 1.3 % | 84.3 % | eltwise | 92 % |
| 27 | multiply_ | [1, 256, 256, 128]BFL|[1, 256, 256, 128]BFL [1, 256, 256, | 2 | 0.38 | 1.1 % | 85.4 % | eltwise | 84 % |
| 28 | layer_norm | [256, 256, 128]BFL|[128]BFL|[128]BFL [256, 256, | 2 | 0.36 | 1.1 % | 86.4 % | layernorm | 85 % |
| 29 | concat |  [1, 256, 256, | 2 | 0.35 | 1.0 % | 87.5 % | layout | 38 % |
| 30 | add_ | [13, 4, 256, 256]FLO|[1, 4, 256, 256]FLO [13, 4, 256, | 2 | 0.35 | 1.0 % | 88.5 % | eltwise | 63 % |
| 31 | linear | [256, 256, 128]BFL|[128, 128]BFL|[128]BFL [256, 256, | 2 | 0.34 | 1.0 % | 89.5 % | matmul | 75 % |
| 32 | clone | [1, 256, 256, 64]BFL [1, 256, 256, | 4 | 0.34 | 1.0 % | 90.5 % | layout | 73 % |
