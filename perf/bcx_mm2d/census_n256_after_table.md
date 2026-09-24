b = 0.19786 s; sum of per-op replay = 0.19805 s (100.1%); sum of per-op roofs = 0.09464 s; ops >= 200 us per call = 77.4% of b

| # | op (verb:ttnn op, phase) | shape | dtype | calls | time ms | share of b | cum | class | its own roof, us/call | % of its roof |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | triangle_attention:matmul bwd | `[128, 4, 128, 256]:bf16,[128, 4, 128, 32]:bf16 ; transpose_a=True` | bf16 | 16 | 17.24 (1078 us/call) | 8.7% | 8.7% | matmul | 107.2 | 9.9% |
| 2 | reshape:reshape fwd | `[256, 256, 128]:bf16,(256,256,4,32)` | bf16 | 8 | 16.12 (2015 us/call) | 8.1% | 16.9% | data-movement | 400.6 | 19.9% |
| 3 | triangle_attention:matmul bwd | `[128, 4, 128, 256]:bf16,[128, 4, 256, 32]:bf16` | bf16 | 8 | 13.34 (1667 us/call) | 6.7% | 23.6% | matmul | 107.2 | 6.4% |
| 4 | triangle_attention:matmul fwd | `[128, 4, 128, 256]:bf16,[128, 4, 256, 32]:bf16` | bf16 | 8 | 13.29 (1662 us/call) | 6.7% | 30.3% | matmul | 107.2 | 6.5% |
| 5 | reshape:reshape bwd | `[256, 256, 4, 32]:bf16,(256,256,128)` | bf16 | 8 | 12.13 (1517 us/call) | 6.1% | 36.5% | data-movement | 400.6 | 26.4% |
| 6 | triangle_attention:matmul bwd | `[128, 4, 128, 32]:bf16,[128, 4, 256, 32]:bf16 ; transpose_b=True` | bf16 | 16 | 9.25 (578 us/call) | 4.7% | 41.1% | matmul | 107.2 | 18.5% |
| 7 | permute:permute bwd | `[128, 256, 256]:bf16,(1,2,0)` | bf16 | 4 | 6.47 (1618 us/call) | 3.3% | 44.4% | data-movement | 89.0 | 5.5% |
| 8 | permute:permute bwd | `[256, 4, 256, 32]:bf16,(0,2,1,3)` | bf16 | 8 | 5.89 (736 us/call) | 3.0% | 47.4% | data-movement | 400.6 | 54.4% |
| 9 | permute:permute fwd | `[256, 256, 4, 32]:bf16,(0,2,1,3)` | bf16 | 8 | 4.89 (611 us/call) | 2.5% | 49.8% | data-movement | 400.6 | 65.6% |
| 10 | triangle_attention:matmul fwd | `[128, 4, 128, 32]:bf16,[128, 4, 256, 32]:bf16 ; transpose_b=True` | bf16 | 8 | 4.63 (579 us/call) | 2.3% | 52.2% | matmul | 107.2 | 18.5% |
| 11 | reshape:reshape bwd | `[256, 256, 128]:bf16,(256,256,4,32)` | bf16 | 2 | 4.03 (2013 us/call) | 2.0% | 54.2% | data-movement | 400.6 | 19.9% |
| 12 | linear:add bwd | `[256, 256, 128]:fp32,[256, 256, 128]:fp32` | fp32 | 17 | 3.96 (233 us/call) | 2.0% | 56.2% | elementwise | 233.9 | 100.5% |
| 13 | triangle_attention:multiply bwd | `[128, 4, 128, 256]:bf16,[128, 4, 128, 256]:bf16` | bf16 | 16 | 3.77 (236 us/call) | 1.9% | 58.1% | elementwise | 233.9 | 99.2% |
| 14 | pair_contract:permute fwd | `[128, 256, 256]:bf16,(1,2,0)` | bf16 | 2 | 3.35 (1674 us/call) | 1.7% | 59.8% | data-movement | 89.0 | 5.3% |
| 15 | linear:typecast bwd | `[256, 256, 128]:bf16,DataType` | bf16 | 22 | 3.20 (146 us/call) | 1.6% | 61.4% | data-movement | 133.5 | 91.7% |
| 16 | reshape:reshape fwd | `[256, 256, 4, 32]:bf16,(256,256,128)` | bf16 | 2 | 3.03 (1515 us/call) | 1.5% | 63.0% | data-movement | 400.6 | 26.4% |
| 17 | triangle_attention:softmax fwd | `[128, 4, 128, 256]:bf16 ; dim=-1` | bf16 | 8 | 2.85 (356 us/call) | 1.4% | 64.4% | softmax | 178.1 | 50.0% |
| 18 | triangle_attention:softmax bwd | `[128, 4, 128, 256]:bf16 ; dim=-1` | bf16 | 8 | 2.85 (356 us/call) | 1.4% | 65.9% | softmax | 178.1 | 50.0% |
| 19 | linear:linear fwd | `[65536, 128]:bf16,[128, 128]:bf16 ; dtype=None` | bf16 | 22 | 2.77 (126 us/call) | 1.4% | 67.3% | matmul | 78.0 | 62.0% |
| 20 | linear:matmul bwd | `[65536, 128]:bf16,[128, 128]:bf16 ; transpose_b=True` | bf16 | 22 | 2.75 (125 us/call) | 1.4% | 68.6% | matmul | 78.0 | 62.5% |
| 21 | layer_norm:multiply bwd | `[256, 256, 128]:bf16,[256, 256, 1]:bf16` | bf16 | 21 | 2.73 (130 us/call) | 1.4% | 70.0% | elementwise | 87.7 | 67.5% |
| 22 | triangle_attention:add fwd | `[128, 4, 128, 256]:bf16,[1, 4, 128, 256]:bf16` | bf16 | 8 | 1.87 (233 us/call) | 0.9% | 71.0% | elementwise | 156.5 | 67.0% |
| 23 | triangle_attention:add bwd | `[128, 4, 128, 256]:bf16,[1, 4, 128, 256]:bf16` | bf16 | 8 | 1.86 (233 us/call) | 0.9% | 71.9% | elementwise | 156.5 | 67.2% |
| 24 | mul:multiply bwd | `[256, 256, 512]:bf16,[256, 256, 512]:bf16` | bf16 | 4 | 1.84 (461 us/call) | 0.9% | 72.8% | elementwise | 467.8 | 101.5% |
| 25 | layer_norm:subtract bwd | `[256, 256, 128]:bf16,[256, 256, 1]:bf16` | bf16 | 14 | 1.80 (129 us/call) | 0.9% | 73.7% | elementwise | 87.7 | 68.0% |
| 26 | layer_norm:multiply bwd | `[256, 256, 128]:bf16,[256, 256, 128]:bf16` | bf16 | 14 | 1.73 (124 us/call) | 0.9% | 74.6% | elementwise | 116.9 | 94.6% |
| 27 | layer_norm:mean bwd | `[256, 256, 128]:bf16 ; dim=-1,keepdim=True` | bf16 | 28 | 1.66 (59 us/call) | 0.8% | 75.5% | reduction | 55.6 | 93.8% |
| 28 | pair_contract:permute fwd | `[256, 256, 128]:bf16,(2,0,1)` | bf16 | 4 | 1.65 (413 us/call) | 0.8% | 76.3% | data-movement | 89.0 | 21.6% |
| 29 | triangle_attention:sum bwd | `[128, 4, 128, 256]:bf16 ; dim=-1,keepdim=True` | bf16 | 16 | 1.61 (101 us/call) | 0.8% | 77.1% | reduction | 100.2 | 99.5% |
| 30 | triangle_attention:multiply fwd | `[128, 4, 128, 256]:bf16,0.1767766952966369` | bf16 | 8 | 1.59 (199 us/call) | 0.8% | 77.9% | elementwise | 155.9 | 78.3% |
| 31 | triangle_attention:multiply bwd | `[128, 4, 128, 256]:bf16,0.1767766952966369` | bf16 | 8 | 1.58 (198 us/call) | 0.8% | 78.7% | elementwise | 155.9 | 78.8% |
| 32 | mul:multiply bwd | `[256, 256, 128]:bf16,[256, 256, 128]:bf16` | bf16 | 12 | 1.49 (124 us/call) | 0.8% | 79.5% | elementwise | 116.9 | 94.3% |
| 33 | sigmoid:multiply bwd | `[256, 256, 128]:bf16,[256, 256, 128]:bf16` | bf16 | 12 | 1.47 (122 us/call) | 0.7% | 80.2% | elementwise | 116.9 | 95.6% |
| 34 | permute:permute fwd | `[256, 4, 256, 32]:bf16,(0,2,1,3)` | bf16 | 2 | 1.46 (730 us/call) | 0.7% | 80.9% | data-movement | 400.6 | 54.9% |
| 35 | triangle_attention:subtract bwd | `[128, 4, 128, 256]:bf16,[128, 4, 128, 1]:bf16` | bf16 | 8 | 1.42 (178 us/call) | 0.7% | 81.7% | elementwise | 165.7 | 93.1% |
| 36 | permute:permute bwd | `[256, 256, 4, 32]:bf16,(0,2,1,3)` | bf16 | 2 | 1.21 (606 us/call) | 0.6% | 82.3% | data-movement | 400.6 | 66.1% |
| 37 | harness:typecast bwd | `[256, 256, 128]:fp32,DataType` | fp32 | 10 | 1.19 (119 us/call) | 0.6% | 82.9% | data-movement | 133.5 | 112.6% |
| 38 | layer_norm:multiply bwd | `[256, 256, 128]:bf16,[1, 128]:bf16` | bf16 | 7 | 1.17 (168 us/call) | 0.6% | 83.5% | elementwise | 78.0 | 46.6% |
| 39 | layer_norm:typecast bwd | `[256, 256, 128]:bf16,DataType` | bf16 | 8 | 1.16 (145 us/call) | 0.6% | 84.1% | data-movement | 133.5 | 92.1% |
| 40 | sigmoid:typecast bwd | `[256, 256, 512]:bf16,DataType` | bf16 | 2 | 1.12 (559 us/call) | 0.6% | 84.6% | data-movement | 534.2 | 95.6% |
| 41 | linear:matmul bwd | `[65536, 512]:bf16,[128, 512]:bf16 ; transpose_b=True` | bf16 | 2 | 0.97 (487 us/call) | 0.5% | 85.1% | matmul | 195.2 | 40.1% |
| 42 | layer_norm:layer_norm fwd | `[256, 256, 128]:bf16` | bf16 | 7 | 0.94 (135 us/call) | 0.5% | 85.6% | layernorm | 89.1 | 66.1% |
| 43 | mul:multiply fwd | `[256, 256, 512]:bf16,[256, 256, 512]:bf16` | bf16 | 2 | 0.93 (464 us/call) | 0.5% | 86.1% | elementwise | 467.8 | 100.7% |
| 44 | layer_norm:add bwd | `[256, 256, 128]:fp32,[256, 256, 128]:fp32` | fp32 | 4 | 0.93 (232 us/call) | 0.5% | 86.5% | elementwise | 233.9 | 101.0% |
| 45 | sigmoid:multiply bwd | `[256, 256, 512]:bf16,[256, 256, 512]:bf16` | bf16 | 2 | 0.92 (461 us/call) | 0.5% | 87.0% | elementwise | 467.8 | 101.4% |
| 46 | linear:linear fwd | `[65536, 128]:bf16,[128, 512]:bf16 ; dtype=None` | bf16 | 2 | 0.91 (454 us/call) | 0.5% | 87.5% | matmul | 195.2 | 43.0% |
| 47 | sigmoid:add bwd | `[256, 256, 512]:fp32,[256, 256, 512]:fp32` | fp32 | 1 | 0.90 (900 us/call) | 0.5% | 87.9% | elementwise | 935.5 | 103.9% |
| 48 | permute:permute fwd | `[256, 256, 128]:bf16,(1,0,2)` | bf16 | 2 | 0.87 (435 us/call) | 0.4% | 88.4% | data-movement | 89.0 | 20.5% |
| 49 | permute:permute bwd | `[256, 256, 128]:bf16,(1,0,2)` | bf16 | 2 | 0.87 (434 us/call) | 0.4% | 88.8% | data-movement | 89.0 | 20.5% |
| 50 | layer_norm:subtract bwd | `[256, 256, 128]:bf16,[256, 256, 128]:bf16` | bf16 | 7 | 0.85 (122 us/call) | 0.4% | 89.2% | elementwise | 116.9 | 96.2% |
| 51 | permute:permute bwd | `[256, 256, 128]:bf16,(2,0,1)` | bf16 | 2 | 0.81 (404 us/call) | 0.4% | 89.6% | data-movement | 89.0 | 22.0% |
| 52 | triangle_attention:sum bwd | `[128, 4, 128, 256]:bf16 ; dim=0,keepdim=True` | bf16 | 8 | 0.79 (98 us/call) | 0.4% | 90.0% | reduction | 89.7 | 91.1% |
| | 68 further signatures | | | 351 | 19.92 | 10.1% | | | | |
| | **OVERHEAD** = b - sum(per-op) | | | | -0.19 | **-0.1%** | | overhead | | |

| class | calls | time ms | share of b | roof ms | % of roof | brought to roof saves ms | of b |
|---|---|---|---|---|---|---|---|
| data-movement | 325 | 75.57 | 38.2% | 41.24 | 54.6% | 34.33 | 17.4% |
| matmul | 116 | 71.15 | 36.0% | 11.50 | 16.2% | 59.65 | 30.1% |
| elementwise | 261 | 40.25 | 20.3% | 34.54 | 85.8% | 5.71 | 2.9% |
| softmax | 16 | 5.70 | 2.9% | 2.85 | 50.0% | 2.85 | 1.4% |
| reduction | 52 | 4.06 | 2.1% | 3.88 | 95.5% | 0.18 | 0.1% |
| layernorm | 7 | 0.94 | 0.5% | 0.62 | 66.1% | 0.32 | 0.2% |
| dealloc | 24 | 0.37 | 0.2% | 0.00 | 0.0% | 0.37 | 0.2% |

| family | calls | time ms | share of b | roof ms | % of roof | brought to roof saves ms | of b |
|---|---|---|---|---|---|---|---|
| tri-attention batched matmuls (QK^T, PV and their grads) | 56 | 57.76 | 29.2% | 6.00 | 10.4% | 51.76 | 26.2% |
| all other elementwise | 261 | 40.25 | 20.3% | 34.54 | 85.8% | 5.71 | 2.9% |
| heads reshape [N,N,128] <-> [N,N,4,32] | 20 | 35.31 | 17.8% | 8.01 | 22.7% | 27.30 | 13.8% |
| permutes | 40 | 27.75 | 14.0% | 9.49 | 34.2% | 18.26 | 9.2% |
| trimul per-channel matmuls | 60 | 13.40 | 6.8% | 5.50 | 41.1% | 7.89 | 4.0% |
| all other data-movement | 265 | 12.51 | 6.3% | 23.74 | 189.7% | -11.23 | -5.7% |
| all other softmax | 16 | 5.70 | 2.9% | 2.85 | 50.0% | 2.85 | 1.4% |
| all other reduction | 52 | 4.06 | 2.1% | 3.88 | 95.5% | 0.18 | 0.1% |
| all other layernorm | 7 | 0.94 | 0.5% | 0.62 | 66.1% | 0.32 | 0.2% |
| all other dealloc | 24 | 0.37 | 0.2% | 0.00 | 0.0% | 0.37 | 0.2% |
