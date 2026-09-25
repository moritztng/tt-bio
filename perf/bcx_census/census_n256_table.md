b = 0.23090 s; sum of per-op replay = 0.24572 s (106.4%); sum of per-op roofs = 0.08471 s; ops >= 200 us per call = 89.9% of b

| # | op (verb:ttnn op, phase) | shape | dtype | calls | time ms | share of b | cum | class | its own roof, us/call | % of its roof |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | linear:linear fwd | `[256, 256, 128]:bf16,[128, 128]:bf16 ; dtype=None` | bf16 | 22 | 18.19 (827 us/call) | 7.9% | 7.9% | matmul | 77.8 | 9.4% |
| 2 | linear:matmul bwd | `[256, 256, 128]:bf16,[128, 128]:bf16 ; transpose_b=True` | bf16 | 22 | 18.00 (818 us/call) | 7.8% | 15.7% | matmul | 77.8 | 9.5% |
| 3 | triangle_attention:matmul bwd | `[128, 4, 128, 256]:bf16,[128, 4, 128, 32]:bf16 ; transpose_a=True` | bf16 | 16 | 17.62 (1101 us/call) | 7.6% | 23.3% | matmul | 106.9 | 9.7% |
| 4 | reshape:reshape fwd | `[256, 256, 128]:bf16,(256,256,4,32)` | bf16 | 8 | 16.29 (2036 us/call) | 7.1% | 30.4% | data-movement | 403.2 | 19.8% |
| 5 | triangle_attention:matmul bwd | `[128, 4, 128, 256]:bf16,[128, 4, 256, 32]:bf16` | bf16 | 8 | 13.27 (1659 us/call) | 5.7% | 36.1% | matmul | 106.9 | 6.4% |
| 6 | triangle_attention:matmul fwd | `[128, 4, 128, 256]:bf16,[128, 4, 256, 32]:bf16` | bf16 | 8 | 13.23 (1654 us/call) | 5.7% | 41.8% | matmul | 106.9 | 6.5% |
| 7 | reshape:reshape bwd | `[256, 256, 4, 32]:bf16,(256,256,128)` | bf16 | 8 | 12.36 (1545 us/call) | 5.4% | 47.2% | data-movement | 403.2 | 26.1% |
| 8 | triangle_attention:matmul bwd | `[128, 4, 128, 32]:bf16,[128, 4, 256, 32]:bf16 ; transpose_b=True` | bf16 | 16 | 9.60 (600 us/call) | 4.2% | 51.3% | matmul | 106.9 | 17.8% |
| 9 | permute:permute bwd | `[128, 256, 256]:bf16,(1,2,0)` | bf16 | 4 | 6.65 (1662 us/call) | 2.9% | 54.2% | data-movement | 89.6 | 5.4% |
| 10 | permute:permute bwd | `[256, 4, 256, 32]:bf16,(0,2,1,3)` | bf16 | 8 | 5.98 (748 us/call) | 2.6% | 56.8% | data-movement | 403.2 | 53.9% |
| 11 | linear:matmul bwd | `[256, 256, 512]:bf16,[128, 512]:bf16 ; transpose_b=True` | bf16 | 2 | 5.75 (2875 us/call) | 2.5% | 59.3% | matmul | 194.7 | 6.8% |
| 12 | permute:permute fwd | `[256, 256, 4, 32]:bf16,(0,2,1,3)` | bf16 | 8 | 5.01 (626 us/call) | 2.2% | 61.5% | data-movement | 403.2 | 64.4% |
| 13 | triangle_attention:matmul fwd | `[128, 4, 128, 32]:bf16,[128, 4, 256, 32]:bf16 ; transpose_b=True` | bf16 | 8 | 4.71 (589 us/call) | 2.0% | 63.5% | matmul | 106.9 | 18.2% |
| 14 | linear:add bwd | `[256, 256, 128]:fp32,[256, 256, 128]:fp32` | fp32 | 17 | 4.26 (251 us/call) | 1.8% | 65.4% | elementwise | 239.4 | 95.5% |
| 15 | reshape:reshape bwd | `[256, 256, 128]:bf16,(256,256,4,32)` | bf16 | 2 | 4.10 (2052 us/call) | 1.8% | 67.1% | data-movement | 403.2 | 19.6% |
| 16 | triangle_attention:multiply bwd | `[128, 4, 128, 256]:bf16,[128, 4, 128, 256]:bf16` | bf16 | 16 | 3.96 (248 us/call) | 1.7% | 68.9% | elementwise | 239.4 | 96.6% |
| 17 | linear:typecast bwd | `[256, 256, 128]:bf16,DataType` | bf16 | 22 | 3.45 (157 us/call) | 1.5% | 70.4% | data-movement | 134.4 | 85.7% |
| 18 | pair_contract:permute fwd | `[128, 256, 256]:bf16,(1,2,0)` | bf16 | 2 | 3.31 (1656 us/call) | 1.4% | 71.8% | data-movement | 89.6 | 5.4% |
| 19 | reshape:reshape fwd | `[256, 256, 4, 32]:bf16,(256,256,128)` | bf16 | 2 | 3.08 (1542 us/call) | 1.3% | 73.1% | data-movement | 403.2 | 26.1% |
| 20 | triangle_attention:softmax bwd | `[128, 4, 128, 256]:bf16 ; dim=-1` | bf16 | 8 | 2.93 (367 us/call) | 1.3% | 74.4% | softmax | 179.2 | 48.9% |
| 21 | triangle_attention:softmax fwd | `[128, 4, 128, 256]:bf16 ; dim=-1` | bf16 | 8 | 2.90 (362 us/call) | 1.3% | 75.7% | softmax | 179.2 | 49.5% |
| 22 | linear:linear fwd | `[256, 256, 512]:bf16,[512, 128]:bf16 ; dtype=None` | bf16 | 1 | 2.84 (2841 us/call) | 1.2% | 76.9% | matmul | 194.7 | 6.9% |
| 23 | layer_norm:multiply bwd | `[256, 256, 128]:bf16,[256, 256, 1]:bf16` | bf16 | 21 | 2.80 (133 us/call) | 1.2% | 78.1% | elementwise | 89.8 | 67.3% |
| 24 | linear:linear fwd | `[256, 256, 128]:bf16,[128, 512]:bf16 ; dtype=None` | bf16 | 2 | 2.16 (1082 us/call) | 0.9% | 79.0% | matmul | 194.7 | 18.0% |
| 25 | triangle_attention:add bwd | `[128, 4, 128, 256]:bf16,[1, 4, 128, 256]:bf16` | bf16 | 8 | 1.95 (244 us/call) | 0.8% | 79.9% | elementwise | 160.2 | 65.6% |
| 26 | mul:multiply bwd | `[256, 256, 512]:bf16,[256, 256, 512]:bf16` | bf16 | 4 | 1.93 (482 us/call) | 0.8% | 80.7% | elementwise | 478.8 | 99.4% |
| 27 | triangle_attention:add fwd | `[128, 4, 128, 256]:bf16,[1, 4, 128, 256]:bf16` | bf16 | 8 | 1.90 (237 us/call) | 0.8% | 81.5% | elementwise | 160.2 | 67.6% |
| 28 | layer_norm:subtract bwd | `[256, 256, 128]:bf16,[256, 256, 1]:bf16` | bf16 | 14 | 1.89 (135 us/call) | 0.8% | 82.4% | elementwise | 89.8 | 66.5% |
| 29 | layer_norm:multiply bwd | `[256, 256, 128]:bf16,[256, 256, 128]:bf16` | bf16 | 14 | 1.80 (128 us/call) | 0.8% | 83.1% | elementwise | 119.7 | 93.3% |
| 30 | layer_norm:mean bwd | `[256, 256, 128]:bf16 ; dim=-1,keepdim=True` | bf16 | 28 | 1.75 (63 us/call) | 0.8% | 83.9% | reduction | 56.0 | 89.5% |
| 31 | linear:matmul bwd | `[256, 256, 4]:bf16,[128, 4]:bf16 ; transpose_b=True` | bf16 | 2 | 1.73 (864 us/call) | 0.7% | 84.6% | matmul | 48.6 | 5.6% |
| 32 | triangle_attention:multiply bwd | `[128, 4, 128, 256]:bf16,0.1767766952966369` | bf16 | 8 | 1.72 (215 us/call) | 0.7% | 85.4% | elementwise | 159.6 | 74.3% |
| 33 | triangle_attention:sum bwd | `[128, 4, 128, 256]:bf16 ; dim=-1,keepdim=True` | bf16 | 16 | 1.72 (107 us/call) | 0.7% | 86.1% | reduction | 100.8 | 94.0% |
| 34 | pair_contract:permute fwd | `[256, 256, 128]:bf16,(2,0,1)` | bf16 | 4 | 1.69 (423 us/call) | 0.7% | 86.9% | data-movement | 89.6 | 21.2% |
| 35 | triangle_attention:multiply fwd | `[128, 4, 128, 256]:bf16,0.1767766952966369` | bf16 | 8 | 1.63 (203 us/call) | 0.7% | 87.6% | elementwise | 159.6 | 78.5% |
| 36 | mul:multiply bwd | `[256, 256, 128]:bf16,[256, 256, 128]:bf16` | bf16 | 12 | 1.60 (134 us/call) | 0.7% | 88.3% | elementwise | 119.7 | 89.6% |
| 37 | sigmoid:multiply bwd | `[256, 256, 128]:bf16,[256, 256, 128]:bf16` | bf16 | 12 | 1.55 (129 us/call) | 0.7% | 88.9% | elementwise | 119.7 | 92.8% |
| 38 | permute:permute fwd | `[256, 4, 256, 32]:bf16,(0,2,1,3)` | bf16 | 2 | 1.50 (750 us/call) | 0.6% | 89.6% | data-movement | 403.2 | 53.8% |
| 39 | triangle_attention:subtract bwd | `[128, 4, 128, 256]:bf16,[128, 4, 128, 1]:bf16` | bf16 | 8 | 1.48 (184 us/call) | 0.6% | 90.2% | elementwise | 169.6 | 91.9% |
| | 71 further signatures | | | 306 | 37.41 | 16.2% | | | | |
| | **OVERHEAD** = b - sum(per-op) | | | | -14.82 | **-6.4%** | | overhead | | |

| class | calls | time ms | share of b | roof ms | % of roof | brought to roof saves ms | of b |
|---|---|---|---|---|---|---|---|
| matmul | 116 | 113.95 | 49.3% | 11.47 | 10.1% | 102.47 | 44.4% |
| data-movement | 217 | 77.38 | 33.5% | 30.48 | 39.4% | 46.90 | 20.3% |
| elementwise | 261 | 42.79 | 18.5% | 35.36 | 82.6% | 7.43 | 3.2% |
| softmax | 16 | 5.83 | 2.5% | 2.87 | 49.2% | 2.96 | 1.3% |
| reduction | 52 | 4.28 | 1.9% | 3.90 | 91.1% | 0.38 | 0.2% |
| layernorm | 7 | 0.97 | 0.4% | 0.63 | 64.7% | 0.34 | 0.1% |
| dealloc | 24 | 0.52 | 0.2% | 0.00 | 0.0% | 0.52 | 0.2% |

| family | calls | time ms | share of b | roof ms | % of roof | brought to roof saves ms | of b |
|---|---|---|---|---|---|---|---|
| tri-attention batched matmuls (QK^T, PV and their grads) | 56 | 58.44 | 25.3% | 5.99 | 10.2% | 52.46 | 22.7% |
| 3D pair-tensor linears (fwd + dX) | 54 | 50.86 | 22.0% | 4.79 | 9.4% | 46.07 | 20.0% |
| all other elementwise | 261 | 42.79 | 18.5% | 35.36 | 82.6% | 7.43 | 3.2% |
| heads reshape [N,N,128] <-> [N,N,4,32] | 20 | 35.84 | 15.5% | 8.06 | 22.5% | 27.77 | 12.0% |
| permutes | 40 | 28.30 | 12.3% | 9.55 | 33.7% | 18.75 | 8.1% |
| all other data-movement | 157 | 13.24 | 5.7% | 12.87 | 97.2% | 0.38 | 0.2% |
| all other softmax | 16 | 5.83 | 2.5% | 2.87 | 49.2% | 2.96 | 1.3% |
| trimul per-channel matmuls | 6 | 4.64 | 2.0% | 0.70 | 15.1% | 3.94 | 1.7% |
| all other reduction | 52 | 4.28 | 1.9% | 3.90 | 91.1% | 0.38 | 0.2% |
| all other layernorm | 7 | 0.97 | 0.4% | 0.63 | 64.7% | 0.34 | 0.1% |
| all other dealloc | 24 | 0.52 | 0.2% | 0.00 | 0.0% | 0.52 | 0.2% |
