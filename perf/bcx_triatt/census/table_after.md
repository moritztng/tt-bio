b = 0.13684 s; sum of per-op replay = 0.14765 s (107.9%); sum of per-op roofs = 0.07129 s; ops >= 200 us per call = 71.3% of b

| # | op (verb:ttnn op, phase) | shape | dtype | calls | time ms | share of b | cum | class | its own roof, us/call | % of its roof |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | linear:linear fwd | `[256, 256, 128]:bf16,[128, 128]:bf16 ; dtype=None` | bf16 | 22 | 17.81 (810 us/call) | 13.0% | 13.0% | matmul | 76.7 | 9.5% |
| 2 | linear:matmul bwd | `[256, 256, 128]:bf16,[128, 128]:bf16 ; transpose_b=True` | bf16 | 22 | 17.65 (802 us/call) | 12.9% | 25.9% | matmul | 76.7 | 9.6% |
| 3 | permute:permute bwd | `[128, 256, 256]:bf16,(1,2,0)` | bf16 | 4 | 6.52 (1629 us/call) | 4.8% | 30.7% | data-movement | 87.7 | 5.4% |
| 4 | linear:matmul bwd | `[256, 256, 512]:bf16,[128, 512]:bf16 ; transpose_b=True` | bf16 | 2 | 5.68 (2841 us/call) | 4.2% | 34.8% | matmul | 192.0 | 6.8% |
| 5 | linear:add bwd | `[256, 256, 128]:fp32,[256, 256, 128]:fp32` | fp32 | 17 | 4.06 (239 us/call) | 3.0% | 37.8% | elementwise | 235.8 | 98.7% |
| 6 | triangle_attention:multiply bwd | `[128, 4, 128, 256]:bf16,[128, 4, 128, 256]:bf16` | bf16 | 16 | 3.89 (243 us/call) | 2.8% | 40.6% | elementwise | 235.8 | 97.0% |
| 7 | linear:typecast bwd | `[256, 256, 128]:bf16,DataType` | bf16 | 22 | 3.29 (150 us/call) | 2.4% | 43.0% | data-movement | 131.5 | 87.9% |
| 8 | pair_contract:permute fwd | `[128, 256, 256]:bf16,(1,2,0)` | bf16 | 2 | 3.25 (1624 us/call) | 2.4% | 45.4% | data-movement | 87.7 | 5.4% |
| 9 | triangle_attention:softmax fwd | `[128, 4, 128, 256]:bf16 ; dim=-1` | bf16 | 8 | 2.89 (362 us/call) | 2.1% | 47.5% | softmax | 175.4 | 48.5% |
| 10 | triangle_attention:softmax bwd | `[128, 4, 128, 256]:bf16 ; dim=-1` | bf16 | 8 | 2.89 (361 us/call) | 2.1% | 49.6% | softmax | 175.4 | 48.6% |
| 11 | layer_norm:multiply bwd | `[256, 256, 128]:bf16,[256, 256, 1]:bf16` | bf16 | 21 | 2.88 (137 us/call) | 2.1% | 51.7% | elementwise | 88.4 | 64.6% |
| 12 | linear:linear fwd | `[256, 256, 512]:bf16,[512, 128]:bf16 ; dtype=None` | bf16 | 1 | 2.82 (2819 us/call) | 2.1% | 53.8% | matmul | 192.0 | 6.8% |
| 13 | triangle_attention:matmul bwd | `[128, 4, 128, 32]:bf16,[128, 4, 256, 32]:bf16 ; transpose_a=False,transpose_b=True` | bf16 | 16 | 2.34 (146 us/call) | 1.7% | 55.5% | matmul | 105.4 | 72.2% |
| 14 | linear:linear fwd | `[256, 256, 128]:bf16,[128, 512]:bf16 ; dtype=None` | bf16 | 2 | 2.14 (1069 us/call) | 1.6% | 57.1% | matmul | 192.0 | 18.0% |
| 15 | mul:multiply bwd | `[256, 256, 128]:bf16,[256, 256, 128]:bf16` | bf16 | 16 | 2.08 (130 us/call) | 1.5% | 58.6% | elementwise | 117.9 | 90.7% |
| 16 | sigmoid:multiply bwd | `[256, 256, 128]:bf16,[256, 256, 128]:bf16` | bf16 | 16 | 2.08 (130 us/call) | 1.5% | 60.1% | elementwise | 117.9 | 90.9% |
| 17 | triangle_attention:matmul bwd | `[128, 4, 128, 256]:bf16,[128, 4, 128, 32]:bf16 ; transpose_a=True,transpose_b=False` | bf16 | 16 | 2.02 (126 us/call) | 1.5% | 61.6% | matmul | 105.4 | 83.5% |
| 18 | triangle_attention:add fwd | `[128, 4, 128, 256]:bf16,[1, 4, 128, 256]:bf16` | bf16 | 8 | 1.92 (241 us/call) | 1.4% | 63.0% | elementwise | 157.8 | 65.6% |
| 19 | triangle_attention:add bwd | `[128, 4, 128, 256]:bf16,[1, 4, 128, 256]:bf16` | bf16 | 8 | 1.92 (240 us/call) | 1.4% | 64.4% | elementwise | 157.8 | 65.7% |
| 20 | layer_norm:subtract bwd | `[256, 256, 128]:bf16,[256, 256, 1]:bf16` | bf16 | 14 | 1.91 (136 us/call) | 1.4% | 65.8% | elementwise | 88.4 | 64.9% |
| 21 | mul:multiply bwd | `[256, 256, 512]:bf16,[256, 256, 512]:bf16` | bf16 | 4 | 1.88 (469 us/call) | 1.4% | 67.2% | elementwise | 471.6 | 100.5% |
| 22 | layer_norm:multiply bwd | `[256, 256, 128]:bf16,[256, 256, 128]:bf16` | bf16 | 14 | 1.82 (130 us/call) | 1.3% | 68.5% | elementwise | 117.9 | 90.9% |
| 23 | layer_norm:mean bwd | `[256, 256, 128]:bf16 ; dim=-1,keepdim=True` | bf16 | 28 | 1.80 (64 us/call) | 1.3% | 69.8% | reduction | 54.8 | 85.4% |
| 24 | linear:matmul bwd | `[256, 256, 4]:bf16,[128, 4]:bf16 ; transpose_b=True` | bf16 | 2 | 1.69 (847 us/call) | 1.2% | 71.0% | matmul | 47.9 | 5.7% |
| 25 | triangle_attention:sum bwd | `[128, 4, 128, 256]:bf16 ; dim=-1,keepdim=True` | bf16 | 16 | 1.69 (106 us/call) | 1.2% | 72.3% | reduction | 98.7 | 93.2% |
| 26 | pair_contract:permute fwd | `[256, 256, 128]:bf16,(2,0,1)` | bf16 | 4 | 1.65 (412 us/call) | 1.2% | 73.5% | data-movement | 87.7 | 21.3% |
| 27 | triangle_attention:multiply fwd | `[128, 4, 128, 256]:bf16,0.1767766952966369` | bf16 | 8 | 1.64 (205 us/call) | 1.2% | 74.7% | elementwise | 157.2 | 76.8% |
| 28 | triangle_attention:multiply bwd | `[128, 4, 128, 256]:bf16,0.1767766952966369` | bf16 | 8 | 1.63 (204 us/call) | 1.2% | 75.9% | elementwise | 157.2 | 77.0% |
| 29 | triangle_attention:subtract bwd | `[128, 4, 128, 256]:bf16,[128, 4, 128, 1]:bf16` | bf16 | 8 | 1.48 (185 us/call) | 1.1% | 77.0% | elementwise | 167.0 | 90.1% |
| 30 | harness:typecast bwd | `[256, 256, 128]:fp32,DataType` | fp32 | 10 | 1.24 (124 us/call) | 0.9% | 77.9% | data-movement | 131.5 | 106.3% |
| 31 | layer_norm:multiply bwd | `[256, 256, 128]:bf16,[1, 128]:bf16` | bf16 | 7 | 1.22 (174 us/call) | 0.9% | 78.8% | elementwise | 78.6 | 45.2% |
| 32 | layer_norm:typecast bwd | `[256, 256, 128]:bf16,DataType` | bf16 | 8 | 1.20 (150 us/call) | 0.9% | 79.6% | data-movement | 131.5 | 87.7% |
| 33 | triangle_attention:matmul fwd | `[128, 4, 128, 32]:bf16,[128, 4, 256, 32]:bf16 ; transpose_a=False,transpose_b=True` | bf16 | 8 | 1.17 (146 us/call) | 0.9% | 80.5% | matmul | 105.4 | 72.3% |
| 34 | sigmoid:typecast bwd | `[256, 256, 512]:bf16,DataType` | bf16 | 2 | 1.12 (560 us/call) | 0.8% | 81.3% | data-movement | 526.2 | 94.0% |
| 35 | linear:linear fwd | `[256, 256, 128]:bf16,[128, 4]:bf16 ; dtype=None` | bf16 | 2 | 1.10 (549 us/call) | 0.8% | 82.1% | matmul | 47.9 | 8.7% |
| 36 | linear:matmul bwd | `[256, 256, 128]:bf16,[512, 128]:bf16 ; transpose_b=True` | bf16 | 1 | 1.07 (1066 us/call) | 0.8% | 82.9% | matmul | 192.0 | 18.0% |
| 37 | mul:multiply fwd | `[256, 256, 128]:bf16,[256, 256, 128]:bf16` | bf16 | 8 | 1.05 (131 us/call) | 0.8% | 83.7% | elementwise | 117.9 | 90.2% |
| 38 | sigmoid:sigmoid fwd | `[256, 256, 128]:bf16` | bf16 | 8 | 1.01 (127 us/call) | 0.7% | 84.4% | elementwise | 78.6 | 62.0% |
| 39 | layer_norm:layer_norm fwd | `[256, 256, 128]:bf16` | bf16 | 7 | 0.97 (138 us/call) | 0.7% | 85.1% | layernorm | 87.7 | 63.5% |
| 40 | layer_norm:add bwd | `[256, 256, 128]:fp32,[256, 256, 128]:fp32` | fp32 | 4 | 0.96 (239 us/call) | 0.7% | 85.8% | elementwise | 235.8 | 98.7% |
| 41 | mul:multiply fwd | `[256, 256, 512]:bf16,[256, 256, 512]:bf16` | bf16 | 2 | 0.94 (470 us/call) | 0.7% | 86.5% | elementwise | 471.6 | 100.3% |
| 42 | sigmoid:multiply bwd | `[256, 256, 512]:bf16,[256, 256, 512]:bf16` | bf16 | 2 | 0.94 (469 us/call) | 0.7% | 87.2% | elementwise | 471.6 | 100.5% |
| 43 | triangle_attention:matmul bwd | `[128, 4, 128, 256]:bf16,[128, 4, 256, 32]:bf16 ; transpose_a=False,transpose_b=False` | bf16 | 8 | 0.93 (117 us/call) | 0.7% | 87.9% | matmul | 105.4 | 90.4% |
| 44 | triangle_attention:matmul fwd | `[128, 4, 128, 256]:bf16,[128, 4, 256, 32]:bf16 ; transpose_a=False,transpose_b=False` | bf16 | 8 | 0.93 (116 us/call) | 0.7% | 88.5% | matmul | 105.4 | 90.8% |
| 45 | layer_norm:subtract bwd | `[256, 256, 128]:bf16,[256, 256, 128]:bf16` | bf16 | 7 | 0.90 (129 us/call) | 0.7% | 89.2% | elementwise | 117.9 | 91.5% |
| 46 | sigmoid:add bwd | `[256, 256, 512]:fp32,[256, 256, 512]:fp32` | fp32 | 1 | 0.90 (899 us/call) | 0.7% | 89.8% | elementwise | 943.3 | 104.9% |
| 47 | permute:permute bwd | `[256, 256, 128]:bf16,(1,0,2)` | bf16 | 2 | 0.88 (439 us/call) | 0.6% | 90.5% | data-movement | 87.7 | 20.0% |
| | 64 further signatures | | | 305 | 23.83 | 17.4% | | | | |
| | **OVERHEAD** = b - sum(per-op) | | | | -10.82 | **-7.9%** | | overhead | | |

| class | calls | time ms | share of b | roof ms | % of roof | brought to roof saves ms | of b |
|---|---|---|---|---|---|---|---|
| matmul | 116 | 61.88 | 45.2% | 11.31 | 18.3% | 50.57 | 37.0% |
| elementwise | 261 | 42.38 | 31.0% | 34.83 | 82.2% | 7.55 | 5.5% |
| data-movement | 257 | 31.57 | 23.1% | 17.91 | 56.7% | 13.67 | 10.0% |
| softmax | 16 | 5.78 | 4.2% | 2.81 | 48.6% | 2.97 | 2.2% |
| reduction | 52 | 4.31 | 3.2% | 3.82 | 88.5% | 0.49 | 0.4% |
| layernorm | 7 | 0.97 | 0.7% | 0.61 | 63.5% | 0.35 | 0.3% |
| dealloc | 24 | 0.75 | 0.5% | 0.00 | 0.0% | 0.75 | 0.5% |

| family | calls | time ms | share of b | roof ms | % of roof | brought to roof saves ms | of b |
|---|---|---|---|---|---|---|---|
| 3D pair-tensor linears (fwd + dX) | 54 | 49.96 | 36.5% | 4.72 | 9.4% | 45.24 | 33.1% |
| all other elementwise | 261 | 42.38 | 31.0% | 34.83 | 82.2% | 7.55 | 5.5% |
| all other data-movement | 237 | 17.27 | 12.6% | 16.45 | 95.3% | 0.81 | 0.6% |
| permutes | 20 | 14.31 | 10.5% | 1.45 | 10.2% | 12.85 | 9.4% |
| tri-attention batched matmuls (QK^T, PV and their grads) | 56 | 7.39 | 5.4% | 5.90 | 79.9% | 1.48 | 1.1% |
| all other softmax | 16 | 5.78 | 4.2% | 2.81 | 48.6% | 2.97 | 2.2% |
| trimul per-channel matmuls | 6 | 4.53 | 3.3% | 0.69 | 15.2% | 3.84 | 2.8% |
| all other reduction | 52 | 4.31 | 3.2% | 3.82 | 88.5% | 0.49 | 0.4% |
| all other layernorm | 7 | 0.97 | 0.7% | 0.61 | 63.5% | 0.35 | 0.3% |
| all other dealloc | 24 | 0.75 | 0.5% | 0.00 | 0.0% | 0.75 | 0.5% |
