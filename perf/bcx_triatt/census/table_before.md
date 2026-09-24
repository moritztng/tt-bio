b = 0.23240 s; sum of per-op replay = 0.24346 s (104.8%); sum of per-op roofs = 0.08332 s; ops >= 200 us per call = 87.8% of b

| # | op (verb:ttnn op, phase) | shape | dtype | calls | time ms | share of b | cum | class | its own roof, us/call | % of its roof |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | linear:linear fwd | `[256, 256, 128]:bf16,[128, 128]:bf16 ; dtype=None` | bf16 | 22 | 17.82 (810 us/call) | 7.7% | 7.7% | matmul | 77.2 | 9.5% |
| 2 | linear:matmul bwd | `[256, 256, 128]:bf16,[128, 128]:bf16 ; transpose_b=True` | bf16 | 22 | 17.64 (802 us/call) | 7.6% | 15.3% | matmul | 77.2 | 9.6% |
| 3 | triangle_attention:matmul bwd | `[128, 4, 128, 256]:bf16,[128, 4, 128, 32]:bf16 ; transpose_a=True,transpose_b=False` | bf16 | 16 | 17.49 (1093 us/call) | 7.5% | 22.8% | matmul | 106.1 | 9.7% |
| 4 | reshape:reshape fwd | `[256, 256, 128]:bf16,(256,256,4,32)` | bf16 | 8 | 16.16 (2019 us/call) | 7.0% | 29.7% | data-movement | 394.5 | 19.5% |
| 5 | triangle_attention:matmul bwd | `[128, 4, 128, 256]:bf16,[128, 4, 256, 32]:bf16 ; transpose_a=False,transpose_b=False` | bf16 | 8 | 13.09 (1637 us/call) | 5.6% | 35.4% | matmul | 106.1 | 6.5% |
| 6 | triangle_attention:matmul fwd | `[128, 4, 128, 256]:bf16,[128, 4, 256, 32]:bf16 ; transpose_a=False,transpose_b=False` | bf16 | 8 | 13.09 (1637 us/call) | 5.6% | 41.0% | matmul | 106.1 | 6.5% |
| 7 | reshape:reshape bwd | `[256, 256, 4, 32]:bf16,(256,256,128)` | bf16 | 8 | 12.13 (1517 us/call) | 5.2% | 46.2% | data-movement | 394.5 | 26.0% |
| 8 | triangle_attention:matmul bwd | `[128, 4, 128, 32]:bf16,[128, 4, 256, 32]:bf16 ; transpose_a=False,transpose_b=True` | bf16 | 16 | 9.35 (584 us/call) | 4.0% | 50.2% | matmul | 106.1 | 18.2% |
| 9 | permute:permute bwd | `[128, 256, 256]:bf16,(1,2,0)` | bf16 | 4 | 6.49 (1622 us/call) | 2.8% | 53.0% | data-movement | 87.7 | 5.4% |
| 10 | permute:permute bwd | `[256, 4, 256, 32]:bf16,(0,2,1,3)` | bf16 | 8 | 5.85 (731 us/call) | 2.5% | 55.6% | data-movement | 394.5 | 54.0% |
| 11 | linear:matmul bwd | `[256, 256, 512]:bf16,[128, 512]:bf16 ; transpose_b=True` | bf16 | 2 | 5.68 (2838 us/call) | 2.4% | 58.0% | matmul | 193.2 | 6.8% |
| 12 | permute:permute fwd | `[256, 256, 4, 32]:bf16,(0,2,1,3)` | bf16 | 8 | 4.94 (618 us/call) | 2.1% | 60.1% | data-movement | 394.5 | 63.9% |
| 13 | triangle_attention:matmul fwd | `[128, 4, 128, 32]:bf16,[128, 4, 256, 32]:bf16 ; transpose_a=False,transpose_b=True` | bf16 | 8 | 4.67 (584 us/call) | 2.0% | 62.1% | matmul | 106.1 | 18.2% |
| 14 | linear:add bwd | `[256, 256, 128]:fp32,[256, 256, 128]:fp32` | fp32 | 17 | 4.05 (238 us/call) | 1.7% | 63.9% | elementwise | 236.1 | 99.0% |
| 15 | reshape:reshape bwd | `[256, 256, 128]:bf16,(256,256,4,32)` | bf16 | 2 | 4.03 (2013 us/call) | 1.7% | 65.6% | data-movement | 394.5 | 19.6% |
| 16 | triangle_attention:multiply bwd | `[128, 4, 128, 256]:bf16,[128, 4, 128, 256]:bf16` | bf16 | 16 | 3.88 (242 us/call) | 1.7% | 67.3% | elementwise | 236.1 | 97.4% |
| 17 | linear:typecast bwd | `[256, 256, 128]:bf16,DataType` | bf16 | 22 | 3.29 (150 us/call) | 1.4% | 68.7% | data-movement | 131.5 | 87.9% |
| 18 | pair_contract:permute fwd | `[128, 256, 256]:bf16,(1,2,0)` | bf16 | 2 | 3.25 (1626 us/call) | 1.4% | 70.1% | data-movement | 87.7 | 5.4% |
| 19 | reshape:reshape fwd | `[256, 256, 4, 32]:bf16,(256,256,128)` | bf16 | 2 | 3.03 (1515 us/call) | 1.3% | 71.4% | data-movement | 394.5 | 26.0% |
| 20 | triangle_attention:softmax fwd | `[128, 4, 128, 256]:bf16 ; dim=-1` | bf16 | 8 | 2.90 (363 us/call) | 1.2% | 72.6% | softmax | 175.4 | 48.3% |
| 21 | triangle_attention:softmax bwd | `[128, 4, 128, 256]:bf16 ; dim=-1` | bf16 | 8 | 2.88 (361 us/call) | 1.2% | 73.9% | softmax | 175.4 | 48.6% |
| 22 | layer_norm:multiply bwd | `[256, 256, 128]:bf16,[256, 256, 1]:bf16` | bf16 | 21 | 2.87 (137 us/call) | 1.2% | 75.1% | elementwise | 88.5 | 64.8% |
| 23 | linear:linear fwd | `[256, 256, 512]:bf16,[512, 128]:bf16 ; dtype=None` | bf16 | 1 | 2.82 (2819 us/call) | 1.2% | 76.3% | matmul | 193.2 | 6.9% |
| 24 | linear:linear fwd | `[256, 256, 128]:bf16,[128, 512]:bf16 ; dtype=None` | bf16 | 2 | 2.14 (1068 us/call) | 0.9% | 77.3% | matmul | 193.2 | 18.1% |
| 25 | triangle_attention:add fwd | `[128, 4, 128, 256]:bf16,[1, 4, 128, 256]:bf16` | bf16 | 8 | 1.92 (240 us/call) | 0.8% | 78.1% | elementwise | 158.0 | 65.8% |
| 26 | triangle_attention:add bwd | `[128, 4, 128, 256]:bf16,[1, 4, 128, 256]:bf16` | bf16 | 8 | 1.92 (240 us/call) | 0.8% | 78.9% | elementwise | 158.0 | 65.9% |
| 27 | layer_norm:subtract bwd | `[256, 256, 128]:bf16,[256, 256, 1]:bf16` | bf16 | 14 | 1.90 (136 us/call) | 0.8% | 79.7% | elementwise | 88.5 | 65.2% |
| 28 | mul:multiply bwd | `[256, 256, 512]:bf16,[256, 256, 512]:bf16` | bf16 | 4 | 1.87 (467 us/call) | 0.8% | 80.5% | elementwise | 472.2 | 101.1% |
| 29 | layer_norm:multiply bwd | `[256, 256, 128]:bf16,[256, 256, 128]:bf16` | bf16 | 14 | 1.81 (129 us/call) | 0.8% | 81.3% | elementwise | 118.0 | 91.3% |
| 30 | layer_norm:mean bwd | `[256, 256, 128]:bf16 ; dim=-1,keepdim=True` | bf16 | 28 | 1.79 (64 us/call) | 0.8% | 82.1% | reduction | 54.8 | 85.8% |
| 31 | linear:matmul bwd | `[256, 256, 4]:bf16,[128, 4]:bf16 ; transpose_b=True` | bf16 | 2 | 1.69 (847 us/call) | 0.7% | 82.8% | matmul | 48.2 | 5.7% |
| 32 | triangle_attention:sum bwd | `[128, 4, 128, 256]:bf16 ; dim=-1,keepdim=True` | bf16 | 16 | 1.69 (105 us/call) | 0.7% | 83.5% | reduction | 98.6 | 93.6% |
| 33 | pair_contract:permute fwd | `[256, 256, 128]:bf16,(2,0,1)` | bf16 | 4 | 1.65 (412 us/call) | 0.7% | 84.2% | data-movement | 87.7 | 21.3% |
| 34 | triangle_attention:multiply fwd | `[128, 4, 128, 256]:bf16,0.1767766952966369` | bf16 | 8 | 1.63 (204 us/call) | 0.7% | 84.9% | elementwise | 157.4 | 77.1% |
| 35 | triangle_attention:multiply bwd | `[128, 4, 128, 256]:bf16,0.1767766952966369` | bf16 | 8 | 1.63 (204 us/call) | 0.7% | 85.6% | elementwise | 157.4 | 77.2% |
| 36 | sigmoid:multiply bwd | `[256, 256, 128]:bf16,[256, 256, 128]:bf16` | bf16 | 12 | 1.55 (129 us/call) | 0.7% | 86.3% | elementwise | 118.0 | 91.3% |
| 37 | mul:multiply bwd | `[256, 256, 128]:bf16,[256, 256, 128]:bf16` | bf16 | 12 | 1.55 (129 us/call) | 0.7% | 87.0% | elementwise | 118.0 | 91.4% |
| 38 | triangle_attention:subtract bwd | `[128, 4, 128, 256]:bf16,[128, 4, 128, 1]:bf16` | bf16 | 8 | 1.48 (185 us/call) | 0.6% | 87.6% | elementwise | 167.2 | 90.5% |
| 39 | permute:permute fwd | `[256, 4, 256, 32]:bf16,(0,2,1,3)` | bf16 | 2 | 1.46 (730 us/call) | 0.6% | 88.2% | data-movement | 394.5 | 54.0% |
| 40 | harness:typecast bwd | `[256, 256, 128]:fp32,DataType` | fp32 | 10 | 1.23 (123 us/call) | 0.5% | 88.8% | data-movement | 131.5 | 106.6% |
| 41 | permute:permute bwd | `[256, 256, 4, 32]:bf16,(0,2,1,3)` | bf16 | 2 | 1.22 (612 us/call) | 0.5% | 89.3% | data-movement | 394.5 | 64.5% |
| 42 | layer_norm:multiply bwd | `[256, 256, 128]:bf16,[1, 128]:bf16` | bf16 | 7 | 1.22 (174 us/call) | 0.5% | 89.8% | elementwise | 78.7 | 45.3% |
| 43 | layer_norm:typecast bwd | `[256, 256, 128]:bf16,DataType` | bf16 | 8 | 1.19 (149 us/call) | 0.5% | 90.3% | data-movement | 131.5 | 88.3% |
| | 67 further signatures | | | 279 | 33.51 | 14.4% | | | | |
| | **OVERHEAD** = b - sum(per-op) | | | | -11.06 | **-4.8%** | | overhead | | |

| class | calls | time ms | share of b | roof ms | % of roof | brought to roof saves ms | of b |
|---|---|---|---|---|---|---|---|
| matmul | 116 | 112.20 | 48.3% | 11.39 | 10.1% | 100.81 | 43.4% |
| data-movement | 217 | 77.29 | 33.3% | 29.83 | 38.6% | 47.46 | 20.4% |
| elementwise | 261 | 42.19 | 18.2% | 34.87 | 82.6% | 7.32 | 3.2% |
| softmax | 16 | 5.79 | 2.5% | 2.81 | 48.5% | 2.98 | 1.3% |
| reduction | 52 | 4.29 | 1.8% | 3.82 | 89.0% | 0.47 | 0.2% |
| layernorm | 7 | 0.97 | 0.4% | 0.61 | 63.6% | 0.35 | 0.2% |
| dealloc | 24 | 0.74 | 0.3% | 0.00 | 0.0% | 0.74 | 0.3% |

| family | calls | time ms | share of b | roof ms | % of roof | brought to roof saves ms | of b |
|---|---|---|---|---|---|---|---|
| tri-attention batched matmuls (QK^T, PV and their grads) | 56 | 57.70 | 24.8% | 5.94 | 10.3% | 51.76 | 22.3% |
| 3D pair-tensor linears (fwd + dX) | 54 | 49.97 | 21.5% | 4.75 | 9.5% | 45.22 | 19.5% |
| all other elementwise | 261 | 42.19 | 18.2% | 34.87 | 82.6% | 7.32 | 3.2% |
| heads reshape [N,N,128] <-> [N,N,4,32] | 20 | 35.34 | 15.2% | 7.89 | 22.3% | 27.45 | 11.8% |
| permutes | 40 | 27.77 | 11.9% | 9.34 | 33.6% | 18.42 | 7.9% |
| all other data-movement | 157 | 14.18 | 6.1% | 12.59 | 88.8% | 1.58 | 0.7% |
| all other softmax | 16 | 5.79 | 2.5% | 2.81 | 48.5% | 2.98 | 1.3% |
| trimul per-channel matmuls | 6 | 4.53 | 1.9% | 0.69 | 15.3% | 3.84 | 1.7% |
| all other reduction | 52 | 4.29 | 1.8% | 3.82 | 89.0% | 0.47 | 0.2% |
| all other layernorm | 7 | 0.97 | 0.4% | 0.61 | 63.6% | 0.35 | 0.2% |
| all other dealloc | 24 | 0.74 | 0.3% | 0.00 | 0.0% | 0.74 | 0.3% |
