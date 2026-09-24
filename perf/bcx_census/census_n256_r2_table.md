b = 0.22955 s; sum of per-op replay = 0.23997 s (104.5%); sum of per-op roofs = 0.08909 s; ops >= 200 us per call = 88.8% of b

| # | op (verb:ttnn op, phase) | shape | dtype | calls | time ms | share of b | cum | class | its own roof, us/call | % of its roof |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | linear:linear fwd | `[256, 256, 128]:bf16,[128, 128]:bf16 ; dtype=None` | bf16 | 22 | 17.83 (810 us/call) | 7.8% | 7.8% | matmul | 78.2 | 9.6% |
| 2 | linear:matmul bwd | `[256, 256, 128]:bf16,[128, 128]:bf16 ; transpose_b=True` | bf16 | 22 | 17.48 (794 us/call) | 7.6% | 15.4% | matmul | 78.2 | 9.8% |
| 3 | triangle_attention:matmul bwd | `[128, 4, 128, 256]:bf16,[128, 4, 128, 32]:bf16 ; transpose_a=True` | bf16 | 16 | 17.35 (1084 us/call) | 7.6% | 22.9% | matmul | 107.4 | 9.9% |
| 4 | reshape:reshape fwd | `[256, 256, 128]:bf16,(256,256,4,32)` | bf16 | 8 | 16.20 (2025 us/call) | 7.1% | 30.0% | data-movement | 447.7 | 22.1% |
| 5 | triangle_attention:matmul fwd | `[128, 4, 128, 256]:bf16,[128, 4, 256, 32]:bf16` | bf16 | 8 | 13.25 (1657 us/call) | 5.8% | 35.8% | matmul | 107.4 | 6.5% |
| 6 | triangle_attention:matmul bwd | `[128, 4, 128, 256]:bf16,[128, 4, 256, 32]:bf16` | bf16 | 8 | 13.12 (1640 us/call) | 5.7% | 41.5% | matmul | 107.4 | 6.5% |
| 7 | reshape:reshape bwd | `[256, 256, 4, 32]:bf16,(256,256,128)` | bf16 | 8 | 12.16 (1520 us/call) | 5.3% | 46.8% | data-movement | 447.7 | 29.4% |
| 8 | triangle_attention:matmul bwd | `[128, 4, 128, 32]:bf16,[128, 4, 256, 32]:bf16 ; transpose_b=True` | bf16 | 16 | 9.31 (582 us/call) | 4.1% | 50.8% | matmul | 107.4 | 18.5% |
| 9 | permute:permute bwd | `[128, 256, 256]:bf16,(1,2,0)` | bf16 | 4 | 6.58 (1645 us/call) | 2.9% | 53.7% | data-movement | 99.5 | 6.0% |
| 10 | permute:permute bwd | `[256, 4, 256, 32]:bf16,(0,2,1,3)` | bf16 | 8 | 5.86 (733 us/call) | 2.6% | 56.3% | data-movement | 447.7 | 61.1% |
| 11 | linear:matmul bwd | `[256, 256, 512]:bf16,[128, 512]:bf16 ; transpose_b=True` | bf16 | 2 | 5.65 (2827 us/call) | 2.5% | 58.7% | matmul | 195.6 | 6.9% |
| 12 | permute:permute fwd | `[256, 256, 4, 32]:bf16,(0,2,1,3)` | bf16 | 8 | 4.93 (616 us/call) | 2.1% | 60.9% | data-movement | 447.7 | 72.7% |
| 13 | triangle_attention:matmul fwd | `[128, 4, 128, 32]:bf16,[128, 4, 256, 32]:bf16 ; transpose_b=True` | bf16 | 8 | 4.70 (588 us/call) | 2.0% | 62.9% | matmul | 107.4 | 18.3% |
| 14 | reshape:reshape bwd | `[256, 256, 128]:bf16,(256,256,4,32)` | bf16 | 2 | 4.02 (2008 us/call) | 1.7% | 64.7% | data-movement | 447.7 | 22.3% |
| 15 | linear:add bwd | `[256, 256, 128]:fp32,[256, 256, 128]:fp32` | fp32 | 17 | 3.97 (233 us/call) | 1.7% | 66.4% | elementwise | 240.4 | 103.0% |
| 16 | triangle_attention:multiply bwd | `[128, 4, 128, 256]:bf16,[128, 4, 128, 256]:bf16` | bf16 | 16 | 3.81 (238 us/call) | 1.7% | 68.1% | elementwise | 240.4 | 101.0% |
| 17 | pair_contract:permute fwd | `[128, 256, 256]:bf16,(1,2,0)` | bf16 | 2 | 3.30 (1649 us/call) | 1.4% | 69.5% | data-movement | 99.5 | 6.0% |
| 18 | linear:typecast bwd | `[256, 256, 128]:bf16,DataType` | bf16 | 22 | 3.21 (146 us/call) | 1.4% | 70.9% | data-movement | 149.2 | 102.3% |
| 19 | reshape:reshape fwd | `[256, 256, 4, 32]:bf16,(256,256,128)` | bf16 | 2 | 3.05 (1527 us/call) | 1.3% | 72.2% | data-movement | 447.7 | 29.3% |
| 20 | triangle_attention:softmax fwd | `[128, 4, 128, 256]:bf16 ; dim=-1` | bf16 | 8 | 2.93 (366 us/call) | 1.3% | 73.5% | softmax | 199.0 | 54.4% |
| 21 | triangle_attention:softmax bwd | `[128, 4, 128, 256]:bf16 ; dim=-1` | bf16 | 8 | 2.86 (358 us/call) | 1.2% | 74.7% | softmax | 199.0 | 55.6% |
| 22 | linear:linear fwd | `[256, 256, 512]:bf16,[512, 128]:bf16 ; dtype=None` | bf16 | 1 | 2.80 (2802 us/call) | 1.2% | 76.0% | matmul | 195.6 | 7.0% |
| 23 | layer_norm:multiply bwd | `[256, 256, 128]:bf16,[256, 256, 1]:bf16` | bf16 | 21 | 2.73 (130 us/call) | 1.2% | 77.1% | elementwise | 90.1 | 69.5% |
| 24 | linear:linear fwd | `[256, 256, 128]:bf16,[128, 512]:bf16 ; dtype=None` | bf16 | 2 | 2.15 (1075 us/call) | 0.9% | 78.1% | matmul | 195.6 | 18.2% |
| 25 | triangle_attention:add fwd | `[128, 4, 128, 256]:bf16,[1, 4, 128, 256]:bf16` | bf16 | 8 | 1.94 (242 us/call) | 0.8% | 78.9% | elementwise | 160.9 | 66.5% |
| 26 | triangle_attention:add bwd | `[128, 4, 128, 256]:bf16,[1, 4, 128, 256]:bf16` | bf16 | 8 | 1.88 (235 us/call) | 0.8% | 79.7% | elementwise | 160.9 | 68.5% |
| 27 | mul:multiply bwd | `[256, 256, 512]:bf16,[256, 256, 512]:bf16` | bf16 | 4 | 1.84 (460 us/call) | 0.8% | 80.5% | elementwise | 480.8 | 104.5% |
| 28 | layer_norm:subtract bwd | `[256, 256, 128]:bf16,[256, 256, 1]:bf16` | bf16 | 14 | 1.81 (130 us/call) | 0.8% | 81.3% | elementwise | 90.1 | 69.6% |
| 29 | layer_norm:multiply bwd | `[256, 256, 128]:bf16,[256, 256, 128]:bf16` | bf16 | 14 | 1.74 (124 us/call) | 0.8% | 82.1% | elementwise | 120.2 | 96.8% |
| 30 | linear:matmul bwd | `[256, 256, 4]:bf16,[128, 4]:bf16 ; transpose_b=True` | bf16 | 2 | 1.68 (841 us/call) | 0.7% | 82.8% | matmul | 48.8 | 5.8% |
| 31 | layer_norm:mean bwd | `[256, 256, 128]:bf16 ; dim=-1,keepdim=True` | bf16 | 28 | 1.67 (60 us/call) | 0.7% | 83.6% | reduction | 62.2 | 104.4% |
| 32 | triangle_attention:multiply fwd | `[128, 4, 128, 256]:bf16,0.1767766952966369` | bf16 | 8 | 1.66 (208 us/call) | 0.7% | 84.3% | elementwise | 160.3 | 77.2% |
| 33 | pair_contract:permute fwd | `[256, 256, 128]:bf16,(2,0,1)` | bf16 | 4 | 1.65 (412 us/call) | 0.7% | 85.0% | data-movement | 99.5 | 24.2% |
| 34 | triangle_attention:sum bwd | `[128, 4, 128, 256]:bf16 ; dim=-1,keepdim=True` | bf16 | 16 | 1.64 (102 us/call) | 0.7% | 85.7% | reduction | 111.9 | 109.5% |
| 35 | triangle_attention:multiply bwd | `[128, 4, 128, 256]:bf16,0.1767766952966369` | bf16 | 8 | 1.61 (201 us/call) | 0.7% | 86.4% | elementwise | 160.3 | 79.8% |
| 36 | permute:permute fwd | `[256, 4, 256, 32]:bf16,(0,2,1,3)` | bf16 | 2 | 1.50 (749 us/call) | 0.7% | 87.1% | data-movement | 447.7 | 59.8% |
| 37 | mul:multiply bwd | `[256, 256, 128]:bf16,[256, 256, 128]:bf16` | bf16 | 12 | 1.49 (124 us/call) | 0.6% | 87.7% | elementwise | 120.2 | 96.8% |
| 38 | sigmoid:multiply bwd | `[256, 256, 128]:bf16,[256, 256, 128]:bf16` | bf16 | 12 | 1.47 (123 us/call) | 0.6% | 88.4% | elementwise | 120.2 | 97.8% |
| 39 | triangle_attention:subtract bwd | `[128, 4, 128, 256]:bf16,[128, 4, 128, 1]:bf16` | bf16 | 8 | 1.44 (180 us/call) | 0.6% | 89.0% | elementwise | 170.3 | 94.4% |
| 40 | permute:permute bwd | `[256, 256, 4, 32]:bf16,(0,2,1,3)` | bf16 | 2 | 1.22 (609 us/call) | 0.5% | 89.5% | data-movement | 447.7 | 73.6% |
| 41 | harness:typecast bwd | `[256, 256, 128]:fp32,DataType` | fp32 | 10 | 1.19 (119 us/call) | 0.5% | 90.0% | data-movement | 149.2 | 125.4% |
| | 69 further signatures | | | 294 | 33.30 | 14.5% | | | | |
| | **OVERHEAD** = b - sum(per-op) | | | | -10.42 | **-4.5%** | | overhead | | |

| class | calls | time ms | share of b | roof ms | % of roof | brought to roof saves ms | of b |
|---|---|---|---|---|---|---|---|
| matmul | 116 | 112.05 | 48.8% | 11.53 | 10.3% | 100.52 | 43.8% |
| data-movement | 217 | 75.96 | 33.1% | 33.85 | 44.6% | 42.11 | 18.3% |
| elementwise | 261 | 40.66 | 17.7% | 35.50 | 87.3% | 5.16 | 2.2% |
| softmax | 16 | 5.79 | 2.5% | 3.18 | 55.0% | 2.61 | 1.1% |
| reduction | 52 | 4.09 | 1.8% | 4.33 | 105.9% | -0.24 | -0.1% |
| layernorm | 7 | 0.93 | 0.4% | 0.70 | 74.9% | 0.23 | 0.1% |
| dealloc | 24 | 0.49 | 0.2% | 0.00 | 0.0% | 0.49 | 0.2% |

| family | calls | time ms | share of b | roof ms | % of roof | brought to roof saves ms | of b |
|---|---|---|---|---|---|---|---|
| tri-attention batched matmuls (QK^T, PV and their grads) | 56 | 57.74 | 25.2% | 6.01 | 10.4% | 51.72 | 22.5% |
| 3D pair-tensor linears (fwd + dX) | 54 | 49.74 | 21.7% | 4.81 | 9.7% | 44.94 | 19.6% |
| all other elementwise | 261 | 40.66 | 17.7% | 35.50 | 87.3% | 5.16 | 2.2% |
| heads reshape [N,N,128] <-> [N,N,4,32] | 20 | 35.43 | 15.4% | 8.95 | 25.3% | 26.48 | 11.5% |
| permutes | 40 | 27.87 | 12.1% | 10.60 | 38.0% | 17.26 | 7.5% |
| all other data-movement | 157 | 12.66 | 5.5% | 14.29 | 112.9% | -1.63 | -0.7% |
| all other softmax | 16 | 5.79 | 2.5% | 3.18 | 55.0% | 2.61 | 1.1% |
| trimul per-channel matmuls | 6 | 4.57 | 2.0% | 0.70 | 15.4% | 3.86 | 1.7% |
| all other reduction | 52 | 4.09 | 1.8% | 4.33 | 105.9% | -0.24 | -0.1% |
| all other layernorm | 7 | 0.93 | 0.4% | 0.70 | 74.9% | 0.23 | 0.1% |
| all other dealloc | 24 | 0.49 | 0.2% | 0.00 | 0.0% | 0.49 | 0.2% |
