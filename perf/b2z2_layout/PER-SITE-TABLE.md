# Boltz-2 512 aa: every device program of a Pairformer block and a diffusion step, by site

ARCH: BH. qb2 physical card 0, one Blackhole processor of a p300c, 11x10 grid, tt-metal built
from source at v0.68.0 with the device profiler on. Source data is the per-op profiler CSV
taken by `ws:b2z-kernel-cycle-census`; `perf/b2z2_layout/census.py` re-windows it by op-code
signature and reads the dtype, layout, math-fidelity and shape columns the JSON census dropped.
No new device time was spent to produce this table.

A site is one (op, compute kernel, fidelity, accumulator width, input shape, output shape,
input dtype, output dtype, input layout, output layout). `acc` is `fp32acc` when the program
carries `fp32_dest_acc_en=1`, which halves the dest register file from 8 tiles to 4.

## PairformerLayer

272 device programs, span **36.2456 ms**, kernel time **35.7140 ms**.

| | programs | kernel ms | % of span |
|---|---|---|---|
| pure movement (transpose, permute, slice, concat, copy, pad, tilize, heads) | 49 | 3.7059 | **10.22** |
| layout transitions only (tilize / untilize / reshape) | 3 | 0.5109 | **1.41** |
| any fp32 operand or result (storage) | 0 | 0.0000 | **0.00** |
| fp32 dest accumulate | 155 | 12.7725 | **35.24** |

Math fidelity, as executed:

| fidelity | programs | kernel ms |
|---|---|---|
| HiFi4 | 228 | 29.2904 |
| HiFi2 | 4 | 3.6963 |
| - | 40 | 2.7274 |

### Sites, most expensive first

| op | kernel | fid | acc | in0 | out0 | in0 dt | out0 dt | in0 lay | out0 lay | n | kernel ms | TRISC1 ms | cores | % span |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Matmul | bmm_large_block_zm_fused_bias_activation | HiFi4 | fp32acc | 1x16x512x128 | 1x16x512x512 | BFLOAT16 | BFLOAT16 | TILE | TILE | 64 | 4.4778 | 4.0888 | 86.0 | 12.354 |
| GenericOp | compute_reblock_permute_gated | HiFi4 | bf16acc | 1x512x512x512 | 1x128x512x512 | BFLOAT16 | BFLOAT16 | TILE | TILE | 4 | 3.9870 | 3.8107 | 110.0 | 11.000 |
| GenericOp | sdpa | HiFi2 | bf16acc | 512x4x512x32 | 512x4x512x32 | BFLOAT16 | BFLOAT16 | TILE | TILE | 2 | 2.6182 | 2.6158 | 110.0 | 7.224 |
| LayerNorm | layernorm | HiFi4 | fp32acc | 1x512x512x128 | 1x512x512x128 | BFLOAT16 | BFLOAT16 | TILE | TILE | 7 | 2.6062 | 2.6005 | 110.0 | 7.190 |
| GenericOp | compute | HiFi4 | bf16acc | 1x512x512x128 | 512x4x512x32 | BFLOAT16 | BFLOAT16 | TILE | TILE | 4 | 2.5379 | 2.5366 | 110.0 | 7.002 |
| GenericOp | compute | HiFi4 | bf16acc | 1x512x512x128 | 1x512x512x512 | BFLOAT16 | BFLOAT16 | TILE | TILE | 2 | 2.3423 | 2.3417 | 110.0 | 6.462 |
| BinaryNg | eltwise_binary_no_bcast | HiFi4 | bf16acc | 1x512x512x128 | 1x512x512x128 | BFLOAT16 | BFLOAT16 | TILE | TILE | 5 | 1.9146 | 1.9057 | 110.0 | 5.282 |
| Matmul | bmm_large_block_zm_fused_bias_activation | HiFi4 | fp32acc | 1x128x512x512 | 1x128x512x512 | BFLOAT16 | BFLOAT16 | TILE | TILE | 2 | 1.6428 | 1.6410 | 64.0 | 4.532 |
| Transpose | - | - | bf16acc | 1x512x512x128 | 1x512x512x128 | BFLOAT16 | BFLOAT16 | TILE | TILE | 2 | 1.4589 | 0.0000 | 110.0 | 4.025 |
| Matmul | bmm_large_block_zm_fused_bias_activation | HiFi4 | fp32acc | 1x512x512x128 | 1x512x512x128 | BFLOAT16 | BFLOAT16 | TILE | TILE | 4 | 1.1592 | 1.1479 | 110.0 | 3.198 |
| Matmul | bmm_large_block_zm_fused_bias_activation | HiFi4 | fp32acc | 1x16x512x512 | 1x16x512x128 | BFLOAT16 | BFLOAT16 | TILE | TILE | 32 | 1.1428 | 0.7329 | 86.0 | 3.153 |
| GenericOp | compute_reblock_permute | HiFi2 | bf16acc | 1x128x512x512 | 1x512x512x128 | BFLOAT16 | BFLOAT16 | TILE | TILE | 2 | 1.0780 | 1.0730 | 110.0 | 2.974 |
| BinaryNg | eltwise_binary_sfpu_no_bcast | HiFi4 | bf16acc | 512x4x512x32 | 512x4x512x32 | BFLOAT16 | BFLOAT16 | TILE | TILE | 2 | 1.0192 | 1.0184 | 110.0 | 2.812 |
| BinaryNg | eltwise_binary_sfpu_no_bcast | HiFi4 | bf16acc | 1x512x512x128 | 1x512x512x128 | BFLOAT16 | BFLOAT16 | TILE | TILE | 2 | 1.0166 | 1.0158 | 110.0 | 2.805 |
| BinaryNg | eltwise_binary_sfpu_no_bcast | HiFi4 | bf16acc | 1x128x512x512 | 1x128x512x512 | BFLOAT16 | BFLOAT16 | TILE | TILE | 2 | 0.8949 | 0.8941 | 110.0 | 2.469 |
| BinaryNg | eltwise_binary_sfpu_no_bcast | HiFi4 | bf16acc | 1x16x512x512 | 1x16x512x512 | BFLOAT16 | BFLOAT16 | TILE | TILE | 32 | 0.8314 | 0.8188 | 110.0 | 2.294 |
| GenericOp | compute | HiFi4 | bf16acc | 512x4x512x32 | 1x512x512x128 | BFLOAT16 | BFLOAT16 | TILE | TILE | 2 | 0.8252 | 0.8246 | 110.0 | 2.277 |
| LayerNorm | layernorm | HiFi4 | fp32acc | 1x16x512x128 | 1x16x512x128 | BFLOAT16 | BFLOAT16 | TILE | TILE | 32 | 0.7462 | 0.7206 | 110.0 | 2.059 |
| Transpose | transpose_wh | HiFi4 | bf16acc | 1x128x512x512 | 1x128x512x512 | BFLOAT16 | BFLOAT16 | TILE | TILE | 2 | 0.6894 | 0.6864 | 110.0 | 1.902 |
| Matmul | bmm_large_block_zm_fused_bias_activation | HiFi4 | fp32acc | 1x512x512x128 | 1x512x512x32 | BFLOAT16 | BFLOAT16 | TILE | TILE | 3 | 0.6838 | 0.6744 | 110.0 | 1.887 |
| ReshapeView | - | - | bf16acc | 1x1x512x512 | 1x512x512x32 | BFLOAT16 | BFLOAT16 | TILE | TILE | 2 | 0.4989 | 0.0000 | 110.0 | 1.376 |
| Slice | - | - | bf16acc | 1x512x512x128 | 1x16x512x128 | BFLOAT16 | BFLOAT16 | TILE | TILE | 32 | 0.3845 | 0.0000 | 110.0 | 1.061 |
| Concat | - | - | bf16acc | 1x16x512x128 | 1x512x512x128 | BFLOAT16 | BFLOAT16 | TILE | TILE | 1 | 0.3476 | 0.0000 | 110.0 | 0.959 |
| Permute | transpose_wh | HiFi4 | bf16acc | 1x512x512x32 | 1x4x512x512 | BFLOAT16 | BFLOAT16 | TILE | TILE | 2 | 0.1452 | 0.1371 | 110.0 | 0.401 |
| Permute | transpose_wh | HiFi4 | bf16acc | 1x512x512x32 | 1x16x512x512 | BFLOAT16 | BFLOAT16 | TILE | TILE | 1 | 0.1312 | 0.1190 | 110.0 | 0.362 |
| Softmax | softmax | HiFi4 | fp32acc | 1x16x512x512 | 1x16x512x512 | BFLOAT16 | BFLOAT16 | TILE | TILE | 1 | 0.1175 | 0.1163 | 110.0 | 0.324 |
| Matmul | bmm_large_block_zm_fused_bias_activation | HiFi4 | fp32acc | 1x1x512x384 | 1x1x512x1536 | BFLOAT16 | BFLOAT16 | TILE | TILE | 3 | 0.0593 | 0.0508 | 80.0 | 0.164 |
| BinaryNg | eltwise_binary_sfpu_scalar | HiFi4 | bf16acc | 1x16x512x512 | 1x16x512x512 | BFLOAT16 | BFLOAT16 | TILE | TILE | 1 | 0.0469 | 0.0465 | 110.0 | 0.129 |
| Matmul | bmm_large_block_zm_fused_bias_activation | HiFi4 | fp32acc | 1x16x512x32 | 1x16x512x512 | BFLOAT16 | BFLOAT16 | TILE | TILE | 1 | 0.0433 | 0.0115 | 32.0 | 0.119 |
| BinaryNg | eltwise_binary_no_bcast | HiFi4 | bf16acc | 1x16x512x512 | 1x16x512x512 | BFLOAT16 | BFLOAT16 | TILE | TILE | 1 | 0.0406 | 0.0389 | 110.0 | 0.112 |
| Matmul | bmm_large_block_zm_fused_bias_activation | HiFi4 | fp32acc | 1x16x512x512 | 1x16x512x32 | BFLOAT16 | BFLOAT16 | TILE | TILE | 1 | 0.0369 | 0.0360 | 32.0 | 0.102 |
| BinaryNg | eltwise_binary_row_bcast | HiFi4 | bf16acc | 1x4x512x512 | 1x4x512x512 | BFLOAT16 | BFLOAT16 | TILE | TILE | 2 | 0.0311 | 0.0300 | 110.0 | 0.086 |
| BinaryNg | eltwise_binary_row_bcast | HiFi4 | bf16acc | 1x16x512x512 | 1x16x512x512 | BFLOAT16 | BFLOAT16 | TILE | TILE | 1 | 0.0284 | 0.0279 | 110.0 | 0.078 |
| LayerNorm | layernorm | HiFi4 | fp32acc | 1x1x512x384 | 1x1x512x384 | BFLOAT16 | BFLOAT16 | TILE | TILE | 2 | 0.0235 | 0.0218 | 16.0 | 0.065 |
| NlpCreateHeads | - | - | bf16acc | 1x1x512x1536 | 1x16x512x32 | BFLOAT16 | BFLOAT16 | TILE | TILE | 1 | 0.0221 | 0.0000 | 16.0 | 0.061 |
| Matmul | bmm_large_block_zm_fused_bias_activation | HiFi4 | fp32acc | 1x1x512x1536 | 1x1x512x384 | BFLOAT16 | BFLOAT16 | TILE | TILE | 1 | 0.0173 | 0.0150 | 48.0 | 0.048 |
| Matmul | bmm_large_block_zm_fused_bias_activation | HiFi4 | fp32acc | 1x1x512x384 | 1x1x512x384 | BFLOAT16 | BFLOAT16 | TILE | TILE | 2 | 0.0158 | 0.0110 | 48.0 | 0.044 |
| ReshapeView | - | - | bf16acc | 1x16x32x512 | 1x1x384x512 | BFLOAT16 | BFLOAT16 | TILE | TILE | 1 | 0.0120 | 0.0000 | 110.0 | 0.033 |
| BinaryNg | eltwise_binary_no_bcast | HiFi4 | bf16acc | 1x1x512x384 | 1x1x512x384 | BFLOAT16 | BFLOAT16 | TILE | TILE | 2 | 0.0099 | 0.0022 | 110.0 | 0.027 |
| BinaryNg | eltwise_binary_sfpu_no_bcast | HiFi4 | bf16acc | 1x1x512x1536 | 1x1x512x1536 | BFLOAT16 | BFLOAT16 | TILE | TILE | 1 | 0.0068 | 0.0064 | 110.0 | 0.019 |
| Transpose | transpose_wh | HiFi4 | bf16acc | 1x16x512x32 | 1x16x32x512 | BFLOAT16 | BFLOAT16 | TILE | TILE | 2 | 0.0066 | 0.0022 | 110.0 | 0.018 |
| BinaryNg | eltwise_binary_sfpu_no_bcast | HiFi4 | bf16acc | 1x1x512x384 | 1x1x512x384 | BFLOAT16 | BFLOAT16 | TILE | TILE | 1 | 0.0064 | 0.0060 | 110.0 | 0.018 |
| Slice | - | - | bf16acc | 1x16x512x32 | 1x16x512x32 | BFLOAT16 | BFLOAT16 | TILE | TILE | 1 | 0.0035 | 0.0000 | 110.0 | 0.010 |
| Transpose | transpose_wh | HiFi4 | bf16acc | 1x1x512x512 | 1x1x512x512 | BFLOAT16 | BFLOAT16 | TILE | TILE | 1 | 0.0033 | 0.0011 | 110.0 | 0.009 |
| Transpose | transpose_wh | HiFi4 | bf16acc | 1x1x384x512 | 1x1x512x384 | BFLOAT16 | BFLOAT16 | TILE | TILE | 1 | 0.0028 | 0.0012 | 110.0 | 0.008 |

### Every movement program, in issue order

| # | op | ms | cores | in0 | in0 dt | in0 layout | out0 | out0 dt | out0 layout |
|---|---|---|---|---|---|---|---|---|---|
| 1 | ReshapeView | 0.2494 | 110 | 1x1x512x512 | BFLOAT16 | TILE | 1x512x512x32 | BFLOAT16 | TILE |
| 5 | Transpose | 0.3442 | 110 | 1x128x512x512 | BFLOAT16 | TILE | 1x128x512x512 | BFLOAT16 | TILE |
| 15 | ReshapeView | 0.2495 | 110 | 1x1x512x512 | BFLOAT16 | TILE | 1x512x512x32 | BFLOAT16 | TILE |
| 16 | Transpose | 0.0033 | 110 | 1x1x512x512 | BFLOAT16 | TILE | 1x1x512x512 | BFLOAT16 | TILE |
| 19 | Transpose | 0.3453 | 110 | 1x128x512x512 | BFLOAT16 | TILE | 1x128x512x512 | BFLOAT16 | TILE |
| 31 | Permute | 0.0736 | 110 | 1x512x512x32 | BFLOAT16 | TILE | 1x4x512x512 | BFLOAT16 | TILE |
| 39 | Transpose | 0.7296 | 110 | 1x512x512x128 | BFLOAT16 | TILE | 1x512x512x128 | BFLOAT16 | TILE |
| 42 | Permute | 0.0716 | 110 | 1x512x512x32 | BFLOAT16 | TILE | 1x4x512x512 | BFLOAT16 | TILE |
| 49 | Transpose | 0.7293 | 110 | 1x512x512x128 | BFLOAT16 | TILE | 1x512x512x128 | BFLOAT16 | TILE |
| 51 | Slice | 0.0120 | 110 | 1x512x512x128 | BFLOAT16 | TILE | 1x16x512x128 | BFLOAT16 | TILE |
| 52 | Slice | 0.0121 | 110 | 1x512x512x128 | BFLOAT16 | TILE | 1x16x512x128 | BFLOAT16 | TILE |
| 53 | Slice | 0.0115 | 110 | 1x512x512x128 | BFLOAT16 | TILE | 1x16x512x128 | BFLOAT16 | TILE |
| 54 | Slice | 0.0116 | 110 | 1x512x512x128 | BFLOAT16 | TILE | 1x16x512x128 | BFLOAT16 | TILE |
| 55 | Slice | 0.0120 | 110 | 1x512x512x128 | BFLOAT16 | TILE | 1x16x512x128 | BFLOAT16 | TILE |
| 56 | Slice | 0.0120 | 110 | 1x512x512x128 | BFLOAT16 | TILE | 1x16x512x128 | BFLOAT16 | TILE |
| 57 | Slice | 0.0123 | 110 | 1x512x512x128 | BFLOAT16 | TILE | 1x16x512x128 | BFLOAT16 | TILE |
| 58 | Slice | 0.0119 | 110 | 1x512x512x128 | BFLOAT16 | TILE | 1x16x512x128 | BFLOAT16 | TILE |
| 59 | Slice | 0.0114 | 110 | 1x512x512x128 | BFLOAT16 | TILE | 1x16x512x128 | BFLOAT16 | TILE |
| 60 | Slice | 0.0117 | 110 | 1x512x512x128 | BFLOAT16 | TILE | 1x16x512x128 | BFLOAT16 | TILE |
| 61 | Slice | 0.0124 | 110 | 1x512x512x128 | BFLOAT16 | TILE | 1x16x512x128 | BFLOAT16 | TILE |
| 62 | Slice | 0.0120 | 110 | 1x512x512x128 | BFLOAT16 | TILE | 1x16x512x128 | BFLOAT16 | TILE |
| 63 | Slice | 0.0117 | 110 | 1x512x512x128 | BFLOAT16 | TILE | 1x16x512x128 | BFLOAT16 | TILE |
| 64 | Slice | 0.0122 | 110 | 1x512x512x128 | BFLOAT16 | TILE | 1x16x512x128 | BFLOAT16 | TILE |
| 65 | Slice | 0.0121 | 110 | 1x512x512x128 | BFLOAT16 | TILE | 1x16x512x128 | BFLOAT16 | TILE |
| 66 | Slice | 0.0120 | 110 | 1x512x512x128 | BFLOAT16 | TILE | 1x16x512x128 | BFLOAT16 | TILE |
| 67 | Slice | 0.0121 | 110 | 1x512x512x128 | BFLOAT16 | TILE | 1x16x512x128 | BFLOAT16 | TILE |
| 68 | Slice | 0.0121 | 110 | 1x512x512x128 | BFLOAT16 | TILE | 1x16x512x128 | BFLOAT16 | TILE |
| 69 | Slice | 0.0125 | 110 | 1x512x512x128 | BFLOAT16 | TILE | 1x16x512x128 | BFLOAT16 | TILE |
| 70 | Slice | 0.0122 | 110 | 1x512x512x128 | BFLOAT16 | TILE | 1x16x512x128 | BFLOAT16 | TILE |
| 71 | Slice | 0.0122 | 110 | 1x512x512x128 | BFLOAT16 | TILE | 1x16x512x128 | BFLOAT16 | TILE |
| 72 | Slice | 0.0121 | 110 | 1x512x512x128 | BFLOAT16 | TILE | 1x16x512x128 | BFLOAT16 | TILE |
| 73 | Slice | 0.0120 | 110 | 1x512x512x128 | BFLOAT16 | TILE | 1x16x512x128 | BFLOAT16 | TILE |
| 74 | Slice | 0.0125 | 110 | 1x512x512x128 | BFLOAT16 | TILE | 1x16x512x128 | BFLOAT16 | TILE |
| 75 | Slice | 0.0121 | 110 | 1x512x512x128 | BFLOAT16 | TILE | 1x16x512x128 | BFLOAT16 | TILE |
| 76 | Slice | 0.0117 | 110 | 1x512x512x128 | BFLOAT16 | TILE | 1x16x512x128 | BFLOAT16 | TILE |
| 77 | Slice | 0.0121 | 110 | 1x512x512x128 | BFLOAT16 | TILE | 1x16x512x128 | BFLOAT16 | TILE |
| 78 | Slice | 0.0123 | 110 | 1x512x512x128 | BFLOAT16 | TILE | 1x16x512x128 | BFLOAT16 | TILE |
| 79 | Slice | 0.0120 | 110 | 1x512x512x128 | BFLOAT16 | TILE | 1x16x512x128 | BFLOAT16 | TILE |
| 80 | Slice | 0.0120 | 110 | 1x512x512x128 | BFLOAT16 | TILE | 1x16x512x128 | BFLOAT16 | TILE |
| 81 | Slice | 0.0118 | 110 | 1x512x512x128 | BFLOAT16 | TILE | 1x16x512x128 | BFLOAT16 | TILE |
| 82 | Slice | 0.0118 | 110 | 1x512x512x128 | BFLOAT16 | TILE | 1x16x512x128 | BFLOAT16 | TILE |
| 243 | Concat | 0.3476 | 110 | 1x16x512x128 | BFLOAT16 | TILE | 1x512x512x128 | BFLOAT16 | TILE |
| 247 | NlpCreateHeads | 0.0221 | 16 | 1x1x512x1536 | BFLOAT16 | TILE | 1x16x512x32 | BFLOAT16 | TILE |
| 250 | Permute | 0.1312 | 110 | 1x512x512x32 | BFLOAT16 | TILE | 1x16x512x512 | BFLOAT16 | TILE |
| 252 | Transpose | 0.0035 | 110 | 1x16x512x32 | BFLOAT16 | TILE | 1x16x32x512 | BFLOAT16 | TILE |
| 258 | Slice | 0.0035 | 110 | 1x16x512x32 | BFLOAT16 | TILE | 1x16x512x32 | BFLOAT16 | TILE |
| 259 | Transpose | 0.0032 | 110 | 1x16x512x32 | BFLOAT16 | TILE | 1x16x32x512 | BFLOAT16 | TILE |
| 260 | ReshapeView | 0.0120 | 110 | 1x16x32x512 | BFLOAT16 | TILE | 1x1x384x512 | BFLOAT16 | TILE |
| 261 | Transpose | 0.0028 | 110 | 1x1x384x512 | BFLOAT16 | TILE | 1x1x512x384 | BFLOAT16 | TILE |

## DiffusionStep

1066 device programs, span **45.5048 ms**, kernel time **22.0224 ms**.

| | programs | kernel ms | % of span |
|---|---|---|---|
| pure movement (transpose, permute, slice, concat, copy, pad, tilize, heads) | 195 | 5.1155 | **11.24** |
| layout transitions only (tilize / untilize / reshape) | 36 | 1.3603 | **2.99** |
| any fp32 operand or result (storage) | 0 | 0.0000 | **0.00** |
| fp32 dest accumulate | 501 | 10.8060 | **23.75** |

Math fidelity, as executed:

| fidelity | programs | kernel ms |
|---|---|---|
| HiFi4 | 904 | 16.1433 |
| - | 132 | 3.3749 |
| HiFi2 | 30 | 2.5042 |

### Sites, most expensive first

| op | kernel | fid | acc | in0 | out0 | in0 dt | out0 dt | in0 lay | out0 lay | n | kernel ms | TRISC1 ms | cores | % span |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Matmul | bmm_large_block_zm_fused_bias_activation | HiFi4 | fp32acc | 1x1x512x768 | 1x1x512x768 | BFLOAT16 | BFLOAT16 | TILE | TILE | 193 | 3.8768 | 3.2889 | 64.0 | 8.520 |
| SDPA | sdpa | HiFi2 | bf16acc | 1x16x512x64 | 1x16x512x64 | BFLOAT16 | BFLOAT16 | TILE | TILE | 24 | 2.2017 | 2.1658 | 110.0 | 4.838 |
| Matmul | bmm_large_block_zm_fused_bias_activation | HiFi4 | fp32acc | 1x1x512x768 | 1x1x512x1536 | BFLOAT16 | BFLOAT16 | TILE | TILE | 76 | 1.6850 | 1.3169 | 80.0 | 3.703 |
| LayerNorm | layernorm | HiFi4 | fp32acc | 1x1x512x768 | 1x1x512x768 | BFLOAT16 | BFLOAT16 | TILE | TILE | 100 | 1.5942 | 1.4698 | 16.0 | 3.503 |
| BinaryNg | eltwise_binary_sfpu_no_bcast | HiFi4 | bf16acc | 1x1x512x768 | 1x1x512x768 | BFLOAT16 | BFLOAT16 | TILE | TILE | 120 | 1.2905 | 1.2380 | 110.0 | 2.836 |
| NlpCreateHeads | - | - | bf16acc | 1x1x512x3072 | 1x16x512x64 | BFLOAT16 | BFLOAT16 | TILE | TILE | 24 | 1.0215 | 0.0000 | 16.0 | 2.245 |
| Matmul | bmm_large_block_zm_fused_bias_activation | HiFi4 | fp32acc | 1x1x512x768 | 1x1x512x3072 | BFLOAT16 | BFLOAT16 | TILE | TILE | 24 | 0.9522 | 0.6518 | 88.0 | 2.093 |
| BinaryNg | eltwise_binary_no_bcast | HiFi4 | bf16acc | 1x1x512x768 | 1x1x512x768 | BFLOAT16 | BFLOAT16 | TILE | TILE | 99 | 0.8725 | 0.4677 | 110.0 | 1.917 |
| Permute | transpose_wh | HiFi4 | bf16acc | 1x16x128x1120 | 1x1120x32x128 | BFLOAT16 | BFLOAT16 | TILE | TILE | 6 | 0.7417 | 0.5641 | 110.0 | 1.630 |
| Permute | transpose_xw_tiled | HiFi4 | bf16acc | 1x280x32x128 | 1x16x128x288 | BFLOAT16 | BFLOAT16 | TILE | TILE | 6 | 0.7246 | 0.7086 | 110.0 | 1.592 |
| BinaryNg | eltwise_binary_sfpu_no_bcast | HiFi4 | bf16acc | 1x1x512x1536 | 1x1x512x1536 | BFLOAT16 | BFLOAT16 | TILE | TILE | 50 | 0.6486 | 0.6271 | 110.0 | 1.425 |
| ReshapeView | - | - | bf16acc | 1x1120x32x128 | 1x140x128x128 | BFLOAT16 | BFLOAT16 | TILE | TILE | 6 | 0.6444 | 0.0000 | 110.0 | 1.416 |
| Matmul | bmm_large_block_zm_fused_bias_activation | HiFi4 | fp32acc | 1x1x512x1536 | 1x1x512x768 | BFLOAT16 | BFLOAT16 | TILE | TILE | 26 | 0.5959 | 0.5210 | 64.0 | 1.310 |
| ReshapeView | - | - | bf16acc | 1x16x64x512 | 1x1x768x512 | BFLOAT16 | BFLOAT16 | TILE | TILE | 24 | 0.5221 | 0.0000 | 110.0 | 1.147 |
| Matmul | bmm_large_block_zm_fused_bias_activation | HiFi4 | fp32acc | 1x140x128x128 | 1x140x128x256 | BFLOAT16 | BFLOAT16 | TILE | TILE | 6 | 0.4860 | 0.1692 | 94.0 | 1.068 |
| NlpCreateHeads | - | - | bf16acc | 140x1x128x128 | 140x4x128x32 | BFLOAT16 | BFLOAT16 | TILE | TILE | 6 | 0.4485 | 0.0000 | 110.0 | 0.986 |
| Matmul | bmm_large_block_zm_fused_bias_activation | HiFi4 | fp32acc | 1x140x32x128 | 1x140x32x256 | BFLOAT16 | BFLOAT16 | TILE | TILE | 18 | 0.4437 | 0.1529 | 70.0 | 0.975 |
| Matmul | bmm_large_block_zm_fused_bias_activation | HiFi4 | fp32acc | 1x140x32x128 | 1x140x32x128 | BFLOAT16 | BFLOAT16 | TILE | TILE | 24 | 0.3861 | 0.1853 | 70.0 | 0.848 |
| BinaryNg | eltwise_binary_sfpu_no_bcast | HiFi4 | bf16acc | 1x140x32x128 | 1x140x32x128 | BFLOAT16 | BFLOAT16 | TILE | TILE | 30 | 0.3505 | 0.3382 | 110.0 | 0.770 |
| SDPA | sdpa | HiFi2 | bf16acc | 1x560x32x32 | 1x560x32x32 | BFLOAT16 | BFLOAT16 | TILE | TILE | 6 | 0.3025 | 0.2995 | 110.0 | 0.665 |
| BinaryNg | eltwise_binary_sfpu_no_bcast | HiFi4 | bf16acc | 1x140x32x256 | 1x140x32x256 | BFLOAT16 | BFLOAT16 | TILE | TILE | 12 | 0.2400 | 0.2351 | 110.0 | 0.527 |
| Matmul | bmm_large_block_zm_fused_bias_activation | HiFi4 | fp32acc | 1x16x128x288 | 1x16x128x1120 | BFLOAT16 | BFLOAT16 | TILE | TILE | 6 | 0.2217 | 0.1366 | 90.0 | 0.487 |
| ReshapeView | - | - | bf16acc | 1x140x32x128 | 1x280x32x128 | BFLOAT16 | BFLOAT16 | TILE | TILE | 6 | 0.1938 | 0.0000 | 110.0 | 0.426 |
| Matmul | bmm_large_block_zm_fused_bias_activation | HiFi4 | fp32acc | 1x140x32x256 | 1x140x32x128 | BFLOAT16 | BFLOAT16 | TILE | TILE | 6 | 0.1909 | 0.1410 | 70.0 | 0.420 |
| BinaryNg | eltwise_binary_no_bcast | HiFi4 | bf16acc | 1x140x32x128 | 1x140x32x128 | BFLOAT16 | BFLOAT16 | TILE | TILE | 24 | 0.1642 | 0.0942 | 110.0 | 0.361 |
| Transpose | transpose_wh | HiFi4 | bf16acc | 1x16x512x64 | 1x16x64x512 | BFLOAT16 | BFLOAT16 | TILE | TILE | 24 | 0.1492 | 0.0772 | 110.0 | 0.328 |
| Slice | - | - | bf16acc | 1x16x512x64 | 1x16x512x64 | BFLOAT16 | BFLOAT16 | TILE | TILE | 24 | 0.1456 | 0.0000 | 110.0 | 0.320 |
| Pad | - | - | bf16acc | 1x140x32x128 | 1x140x128x128 | BFLOAT16 | BFLOAT16 | TILE | TILE | 6 | 0.1456 | 0.0000 | 110.0 | 0.320 |
| Matmul | bmm_large_block_zm_fused_bias_activation | HiFi4 | fp32acc | 1x1x4480x768 | 1x1x768x512 | BFLOAT16 | BFLOAT16 | TILE | TILE | 1 | 0.1206 | 0.1172 | 64.0 | 0.265 |
| Transpose | transpose_wh | HiFi4 | bf16acc | 1x1x768x512 | 1x1x512x768 | BFLOAT16 | BFLOAT16 | TILE | TILE | 25 | 0.1179 | 0.0260 | 110.0 | 0.259 |
| LayerNorm | layernorm | HiFi4 | fp32acc | 1x140x32x128 | 1x140x32x128 | BFLOAT16 | BFLOAT16 | TILE | TILE | 12 | 0.1136 | 0.1047 | 110.0 | 0.250 |
| Copy | - | - | bf16acc | 1x140x32x128 | 1x140x32x128 | BFLOAT16 | BFLOAT16 | TILE | TILE | 24 | 0.1074 | 0.0000 | 110.0 | 0.236 |
| Slice | - | - | bf16acc | 1x560x128x32 | 1x560x32x32 | BFLOAT16 | BFLOAT16 | TILE | TILE | 6 | 0.0814 | 0.0000 | 110.0 | 0.179 |
| NLPConcatHeads | - | - | bf16acc | 140x4x32x32 | 140x1x32x128 | BFLOAT16 | BFLOAT16 | TILE | TILE | 6 | 0.0646 | 0.0000 | 110.0 | 0.142 |
| Matmul | bmm_large_block_zm_fused_bias_activation | HiFi4 | fp32acc | 1x1x4480x128 | 1x1x4480x768 | BFLOAT16 | BFLOAT16 | TILE | TILE | 1 | 0.0356 | 0.0150 | 80.0 | 0.078 |
| Matmul | bmm_large_block_zm_fused_bias_activation | HiFi4 | fp32acc | 1x1x128x512 | 1x1x128x4480 | BFLOAT16 | BFLOAT16 | TILE | TILE | 1 | 0.0274 | 0.0211 | 70.0 | 0.060 |
| LayerNorm | layernorm | HiFi4 | fp32acc | 1x1x4480x128 | 1x1x4480x128 | BFLOAT16 | BFLOAT16 | TILE | TILE | 1 | 0.0212 | 0.0204 | 110.0 | 0.047 |
| BinaryNg | eltwise_binary_no_bcast | HiFi4 | bf16acc | 1x1x4480x128 | 1x1x4480x128 | BFLOAT16 | BFLOAT16 | TILE | TILE | 2 | 0.0173 | 0.0093 | 110.0 | 0.038 |
| Matmul | bmm_large_block_zm_fused_bias_activation | HiFi4 | fp32acc | 1x1x4480x32 | 1x1x4480x128 | BFLOAT16 | BFLOAT16 | TILE | TILE | 1 | 0.0148 | 0.0004 | 70.0 | 0.033 |
| Matmul | bmm_large_block_zm_fused_bias_activation | HiFi4 | fp32acc | 1x1x512x768 | 1x1x512x128 | BFLOAT16 | BFLOAT16 | TILE | TILE | 1 | 0.0104 | 0.0095 | 32.0 | 0.023 |
| Matmul | bmm_large_block_zm_fused_bias_activation | HiFi4 | fp32acc | 1x1x4480x128 | 1x1x4480x32 | BFLOAT16 | BFLOAT16 | TILE | TILE | 1 | 0.0091 | 0.0066 | 70.0 | 0.020 |
| BinaryNg | eltwise_binary_row_bcast | HiFi4 | bf16acc | 1x1x512x768 | 1x1x512x768 | BFLOAT16 | BFLOAT16 | TILE | TILE | 1 | 0.0090 | 0.0085 | 110.0 | 0.020 |
| LayerNorm | layernorm | HiFi4 | fp32acc | 1x1x32x256 | 1x1x32x256 | BFLOAT16 | BFLOAT16 | TILE | TILE | 1 | 0.0085 | 0.0077 | 1.0 | 0.019 |
| Matmul | bmm_large_block_zm_fused_bias_activation | HiFi4 | fp32acc | 1x1x32x256 | 1x1x32x768 | BFLOAT16 | BFLOAT16 | TILE | TILE | 1 | 0.0077 | 0.0069 | 24.0 | 0.017 |
| Transpose | transpose_wh | HiFi4 | bf16acc | 1x1x128x4480 | 1x1x4480x128 | BFLOAT16 | BFLOAT16 | TILE | TILE | 1 | 0.0058 | 0.0029 | 110.0 | 0.013 |
| Matmul | bmm_large_block_zm_fused_bias_activation | HiFi4 | fp32acc | 1x1x32x32 | 1x1x32x256 | BFLOAT16 | BFLOAT16 | TILE | TILE | 1 | 0.0046 | 0.0007 | 8.0 | 0.010 |
| BinaryNg | eltwise_binary_sfpu_scalar | HiFi4 | bf16acc | 1x1x32x256 | 1x1x32x256 | BFLOAT16 | BFLOAT16 | TILE | TILE | 1 | 0.0025 | 0.0020 | 110.0 | 0.005 |
| UnaryNg | eltwise_sfpu | HiFi4 | bf16acc | 1x1x32x256 | 1x1x32x256 | BFLOAT16 | BFLOAT16 | TILE | TILE | 1 | 0.0018 | 0.0014 | 110.0 | 0.004 |
| Transpose | transpose_wh | HiFi4 | bf16acc | 1x1x512x128 | 1x1x128x512 | BFLOAT16 | BFLOAT16 | TILE | TILE | 1 | 0.0014 | 0.0003 | 110.0 | 0.003 |

### Every movement program, in issue order

| # | op | ms | cores | in0 | in0 dt | in0 layout | out0 | out0 dt | out0 layout |
|---|---|---|---|---|---|---|---|---|---|
| 2 | Copy | 0.0039 | 110 | 1x140x32x128 | BFLOAT16 | TILE | 1x140x32x128 | BFLOAT16 | TILE |
| 6 | Copy | 0.0048 | 110 | 1x140x32x128 | BFLOAT16 | TILE | 1x140x32x128 | BFLOAT16 | TILE |
| 7 | ReshapeView | 0.0325 | 110 | 1x140x32x128 | BFLOAT16 | TILE | 1x280x32x128 | BFLOAT16 | TILE |
| 8 | Permute | 0.1212 | 110 | 1x280x32x128 | BFLOAT16 | TILE | 1x16x128x288 | BFLOAT16 | TILE |
| 10 | Permute | 0.1179 | 110 | 1x16x128x1120 | BFLOAT16 | TILE | 1x1120x32x128 | BFLOAT16 | TILE |
| 11 | ReshapeView | 0.1078 | 110 | 1x1120x32x128 | BFLOAT16 | TILE | 1x140x128x128 | BFLOAT16 | TILE |
| 14 | Pad | 0.0234 | 110 | 1x140x32x128 | BFLOAT16 | TILE | 1x140x128x128 | BFLOAT16 | TILE |
| 15 | NlpCreateHeads | 0.0741 | 110 | 140x1x128x128 | BFLOAT16 | TILE | 140x4x128x32 | BFLOAT16 | TILE |
| 16 | Slice | 0.0138 | 110 | 1x560x128x32 | BFLOAT16 | TILE | 1x560x32x32 | BFLOAT16 | TILE |
| 18 | NLPConcatHeads | 0.0108 | 110 | 140x4x32x32 | BFLOAT16 | TILE | 140x1x32x128 | BFLOAT16 | TILE |
| 24 | Copy | 0.0039 | 110 | 1x140x32x128 | BFLOAT16 | TILE | 1x140x32x128 | BFLOAT16 | TILE |
| 28 | Copy | 0.0050 | 110 | 1x140x32x128 | BFLOAT16 | TILE | 1x140x32x128 | BFLOAT16 | TILE |
| 38 | Copy | 0.0039 | 110 | 1x140x32x128 | BFLOAT16 | TILE | 1x140x32x128 | BFLOAT16 | TILE |
| 42 | Copy | 0.0049 | 110 | 1x140x32x128 | BFLOAT16 | TILE | 1x140x32x128 | BFLOAT16 | TILE |
| 43 | ReshapeView | 0.0323 | 110 | 1x140x32x128 | BFLOAT16 | TILE | 1x280x32x128 | BFLOAT16 | TILE |
| 44 | Permute | 0.1225 | 110 | 1x280x32x128 | BFLOAT16 | TILE | 1x16x128x288 | BFLOAT16 | TILE |
| 46 | Permute | 0.1173 | 110 | 1x16x128x1120 | BFLOAT16 | TILE | 1x1120x32x128 | BFLOAT16 | TILE |
| 47 | ReshapeView | 0.1077 | 110 | 1x1120x32x128 | BFLOAT16 | TILE | 1x140x128x128 | BFLOAT16 | TILE |
| 50 | Pad | 0.0246 | 110 | 1x140x32x128 | BFLOAT16 | TILE | 1x140x128x128 | BFLOAT16 | TILE |
| 51 | NlpCreateHeads | 0.0749 | 110 | 140x1x128x128 | BFLOAT16 | TILE | 140x4x128x32 | BFLOAT16 | TILE |
| 52 | Slice | 0.0136 | 110 | 1x560x128x32 | BFLOAT16 | TILE | 1x560x32x32 | BFLOAT16 | TILE |
| 54 | NLPConcatHeads | 0.0106 | 110 | 140x4x32x32 | BFLOAT16 | TILE | 140x1x32x128 | BFLOAT16 | TILE |
| 60 | Copy | 0.0039 | 110 | 1x140x32x128 | BFLOAT16 | TILE | 1x140x32x128 | BFLOAT16 | TILE |
| 64 | Copy | 0.0052 | 110 | 1x140x32x128 | BFLOAT16 | TILE | 1x140x32x128 | BFLOAT16 | TILE |
| 74 | Copy | 0.0039 | 110 | 1x140x32x128 | BFLOAT16 | TILE | 1x140x32x128 | BFLOAT16 | TILE |
| 78 | Copy | 0.0049 | 110 | 1x140x32x128 | BFLOAT16 | TILE | 1x140x32x128 | BFLOAT16 | TILE |
| 79 | ReshapeView | 0.0324 | 110 | 1x140x32x128 | BFLOAT16 | TILE | 1x280x32x128 | BFLOAT16 | TILE |
| 80 | Permute | 0.1213 | 110 | 1x280x32x128 | BFLOAT16 | TILE | 1x16x128x288 | BFLOAT16 | TILE |
| 82 | Permute | 0.1271 | 110 | 1x16x128x1120 | BFLOAT16 | TILE | 1x1120x32x128 | BFLOAT16 | TILE |
| 83 | ReshapeView | 0.1076 | 110 | 1x1120x32x128 | BFLOAT16 | TILE | 1x140x128x128 | BFLOAT16 | TILE |
| 86 | Pad | 0.0246 | 110 | 1x140x32x128 | BFLOAT16 | TILE | 1x140x128x128 | BFLOAT16 | TILE |
| 87 | NlpCreateHeads | 0.0748 | 110 | 140x1x128x128 | BFLOAT16 | TILE | 140x4x128x32 | BFLOAT16 | TILE |
| 88 | Slice | 0.0134 | 110 | 1x560x128x32 | BFLOAT16 | TILE | 1x560x32x32 | BFLOAT16 | TILE |
| 90 | NLPConcatHeads | 0.0110 | 110 | 140x4x32x32 | BFLOAT16 | TILE | 140x1x32x128 | BFLOAT16 | TILE |
| 96 | Copy | 0.0039 | 110 | 1x140x32x128 | BFLOAT16 | TILE | 1x140x32x128 | BFLOAT16 | TILE |
| 100 | Copy | 0.0051 | 110 | 1x140x32x128 | BFLOAT16 | TILE | 1x140x32x128 | BFLOAT16 | TILE |
| 112 | Transpose | 0.0046 | 110 | 1x1x768x512 | BFLOAT16 | TILE | 1x1x512x768 | BFLOAT16 | TILE |
| 141 | NlpCreateHeads | 0.0424 | 16 | 1x1x512x3072 | BFLOAT16 | TILE | 1x16x512x64 | BFLOAT16 | TILE |
| 143 | Slice | 0.0061 | 110 | 1x16x512x64 | BFLOAT16 | TILE | 1x16x512x64 | BFLOAT16 | TILE |
| 144 | Transpose | 0.0062 | 110 | 1x16x512x64 | BFLOAT16 | TILE | 1x16x64x512 | BFLOAT16 | TILE |
| 145 | ReshapeView | 0.0217 | 110 | 1x16x64x512 | BFLOAT16 | TILE | 1x1x768x512 | BFLOAT16 | TILE |
| 146 | Transpose | 0.0049 | 110 | 1x1x768x512 | BFLOAT16 | TILE | 1x1x512x768 | BFLOAT16 | TILE |
| 175 | NlpCreateHeads | 0.0428 | 16 | 1x1x512x3072 | BFLOAT16 | TILE | 1x16x512x64 | BFLOAT16 | TILE |
| 177 | Slice | 0.0060 | 110 | 1x16x512x64 | BFLOAT16 | TILE | 1x16x512x64 | BFLOAT16 | TILE |
| 178 | Transpose | 0.0060 | 110 | 1x16x512x64 | BFLOAT16 | TILE | 1x16x64x512 | BFLOAT16 | TILE |
| 179 | ReshapeView | 0.0218 | 110 | 1x16x64x512 | BFLOAT16 | TILE | 1x1x768x512 | BFLOAT16 | TILE |
| 180 | Transpose | 0.0046 | 110 | 1x1x768x512 | BFLOAT16 | TILE | 1x1x512x768 | BFLOAT16 | TILE |
| 209 | NlpCreateHeads | 0.0424 | 16 | 1x1x512x3072 | BFLOAT16 | TILE | 1x16x512x64 | BFLOAT16 | TILE |
| 211 | Slice | 0.0060 | 110 | 1x16x512x64 | BFLOAT16 | TILE | 1x16x512x64 | BFLOAT16 | TILE |
| 212 | Transpose | 0.0061 | 110 | 1x16x512x64 | BFLOAT16 | TILE | 1x16x64x512 | BFLOAT16 | TILE |
| 213 | ReshapeView | 0.0218 | 110 | 1x16x64x512 | BFLOAT16 | TILE | 1x1x768x512 | BFLOAT16 | TILE |
| 214 | Transpose | 0.0046 | 110 | 1x1x768x512 | BFLOAT16 | TILE | 1x1x512x768 | BFLOAT16 | TILE |
| 243 | NlpCreateHeads | 0.0425 | 16 | 1x1x512x3072 | BFLOAT16 | TILE | 1x16x512x64 | BFLOAT16 | TILE |
| 245 | Slice | 0.0061 | 110 | 1x16x512x64 | BFLOAT16 | TILE | 1x16x512x64 | BFLOAT16 | TILE |
| 246 | Transpose | 0.0062 | 110 | 1x16x512x64 | BFLOAT16 | TILE | 1x16x64x512 | BFLOAT16 | TILE |
| 247 | ReshapeView | 0.0217 | 110 | 1x16x64x512 | BFLOAT16 | TILE | 1x1x768x512 | BFLOAT16 | TILE |
| 248 | Transpose | 0.0049 | 110 | 1x1x768x512 | BFLOAT16 | TILE | 1x1x512x768 | BFLOAT16 | TILE |
| 277 | NlpCreateHeads | 0.0422 | 16 | 1x1x512x3072 | BFLOAT16 | TILE | 1x16x512x64 | BFLOAT16 | TILE |
| 279 | Slice | 0.0060 | 110 | 1x16x512x64 | BFLOAT16 | TILE | 1x16x512x64 | BFLOAT16 | TILE |
| 280 | Transpose | 0.0062 | 110 | 1x16x512x64 | BFLOAT16 | TILE | 1x16x64x512 | BFLOAT16 | TILE |
| 281 | ReshapeView | 0.0217 | 110 | 1x16x64x512 | BFLOAT16 | TILE | 1x1x768x512 | BFLOAT16 | TILE |
| 282 | Transpose | 0.0047 | 110 | 1x1x768x512 | BFLOAT16 | TILE | 1x1x512x768 | BFLOAT16 | TILE |
| 311 | NlpCreateHeads | 0.0425 | 16 | 1x1x512x3072 | BFLOAT16 | TILE | 1x16x512x64 | BFLOAT16 | TILE |
| 313 | Slice | 0.0060 | 110 | 1x16x512x64 | BFLOAT16 | TILE | 1x16x512x64 | BFLOAT16 | TILE |
| 314 | Transpose | 0.0061 | 110 | 1x16x512x64 | BFLOAT16 | TILE | 1x16x64x512 | BFLOAT16 | TILE |
| 315 | ReshapeView | 0.0218 | 110 | 1x16x64x512 | BFLOAT16 | TILE | 1x1x768x512 | BFLOAT16 | TILE |
| 316 | Transpose | 0.0046 | 110 | 1x1x768x512 | BFLOAT16 | TILE | 1x1x512x768 | BFLOAT16 | TILE |
| 345 | NlpCreateHeads | 0.0430 | 16 | 1x1x512x3072 | BFLOAT16 | TILE | 1x16x512x64 | BFLOAT16 | TILE |
| 347 | Slice | 0.0062 | 110 | 1x16x512x64 | BFLOAT16 | TILE | 1x16x512x64 | BFLOAT16 | TILE |
| 348 | Transpose | 0.0062 | 110 | 1x16x512x64 | BFLOAT16 | TILE | 1x16x64x512 | BFLOAT16 | TILE |
| 349 | ReshapeView | 0.0218 | 110 | 1x16x64x512 | BFLOAT16 | TILE | 1x1x768x512 | BFLOAT16 | TILE |
| 350 | Transpose | 0.0047 | 110 | 1x1x768x512 | BFLOAT16 | TILE | 1x1x512x768 | BFLOAT16 | TILE |
| 379 | NlpCreateHeads | 0.0428 | 16 | 1x1x512x3072 | BFLOAT16 | TILE | 1x16x512x64 | BFLOAT16 | TILE |
| 381 | Slice | 0.0062 | 110 | 1x16x512x64 | BFLOAT16 | TILE | 1x16x512x64 | BFLOAT16 | TILE |
| 382 | Transpose | 0.0061 | 110 | 1x16x512x64 | BFLOAT16 | TILE | 1x16x64x512 | BFLOAT16 | TILE |
| 383 | ReshapeView | 0.0218 | 110 | 1x16x64x512 | BFLOAT16 | TILE | 1x1x768x512 | BFLOAT16 | TILE |
| 384 | Transpose | 0.0049 | 110 | 1x1x768x512 | BFLOAT16 | TILE | 1x1x512x768 | BFLOAT16 | TILE |
| 413 | NlpCreateHeads | 0.0428 | 16 | 1x1x512x3072 | BFLOAT16 | TILE | 1x16x512x64 | BFLOAT16 | TILE |
| 415 | Slice | 0.0060 | 110 | 1x16x512x64 | BFLOAT16 | TILE | 1x16x512x64 | BFLOAT16 | TILE |
| 416 | Transpose | 0.0063 | 110 | 1x16x512x64 | BFLOAT16 | TILE | 1x16x64x512 | BFLOAT16 | TILE |
| 417 | ReshapeView | 0.0218 | 110 | 1x16x64x512 | BFLOAT16 | TILE | 1x1x768x512 | BFLOAT16 | TILE |
| 418 | Transpose | 0.0045 | 110 | 1x1x768x512 | BFLOAT16 | TILE | 1x1x512x768 | BFLOAT16 | TILE |
| 447 | NlpCreateHeads | 0.0427 | 16 | 1x1x512x3072 | BFLOAT16 | TILE | 1x16x512x64 | BFLOAT16 | TILE |
| 449 | Slice | 0.0061 | 110 | 1x16x512x64 | BFLOAT16 | TILE | 1x16x512x64 | BFLOAT16 | TILE |
| 450 | Transpose | 0.0061 | 110 | 1x16x512x64 | BFLOAT16 | TILE | 1x16x64x512 | BFLOAT16 | TILE |
| 451 | ReshapeView | 0.0218 | 110 | 1x16x64x512 | BFLOAT16 | TILE | 1x1x768x512 | BFLOAT16 | TILE |
| 452 | Transpose | 0.0048 | 110 | 1x1x768x512 | BFLOAT16 | TILE | 1x1x512x768 | BFLOAT16 | TILE |
| 481 | NlpCreateHeads | 0.0426 | 16 | 1x1x512x3072 | BFLOAT16 | TILE | 1x16x512x64 | BFLOAT16 | TILE |
| 483 | Slice | 0.0062 | 110 | 1x16x512x64 | BFLOAT16 | TILE | 1x16x512x64 | BFLOAT16 | TILE |
| 484 | Transpose | 0.0062 | 110 | 1x16x512x64 | BFLOAT16 | TILE | 1x16x64x512 | BFLOAT16 | TILE |
| 485 | ReshapeView | 0.0218 | 110 | 1x16x64x512 | BFLOAT16 | TILE | 1x1x768x512 | BFLOAT16 | TILE |
| 486 | Transpose | 0.0047 | 110 | 1x1x768x512 | BFLOAT16 | TILE | 1x1x512x768 | BFLOAT16 | TILE |
| 515 | NlpCreateHeads | 0.0426 | 16 | 1x1x512x3072 | BFLOAT16 | TILE | 1x16x512x64 | BFLOAT16 | TILE |
| 517 | Slice | 0.0059 | 110 | 1x16x512x64 | BFLOAT16 | TILE | 1x16x512x64 | BFLOAT16 | TILE |
| 518 | Transpose | 0.0061 | 110 | 1x16x512x64 | BFLOAT16 | TILE | 1x16x64x512 | BFLOAT16 | TILE |
| 519 | ReshapeView | 0.0217 | 110 | 1x16x64x512 | BFLOAT16 | TILE | 1x1x768x512 | BFLOAT16 | TILE |
| 520 | Transpose | 0.0047 | 110 | 1x1x768x512 | BFLOAT16 | TILE | 1x1x512x768 | BFLOAT16 | TILE |
| 549 | NlpCreateHeads | 0.0426 | 16 | 1x1x512x3072 | BFLOAT16 | TILE | 1x16x512x64 | BFLOAT16 | TILE |
| 551 | Slice | 0.0059 | 110 | 1x16x512x64 | BFLOAT16 | TILE | 1x16x512x64 | BFLOAT16 | TILE |
| 552 | Transpose | 0.0064 | 110 | 1x16x512x64 | BFLOAT16 | TILE | 1x16x64x512 | BFLOAT16 | TILE |
| 553 | ReshapeView | 0.0217 | 110 | 1x16x64x512 | BFLOAT16 | TILE | 1x1x768x512 | BFLOAT16 | TILE |
| 554 | Transpose | 0.0048 | 110 | 1x1x768x512 | BFLOAT16 | TILE | 1x1x512x768 | BFLOAT16 | TILE |
| 583 | NlpCreateHeads | 0.0428 | 16 | 1x1x512x3072 | BFLOAT16 | TILE | 1x16x512x64 | BFLOAT16 | TILE |
| 585 | Slice | 0.0060 | 110 | 1x16x512x64 | BFLOAT16 | TILE | 1x16x512x64 | BFLOAT16 | TILE |
| 586 | Transpose | 0.0064 | 110 | 1x16x512x64 | BFLOAT16 | TILE | 1x16x64x512 | BFLOAT16 | TILE |
| 587 | ReshapeView | 0.0217 | 110 | 1x16x64x512 | BFLOAT16 | TILE | 1x1x768x512 | BFLOAT16 | TILE |
| 588 | Transpose | 0.0048 | 110 | 1x1x768x512 | BFLOAT16 | TILE | 1x1x512x768 | BFLOAT16 | TILE |
| 617 | NlpCreateHeads | 0.0424 | 16 | 1x1x512x3072 | BFLOAT16 | TILE | 1x16x512x64 | BFLOAT16 | TILE |
| 619 | Slice | 0.0062 | 110 | 1x16x512x64 | BFLOAT16 | TILE | 1x16x512x64 | BFLOAT16 | TILE |
| 620 | Transpose | 0.0063 | 110 | 1x16x512x64 | BFLOAT16 | TILE | 1x16x64x512 | BFLOAT16 | TILE |
| 621 | ReshapeView | 0.0219 | 110 | 1x16x64x512 | BFLOAT16 | TILE | 1x1x768x512 | BFLOAT16 | TILE |
| 622 | Transpose | 0.0047 | 110 | 1x1x768x512 | BFLOAT16 | TILE | 1x1x512x768 | BFLOAT16 | TILE |
| 651 | NlpCreateHeads | 0.0424 | 16 | 1x1x512x3072 | BFLOAT16 | TILE | 1x16x512x64 | BFLOAT16 | TILE |
| 653 | Slice | 0.0063 | 110 | 1x16x512x64 | BFLOAT16 | TILE | 1x16x512x64 | BFLOAT16 | TILE |
| 654 | Transpose | 0.0065 | 110 | 1x16x512x64 | BFLOAT16 | TILE | 1x16x64x512 | BFLOAT16 | TILE |
| 655 | ReshapeView | 0.0217 | 110 | 1x16x64x512 | BFLOAT16 | TILE | 1x1x768x512 | BFLOAT16 | TILE |
| 656 | Transpose | 0.0047 | 110 | 1x1x768x512 | BFLOAT16 | TILE | 1x1x512x768 | BFLOAT16 | TILE |
| 685 | NlpCreateHeads | 0.0425 | 16 | 1x1x512x3072 | BFLOAT16 | TILE | 1x16x512x64 | BFLOAT16 | TILE |
| 687 | Slice | 0.0061 | 110 | 1x16x512x64 | BFLOAT16 | TILE | 1x16x512x64 | BFLOAT16 | TILE |
| 688 | Transpose | 0.0061 | 110 | 1x16x512x64 | BFLOAT16 | TILE | 1x16x64x512 | BFLOAT16 | TILE |
| 689 | ReshapeView | 0.0219 | 110 | 1x16x64x512 | BFLOAT16 | TILE | 1x1x768x512 | BFLOAT16 | TILE |
| 690 | Transpose | 0.0047 | 110 | 1x1x768x512 | BFLOAT16 | TILE | 1x1x512x768 | BFLOAT16 | TILE |
| 719 | NlpCreateHeads | 0.0428 | 16 | 1x1x512x3072 | BFLOAT16 | TILE | 1x16x512x64 | BFLOAT16 | TILE |
| 721 | Slice | 0.0061 | 110 | 1x16x512x64 | BFLOAT16 | TILE | 1x16x512x64 | BFLOAT16 | TILE |
| 722 | Transpose | 0.0064 | 110 | 1x16x512x64 | BFLOAT16 | TILE | 1x16x64x512 | BFLOAT16 | TILE |
| 723 | ReshapeView | 0.0217 | 110 | 1x16x64x512 | BFLOAT16 | TILE | 1x1x768x512 | BFLOAT16 | TILE |
| 724 | Transpose | 0.0048 | 110 | 1x1x768x512 | BFLOAT16 | TILE | 1x1x512x768 | BFLOAT16 | TILE |
| 753 | NlpCreateHeads | 0.0424 | 16 | 1x1x512x3072 | BFLOAT16 | TILE | 1x16x512x64 | BFLOAT16 | TILE |
| 755 | Slice | 0.0060 | 110 | 1x16x512x64 | BFLOAT16 | TILE | 1x16x512x64 | BFLOAT16 | TILE |
| 756 | Transpose | 0.0063 | 110 | 1x16x512x64 | BFLOAT16 | TILE | 1x16x64x512 | BFLOAT16 | TILE |
| 757 | ReshapeView | 0.0217 | 110 | 1x16x64x512 | BFLOAT16 | TILE | 1x1x768x512 | BFLOAT16 | TILE |
| 758 | Transpose | 0.0047 | 110 | 1x1x768x512 | BFLOAT16 | TILE | 1x1x512x768 | BFLOAT16 | TILE |
| 787 | NlpCreateHeads | 0.0426 | 16 | 1x1x512x3072 | BFLOAT16 | TILE | 1x16x512x64 | BFLOAT16 | TILE |
| 789 | Slice | 0.0060 | 110 | 1x16x512x64 | BFLOAT16 | TILE | 1x16x512x64 | BFLOAT16 | TILE |
| 790 | Transpose | 0.0064 | 110 | 1x16x512x64 | BFLOAT16 | TILE | 1x16x64x512 | BFLOAT16 | TILE |
| 791 | ReshapeView | 0.0218 | 110 | 1x16x64x512 | BFLOAT16 | TILE | 1x1x768x512 | BFLOAT16 | TILE |
| 792 | Transpose | 0.0048 | 110 | 1x1x768x512 | BFLOAT16 | TILE | 1x1x512x768 | BFLOAT16 | TILE |
| 821 | NlpCreateHeads | 0.0423 | 16 | 1x1x512x3072 | BFLOAT16 | TILE | 1x16x512x64 | BFLOAT16 | TILE |
| 823 | Slice | 0.0063 | 110 | 1x16x512x64 | BFLOAT16 | TILE | 1x16x512x64 | BFLOAT16 | TILE |
| 824 | Transpose | 0.0060 | 110 | 1x16x512x64 | BFLOAT16 | TILE | 1x16x64x512 | BFLOAT16 | TILE |
| 825 | ReshapeView | 0.0218 | 110 | 1x16x64x512 | BFLOAT16 | TILE | 1x1x768x512 | BFLOAT16 | TILE |
| 826 | Transpose | 0.0046 | 110 | 1x1x768x512 | BFLOAT16 | TILE | 1x1x512x768 | BFLOAT16 | TILE |
| 855 | NlpCreateHeads | 0.0430 | 16 | 1x1x512x3072 | BFLOAT16 | TILE | 1x16x512x64 | BFLOAT16 | TILE |
| 857 | Slice | 0.0061 | 110 | 1x16x512x64 | BFLOAT16 | TILE | 1x16x512x64 | BFLOAT16 | TILE |
| 858 | Transpose | 0.0062 | 110 | 1x16x512x64 | BFLOAT16 | TILE | 1x16x64x512 | BFLOAT16 | TILE |
| 859 | ReshapeView | 0.0217 | 110 | 1x16x64x512 | BFLOAT16 | TILE | 1x1x768x512 | BFLOAT16 | TILE |
| 860 | Transpose | 0.0046 | 110 | 1x1x768x512 | BFLOAT16 | TILE | 1x1x512x768 | BFLOAT16 | TILE |
| 889 | NlpCreateHeads | 0.0424 | 16 | 1x1x512x3072 | BFLOAT16 | TILE | 1x16x512x64 | BFLOAT16 | TILE |
| 891 | Slice | 0.0059 | 110 | 1x16x512x64 | BFLOAT16 | TILE | 1x16x512x64 | BFLOAT16 | TILE |
| 892 | Transpose | 0.0063 | 110 | 1x16x512x64 | BFLOAT16 | TILE | 1x16x64x512 | BFLOAT16 | TILE |
| 893 | ReshapeView | 0.0218 | 110 | 1x16x64x512 | BFLOAT16 | TILE | 1x1x768x512 | BFLOAT16 | TILE |
| 894 | Transpose | 0.0046 | 110 | 1x1x768x512 | BFLOAT16 | TILE | 1x1x512x768 | BFLOAT16 | TILE |
| 923 | NlpCreateHeads | 0.0421 | 16 | 1x1x512x3072 | BFLOAT16 | TILE | 1x16x512x64 | BFLOAT16 | TILE |
| 925 | Slice | 0.0060 | 110 | 1x16x512x64 | BFLOAT16 | TILE | 1x16x512x64 | BFLOAT16 | TILE |
| 926 | Transpose | 0.0062 | 110 | 1x16x512x64 | BFLOAT16 | TILE | 1x16x64x512 | BFLOAT16 | TILE |
| 927 | ReshapeView | 0.0218 | 110 | 1x16x64x512 | BFLOAT16 | TILE | 1x1x768x512 | BFLOAT16 | TILE |
| 928 | Transpose | 0.0047 | 110 | 1x1x768x512 | BFLOAT16 | TILE | 1x1x512x768 | BFLOAT16 | TILE |
| 952 | Transpose | 0.0014 | 110 | 1x1x512x128 | BFLOAT16 | TILE | 1x1x128x512 | BFLOAT16 | TILE |
| 954 | Transpose | 0.0058 | 110 | 1x1x128x4480 | BFLOAT16 | TILE | 1x1x4480x128 | BFLOAT16 | TILE |
| 956 | Copy | 0.0039 | 110 | 1x140x32x128 | BFLOAT16 | TILE | 1x140x32x128 | BFLOAT16 | TILE |
| 960 | Copy | 0.0052 | 110 | 1x140x32x128 | BFLOAT16 | TILE | 1x140x32x128 | BFLOAT16 | TILE |
| 961 | ReshapeView | 0.0323 | 110 | 1x140x32x128 | BFLOAT16 | TILE | 1x280x32x128 | BFLOAT16 | TILE |
| 962 | Permute | 0.1200 | 110 | 1x280x32x128 | BFLOAT16 | TILE | 1x16x128x288 | BFLOAT16 | TILE |
| 964 | Permute | 0.1269 | 110 | 1x16x128x1120 | BFLOAT16 | TILE | 1x1120x32x128 | BFLOAT16 | TILE |
| 965 | ReshapeView | 0.1073 | 110 | 1x1120x32x128 | BFLOAT16 | TILE | 1x140x128x128 | BFLOAT16 | TILE |
| 968 | Pad | 0.0247 | 110 | 1x140x32x128 | BFLOAT16 | TILE | 1x140x128x128 | BFLOAT16 | TILE |
| 969 | NlpCreateHeads | 0.0754 | 110 | 140x1x128x128 | BFLOAT16 | TILE | 140x4x128x32 | BFLOAT16 | TILE |
| 970 | Slice | 0.0134 | 110 | 1x560x128x32 | BFLOAT16 | TILE | 1x560x32x32 | BFLOAT16 | TILE |
| 972 | NLPConcatHeads | 0.0106 | 110 | 140x4x32x32 | BFLOAT16 | TILE | 140x1x32x128 | BFLOAT16 | TILE |
| 978 | Copy | 0.0039 | 110 | 1x140x32x128 | BFLOAT16 | TILE | 1x140x32x128 | BFLOAT16 | TILE |
| 982 | Copy | 0.0052 | 110 | 1x140x32x128 | BFLOAT16 | TILE | 1x140x32x128 | BFLOAT16 | TILE |
| 992 | Copy | 0.0038 | 110 | 1x140x32x128 | BFLOAT16 | TILE | 1x140x32x128 | BFLOAT16 | TILE |
| 996 | Copy | 0.0053 | 110 | 1x140x32x128 | BFLOAT16 | TILE | 1x140x32x128 | BFLOAT16 | TILE |
| 997 | ReshapeView | 0.0321 | 110 | 1x140x32x128 | BFLOAT16 | TILE | 1x280x32x128 | BFLOAT16 | TILE |
| 998 | Permute | 0.1184 | 110 | 1x280x32x128 | BFLOAT16 | TILE | 1x16x128x288 | BFLOAT16 | TILE |
| 1000 | Permute | 0.1294 | 110 | 1x16x128x1120 | BFLOAT16 | TILE | 1x1120x32x128 | BFLOAT16 | TILE |
| 1001 | ReshapeView | 0.1072 | 110 | 1x1120x32x128 | BFLOAT16 | TILE | 1x140x128x128 | BFLOAT16 | TILE |
| 1004 | Pad | 0.0236 | 110 | 1x140x32x128 | BFLOAT16 | TILE | 1x140x128x128 | BFLOAT16 | TILE |
| 1005 | NlpCreateHeads | 0.0754 | 110 | 140x1x128x128 | BFLOAT16 | TILE | 140x4x128x32 | BFLOAT16 | TILE |
| 1006 | Slice | 0.0137 | 110 | 1x560x128x32 | BFLOAT16 | TILE | 1x560x32x32 | BFLOAT16 | TILE |
| 1008 | NLPConcatHeads | 0.0105 | 110 | 140x4x32x32 | BFLOAT16 | TILE | 140x1x32x128 | BFLOAT16 | TILE |
| 1014 | Copy | 0.0039 | 110 | 1x140x32x128 | BFLOAT16 | TILE | 1x140x32x128 | BFLOAT16 | TILE |
| 1018 | Copy | 0.0052 | 110 | 1x140x32x128 | BFLOAT16 | TILE | 1x140x32x128 | BFLOAT16 | TILE |
| 1028 | Copy | 0.0038 | 110 | 1x140x32x128 | BFLOAT16 | TILE | 1x140x32x128 | BFLOAT16 | TILE |
| 1032 | Copy | 0.0052 | 110 | 1x140x32x128 | BFLOAT16 | TILE | 1x140x32x128 | BFLOAT16 | TILE |
| 1033 | ReshapeView | 0.0323 | 110 | 1x140x32x128 | BFLOAT16 | TILE | 1x280x32x128 | BFLOAT16 | TILE |
| 1034 | Permute | 0.1213 | 110 | 1x280x32x128 | BFLOAT16 | TILE | 1x16x128x288 | BFLOAT16 | TILE |
| 1036 | Permute | 0.1231 | 110 | 1x16x128x1120 | BFLOAT16 | TILE | 1x1120x32x128 | BFLOAT16 | TILE |
| 1037 | ReshapeView | 0.1069 | 110 | 1x1120x32x128 | BFLOAT16 | TILE | 1x140x128x128 | BFLOAT16 | TILE |
| 1040 | Pad | 0.0246 | 110 | 1x140x32x128 | BFLOAT16 | TILE | 1x140x128x128 | BFLOAT16 | TILE |
| 1041 | NlpCreateHeads | 0.0739 | 110 | 140x1x128x128 | BFLOAT16 | TILE | 140x4x128x32 | BFLOAT16 | TILE |
| 1042 | Slice | 0.0135 | 110 | 1x560x128x32 | BFLOAT16 | TILE | 1x560x32x32 | BFLOAT16 | TILE |
| 1044 | NLPConcatHeads | 0.0111 | 110 | 140x4x32x32 | BFLOAT16 | TILE | 140x1x32x128 | BFLOAT16 | TILE |
| 1050 | Copy | 0.0039 | 110 | 1x140x32x128 | BFLOAT16 | TILE | 1x140x32x128 | BFLOAT16 | TILE |
| 1054 | Copy | 0.0051 | 110 | 1x140x32x128 | BFLOAT16 | TILE | 1x140x32x128 | BFLOAT16 | TILE |
