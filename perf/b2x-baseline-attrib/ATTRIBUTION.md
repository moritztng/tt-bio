# Boltz-2 512 aa: every byte the fold moves, by module

One instrumented fold on tt-quietbox2 card 1 (p300c), ttnn 0.68.0, commit `13604183`, 3 recycles / 200 sampling steps / seed 0 on `perf/size512/fixtures/cdk2x2_512.yaml`.

The fold moves **5.386 TB**, not the 6.45 TB the published budget estimated. 99.9 % of it is inside a named module; the ops that belong to no module move 6.3 GB.

| phase | calls | ms/call | GB/call | Z/call | TB/fold | share | GB/s | % of roof |
|---|---|---|---|---|---|---|---|---|
| TrunkModule/Pairformer/PairformerLayer | 256 | 42.199 | 12.169 | 181.3 | 3.115 | 57.8 % | 288.4 | 67.1 % |
| DiffusionModule | 200 | 40.263 | 8.254 | 123.0 | 1.651 | 30.7 % | 205.0 | 47.7 % |
| TrunkModule/MSA/MSALayer | 16 | 122.781 | 32.244 | 480.5 | 0.516 | 9.6 % | 262.6 | 61.1 % |
| PairformerModule/Pairformer/PairformerLayer | 8 | 42.199 | 12.169 | 181.3 | 0.097 | 1.8 % | 288.4 | 67.1 % |
| glue (no module) | | | | | 0.006 | 0.1 % | | |

Z is one bf16 pair tensor at 512 tokens, 67.11 MB. ms/call is the median over that unit's calls; every nested bracket carries a device sync, which inflates the smaller units, so each GB/s here is a floor.

**REMAINDER-ATTRIBUTED: 98.1 %** — the four signatures the published budget instrumented carry 5051 GB, leaving 334 GB outside them, of which 328 GB now sits in a named module.


## MSALayer|1x512x512x128,1x1024x512x64 — 32.244 GB/call, 480.5 Z, 1104 ttnn ops

| sub-unit | GB read | GB written | ttnn ops |
|---|---|---|---|
| OuterProductMean | 8.7580 | 3.0199 | 24 |
| PairWeightedAveraging | 5.7047 | 1.9463 | 187 |
| TriangleMultiplication | 5.3691 | 2.0133 | 62 |
| TriangleAttention | 2.7099 | 0.9773 | 54 |
| Transition | 1.2085 | 0.4698 | 776 |
| MSALayer | 0.0671 | 0.0000 | 1 |

| sub-unit | op | GB read | GB written |
|---|---|---|---|
| TriangleMultiplication | ttnn.generic_op | 4.5638 | 1.3422 |
| OuterProductMean | ttnn.to_layout | 2.2817 | 1.1409 |
| OuterProductMean | ttnn.permute | 2.4159 | 0.6040 |
| PairWeightedAveraging | ttnn.permute | 2.1475 | 0.5369 |
| OuterProductMean | ttnn.reshape | 2.0133 | 0.5369 |
| PairWeightedAveraging | ttnn.linear | 1.4094 | 1.0737 |
| TriangleAttention | ttnn.generic_op | 1.5480 | 0.4027 |
| OuterProductMean | ttnn.matmul | 0.6040 | 0.5369 |
| TriangleMultiplication | ttnn.allocate_tensor_on_device | 0.5369 | 0.5369 |
| OuterProductMean | ttnn.linear | 0.7385 | 0.1342 |
| PairWeightedAveraging | ttnn.matmul | 0.5369 | 0.2684 |
| TriangleAttention | ttnn.allocate_tensor_on_device | 0.4027 | 0.4027 |
| Transition | ttnn.concat | 0.5369 | 0.2013 |
| OuterProductMean | ttnn.multiply_ | 0.5704 | 0.0000 |
| PairWeightedAveraging | ttnn.multiply_ | 0.5369 | 0.0000 |
| PairWeightedAveraging | ttnn.add_ | 0.5369 | 0.0000 |
| Transition | ttnn.chunk | 0.4027 | 0.1342 |
| TriangleMultiplication | ttnn.layer_norm | 0.2684 | 0.1342 |

dtype, by DRAM bytes read: BFLOAT16 100.00 % (23817.2 MB)

## PairformerLayer|1x512x384,1x512x512x128 — 12.169 GB/call, 181.3 Z, 428 ttnn ops

| sub-unit | GB read | GB written | ttnn ops |
|---|---|---|---|
| TriangleMultiplication | 5.5070 | 2.0468 | 66 |
| TriangleAttention | 2.7182 | 0.9815 | 56 |
| Transition | 0.5432 | 0.2021 | 274 |
| AttentionPairBias | 0.1464 | 0.0240 | 32 |

| sub-unit | op | GB read | GB written |
|---|---|---|---|
| TriangleMultiplication | ttnn.generic_op | 4.6991 | 1.3422 |
| TriangleAttention | ttnn.generic_op | 1.5564 | 0.4068 |
| TriangleMultiplication | ttnn.allocate_tensor_on_device | 0.5369 | 0.5369 |
| TriangleAttention | ttnn.allocate_tensor_on_device | 0.4027 | 0.4027 |
| TriangleMultiplication | ttnn.layer_norm | 0.2684 | 0.1342 |
| TriangleAttention | ttnn.layer_norm | 0.2013 | 0.1342 |
| Transition | ttnn.concat | 0.2029 | 0.0675 |
| TriangleAttention | ttnn.reshape | 0.2684 | 0.0000 |
| Transition | ttnn.chunk | 0.2013 | 0.0671 |
| TriangleAttention | ttnn.linear | 0.1384 | 0.0336 |
| TriangleAttention | ttnn.permute | 0.1426 | 0.0042 |
| Transition | ttnn.linear | 0.0714 | 0.0675 |
| Transition | ttnn.layer_norm | 0.0675 | 0.0000 |
| AttentionPairBias | ttnn.layer_norm | 0.0671 | 0.0000 |
| TriangleMultiplication | ttnn.unsqueeze | 0.0026 | 0.0336 |
| AttentionPairBias | ttnn.softmax | 0.0252 | 0.0084 |
| AttentionPairBias | ttnn.matmul | 0.0189 | 0.0089 |
| TriangleAttention | ttnn.unsqueeze | 0.0084 | 0.0000 |

dtype, by DRAM bytes read: BFLOAT16 100.00 % (8914.7 MB)

## PairformerLayer|1x512x512x128 — 11.808 GB/call, 176.0 Z, 376 ttnn ops

| sub-unit | GB read | GB written | ttnn ops |
|---|---|---|---|
| TriangleMultiplication | 5.3691 | 2.0133 | 62 |
| TriangleAttention | 2.7099 | 0.9773 | 54 |
| Transition | 0.5373 | 0.2013 | 260 |

| sub-unit | op | GB read | GB written |
|---|---|---|---|
| TriangleMultiplication | ttnn.generic_op | 4.5638 | 1.3422 |
| TriangleAttention | ttnn.generic_op | 1.5480 | 0.4027 |
| TriangleMultiplication | ttnn.allocate_tensor_on_device | 0.5369 | 0.5369 |
| TriangleAttention | ttnn.allocate_tensor_on_device | 0.4027 | 0.4027 |
| TriangleMultiplication | ttnn.layer_norm | 0.2684 | 0.1342 |
| TriangleAttention | ttnn.layer_norm | 0.2013 | 0.1342 |
| TriangleAttention | ttnn.reshape | 0.2684 | 0.0000 |
| Transition | ttnn.chunk | 0.2013 | 0.0671 |
| Transition | ttnn.concat | 0.2013 | 0.0671 |
| TriangleAttention | ttnn.linear | 0.1384 | 0.0336 |
| TriangleAttention | ttnn.permute | 0.1426 | 0.0042 |
| Transition | ttnn.linear | 0.0675 | 0.0671 |
| Transition | ttnn.layer_norm | 0.0671 | 0.0000 |
| TriangleAttention | ttnn.unsqueeze | 0.0084 | 0.0000 |
| Transition | ttnn.multiply_ | 0.0000 | 0.0000 |

dtype, by DRAM bytes read: BFLOAT16 100.00 % (8616.2 MB)

## OuterProductMean|1x1024x512x64,1024x1x1 — 11.778 GB/call, 175.5 Z, 24 ttnn ops

| sub-unit | GB read | GB written | ttnn ops |
|---|---|---|---|
| OuterProductMean | 8.7580 | 3.0199 | 24 |

| sub-unit | op | GB read | GB written |
|---|---|---|---|
| OuterProductMean | ttnn.to_layout | 2.2817 | 1.1409 |
| OuterProductMean | ttnn.permute | 2.4159 | 0.6040 |
| OuterProductMean | ttnn.reshape | 2.0133 | 0.5369 |
| OuterProductMean | ttnn.matmul | 0.6040 | 0.5369 |
| OuterProductMean | ttnn.linear | 0.7385 | 0.1342 |
| OuterProductMean | ttnn.multiply_ | 0.5704 | 0.0000 |
| OuterProductMean | ttnn.layer_norm | 0.1342 | 0.0671 |

dtype, by DRAM bytes read: BFLOAT16 100.00 % (8758.0 MB)

## DiffusionModule| — 8.254 GB/call, 123.0 Z, 1816 ttnn ops

| sub-unit | GB read | GB written | ttnn ops |
|---|---|---|---|
| AttentionPairBias | 4.5648 | 1.2992 | 792 |
| ConditionedTransitionBlock | 0.8860 | 0.2391 | 360 |
| AdaLN | 0.5790 | 0.1730 | 492 |
| DiffusionTransformerLayer | 0.2564 | 0.1085 | 114 |
| Diffusion | 0.0983 | 0.0299 | 38 |
| Transition | 0.0173 | 0.0016 | 16 |
| DiffusionModule | 0.0001 | 0.0005 | 4 |

| sub-unit | op | GB read | GB written |
|---|---|---|---|
| AttentionPairBias | ttnn.reshape | 0.9374 | 0.0849 |
| AttentionPairBias | ttnn.matmul | 0.5592 | 0.2705 |
| ConditionedTransitionBlock | ttnn.linear | 0.5872 | 0.2391 |
| AttentionPairBias | ttnn.softmax | 0.6040 | 0.2013 |
| AttentionPairBias | ttnn.linear | 0.4982 | 0.2344 |
| AttentionPairBias | ttnn.experimental.nlp_create_qkv_heads | 0.4152 | 0.2076 |
| AttentionPairBias | ttnn.add_ | 0.4027 | 0.0000 |
| AttentionPairBias | ttnn.permute | 0.2233 | 0.1431 |
| AdaLN | ttnn.linear | 0.2266 | 0.0755 |
| ConditionedTransitionBlock | ttnn.multiply_ | 0.2988 | 0.0000 |
| AttentionPairBias | ttnn.multiply_ | 0.2013 | 0.0000 |
| AdaLN | ttnn.layer_norm | 0.1133 | 0.0755 |
| DiffusionTransformerLayer | ttnn.add | 0.1195 | 0.0598 |
| AttentionPairBias | ttnn.Tensor.__getitem__ | 0.1353 | 0.0362 |
| AttentionPairBias | ttnn.transformer.scaled_dot_product_attention | 0.1541 | 0.0110 |
| AttentionPairBias | ttnn.unsqueeze | 0.1510 | 0.0000 |
| AttentionPairBias | ttnn.pad | 0.0991 | 0.0440 |
| AttentionPairBias | ttnn.multiply | 0.0897 | 0.0299 |

dtype, by DRAM bytes read: BFLOAT16 99.99 % (6401.1 MB), BFLOAT4_B 0.01 % (0.8 MB)

## Diffusion|1x7168x3,1 — 8.253 GB/call, 123.0 Z, 1812 ttnn ops

| sub-unit | GB read | GB written | ttnn ops |
|---|---|---|---|
| AttentionPairBias | 4.5648 | 1.2992 | 792 |
| ConditionedTransitionBlock | 0.8860 | 0.2391 | 360 |
| AdaLN | 0.5790 | 0.1730 | 492 |
| DiffusionTransformerLayer | 0.2564 | 0.1085 | 114 |
| Diffusion | 0.0983 | 0.0299 | 38 |
| Transition | 0.0173 | 0.0016 | 16 |

| sub-unit | op | GB read | GB written |
|---|---|---|---|
| AttentionPairBias | ttnn.reshape | 0.9374 | 0.0849 |
| AttentionPairBias | ttnn.matmul | 0.5592 | 0.2705 |
| ConditionedTransitionBlock | ttnn.linear | 0.5872 | 0.2391 |
| AttentionPairBias | ttnn.softmax | 0.6040 | 0.2013 |
| AttentionPairBias | ttnn.linear | 0.4982 | 0.2344 |
| AttentionPairBias | ttnn.experimental.nlp_create_qkv_heads | 0.4152 | 0.2076 |
| AttentionPairBias | ttnn.add_ | 0.4027 | 0.0000 |
| AttentionPairBias | ttnn.permute | 0.2233 | 0.1431 |
| AdaLN | ttnn.linear | 0.2266 | 0.0755 |
| ConditionedTransitionBlock | ttnn.multiply_ | 0.2988 | 0.0000 |
| AttentionPairBias | ttnn.multiply_ | 0.2013 | 0.0000 |
| AdaLN | ttnn.layer_norm | 0.1133 | 0.0755 |
| DiffusionTransformerLayer | ttnn.add | 0.1195 | 0.0598 |
| AttentionPairBias | ttnn.Tensor.__getitem__ | 0.1353 | 0.0362 |
| AttentionPairBias | ttnn.transformer.scaled_dot_product_attention | 0.1541 | 0.0110 |
| AttentionPairBias | ttnn.unsqueeze | 0.1510 | 0.0000 |
| AttentionPairBias | ttnn.pad | 0.0991 | 0.0440 |
| AttentionPairBias | ttnn.multiply | 0.0897 | 0.0299 |

dtype, by DRAM bytes read: BFLOAT16 99.99 % (6401.0 MB), BFLOAT4_B 0.01 % (0.8 MB)

## PairWeightedAveraging|1x1024x512x64,1x512x512x128 — 7.651 GB/call, 114.0 Z, 187 ttnn ops

| sub-unit | GB read | GB written | ttnn ops |
|---|---|---|---|
| PairWeightedAveraging | 5.7047 | 1.9463 | 187 |

| sub-unit | op | GB read | GB written |
|---|---|---|---|
| PairWeightedAveraging | ttnn.permute | 2.1475 | 0.5369 |
| PairWeightedAveraging | ttnn.linear | 1.4094 | 1.0737 |
| PairWeightedAveraging | ttnn.matmul | 0.5369 | 0.2684 |
| PairWeightedAveraging | ttnn.multiply_ | 0.5369 | 0.0000 |
| PairWeightedAveraging | ttnn.add_ | 0.5369 | 0.0000 |
| PairWeightedAveraging | ttnn.reshape | 0.3355 | 0.0000 |
| PairWeightedAveraging | ttnn.layer_norm | 0.2013 | 0.0671 |
| PairWeightedAveraging | ttnn.Tensor.__getitem__ | 0.0003 | 0.0002 |
| PairWeightedAveraging | ttnn.softmax | 0.0000 | 0.0000 |

dtype, by DRAM bytes read: BFLOAT16 100.00 % (5704.7 MB)

## DiffusionTransformer|1x512x768,1x512x768 — 5.129 GB/call, 76.4 Z, 1440 ttnn ops

| sub-unit | GB read | GB written | ttnn ops |
|---|---|---|---|
| AttentionPairBias | 2.6803 | 0.7487 | 624 |
| ConditionedTransitionBlock | 0.6512 | 0.1510 | 288 |
| AdaLN | 0.4917 | 0.1510 | 432 |
| DiffusionTransformerLayer | 0.1793 | 0.0755 | 96 |

| sub-unit | op | GB read | GB written |
|---|---|---|---|
| AttentionPairBias | ttnn.softmax | 0.6040 | 0.2013 |
| AttentionPairBias | ttnn.matmul | 0.5033 | 0.2265 |
| ConditionedTransitionBlock | ttnn.linear | 0.4625 | 0.1510 |
| AttentionPairBias | ttnn.linear | 0.3210 | 0.1132 |
| AttentionPairBias | ttnn.add_ | 0.4027 | 0.0000 |
| AdaLN | ttnn.linear | 0.2266 | 0.0755 |
| AttentionPairBias | ttnn.experimental.nlp_create_qkv_heads | 0.1510 | 0.0755 |
| AttentionPairBias | ttnn.multiply_ | 0.2013 | 0.0000 |
| AdaLN | ttnn.layer_norm | 0.1141 | 0.0755 |
| ConditionedTransitionBlock | ttnn.multiply_ | 0.1887 | 0.0000 |
| AttentionPairBias | ttnn.permute | 0.1132 | 0.0440 |
| AttentionPairBias | ttnn.unsqueeze | 0.1510 | 0.0000 |
| DiffusionTransformerLayer | ttnn.add | 0.0755 | 0.0377 |
| AttentionPairBias | ttnn.Tensor.__getitem__ | 0.0692 | 0.0252 |
| AdaLN | ttnn.multiply_ | 0.0755 | 0.0000 |
| AdaLN | ttnn.add_ | 0.0755 | 0.0000 |
| AttentionPairBias | ttnn.transpose | 0.0503 | 0.0252 |
| AttentionPairBias | ttnn.reshape | 0.0566 | 0.0189 |

dtype, by DRAM bytes read: BFLOAT16 100.00 % (4002.5 MB)

## TriangleMultiplication|1x512x512x128,1x512x512 — 3.777 GB/call, 56.3 Z, 31 ttnn ops

| sub-unit | GB read | GB written | ttnn ops |
|---|---|---|---|
| TriangleMultiplication | 2.7538 | 1.0234 | 31 |

| sub-unit | op | GB read | GB written |
|---|---|---|---|
| TriangleMultiplication | ttnn.generic_op | 2.3495 | 0.6711 |
| TriangleMultiplication | ttnn.allocate_tensor_on_device | 0.2684 | 0.2684 |
| TriangleMultiplication | ttnn.layer_norm | 0.1342 | 0.0671 |
| TriangleMultiplication | ttnn.unsqueeze | 0.0016 | 0.0168 |

dtype, by DRAM bytes read: BFLOAT16 100.00 % (2753.8 MB)

## TriangleMultiplication|1x512x512x128 — 3.691 GB/call, 55.0 Z, 29 ttnn ops

| sub-unit | GB read | GB written | ttnn ops |
|---|---|---|---|
| TriangleMultiplication | 2.6846 | 1.0066 | 29 |

| sub-unit | op | GB read | GB written |
|---|---|---|---|
| TriangleMultiplication | ttnn.generic_op | 2.2819 | 0.6711 |
| TriangleMultiplication | ttnn.allocate_tensor_on_device | 0.2684 | 0.2684 |
| TriangleMultiplication | ttnn.layer_norm | 0.1342 | 0.0671 |

dtype, by DRAM bytes read: BFLOAT16 100.00 % (2684.6 MB)

## TriangleAttention|1x512x512x128,1x1x1x512 — 1.850 GB/call, 27.6 Z, 27 ttnn ops

| sub-unit | GB read | GB written | ttnn ops |
|---|---|---|---|
| TriangleAttention | 1.3591 | 0.4907 | 27 |

| sub-unit | op | GB read | GB written |
|---|---|---|---|
| TriangleAttention | ttnn.generic_op | 0.7447 | 0.2034 |
| TriangleAttention | ttnn.allocate_tensor_on_device | 0.2013 | 0.2013 |
| TriangleAttention | ttnn.permute | 0.1384 | 0.0021 |
| TriangleAttention | ttnn.layer_norm | 0.0671 | 0.0671 |
| TriangleAttention | ttnn.reshape | 0.1342 | 0.0000 |
| TriangleAttention | ttnn.linear | 0.0692 | 0.0168 |
| TriangleAttention | ttnn.unsqueeze | 0.0042 | 0.0000 |

dtype, by DRAM bytes read: BFLOAT16 100.00 % (1359.1 MB)

## TriangleAttention|1x512x512x128 — 1.844 GB/call, 27.5 Z, 26 ttnn ops

| sub-unit | GB read | GB written | ttnn ops |
|---|---|---|---|
| TriangleAttention | 1.3549 | 0.4886 | 26 |

| sub-unit | op | GB read | GB written |
|---|---|---|---|
| TriangleAttention | ttnn.generic_op | 0.7405 | 0.2013 |
| TriangleAttention | ttnn.allocate_tensor_on_device | 0.2013 | 0.2013 |
| TriangleAttention | ttnn.permute | 0.1384 | 0.0021 |
| TriangleAttention | ttnn.layer_norm | 0.0671 | 0.0671 |
| TriangleAttention | ttnn.reshape | 0.1342 | 0.0000 |
| TriangleAttention | ttnn.linear | 0.0692 | 0.0168 |
| TriangleAttention | ttnn.unsqueeze | 0.0042 | 0.0000 |

dtype, by DRAM bytes read: BFLOAT16 100.00 % (1354.9 MB)

## DiffusionTransformer|1x224x32x128,1x224x32x128 — 1.490 GB/call, 22.2 Z, 159 ttnn ops

| sub-unit | GB read | GB written | ttnn ops |
|---|---|---|---|
| AttentionPairBias | 0.9427 | 0.2753 | 84 |
| ConditionedTransitionBlock | 0.1183 | 0.0440 | 36 |
| AdaLN | 0.0440 | 0.0110 | 30 |
| DiffusionTransformerLayer | 0.0385 | 0.0165 | 9 |

| sub-unit | op | GB read | GB written |
|---|---|---|---|
| AttentionPairBias | ttnn.reshape | 0.4404 | 0.0330 |
| AttentionPairBias | ttnn.experimental.nlp_create_qkv_heads | 0.1321 | 0.0661 |
| AttentionPairBias | ttnn.linear | 0.0886 | 0.0606 |
| ConditionedTransitionBlock | ttnn.linear | 0.0633 | 0.0440 |
| AttentionPairBias | ttnn.permute | 0.0551 | 0.0495 |
| AttentionPairBias | ttnn.transformer.scaled_dot_product_attention | 0.0771 | 0.0055 |
| AttentionPairBias | ttnn.pad | 0.0495 | 0.0220 |
| ConditionedTransitionBlock | ttnn.multiply_ | 0.0551 | 0.0000 |
| AttentionPairBias | ttnn.matmul | 0.0283 | 0.0220 |
| AttentionPairBias | ttnn.Tensor.__getitem__ | 0.0330 | 0.0055 |
| AdaLN | ttnn.to_memory_config | 0.0220 | 0.0110 |
| DiffusionTransformerLayer | ttnn.add | 0.0220 | 0.0110 |
| AttentionPairBias | ttnn.multiply | 0.0165 | 0.0055 |
| DiffusionTransformerLayer | ttnn.multiply | 0.0165 | 0.0055 |
| AttentionPairBias | ttnn.experimental.nlp_concat_heads | 0.0110 | 0.0055 |
| AdaLN | ttnn.multiply_ | 0.0110 | 0.0000 |
| AdaLN | ttnn.add_ | 0.0110 | 0.0000 |
| AttentionPairBias | ttnn.squeeze | 0.0110 | 0.0000 |

dtype, by DRAM bytes read: BFLOAT16 99.93 % (1142.8 MB), BFLOAT4_B 0.07 % (0.8 MB)

## Transition|1x1024x512x64 — 0.805 GB/call, 12.0 Z, 514 ttnn ops

| sub-unit | GB read | GB written | ttnn ops |
|---|---|---|---|
| Transition | 0.5370 | 0.2684 | 514 |

| sub-unit | op | GB read | GB written |
|---|---|---|---|
| Transition | ttnn.concat | 0.2013 | 0.1342 |
| Transition | ttnn.chunk | 0.2013 | 0.0671 |
| Transition | ttnn.linear | 0.0672 | 0.0671 |
| Transition | ttnn.layer_norm | 0.0671 | 0.0000 |
| Transition | ttnn.multiply_ | 0.0000 | 0.0000 |

dtype, by DRAM bytes read: BFLOAT16 100.00 % (537.0 MB)

## Transition|1x512x512x128 — 0.671 GB/call, 10.0 Z, 258 ttnn ops

| sub-unit | GB read | GB written | ttnn ops |
|---|---|---|---|
| Transition | 0.4702 | 0.2013 | 258 |

| sub-unit | op | GB read | GB written |
|---|---|---|---|
| Transition | ttnn.chunk | 0.2013 | 0.0671 |
| Transition | ttnn.concat | 0.1342 | 0.0671 |
| Transition | ttnn.linear | 0.0675 | 0.0671 |
| Transition | ttnn.layer_norm | 0.0671 | 0.0000 |
| Transition | ttnn.multiply_ | 0.0000 | 0.0000 |

dtype, by DRAM bytes read: BFLOAT16 100.00 % (470.2 MB)

## DiffusionTransformerLayer|1x224x32x128,1x224x32x128 — 0.517 GB/call, 7.7 Z, 66 ttnn ops

| sub-unit | GB read | GB written | ttnn ops |
|---|---|---|---|
| AttentionPairBias | 0.3148 | 0.0917 | 28 |
| ConditionedTransitionBlock | 0.0388 | 0.0147 | 12 |
| AdaLN | 0.0240 | 0.0110 | 22 |
| DiffusionTransformerLayer | 0.0147 | 0.0073 | 4 |

| sub-unit | op | GB read | GB written |
|---|---|---|---|
| AttentionPairBias | ttnn.reshape | 0.1468 | 0.0110 |
| AttentionPairBias | ttnn.experimental.nlp_create_qkv_heads | 0.0440 | 0.0220 |
| AttentionPairBias | ttnn.linear | 0.0295 | 0.0202 |
| ConditionedTransitionBlock | ttnn.linear | 0.0205 | 0.0147 |
| AttentionPairBias | ttnn.permute | 0.0184 | 0.0165 |
| AdaLN | ttnn.to_memory_config | 0.0165 | 0.0110 |
| AttentionPairBias | ttnn.transformer.scaled_dot_product_attention | 0.0257 | 0.0018 |
| AttentionPairBias | ttnn.pad | 0.0165 | 0.0073 |
| ConditionedTransitionBlock | ttnn.multiply_ | 0.0184 | 0.0000 |
| AttentionPairBias | ttnn.matmul | 0.0100 | 0.0073 |
| AttentionPairBias | ttnn.Tensor.__getitem__ | 0.0110 | 0.0018 |
| DiffusionTransformerLayer | ttnn.add | 0.0073 | 0.0037 |
| AttentionPairBias | ttnn.multiply | 0.0055 | 0.0018 |
| DiffusionTransformerLayer | ttnn.multiply | 0.0055 | 0.0018 |
| AttentionPairBias | ttnn.experimental.nlp_concat_heads | 0.0037 | 0.0018 |
| DiffusionTransformerLayer | ttnn.linear | 0.0019 | 0.0018 |
| AdaLN | ttnn.multiply_ | 0.0037 | 0.0000 |
| AdaLN | ttnn.add_ | 0.0037 | 0.0000 |

dtype, by DRAM bytes read: BFLOAT16 99.80 % (391.5 MB), BFLOAT4_B 0.20 % (0.8 MB)

## AttentionPairBias|1x224x32x128,224x4x32x128 — 0.407 GB/call, 6.1 Z, 28 ttnn ops

| sub-unit | GB read | GB written | ttnn ops |
|---|---|---|---|
| AttentionPairBias | 0.3148 | 0.0917 | 28 |

| sub-unit | op | GB read | GB written |
|---|---|---|---|
| AttentionPairBias | ttnn.reshape | 0.1468 | 0.0110 |
| AttentionPairBias | ttnn.experimental.nlp_create_qkv_heads | 0.0440 | 0.0220 |
| AttentionPairBias | ttnn.linear | 0.0295 | 0.0202 |
| AttentionPairBias | ttnn.permute | 0.0184 | 0.0165 |
| AttentionPairBias | ttnn.transformer.scaled_dot_product_attention | 0.0257 | 0.0018 |
| AttentionPairBias | ttnn.pad | 0.0165 | 0.0073 |
| AttentionPairBias | ttnn.matmul | 0.0100 | 0.0073 |
| AttentionPairBias | ttnn.Tensor.__getitem__ | 0.0110 | 0.0018 |
| AttentionPairBias | ttnn.multiply | 0.0055 | 0.0018 |
| AttentionPairBias | ttnn.experimental.nlp_concat_heads | 0.0037 | 0.0018 |
| AttentionPairBias | ttnn.squeeze | 0.0037 | 0.0000 |

dtype, by DRAM bytes read: BFLOAT16 99.74 % (314.0 MB), BFLOAT4_B 0.26 % (0.8 MB)

## DiffusionTransformerLayer|1x512x768,1x512x768 — 0.214 GB/call, 3.2 Z, 60 ttnn ops

| sub-unit | GB read | GB written | ttnn ops |
|---|---|---|---|
| AttentionPairBias | 0.1117 | 0.0312 | 26 |
| ConditionedTransitionBlock | 0.0271 | 0.0063 | 12 |
| AdaLN | 0.0212 | 0.0063 | 18 |
| DiffusionTransformerLayer | 0.0075 | 0.0031 | 4 |

| sub-unit | op | GB read | GB written |
|---|---|---|---|
| AttentionPairBias | ttnn.softmax | 0.0252 | 0.0084 |
| AttentionPairBias | ttnn.matmul | 0.0210 | 0.0094 |
| ConditionedTransitionBlock | ttnn.linear | 0.0193 | 0.0063 |
| AttentionPairBias | ttnn.linear | 0.0134 | 0.0047 |
| AttentionPairBias | ttnn.add_ | 0.0168 | 0.0000 |
| AdaLN | ttnn.linear | 0.0094 | 0.0031 |
| AttentionPairBias | ttnn.experimental.nlp_create_qkv_heads | 0.0063 | 0.0031 |
| AdaLN | ttnn.layer_norm | 0.0055 | 0.0031 |
| AttentionPairBias | ttnn.multiply_ | 0.0084 | 0.0000 |
| ConditionedTransitionBlock | ttnn.multiply_ | 0.0079 | 0.0000 |
| AttentionPairBias | ttnn.permute | 0.0047 | 0.0018 |
| AttentionPairBias | ttnn.unsqueeze | 0.0063 | 0.0000 |
| DiffusionTransformerLayer | ttnn.add | 0.0031 | 0.0016 |
| AttentionPairBias | ttnn.Tensor.__getitem__ | 0.0029 | 0.0010 |
| AdaLN | ttnn.multiply_ | 0.0031 | 0.0000 |
| AdaLN | ttnn.add_ | 0.0031 | 0.0000 |
| AttentionPairBias | ttnn.transpose | 0.0021 | 0.0010 |
| AttentionPairBias | ttnn.reshape | 0.0024 | 0.0008 |

dtype, by DRAM bytes read: BFLOAT16 100.00 % (167.5 MB)

## AttentionPairBias|1x512x384,1x512x512x128 — 0.170 GB/call, 2.5 Z, 32 ttnn ops

| sub-unit | GB read | GB written | ttnn ops |
|---|---|---|---|
| AttentionPairBias | 0.1464 | 0.0240 | 32 |

| sub-unit | op | GB read | GB written |
|---|---|---|---|
| AttentionPairBias | ttnn.layer_norm | 0.0671 | 0.0000 |
| AttentionPairBias | ttnn.softmax | 0.0252 | 0.0084 |
| AttentionPairBias | ttnn.matmul | 0.0189 | 0.0089 |
| AttentionPairBias | ttnn.add_ | 0.0084 | 0.0000 |
| AttentionPairBias | ttnn.multiply_ | 0.0084 | 0.0000 |
| AttentionPairBias | ttnn.linear | 0.0049 | 0.0024 |
| AttentionPairBias | ttnn.experimental.nlp_create_qkv_heads | 0.0031 | 0.0016 |
| AttentionPairBias | ttnn.permute | 0.0024 | 0.0009 |
| AttentionPairBias | ttnn.unsqueeze | 0.0031 | 0.0000 |
| AttentionPairBias | ttnn.Tensor.__getitem__ | 0.0014 | 0.0005 |
| AttentionPairBias | ttnn.transpose | 0.0010 | 0.0005 |
| AttentionPairBias | ttnn.reshape | 0.0012 | 0.0004 |
| AttentionPairBias | ttnn.multiply | 0.0012 | 0.0004 |

dtype, by DRAM bytes read: BFLOAT16 100.00 % (146.4 MB)

## AttentionPairBias|1x512x768,1x16x512x512 — 0.143 GB/call, 2.1 Z, 26 ttnn ops

| sub-unit | GB read | GB written | ttnn ops |
|---|---|---|---|
| AttentionPairBias | 0.1117 | 0.0312 | 26 |

| sub-unit | op | GB read | GB written |
|---|---|---|---|
| AttentionPairBias | ttnn.softmax | 0.0252 | 0.0084 |
| AttentionPairBias | ttnn.matmul | 0.0210 | 0.0094 |
| AttentionPairBias | ttnn.linear | 0.0134 | 0.0047 |
| AttentionPairBias | ttnn.add_ | 0.0168 | 0.0000 |
| AttentionPairBias | ttnn.experimental.nlp_create_qkv_heads | 0.0063 | 0.0031 |
| AttentionPairBias | ttnn.multiply_ | 0.0084 | 0.0000 |
| AttentionPairBias | ttnn.permute | 0.0047 | 0.0018 |
| AttentionPairBias | ttnn.unsqueeze | 0.0063 | 0.0000 |
| AttentionPairBias | ttnn.Tensor.__getitem__ | 0.0029 | 0.0010 |
| AttentionPairBias | ttnn.transpose | 0.0021 | 0.0010 |
| AttentionPairBias | ttnn.reshape | 0.0024 | 0.0008 |
| AttentionPairBias | ttnn.multiply | 0.0024 | 0.0008 |

dtype, by DRAM bytes read: BFLOAT16 100.00 % (111.7 MB)
