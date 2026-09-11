## env
`{"started": "2026-09-11T16:16:39+0000", "card": "0", "trace_region": true, "commit": "6f150345", "loadavg": ["0.00", "0.16", "1.69"], "size": 512, "phases": ["plain", "table", "census", "block", "control", "slope"], "hardware": "blackhole", "grid": [11, 10], "load_s": 3.02, "n_msa": 35, "card_type": "p300c"}`

## plain folds (no instrument, benchlock)
| arm | wall s | main-thread CPU s | CPU % | loadavg |
|---|---|---|---|---|
| a0 | 23.707 | 18.897 | 79.7 | 1.24 0.43 1.69 |
| a1 | 23.625 | 18.801 | 79.6 | 1.68 0.58 1.71 |

## per-phase wall and main-thread CPU (bracketed fold, wall 23.710 s, CPU 18.960 s = 80.0 %)
| phase | calls | wall s | host CPU s | CPU % of wall | blocked s | ttnn calls | us/call CPU |
|---|---|---|---|---|---|---|---|
| TrunkModule | 1 | 12.882 | 12.653 | 98.2 | 0.228 | 128654 | 94.4 |
| DiffusionModule | 200 | 7.151 | 2.859 | 40.0 | 4.292 | 372778 | 7.5 |
| PairformerModule | 1 | 0.465 | 0.242 | 52.2 | 0.222 | 3470 | 63.3 |
| **residual** | | 3.212 | 3.205 | 99.8 | 0.006 | 79 | |

## where the trunk's host CPU is, by sub-unit (census fold)
| path | ttnn calls | CPU inside ttnn s | us/call | bracket incl CPU s | bracket incl wall s |
|---|---|---|---|---|---|
| TrunkModule | 128654 | 12.143 | 94.4 | 12.6052 | 12.83446 |
| TrunkModule/Pairformer | 110853 | 10.271 | 92.7 | 10.66778 | 10.67135 |
| TrunkModule/Pairformer/PairformerLayer | 110848 | 10.270 | 92.7 | 10.66506 | 10.66851 |
| TrunkModule/Pairformer/PairformerLayer/Transition | 68096 | 9.453 | 138.8 | 9.48948 | 9.49223 |
| DiffusionModule | 372778 | 2.779 | 7.5 | 3.18015 | 7.16858 |
| DiffusionModule/Diffusion | 372083 | 2.237 | 6.0 | 2.60691 | 2.60905 |
| DiffusionModule/Diffusion/DiffusionTransformer | 361278 | 2.157 | 6.0 | 2.51599 | 2.51746 |
| DiffusionModule/Diffusion/DiffusionTransformer/DiffusionTransformerLayer | 361278 | 2.157 | 6.0 | 2.50483 | 2.50629 |
| TrunkModule/MSA | 17724 | 1.604 | 90.5 | 1.63922 | 1.64262 |
| TrunkModule/MSA/MSALayer | 17712 | 1.603 | 90.5 | 1.63809 | 1.64142 |
| DiffusionModule/Diffusion/DiffusionTransformer/DiffusionTransformerLayer/AttentionPairBias | 168000 | 0.929 | 5.5 | 1.11995 | 1.12065 |
| TrunkModule/MSA/MSALayer/Transition | 8224 | 0.898 | 109.2 | 0.90208 | 0.90321 |
| DiffusionModule/Diffusion/DiffusionTransformer/DiffusionTransformerLayer/ConditionedTransitionBlock | 121236 | 0.705 | 5.8 | 0.77516 | 0.77539 |
| TrunkModule/MSA/MSALayer/PairformerLayer | 6048 | 0.377 | 62.3 | 0.40219 | 0.40322 |
| TrunkModule/Pairformer/PairformerLayer/TriangleMultiplication | 15872 | 0.364 | 22.9 | 0.48142 | 0.48156 |
| DiffusionModule/Diffusion/DiffusionTransformer/DiffusionTransformerLayer/ConditionedTransitionBlock/AdaLN | 49236 | 0.300 | 6.1 | 0.32866 | 0.32864 |
| DiffusionModule/Diffusion/DiffusionTransformer/DiffusionTransformerLayer/AdaLN | 49236 | 0.286 | 5.8 | 0.31381 | 0.3141 |
| TrunkModule/MSA/MSALayer/PairformerLayer/Transition | 4128 | 0.272 | 66.0 | 0.27473 | 0.2753 |
| TrunkModule/Pairformer/PairformerLayer/AttentionPairBias | 8960 | 0.242 | 27.0 | 0.25826 | 0.25873 |
| TrunkModule/MSA/MSALayer/PairWeightedAveraging | 3008 | 0.232 | 77.3 | 0.23743 | 0.23816 |
| PairformerModule | 3470 | 0.220 | 63.3 | 0.24327 | 0.46507 |
| TrunkModule/Pairformer/PairformerLayer/TriangleAttention | 13824 | 0.137 | 9.9 | 0.35462 | 0.35467 |
| PairformerModule/Pairformer | 3464 | 0.108 | 31.3 | 0.12173 | 0.12173 |
| PairformerModule/Pairformer/PairformerLayer | 3464 | 0.108 | 31.3 | 0.12163 | 0.12163 |
| PairformerModule/Pairformer/PairformerLayer/Transition | 2128 | 0.094 | 44.1 | 0.0951 | 0.0951 |
| TrunkModule/MSA/MSALayer/PairformerLayer/TriangleMultiplication | 928 | 0.092 | 99.4 | 0.10008 | 0.10033 |
| TrunkModule/MSA/MSALayer/OuterProductMean | 384 | 0.083 | 216.4 | 0.08374 | 0.08418 |
| DiffusionModule/Diffusion/Transition | 3200 | 0.018 | 5.8 | 0.02073 | 0.02085 |

## the ops that carry the host CPU
| op | calls | CPU s | us/call |
|---|---|---|---|
| ttnn.linear | 109887 | 6.853 | 62.36 |
| ttnn.layer_norm | 35325 | 2.462 | 69.69 |
| ttnn.multiply_ | 48643 | 2.395 | 49.23 |
| ttnn.from_torch | 423 | 0.835 | 1975.02 |
| ttnn.chunk | 856 | 0.505 | 590.52 |
| ttnn.add_ | 19422 | 0.290 | 14.95 |
| ttnn.permute | 14949 | 0.228 | 15.27 |
| ttnn.generic_op | 4480 | 0.172 | 38.30 |
| ttnn.reshape | 19117 | 0.170 | 8.90 |
| ttnn.multiply | 12467 | 0.162 | 13.01 |
| ttnn.add | 13742 | 0.157 | 11.45 |
| ttnn.slice | 6806 | 0.150 | 22.01 |
| ttnn.matmul | 12433 | 0.141 | 11.37 |
| ttnn.deallocate | 138213 | 0.112 | 0.81 |
| ttnn.softmax | 5192 | 0.081 | 15.56 |
| ttnn.transpose | 5624 | 0.071 | 12.72 |
| ttnn.experimental.nlp_create_qkv_heads | 6264 | 0.070 | 11.16 |
| ttnn.to_torch | 204 | 0.069 | 337.97 |
| ttnn.to_memory_config | 15636 | 0.048 | 3.05 |
| ttnn.concat | 297 | 0.040 | 135.67 |

## slope: inject measured host CPU per ttnn call, read d(wall)/d(CPU)
| spin us/call | fold wall s | fold CPU s | added CPU s | added wall s | slope |
|---|---|---|---|---|---|
| 0.0 | 23.661 | 19.207 | +0.000 | +0.000 | nan |
| 10.0 | 23.931 | 22.971 | +3.764 | +0.270 | 0.072 |
| 25.0 | 28.994 | 28.508 | +9.301 | +5.332 | 0.573 |
| 0.0 | 23.619 | 19.141 | -0.067 | -0.043 | 0.639 |

| phase | spin | wall s | CPU s | added CPU s | added wall s | slope |
|---|---|---|---|---|---|---|
| DiffusionModule | 10 | 7.449 | 6.945 | +3.782 | +0.289 | 0.076 |
| DiffusionModule | 25 | 12.560 | 12.524 | +9.361 | +5.399 | 0.577 |
| DiffusionModule | 0 | 7.137 | 3.117 | -0.047 | -0.024 | nan |
| PairformerModule | 10 | 0.463 | 0.242 | -0.001 | -0.002 | nan |
| PairformerModule | 25 | 0.469 | 0.249 | +0.006 | +0.004 | nan |
| PairformerModule | 0 | 0.465 | 0.243 | -0.000 | +0.000 | nan |
| TrunkModule | 10 | 12.838 | 12.608 | -0.016 | -0.014 | nan |
| TrunkModule | 25 | 12.862 | 12.635 | +0.011 | +0.010 | nan |
| TrunkModule | 0 | 12.854 | 12.625 | +0.001 | +0.002 | nan |

## controls: is main-thread CPU a host/device discriminator on this stack?
| loop | reps | host us/call | device us/call | CPU/wall | drain s |
|---|---|---|---|---|---|
| device_bound_4096_matmul | 400 | 6.50 | 944.78 | 0.007 | 0.375 |
| device_bound_2048_matmul | 800 | 6.12 | 158.29 | 0.039 | 0.122 |
| host_bound_1tile_add | 20000 | 8.39 | 8.40 | 0.999 | 0.000 |

## pairformer block device floor (ttnn trace capture of shipped PairformerLayer.__call__)
`{"note": "trace capture of one shipped PairformerLayer.__call__, replayed back-to-back", "eager_synced_ms": 41.4546, "eager_host_cpu_ms": 4.0663, "replay_issue_cpu_us_per_block": 0.545, "replay_device_ms_per_block": 41.4012, "replay_all_ms": [41.4046, 41.2888, 41.4012, 41.3483, 41.4703]}`
