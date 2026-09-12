# Boltz-2 512 aa: every device op instance, replayed standalone

whglx (j10glx02) card 1, Wormhole, compute grid (x=8,y=9), ttnn 0.68.0, 20 reps x 5 bursts, program cache hot.

**These microseconds are not a baseline.** whglx was carrying ten other swarm rows while this ran
and UMD brings up all 32 chips on every device open, so the absolutes are inflated by an unknown,
varying amount. The column that matters is *share*: which instances are worth sweeping. Every
perf claim in `BEST-CONFIG-TABLE.md` is a paired interleaved ratio taken in one process.

Replayed total 55.84 s/fold over 323 instances (11 instances failed to build).

| # | instance | op | share | ms/fold | us/call | calls/fold | operands |
|---|---|---|---|---|---|---|---|
| 1 | `PairformerLayer#002` | matmul | 8.33 % | 4651.7 | 550.63 | 8448 | `1x16x512x128 , 128x512` |
| 2 | `PairformerLayer#001` | matmul | 7.16 % | 3998.9 | 473.35 | 8448 | `1x16x512x128 , 128x512` |
| 3 | `PairformerLayer#007` | matmul | 6.02 % | 3362.5 | 3184.22 | 1056 | `1x512x512x128 , 128x128` |
| 4 | `DiffusionStep#000` | matmul | 5.16 % | 2878.5 | 199.90 | 14400 | `1x512x768 , 768x1536` |
| 5 | `DiffusionStep#017` | matmul | 4.28 % | 2387.8 | 497.45 | 4800 | `1x16x512x512 , 1x16x512x64` |
| 6 | `PairformerLayer#004` | matmul | 4.16 % | 2320.7 | 274.70 | 8448 | `1x16x512x512 , 512x128` |
| 7 | `DiffusionStep#029` | matmul | 3.56 % | 1987.7 | 552.13 | 3600 | `1x224x32x128 , 128x256` |
| 8 | `DiffusionStep#035` | matmul | 2.39 % | 1331.9 | 554.94 | 2400 | `1x224x32x128 , 128x128 , 128` |
| 9 | `DiffusionStep#036` | matmul | 2.36 % | 1317.3 | 548.87 | 2400 | `1x224x32x128 , 128x128` |
| 10 | `DiffusionStep#043` | matmul | 2.25 % | 1258.4 | 1048.68 | 1200 | `1x224x128x128 , 128x256` |
| 11 | `PairformerLayer#017` | matmul | 2.21 % | 1233.1 | 2335.42 | 528 | `1x128x512x512 , 1x128x512x512` |
| 12 | `DiffusionStep#016` | softmax | 1.95 % | 1088.9 | 226.85 | 4800 | `1x16x512x512` |
| 13 | `MSALayer#015` | matmul | 1.89 % | 1052.6 | 8223.50 | 128 | `1024x32x512 , 1x512x512` |
| 14 | `PairformerLayer#005` | layernorm | 1.77 % | 988.5 | 936.05 | 1056 | `1x512x512x128 , 128 , 128` |
| 15 | `DiffusionStep#053` | matmul | 1.76 % | 985.0 | 820.84 | 1200 | `1x224x32x256 , 256x128` |
| 16 | `PairformerLayer#006` | binary | 1.60 % | 895.1 | 847.66 | 1056 | `1x512x512x128 , 1x512x512x128 , 1x512x512x128` |
| 17 | `PairformerLayer#019` | matmul | 1.53 % | 855.4 | 1620.02 | 528 | `512x512x128 , 128x4` |
| 18 | `PairformerLayer#014` | binary | 1.50 % | 834.8 | 1581.09 | 528 | `1x512x512x128 , 1x512x512x1 , 1x512x512x128` |
| 19 | `PairformerLayer#025` | transpose | 1.43 % | 800.2 | 1515.61 | 528 | `1x512x512x128` |
| 20 | `DiffusionStep#015` | binary | 1.33 % | 745.2 | 155.25 | 4800 | `1x16x512x512 , 1x16x512x512` |
| 21 | `DiffusionStep#013` | matmul | 1.29 % | 719.0 | 149.80 | 4800 | `1x16x512x64 , 1x16x64x512` |
| 22 | `DiffusionStep#040` | matmul | 1.28 % | 715.6 | 596.32 | 1200 | `1x16x128x448 , 448x1792` |
| 23 | `PairformerLayer#008` | binary | 1.21 % | 674.4 | 851.48 | 792 | `1x512x512x128 , 1x512x512x128 , 1x512x512x128` |
| 24 | `MSALayer#010` | matmul | 1.17 % | 654.3 | 2271.85 | 288 | `1024x512x64 , 64x32` |
| 25 | `DiffusionStep#042` | reshape | 1.05 % | 586.8 | 489.01 | 1200 | `1792x16x128` |
| 26 | `DiffusionStep#010` | matmul | 0.98 % | 544.7 | 113.49 | 4800 | `1x512x768 , 768x3072 , 3072` |
| 27 | `DiffusionStep#014` | binary | 0.97 % | 541.1 | 112.73 | 4800 | `1x16x512x512 , 1x16x512x512 , 1x16x512x512` |
| 28 | `PairformerLayer#003` | binary | 0.96 % | 536.1 | 63.46 | 8448 | `1x16x512x512 , 1x16x512x512 , 1x16x512x512` |
| 29 | `PairformerLayer#023` | binary | 0.90 % | 501.9 | 950.51 | 528 | `512x4x512x32 , 512x4x512x32 , 512x4x512x32` |
| 30 | `DiffusionStep#020` | reshape | 0.89 % | 498.6 | 103.87 | 4800 | `16x48x512` |
| 31 | `PairformerLayer#012` | slice | 0.83 % | 461.9 | 874.72 | 528 | `1x512x512x512` |
| 32 | `PairformerLayer#024` | binary | 0.80 % | 444.7 | 842.29 | 528 | `1x512x512x128 , 1x512x512x128 , 1x512x512x128` |
| 33 | `PairformerLayer#079` | matmul | 0.72 % | 403.1 | 1526.95 | 264 | `1x512x512x128 , 128x16` |
| 34 | `PairformerLayer#000` | layernorm | 0.72 % | 403.1 | 47.71 | 8448 | `1x16x512x128 , 128 , 128` |
| 35 | `DiffusionStep#041` | permute | 0.71 % | 395.4 | 329.47 | 1200 | `1x16x128x1792` |
| 36 | `PairformerLayer#016` | transpose | 0.71 % | 395.2 | 748.48 | 528 | `1x128x512x512` |
| 37 | `PairformerLayer#010` | slice | 0.69 % | 384.4 | 727.98 | 528 | `1x512x512x512` |
| 38 | `MSALayer#018` | matmul | 0.67 % | 375.3 | 2931.95 | 128 | `1024x512x32 , 32x64` |
| 39 | `PairformerLayer#013` | slice | 0.62 % | 347.7 | 658.47 | 528 | `1x512x512x512` |
| 40 | `PairformerLayer#011` | slice | 0.61 % | 343.2 | 650.00 | 528 | `1x512x512x512` |
| 41 | `DiffusionStep#004` | layernorm | 0.58 % | 323.7 | 33.72 | 9600 | `1x512x768 , 768` |
| 42 | `DiffusionStep#006` | matmul | 0.58 % | 323.0 | 33.65 | 9600 | `1x512x768 , 768x768` |
| 43 | `DiffusionStep#011` | nlp_create_qkv_heads | 0.58 % | 321.7 | 67.02 | 4800 | `1x1x512x3072` |
| 44 | `DiffusionStep#005` | matmul | 0.56 % | 314.7 | 32.78 | 9600 | `1x512x768 , 768x768 , 768` |
| 45 | `DiffusionStep#002` | matmul | 0.56 % | 314.5 | 32.09 | 9800 | `1x512x768 , 768x768` |
| 46 | `DiffusionStep#039` | permute | 0.54 % | 298.8 | 249.03 | 1200 | `1x448x16x128` |
| 47 | `DiffusionStep#003` | layernorm | 0.51 % | 283.7 | 29.55 | 9600 | `1x512x768` |
| 48 | `PairformerLayer#078` | layernorm | 0.49 % | 274.8 | 1040.84 | 264 | `1x512x512x128 , 128 , 128` |
| 49 | `MSALayer#145` | matmul | 0.45 % | 250.3 | 15642.23 | 16 | `512x512x1024 , 1024x128 , 128` |
| 50 | `PairformerLayer#034` | layernorm | 0.44 % | 246.7 | 934.51 | 264 | `512x512x128 , 128 , 128` |
| 51 | `PairformerLayer#038` | layernorm | 0.44 % | 244.5 | 926.00 | 264 | `512x512x128 , 128 , 128` |
| 52 | `MSALayer#007` | matmul | 0.44 % | 243.0 | 474.65 | 512 | `1x16x512x128 , 128x512` |
| 53 | `MSALayer#006` | matmul | 0.43 % | 242.2 | 473.10 | 512 | `1x16x512x128 , 128x512` |
| 54 | `MSALayer#001` | matmul | 0.42 % | 235.3 | 229.76 | 1024 | `1x16x512x64 , 64x256` |
| 55 | `DiffusionStep#026` | matmul | 0.41 % | 227.7 | 47.43 | 4800 | `1x512x768 , 768x768 , 768` |
| 56 | `DiffusionStep#047` | sdpa | 0.41 % | 226.9 | 189.11 | 1200 | `1x896x32x32 , 1x896x128x32 , 1x896x128x32 , 1x896x32x128` |
| 57 | `DiffusionStep#001` | binary | 0.40 % | 222.0 | 21.76 | 10200 | `1x512x768 , 1x512x768` |
| 58 | `DiffusionStep#027` | matmul | 0.39 % | 215.6 | 44.92 | 4800 | `1x512x1536 , 1536x768` |
| 59 | `MSALayer#002` | matmul | 0.37 % | 205.2 | 200.36 | 1024 | `1x16x512x64 , 64x256` |
| 60 | `MSALayer#024` | matmul | 0.36 % | 203.7 | 3183.26 | 64 | `1x512x512x128 , 128x128` |

Cumulative share: 
top 10 = 45.7 %, top 20 = 62.6 %, top 30 = 73.3 %, top 50 = 85.7 %.

## Errors

| instance | operands | error |
|---|---|---|
| `PairformerLayer#009` | `1x512x512` | RuntimeError: TT_FATAL @ /project/ttnn/cpp/ttnn/operations/data_movement/reshape_view/reshape_common.cpp:50: new_volume == old_volume
info:
Invalid ar |
| `PairformerLayer#018` | `1x512x512x128 , 1x512x512x128 , 1x512x512x128` | RuntimeError: TT_FATAL @ /project/tt_metal/impl/allocator/bank_manager.cpp:439: false
info:
Out of Memory: Not enough space to allocate 67108864 B L1  |
| `DiffusionStep#038` | `224x32x128` | RuntimeError: TT_FATAL @ /project/ttnn/cpp/ttnn/operations/data_movement/reshape_view/reshape_common.cpp:50: new_volume == old_volume
info:
Invalid ar |
| `DiffusionStep#044` | `1x224x32x128` | no builder |
| `DiffusionStep#045` | `224x1x128x128 , 224x1x128x256` | RuntimeError: TT_FATAL @ /project/ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads/nlp_create_qkv_heads.cpp:30: input_tensor_q.p |
| `DiffusionStep#064` | `1x7168x768 , 1x7168x512` | RuntimeError: TT_FATAL @ /project/ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation.cpp:126: a_shape[-1] == b_shape[-2]
info:
The width o |
| `DiffusionStep#066` | `1x256` | IndexError: list index out of range |
| `DiffusionStep#073` | `1x128x512 , 1x7168x512` | RuntimeError: TT_FATAL @ /project/ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation.cpp:126: a_shape[-1] == b_shape[-2]
info:
The width o |
| `MSALayer#044` | `1x512x512x128 , 1x512x512x128 , 1x512x512x128` | RuntimeError: TT_FATAL @ /project/tt_metal/impl/allocator/bank_manager.cpp:439: false
info:
Out of Memory: Not enough space to allocate 67108864 B L1  |
| `MSALayer#139` | `16384x1024 , 16384x1024` | RuntimeError: TT_FATAL @ /project/ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation.cpp:126: a_shape[-1] == b_shape[-2]
info:
The width o |
| `MSALayer#144` | `512x512x1024 , 512x512x1024` | operands 1074 MB over cap |
