# Boltz-2 512 aa: every named phase against both measured roofs

Card 1 of qb2 (p300c), ttnn 0.68.0, fold 23.841 s (device 23.465 s). Bytes recounted from this task's own captures with the corrected buffer-address rule (`real_traffic.py`, origin/wk/b2x-diffusion-layer-bytes); FLOP/call from `flops_bytes_512.json`; ms/call is the median over that unit's calls in the bracketed fold, which syncs on both sides of every bracket, so every rate here is a floor.

Roofs, measured on this part: streaming **429.9 GB/s**, dense bf16 HiFi4 **85.96 TFLOP/s**.

| phase | calls | ms/call | MB/call | GB/s | % stream roof | GFLOP/call | % compute roof | ops/call | ops/fold | kB/op | us/op | s/fold | s at 80 % roof | **deficit s** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pairformer block | 264 | 42.199 | 6650.7 | 157.6 | 36.7 % | 502.8 | 13.9 % | 428 | 112992 | 15539.1 | 98.6 | 11.141 | 5.105 | **6.035** |
| diffusion step (whole denoiser) | 200 | 40.263 | 6600.4 | 163.9 | 38.1 % | 322.22 | 9.3 % | 1816 | 363200 | 3634.6 | 22.2 | 8.053 | 3.838 | **4.214** |
| token DiT layer | 4800 | 1.078 | 198.9 | 184.5 | 42.9 % | 12.28 | 13.3 % | 60 | 288000 | 3315.6 | 18.0 | 5.175 | 2.776 | **2.398** |
| atom transformer layer | 1200 | 1.957 | 303.9 | 155.3 | 36.1 % | 4.58 | 2.7 % | 66 | 79200 | 4604.7 | 29.7 | 2.349 | 1.06 | **1.288** |
| MSA block | 16 | 122.781 | 20550.7 | 167.4 | 38.9 % | 595.17 | 5.6 % | 1104 | 17664 | 18614.8 | 111.2 | 1.965 | 0.956 | **1.008** |

## Every captured unit, same counter, ranked the same way

| unit | calls | ms/call | MB/call | published MB/call | GB/s | % roof | ops/call | kB/op | s/fold | deficit s |
|---|---|---|---|---|---|---|---|---|---|---|
| `PairformerLayer|1x512x384,1x512x512x128` | 264 | 42.199 | 6650.7 | | 157.6 | 36.7 % | 428 | 15539.1 | 11.141 | 6.035 |
| `DiffusionModule|` | 200 | 40.263 | 6600.4 | | 163.9 | 38.1 % | 1816 | 3634.6 | 8.053 | 4.214 |
| `Diffusion|1x7168x3,1` | 200 | 39.377 | 6599.9 | | 167.6 | 39.0 % | 1812 | 3642.3 | 7.875 | 4.037 |
| `DiffusionTransformer|1x512x768,1x512x768` | 200 | 26.359 | 4756.3 | | 180.4 | 42.0 % | 1440 | 3303.0 | 5.272 | 2.506 |
| `DiffusionTransformerLayer|1x512x768,1x512x768` | 4800 | 1.078 | 198.9 | | 184.5 | 42.9 % | 60 | 3315.6 | 5.175 | 2.398 |
| `TriangleMultiplication|1x512x512x128,1x512x512` | 528 | 10.196 | 2114.7 | | 207.4 | 48.2 % | 31 | 68215.6 | 5.383 | 2.137 |
| `Transition|1x512x512x128` | 280 | 7.993 | 474.2 | | 59.3 | 13.8 % | 258 | 1838.1 | 2.238 | 1.852 |
| `DiffusionTransformer|1x224x32x128,1x224x32x128` | 400 | 5.954 | 874.4 | | 146.9 | 34.2 % | 159 | 5499.4 | 2.382 | 1.365 |
| `TriangleAttention|1x512x512x128,1x1x1x512` | 528 | 5.770 | 1115.9 | | 193.4 | 45.0 % | 27 | 41329.9 | 3.047 | 1.333 |
| `DiffusionTransformerLayer|1x224x32x128,1x224x32x128` | 1200 | 1.957 | 303.9 | | 155.3 | 36.1 % | 66 | 4604.7 | 2.349 | 1.288 |
| `MSALayer|1x512x512x128,1x1024x512x64` | 16 | 122.781 | 20550.7 | | 167.4 | 38.9 % | 1104 | 18614.8 | 1.965 | 1.008 |
| `AttentionPairBias|1x224x32x128,224x4x32x128` | 1200 | 1.293 | 195.1 | | 150.9 | 35.1 % | 28 | 6969.1 | 1.552 | 0.871 |
| `AttentionPairBias|1x512x768,1x16x512x512` | 4800 | 0.516 | 121.6 | | 235.5 | 54.8 % | 26 | 4675.7 | 2.478 | 0.781 |
| `ConditionedTransitionBlock|1x512x768,1x512x768` | 4800 | 0.304 | 52.4 | | 172.6 | 40.2 % | 21 | 2497.4 | 1.458 | 0.726 |
| `AdaLN|1x512x768,1x512x768` | 9600 | 0.118 | 15.0 | | 127.4 | 29.6 % | 9 | 1671.2 | 1.133 | 0.713 |
| `PairformerLayer|1x512x512x128` | 16 | 39.499 | 6519.2 | | 165.0 | 38.4 % | 376 | 17338.3 | 0.632 | 0.329 |
| `OuterProductMean|1x1024x512x64,1024x1x1` | 16 | 41.216 | 7319.3 | | 177.6 | 41.3 % | 24 | 304972.8 | 0.659 | 0.319 |
| `ConditionedTransitionBlock|1x224x32x128,1x224x32x128` | 1200 | 0.411 | 77.5 | | 188.2 | 43.8 % | 23 | 3367.6 | 0.494 | 0.224 |
| `AttentionPairBias|1x512x384,1x512x512x128` | 264 | 1.064 | 151.4 | | 142.3 | 33.1 % | 32 | 4730.6 | 0.281 | 0.165 |
| `PairWeightedAveraging|1x1024x512x64,1x512x512x128` | 16 | 30.710 | 7181.9 | | 233.9 | 54.4 % | 187 | 38405.8 | 0.491 | 0.157 |
| `Transition|1x1024x512x64` | 16 | 9.310 | 606.2 | | 65.1 | 15.1 % | 514 | 1179.3 | 0.149 | 0.121 |
| `TriangleMultiplication|1x512x512x128` | 32 | 9.524 | 2080.6 | | 218.5 | 50.8 % | 29 | 71745.0 | 0.305 | 0.111 |
| `AdaLN|1x224x32x128,1x224x32x128` | 2400 | 0.091 | 18.4 | | 202.1 | 47.0 % | 11 | 1675.6 | 0.219 | 0.09 |
| `TriangleAttention|1x512x512x128` | 32 | 5.783 | 1111.7 | | 192.2 | 44.7 % | 26 | 42756.9 | 0.185 | 0.082 |
| `Transition|1x512x768` | 400 | 0.131 | 9.5 | | 72.9 | 17.0 % | 8 | 1191.9 | 0.052 | 0.041 |
| `Transition|1x512x384` | 264 | 0.116 | 4.8 | | 41.0 | 9.5 % | 8 | 596.0 | 0.031 | 0.027 |

## What the pair of roofs says

Every phase sits at 36-43 % of the streaming roof and 3-14 % of the compute roof. Nothing is bandwidth-bound and nothing is compute-bound, so the diagnostic's third row is the one that fires: latency or dispatch. Of 21.159 s of phase time, 7.920 s is explained by moving those bytes at the streaming roof and 13.239 s is not. The whole fold's 206.71 TFLOP is 2.40 s at the compute roof.

**Delete every byte of every named phase and the fold is still 15.921 s = 1.497x.**

## Where the rest goes: the host

One fold issues **487,202 ttnn calls**. Measured on the calling thread with `time.thread_time()`, which counts only the CPU that thread burns: **21.846 s of main-thread CPU in a 26.037 s fold, 83.9 %**. 19.747 s of the wall is spent inside ttnn entry points, 40.5 us per call on average.

The host is issuing ops for essentially the whole fold while the device runs at 37 % of one roof and 10 % of the other. Op count, not bytes, is what that buys back. (Probe ran co-tenanted at loadavg 6.53, 6.44, 5.70, so its wall is ~9 % above the benchlocked 23.841 s; the ratio is the result, not the wall.)


## Summary

```
{
 "fold_s": 23.841,
 "device_s": 23.4651,
 "stream_roof_GBps": 429.9,
 "compute_roof_TFLOPs": 85.96,
 "top_level_TB_per_fold": 3.4047,
 "top_level_s_per_fold": 21.159,
 "top_level_share_of_fold_pct": 88.8,
 "fold_achieved_GBps_on_device_s": 145.1,
 "fold_pct_stream_roof": 33.8,
 "fold_TFLOP": 206.705568776192,
 "fold_achieved_TFLOPs": 8.81,
 "fold_pct_compute_roof": 10.2,
 "s_explained_by_bytes_at_roof": 7.92,
 "s_of_phase_time_not_explained_by_bytes": 13.239,
 "s_explained_by_flops_at_roof": 2.405,
 "fold_s_if_every_phase_byte_deleted": 15.921,
 "ceiling_x_if_every_phase_byte_deleted": 1.497,
 "counter": "real_traffic.py from origin/wk/b2x-diffusion-layer-bytes",
 "dispatch": {
  "probe_fold_s": 26.037,
  "main_thread_cpu_s": 21.846,
  "main_thread_cpu_pct_of_wall": 83.9,
  "n_ttnn_calls_per_fold": 487202,
  "s_inside_ttnn_calls": 19.747,
  "mean_us_per_ttnn_call": 40.53,
  "loadavg_during_probe": [
   "6.53",
   "6.44",
   "5.70"
  ]
 }
}
```
