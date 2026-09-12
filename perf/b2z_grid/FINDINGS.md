# The Boltz-2 pairformer block already runs on 97 % of the grid. Grid under-utilization is not the deficit.

`ws:b2z-grid-utilization`, a child of the RADICAL 2x campaign. **ARCH: BH** — pc card 0, p150a,
13x10 = 130 Tensix cores. Every number below is from the real dispatched programs, read out of
tt-metal's own device profiler, not from a program config.

## The bet, and how it died

The campaign's open quantity is a **2.34x** miss between the pairformer block's measured 41.4 ms and
the 17.7 ms its bytes and op count predict. The hypothesis was that ops run on a fraction of the
grid, so they see a fraction of the 445 GB/s that roof was calibrated on. For that to be the whole
story, the duration-weighted mean core utilization of the block would have to be `1/2.34 = 42.7 %`.

**Measured: 97.28 %.** Prediction was 55-70 %; the stated falsifier was ">70 % refutes". Refuted.

| | |
|---|---|
| ops in one `PairformerLayer` | **274** |
| unit device time | **39.357 ms** (x280 calls = 11.020 s/fold) |
| duration-weighted mean core utilization | **97.28 %** |
| UNDER-GRID-OPS (< half the grid) | **7 of 274**, 4.5 % of unit device time |
| CORE-SECONDS-LOST (idle core-fraction-seconds) | **0.2999 s/fold** |
| of the 6.6 s/fold deficit, explained by narrow grids | **0.30 s, 4.5 %** |

Duration-weighted, by how much of the grid an op takes:

| cores used | share of unit device time | ops |
|---|---|---|
| 100 % of grid | **78.79 %** | 158 |
| 90-100 % | 16.51 % | 103 |
| 75-90 % | 0.00 % | 0 |
| 50-75 % | 0.19 % | 6 |
| **< 50 %** | **4.50 %** | **7** |

## Per op class

| op class | n | ms/unit | s/fold | mean util | idle s/fold |
|---|---|---|---|---|---|
| GenericOpDeviceOperation | 14 | 11.672 | 3.268 | 100.00 % | 0.0000 |
| MatmulDeviceOperation | 113 | 8.303 | 2.325 | 87.58 % | **0.2888** |
| SDPAOperation | 2 | 6.531 | 1.829 | 100.00 % | 0.0000 |
| BinaryNgDeviceOperation | 54 | 5.494 | 1.538 | 100.00 % | 0.0000 |
| LayerNormDeviceOperation | 41 | 3.783 | 1.059 | 99.46 % | 0.0057 |
| TransposeDeviceOperation | 8 | 1.866 | 0.522 | 100.00 % | 0.0000 |
| SliceDeviceOperation | 33 | 0.580 | 0.162 | 100.00 % | 0.0000 |
| ReshapeViewDeviceOperation | 3 | 0.443 | 0.124 | 100.00 % | 0.0000 |
| ConcatDeviceOperation | 1 | 0.328 | 0.092 | 100.00 % | 0.0000 |
| PermuteDeviceOperation | 3 | 0.252 | 0.070 | 100.00 % | 0.0000 |
| SoftmaxDeviceOperation | 1 | 0.084 | 0.024 | 100.00 % | 0.0000 |
| NlpCreateHeadsDeviceOperation | 1 | 0.022 | 0.006 | 12.31 % | 0.0054 |

All of the idle time is in matmul, and 78 % of *that* is two ops.

## Every op below half the grid, in full

| op | cores | us | in0 shape | in0 memory |
|---|---|---|---|---|
| Matmul | **64/130** | 823.33 | 1x128x512x512 bf16 | DRAM interleaved |
| Matmul | **64/130** | 822.25 | 1x128x512x512 bf16 | DRAM interleaved |
| Matmul | 32/130 | 44.87 | 1x16x512x32 bf16 | DRAM interleaved |
| Matmul | 32/130 | 36.38 | 1x16x512x512 bf16 | DRAM interleaved |
| NlpCreateHeads | 16/130 | 22.05 | 1x1x512x1536 bf16 | DRAM interleaved |
| LayerNorm | 16/130 | 11.86 | 1x1x512x384 bf16 | DRAM interleaved |
| LayerNorm | 16/130 | 11.26 | 1x1x512x384 bf16 | DRAM interleaved |

The two 823 us matmuls are the whole lever. They are batched `1x128x512x512`: 128 independent
512x512 products, and ttnn's batched matmul parallelises the **output tile grid of one batch row**,
not the batch axis, so 16x16 output tiles land on an 8x8 = 64-core subgrid and the other 66 cores
idle for 1.65 ms per block. Perfect repair is 0.234 s/fold, **1.0 % of the fold**. The remaining
five are microseconds; they are narrow for the ordinary reason (512 rows / 32 = 16 tiles, one tile
row per core, 16 cores) and are not worth a config.

## What this means for the campaign

`CORE-SECONDS-LOST` is 0.30 s of a 6.6 s deficit. Even a *perfect* fix — every op on all 130 cores
with linear speedup — returns 1.3 % of the fold. **This campaign is not won by memory configs and
shard specs.** The deficit is inside the kernels or between them, which is
`b2z-kernel-cycle-census`'s GAP-FRACTION question, not this one.

Two things the census hands the swarm for free:

* `GenericOpDeviceOperation` is the single largest class in the block at **11.67 ms/unit,
  3.27 s/fold, 29.7 % of the block** — the fused trimul (E6). It already runs on all 130 cores, so
  whatever it is spending its time on, it is not waiting for cores.
* The block's 274 ops at 39.357 ms reproduce qb2's 41.415 ms on a different Blackhole part with a
  different grid (130 vs 110 cores) and a different tt-metal (0.72.0-dev vs 0.68.0), with the stock
  SDPA substituted for the fused kernel. The unit is stable across parts; the deficit travels
  with it.

## Instrument, and its one substitution

The shipped wheel `ttnn==0.68.0` is not a Tracy build (`TT_METAL_DEVICE_PROFILER=1` throws), so the
census runs against the Tracy-enabled source build at `/home/moritz/tt-metal`,
`v0.72.0-dev20260524-13-g7a6f782e3f8`. tt-bio's custom fused SDPA kernel does not compile against
0.72's LLK API (`DataCopyType`, `PackMode` and the tensor-accessor args all moved), so the census
ran with `TT_BIO_TRIATT_PERSISTENT_MASK=0` and the block's two attention calls served on the stock
`SDPAOperation` instead. Both stock calls read **100 % of the grid**, and the fused kernel is
explicitly one-q-chunk-per-core by construction, so neither is a candidate for the missing cores.

`perf/b2z_grid/census.py` grabs one settled call of a class out of a real fold, aborts the fold
immediately so the profiler only sees a few thousand programs (the device marker buffers overflow
at ~1000 undrained), then replays the grabbed call `--repeats` times warm. `perf/b2z_grid/rank.py`
recovers the per-unit op period from the tail of the CSV by finding the smallest repeating OP CODE
sequence, so no marker bookkeeping is needed.

One forward-compat fix was needed to get that far and is on the branch:
`binary_max_tile`'s vector-mode parameter became a scoped enum in 0.72, so
`tt_bio/kernels/triatt_sdpa/compute/compute_common.hpp` now passes `VectorMode::C` instead of
`static_cast<int>(VectorMode::C)`. Identical value under 0.68's unscoped enum.

## The diffusion step is not censused, and why

Two attempts, both on pc card 0. The census itself ran fine — `Diffusion` was grabbed on call 3 and
replayed warm at **53.187 / 53.257 / 53.759 ms** per step under the profiler (qb2 measures 32.518 ms
bare; the gap is profiler overhead plus a different part, not a finding). What fails is tracy's
**host post-processing**: reaching a diffusion step needs the whole trunk first, ~25 000 dispatched
programs even at `--recycles 1`, and the post-process is OOM-killed on pc's 30 GB host
(`--op-support-count 150000` aborted, `40000` was `Killed`).

So the 6.5 s/fold sampler is untouched by this pass. Note that it shares the
`DiffusionTransformerLayer` op mix with the trunk's attention, which censuses at 100 % of the grid,
so the prior is that it looks like the pairformer block — but that is a prior, not a measurement,
and it is stated as one. Whoever picks this up should drain the profiler mid-run
(`ttnn.ReadDeviceProfiler` + `ttnn.profiler.get_all_programs_perf_data()`, which exposes
`core_count`/`num_available_cores` directly and never builds the giant host log) rather than raise
the tracy budget again.
