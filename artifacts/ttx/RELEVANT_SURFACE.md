# What a Boltz-2 fold actually reaches

Most of the 2.9 M-line diff is code our fold never executes. This is the filter that gets applied
before anyone reads: which upstream files a fold touches, how it was derived, and how much of the
fold's time sits behind each one.

## How this was derived

Three independent sources, none of them a release note.

**1. The op census, from our own source.** Every `ttnn.*` symbol in the eleven modules on the
Boltz-2 device path (`tenstorrent.py` 12,601 lines, `boltz2.py` 6,228, plus `triatt_sdpa`,
`triatt_qkv`, `trimul_tail`, `reblock_permute`, `mm_generic`, `mm_dualnoc`, `sdpa_generic`,
`softmax_generic`, `token_axis` — 22,829 lines in total). 127 distinct symbols. The counts below
are call sites, not dynamic calls.

**2. The upstream paths our tree names outright**, from grepping `tt_bio/`, `perf/` and `docs/` for
upstream path strings. Ranked: `minimal_matmul_device_operation.cpp` 289 mentions,
`tt_metal/impl/program/program.cpp` 219, `tt_metal/impl/allocator/bank_manager.cpp` 106,
`matmul_device_operation.cpp` 53, `scatter.cpp` 6, `matmul_multicore_reuse_mcast_2d_program_factory.cpp` 5,
`softmax_device_operation.cpp` 4, `layernorm_device_operation.cpp` 4, `sdpa_device_operation.cpp` 1.
These are the files we already read closely enough to cite, so they are the files whose change is
most likely to matter.

**3. The executed op mix, from a committed on-device trace.** `perf/b2z2_byte_floor/CENSUS.md`
traces one 512 aa `PairformerLayer` operand by operand on both parts: **272 ops per block, of which
113 matmuls, 41 layer norms and 16 `generic_op`s, and Wormhole and Blackhole agree op for op.**
A diffusion step is 1,066 ops (`perf/k10_diffusion/src/step_census_qb2c0.json`).

## Where the fold's time is

512 aa on Blackhole p300c, `perf/b2x_op_cost/device_floor_512_qb2c0.json` (card 0, 11x10 grid,
commit `49025ea2`): fold 23.915 s, of which `PairformerLayer` 41.4152 ms x 280 calls = **11.5963 s**,
`Diffusion` 32.5179 ms x 200 = **6.5036 s**, `MSALayer` 120.2894 ms x 16 = **1.9246 s**. The
benchlocked cell of record is 17.340 s.

From `perf/roof_budget/ROOF_BUDGET.md` on the same part: **219.49 TFLOP executed, 2.9449 TB moved,
floor 6.934 s set by bandwidth** at a measured 424.7 GB/s, against a dense bf16 HiFi4 roof of
104.93 TFLOP/s and a machine balance of 247.1 FLOP/byte. Every unit but the pair `Transition` is
under machine balance, so **traffic binds, not arithmetic.** A package whose only plausible unlock
is arithmetic is worth less here than its line count suggests.

## Op -> upstream path -> package

| our call | sites | upstream path | pkg |
|---|---|---|---|
| `linear` / `matmul` | 87 / 20 | `ttnn/cpp/ttnn/operations/matmul/` | P1 |
| `experimental.minimal_matmul` | 11 | `.../operations/experimental/minimal_matmul/` | P1 |
| `layer_norm` | 51 | `.../operations/normalization/layernorm/` | P9 |
| `softmax`, `softmax_in_place` | 27 | `.../operations/normalization/softmax/` | P9 |
| `transformer.scaled_dot_product_attention` | 10 | `.../operations/transformer/sdpa/` | P2 |
| `experimental.nlp_create_qkv_heads`, `nlp_concat_heads` | 9 | `.../operations/experimental/transformer/` | P2 |
| `add`, `add_`, `multiply`, `multiply_`, `subtract`, `divide`, `where`, `max` | 145 | `.../operations/eltwise/binary`, `binary_ng`, `ternary` | P10 |
| `relu`, `silu`, `exp`, `cos`, `reciprocal`, `sigmoid` via `UnaryOpType` | 24 | `.../operations/eltwise/unary`, `unary_ng` | P10 |
| `typecast`, `clone`, `to_layout` | 47 | `.../operations/copy/`, `.../data_movement/` | P5 |
| `permute`, `transpose`, `reshape`, `concat`, `slice`, `pad`, `chunk`, `squeeze`, `unsqueeze`, `repeat`, `scatter`, `embedding` | 210 | `.../operations/data_movement/`, `.../operations/embedding/` | P5 |
| `sum` | 2 | `.../operations/reduction/generic/` | P10 |
| `generic_op` + `ProgramDescriptor`/`KernelDescriptor`/`CBDescriptor`/`TensorAccessorArgs`/`SemaphoreDescriptor` | 12 + 60 | `.../operations/generic/` | P6 |
| `begin_trace_capture`, `execute_trace`, `release_trace` | 3 | `tt_metal/impl/dispatch/`, `tt_metal/distributed/` | P6 |
| `allocate_tensor_on_device`, `reallocate`, `deallocate`, `copy_host_to_device_tensor` | 340 | `ttnn/core/tensor/`, `tt_metal/impl/allocator/` | P4/P14 |
| `create_sharded_memory_config`, `to_memory_config`, `MemoryConfig` | 29 | `.../data_movement/sharded/`, `.../operations/core/` | P4 |
| `DeviceComputeKernelConfig`, `MathFidelity` | 44 | `.../operations/core/compute_kernel/` | P13 |

`ttnn/cpp/ttnn/operations/generic/` changed by only **214 lines over 9 files** across the whole
range, and `generic_op` still exists at the tip with the same file set. Our nine hand-written kernel
directories keep their entry point.

## The kernels we transcribed from upstream, and what they cost to re-derive

`tt_bio/kernels/mm_split/patch_mm_split.py` and `tt_bio/kernels/trimul_tail/patch_trimul_tail.py`
regenerate our kernels *from the installed wheel's own* `minimal_matmul` kernels, asserting exact
anchor counts so a wheel bump raises rather than binding a stale kernel. Those four upstream files
moved by **886 added / 299 removed lines** in the range:

| upstream file | +/- |
|---|---|
| `.../minimal_matmul/device/kernels/compute.cpp` | +235 / -88 |
| `.../minimal_matmul/device/kernels/dm_in0_sender.cpp` | +253 / -75 |
| `.../minimal_matmul/device/kernels/dm_in1_sender_out.cpp` | +220 / -56 |
| `.../minimal_matmul/device/kernels/matmul_dataflow_common.hpp` | +178 / -80 |

Three of our nine kernel directories (`mm_split`, `trimul_tail`, `triatt`) are generated from or
share `matmul_dataflow_common.hpp`, so that is the porting bill for our matmul kernels, and it is
paid before any P1 unlock can be measured. Four new `fabric_bound_*` kernels appeared alongside
them; they are for fabric-connected matmul and are not on our path.

## A verified porting-bill item outside Boltz-2

`tt_bio/kernels/rfd3_softmax/` includes three headers that **do not exist at the tip**:
`experimental/noc.h`, `experimental/circular_buffer.h` and `experimental/tensor.h`. The first two
were renamed to `tt_metal/hw/inc/api/dataflow/noc.h` and `.../api/dataflow/circular_buffer.h`, so
they cost three `#include` lines. `experimental/tensor.h` has no obvious successor and P3 owes an
answer. This affects RFdiffusion3, not the Boltz-2 fold, but it is a shipped model.

## The compute-API headers our kernels include, and their churn

Every `tt_metal/hw/inc/api/` header our nine kernel directories include, with lines added/removed in
the range. This is the only part of P7 (126,338 lines) worth reading:

`compute_kernel_api.h` +453/-270, `eltwise_binary.h` +487/-117, `typecast.h` +414/-5,
`bcast.h` +285/-94, `eltwise_binary_sfpu.h` +236/-22, `tilize.h` +235/-90,
`tile_move_copy.h` +212/-51, `reduce_custom.h` +206/-52, `reduce.h` +102/-44,
`sfpu_split_includes.h` +95/-59, `matmul.h` +86/-239, `pack.h` +87/-71,
`transpose_wh.h` +85/-78, `binary_max_min.h` +116/-11, `dataflow_api.h` +76/-54,
`eltwise_unary.h` +56/-30, `softplus.h` +33/-4, `exp.h` +25/-48, `untilize.h` +23/-9,
`fill.h` +26/-5, `recip.h` +15/-10, `negative.h` +10/-4, `sfpu_int_sum.h` +6/-5,
`binop_with_scalar.h` +82/-9, `matmul_custom.h` +10/-20, `sdpa_sub_custom.h` +18/-9,
`common.h` +1/-1, `softmax.h` +1/-1, `assert.h` +45/-61.

`api/compute/matmul.h` is the one that lost more than it gained. There is also a whole new
`tt_metal/hw/inc/api/compute/experimental/2_0/` generation of the compute API (bcast, matmul, pack,
reduce, `pack_untilize`, `reconfig_data_format`, `llk_descriptor`) that did not exist at 0.68.0.
P7 owes what "2_0" is and whether it is reachable from a `generic_op` kernel.

## Dead levers, and which package owns the gate that killed each

| lever | the gate | pkg |
|---|---|---|
| pair-tensor L1 residency | `ttnn.matmul` refuses a sharded in0 unless `fuse_batch`. At 0.68.0 that is three separate `TT_FATAL`s in `matmul_device_operation.cpp` (lines 493, 588, 776). See the worked example in `READING_CONTRACT.md`: two of the three are gone at the tip and one survives, and which one our call hits is exactly what P1 owes | P1 |
| `minimal_matmul` daisy chain | `K_block == full K` leaves nothing to pipeline; `K_block=2` measured 1.159x | P1 |
| per-op launch floor, 20.6 µs fixed + 8.3 µs/block | no way to submit many programs as one | P6, P12 |
| CB depth / prefetch | the producer emits one block, so depth buys nothing | P3 |
| DRAM result write, 3.358 ns/tile-MAC vs 1.177 to L1 (2.85x) | no L1-resident epilogue | P1, P3, P4 |
| bfp8 pair track | the block format flipped `reblock_permute.eligible_back` from 24/24 served to 24/24 declined — it lost on eligibility, not physics | P11, P5 |
| persistent-mask SDPA | stock SDPA could not take a per-head mask, so `triatt_sdpa.py` exists | P2 |
| E6 `reblock_permute_gated` | required `mask_u is None`; every Boltz-2 pairformer site passes a pair mask | P5 |
| Transition row block | ours, not upstream: `_l1_rows_at` and the `SMALL_GRID_TRANSITION_ELEMS` raise are fenced behind `_IS_SMALL_GRID` | none — do not send an agent |

## What no agent should be sent to read

`models/`, `tests/`, `tt-train/`, `.github/`, quasar, CCL and fabric, deepseek, moreh, conv, pool,
sliding_window, FFT, KDA, and the scaleout cabling tooling. That is 75.1 % of the diff by line.
The full list with line counts is in `DIFF_PACKAGES.md`.
