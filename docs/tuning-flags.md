# Tuning flags

tt-bio ships its device optimizations on by default. This page says what each one changes and what it
was measured against, so you can decide whether to turn one off. The flags are environment
variables. A boolean flag reads `1`, `true`, `yes` or `on` as on and `0`, `false`, `no` or `off` as
off; unset or empty means the default, and any other value raises instead of guessing.

Reference numbers are Boltz-2 on one Blackhole processor of a p300c (Tenstorrent QuietBox, the host
called qb2 below), 512 aa, 200 sampling steps, 3 recycles, `perf/size512/fixtures/cdk2x2_512.yaml`.
Fold ratios are paired: both arms are interleaved inside one process, so a ratio is never read
across two sessions.

"Identical" in the table below means the flag writes the same structure byte for byte on the shapes
it was measured on. "Moves" means the structure changes; the flag's section gives the size of the
move against the accuracy bar and the seed-to-seed spread.

| flag | default | scope | output with the flag on |
|---|---|---|---|
| [`BOLTZ2_TOKEN_DIT_SDPA`](#boltz2_token_dit_sdpa) | on | Boltz-2 | moves, inside the 298-residue bar |
| [`TT_BIO_AF2_G_BIAS_IN_MATMUL`](#bindcraft-2-round-kernels) | on in a BindCraft 2 round | BindCraft 2, Blackhole | gradients only |
| [`TT_BIO_AF2_OPM_ROWS_IN_K`](#bindcraft-2-round-kernels) | on in a BindCraft 2 round | BindCraft 2 | moves the forward, closer to float64 |
| [`TT_BIO_ATOM_AXIS_BUCKET`](#tt_bio_atom_axis_bucket) | on | | identical at 298 residues, not guaranteed at 512 |
| [`TT_BIO_ATOM_SHIFT_GATHER`](#tt_bio_atom_shift_gather) | on | | identical |
| [`TT_BIO_ATOM_KV_WINDOW`](#tt_bio_atom_tile_heads-tt_bio_atom_kv_window) | on | Protenix-v2, OpenDDE, PXDesign | identical |
| [`TT_BIO_ATOM_SDPA32`](#tt_bio_atom_sdpa32) | on | fp32 atom attention (normal mode) | moves, closer to float64 |
| [`TT_BIO_ATOM_SUPERSET_WINDOW`](#tt_bio_atom_superset_window) | on | Protenix-v2, OpenDDE, PXDesign | moves, inside the bar |
| [`TT_BIO_ATOM_TILE_HEADS`](#tt_bio_atom_tile_heads-tt_bio_atom_kv_window) | on | Protenix-v2, OpenDDE, PXDesign | identical |
| [`TT_BIO_BH_DRAM_READ_SPLIT`](#tt_bio_bh_dram_read_split) | on | Blackhole | identical |
| [`TT_BIO_BH_ETH_DISPATCH`](#tt_bio_bh_eth_dispatch) | on where the installed ttnn supports it | Blackhole | identical |
| [`TT_BIO_DEVICE_CONDITIONING`](#tt_bio_device_conditioning) | on | Boltz-2 | moves, closer to the experimental structure |
| [`TT_BIO_DEVICE_CONFIDENCE`, `TT_BIO_DEVICE_CONF_HEADS`](#tt_bio_device_confidence-tt_bio_device_conf_heads) | on | Boltz-2 | coordinates identical, confidence scores move |
| [`TT_BIO_DEVICE_TILIZE`](#tt_bio_device_tilize) | on | Protenix-v2, OpenDDE, PXDesign | identical |
| [`TT_BIO_DEVICE_ZINIT`](#tt_bio_device_zinit) | on | Boltz-2 | moves, flat against the experimental structure |
| [`TT_BIO_DIT_COND_HOIST`](#tt_bio_dit_cond_hoist) | on | Boltz-2, RF3 token DiT | moves, inside the 298-residue bar |
| [`TT_BIO_DIT_SHARED_COND`](#tt_bio_dit_shared_cond) | on | Boltz-2, BoltzGen | identical |
| [`TT_BIO_FANIN_CAST_FUSED`](#bindcraft-2-round-kernels) | on in a BindCraft 2 round | BindCraft 2, Blackhole | gradients only |
| [`TT_BIO_FUSE_BIAS_STACKS`](#tt_bio_fuse_bias_stacks) | on | Boltz-2 | moves, inside the 298-residue bar |
| [`TT_BIO_FUSE_MASK_ADD`](#tt_bio_fuse_mask_add) | on | | identical |
| [`TT_BIO_FUSE_NORM_RESIDUAL`](#tt_bio_fuse_norm_residual) | on | | identical |
| [`TT_BIO_FUSE_SCALE_ADD`](#tt_bio_fuse_scale_add) | on | fp32 operands | identical |
| [`TT_BIO_GATED_BW_FUSED`](#bindcraft-2-round-kernels) | on in a BindCraft 2 round | BindCraft 2, Blackhole | gradients only |
| [`TT_BIO_GATED_GRAD_PACKED`](#bindcraft-2-round-kernels) | on in a BindCraft 2 round | BindCraft 2, Blackhole | gradients only |
| [`TT_BIO_GATE_BW_FUSED`](#bindcraft-2-round-kernels) | on in a BindCraft 2 round | BindCraft 2, Blackhole | gradients only |
| [`TT_BIO_GATE_GRANULARITY`](#tt_bio_gate_granularity) | 2 | | identical at every value |
| [`TT_BIO_HOST_LANE`](#tt_bio_host_lane) | on | Protenix-v2 | identical |
| [`TT_BIO_HOST_LEVERS`](#tt_bio_host_levers) | on | Boltz-2 | switches two other flags together |
| [`TT_BIO_LEAD_SUM_FUSED`](#bindcraft-2-round-kernels) | on in a BindCraft 2 round | BindCraft 2, Blackhole | gradients only |
| [`TT_BIO_LEVERS`](#tt_bio_levers) | each model's graded set | Protenix-v2, Boltz-2 | moves, inside the seed-to-seed spread |
| [`TT_BIO_LNBW_FUSED`](#tt_bio_lnbw_fused) | on in a BindCraft 2 round, off elsewhere | BindCraft 2, Blackhole | gradients only |
| [`TT_BIO_MM_LAYOUT`](#tt_bio_mm_layout) | off | training | moves |
| [`TT_BIO_MSA_LADDER`](#tt_bio_msa_ladder) | on | Boltz-2, BoltzGen | moves, closer to the experimental structure |
| [`TT_BIO_NOGRAD_INFERENCE`](#bindcraft-2-round-kernels) | on in a BindCraft 2 round | BindCraft 2, Blackhole | gradients only |
| [`TT_BIO_OPM_JOIN_PARTS`, `TT_BIO_OPM_PROJ_BATCH`](#msa-module-flags) | on | MSA models; graded on Protenix-v2 | join moves, inside the bar; projection batch identical |
| [`TT_BIO_OPM_LEGACY_LAYOUT`](#tt_bio_opm_legacy_layout) | off | | moves, inside the seed spread |
| [`TT_BIO_PAIR_FFN_L1_FC1`](#tt_bio_pair_ffn_l1_fc1) | on | ESMFold2 | identical |
| [`TT_BIO_PAIR_INPLACE`, `TT_BIO_TRIMUL_INPROJ_ROWBLOCK_NORM`](#tt_bio_pair_inplace-tt_bio_trimul_inproj_rowblock_norm) | on | large pair tensors | identical |
| [`TT_BIO_PAIR_MM`](#bindcraft-2-round-kernels) | on in a BindCraft 2 round | BindCraft 2, Blackhole | gradients only |
| [`TT_BIO_PAIR_TRANSPOSE_FUSED`](#tt_bio_pair_transpose_fused) | on | pair tensors, forward and backward | identical |
| [`TT_BIO_PWA_FUSED_HEADS`, `TT_BIO_PWA_UNPADDED`](#msa-module-flags) | on | MSA models; graded on Protenix-v2 | moves, inside the bar |
| [`TT_BIO_PWA_BATCH_HEAD_WEIGHTS`](#tt_bio_pwa_batch_head_weights) | on | | identical |
| [`TT_BIO_PWA_FULL_HEADS_FUSED`](#tt_bio_pwa_full_heads_fused) | on | Boltz-2 (32-wide heads) | identical |
| [`TT_BIO_REBLOCK_PERMUTE_GATED`](#tt_bio_reblock_permute_gated) | on | | identical |
| [`TT_BIO_RESIDUAL_L1`](#tt_bio_residual_l1) | on | | identical |
| [`TT_BIO_SDPA_ADD_GRANULARITY`](#tt_bio_sdpa_add_granularity) | auto | | identical at every value |
| [`TT_BIO_SDPA_BAND_DIV_K`](#tt_bio_sdpa_band_div_k) | on | Blackhole | moves, inside the bar |
| [`TT_BIO_SDPA_FUSED_LARGE_S`](#tt_bio_sdpa_fused_large_s) | on | above 1024 tokens | moves, inside the seed spread |
| [`TT_BIO_SDPA_FUSED_PADDED`](#tt_bio_sdpa_fused_padded) | on | lengths with no dividing chunk (736, 928, 992 ...) | moves |
| [`TT_BIO_SDPA_GRID_Q_CHUNK`](#tt_bio_sdpa_grid_q_chunk) | on | | identical |
| [`TT_BIO_SDPA_WIDE_K`](#tt_bio_sdpa_wide_k) | on | twenty padded lengths | moves, inside the seed spread |
| [`TT_BIO_SOFTMAX_BW_FP32`](#tt_bio_softmax_bw_fp32) | on | training | gradients only |
| [`TT_BIO_TOKEN_BUCKET`](#tt_bio_token_bucket) | on | | switches every model's token bucket |
| [`TT_BIO_TRANSITION_L1_ROWS`](#tt_bio_transition_l1_rows) | on | Blackhole | identical on the measured shapes |
| [`TT_BIO_TRIATT_B8`](#tt_bio_triatt_b8) | off | | moves, and depends on the core grid |
| [`TT_BIO_TRIATT_BW_EXP_21F`](#bindcraft-2-round-kernels) | on in a BindCraft 2 round | BindCraft 2, Blackhole | gradients only |
| [`TT_BIO_TRIATT_BW_FUSED`](#tt_bio_triatt_bw_fused) | on in a BindCraft 2 round, off elsewhere | BindCraft 2 | gradients only |
| [`TT_BIO_TRIATT_BW_QKV_PACKED`](#bindcraft-2-round-kernels) | on in a BindCraft 2 round | BindCraft 2, Blackhole | gradients only |
| [`TT_BIO_TRIATT_DIVIDING_K`](#tt_bio_triatt_dividing_k) | on | OpenFold3 at 832 tokens | moves, inside the bar |
| [`TT_BIO_TRIATT_FUSED_QKVG`](#tt_bio_triatt_fused_qkvg) | on | | identical |
| [`TT_BIO_TRIATT_FUSED_QKVGB`](#tt_bio_triatt_fused_qkvgb) | on | | identical |
| [`TT_BIO_TRIATT_QK_MASK_PRELOAD`](#tt_bio_triatt_qk_mask_preload) | on | fused triangle attention | moves, inside the bar |
| [`TT_BIO_TRIMUL_FUSED_GOUT`](#tt_bio_trimul_fused_gout) | on | | identical |
| [`TT_BIO_TRIATT_GATE_EPILOGUE`](#tt_bio_triatt_gate_epilogue) | off | | identical |
| [`TT_BIO_TRIATT_HIFI_PAD_UP`](#tt_bio_triatt_hifi_pad_up) | on | BindCraft 2, and OpenFold3 at 544 and 608 tokens | moves, far inside the bar |
| [`TT_BIO_TRIATT_NARROW_Q_FALLBACK`](#tt_bio_triatt_narrow_q_fallback) | on | | identical on RoseTTAFold3, moves on OpenBind |
| [`TT_BIO_TRIATT_SDPA_HIFI_AB`](#tt_bio_triatt_sdpa_hifi_ab) | on at `openfold3.trunk` | OpenFold3 | moves, inside the seed spread |
| [`TT_BIO_TRIMUL_GP_BANK_SPLIT`](#tt_bio_trimul_gp_bank_split) | on | | identical |
| [`TT_BIO_TRIMUL_MASK_AFTER_MOVE`](#tt_bio_trimul_mask_after_move) | on | | identical |
| [`TT_BIO_TRIMUL_MASK_L1`](#tt_bio_trimul_mask_l1) | on | | identical |
| [`TT_BIO_TRIMUL_MM_TRANSPOSE`](#tt_bio_trimul_mm_transpose) | on | | identical |
| [`TT_BIO_TRIMUL_TAIL_F1`](#tt_bio_trimul_tail_f1) | on | | identical |
| [`TT_BIO_TRIMUL_TAIL_F1_L1_OUT`](#tt_bio_trimul_tail_f1_l1_out) | on | | identical |
| [`TT_BIO_UNFUSED_SILU`](#tt_bio_unfused_silu) | off | | moves, and costs Protenix-v2 accuracy |
| [`TT_PROTENIX_CONF_DEVICE`](#tt_protenix_conf_device) | on | Protenix-v2, OpenDDE | coordinates identical, confidence within 1e-4 |

A blank scope means the flag names no model: it applies wherever a model reaches the code it
changes, and its section says which ones do. The OpenMP thread settings tt-bio fills in for per-card
workers are not flags of ours; they are covered at the end, under
[Idle host threads when a box is full](#idle-host-threads-when-a-box-is-full).

## BindCraft 2 round kernels

Default: on inside a BindCraft 2 round on Blackhole, off elsewhere.

Kernels that delete DRAM round trips from the AlphaFold 2 Evoformer's gradient. Each is armed by
`bindcraft2.predictor(exact=False)` through `fast_round`, alongside
[`TT_BIO_LNBW_FUSED`](#tt_bio_lnbw_fused) and [`TT_BIO_TRIATT_BW_FUSED`](#tt_bio_triatt_bw_fused).
Outside a BindCraft 2 round every one keeps the composed path, and so do the first eleven on
Wormhole, where they have not been graded. Setting a flag to `0` turns that kernel off even inside
the round; `bindcraft2.predictor(fast=False)` turns all of them off together.

| flag | what it replaces |
|---|---|
| `TT_BIO_GATED_BW_FUSED` | The triangle multiplication's move back plus its two sigmoid-gate gradients, about 16 calls, become one kernel. |
| `TT_BIO_GATED_GRAD_PACKED` | That kernel writes its gradients straight into the in-projection's gradient, so the concat that joined four slices is gone. |
| `TT_BIO_GATE_BW_FUSED` | Every other sigmoid-gate multiply's gradient as one kernel instead of seven eltwise calls. |
| `TT_BIO_NOGRAD_INFERENCE` | The round's two forwards that record nothing run the fused forward kernels the gradient hook used to turn off. |
| `TT_BIO_AF2_G_BIAS_IN_MATMUL` | Triangle attention's gate bias joins its matmul instead of a separate broadcast add. |
| `TT_BIO_LEAD_SUM_FUSED` | The triangle-attention backward's bias-gradient sum as one kernel instead of permute, reduce, permute. |
| `TT_BIO_TRIATT_BW_QKV_PACKED` | The triangle-attention backward writes q, k and v gradients packed, so their concat is gone. |
| `TT_BIO_TRIATT_BW_EXP_21F` | A shorter exp in the triangle-attention backward's softmax recompute. |
| `TT_BIO_FANIN_CAST_FUSED` | A gradient with several consumers is summed and cast in one pass instead of two. |
| `TT_BIO_PAIR_TRANSPOSE_FUSED` | The pair tensor's i/j swap as one move kernel, forward and backward. |
| `TT_BIO_PAIR_MM` | Pair-track linears and their input gradients on `minimal_matmul` with a per-shape block table, ReLU fused at pack. |
| `TT_BIO_AF2_OPM_ROWS_IN_K` | The outer product mean sums its MSA rows inside one contraction instead of one product per row. Acts only with more than one MSA row, which a BindCraft 2 round has. Both boards. |

**Accuracy.** Each was graded on its own against a float64 reference of an Evoformer block's VJP
at 288 tokens, blocks 0, 3 and 7. The largest move any of them makes is 6e-4 rel L2 (pair_mm on
block 0's MSA gradient) against a bf16 floor of 0.034 to 0.074; four move nothing at six digits,
and the kernels that replace eltwise chains are closer to float64 than the chains they replace
(the fused gate gradient reads 1.66e-3 against 2.2e-3 to 4.2e-3). With a float32 cotangent the
composed gate gradient came back bfloat16 at 0.117 rel L2; the fused one keeps float32.

**Speed.** On a BindCraft 2 round at 288 tokens, three trajectories on one p300c chip, the eleven
together take the round from 5.70 to 4.50 s (1.27x; arm means of three, 5.48-5.81 against 4.40-4.64), arms alternated in one
sitting at AICLK 1350 sampled during every arm. The host was shared with other campaigns
(load1 8 to 13), which slows both arms; a second sitting under the same load read 1.275x.

**Sizes.** With all of them on, a p300c chip runs 192, 352, 576 and 864 tokens to a completed
gradient round; 864 peaks at 27.04 GB resident with 7.19 GB free.

`TT_BIO_AF2_OPM_ROWS_IN_K` came later and is graded separately. It changes the forward, closer to
float64 (outer-product rel L2 3.52e-3 to 2.93e-3), and leaves the VJP's per-row products as they
were; on the teacher-forced float64 block VJP with two MSA rows no block moves by more than
0.0017. Together with a ReLU backward that now gates its multiply in place (bit-exact, no flag),
it takes the 288-token round on a p300c from 4.055 to 3.946 s, 1.028x, six arms alternated at
AICLK 1350 with disjoint ranges.

## `BOLTZ2_TOKEN_DIT_SDPA`

Default: on, Boltz-2 only.

The token-level DiT attention ran as an explicit score matrix: materialise 16x512x512 scores, read them back, softmax, read them again. The fused SDPA never writes them out.

**Accuracy: not bit-exact.** The kernel holds its exponentiated scores in bf16 where the explicit path held fp32. A 298-residue control moves 0.1775 Å CA and 0.3837 Å all-atom, inside its 0.35 Å CA bar, and pLDDT goes up rather than down, 0.909487 to 0.913597.

**Speed: 1.105x on the fold** with `TT_BIO_ATOM_AXIS_BUCKET` (22.195 s to 20.079 s at 512 residues on Blackhole) and 1.400x on the diffusion sampler. On its own it is worth 1.100 s.

## `TT_BIO_ATOM_AXIS_BUCKET`

Default: on.

The atom axis used to be bounded by the worst case, every token a tryptophan. Sizing it on the real atom count gives 4480 atoms at 512 residues where the old bound gave 7168, which takes the atom transformer from 224 attention windows to 140. Ninety-five of the windows it deletes never held an atom.

The bucket is a multiple of 32 for any composition. That is what lets `TT_BIO_TOKEN_BUCKET=0` and an off-lattice token count be used together.

**Accuracy: byte-identical at 298 residues.** At 512 it reassociates one matmul's contraction, so the structure is not guaranteed bit-for-bit there.

**Speed: 0.997 s of a 512-residue Boltz-2 fold** on its own.

## `TT_BIO_ATOM_SHIFT_GATHER`

Default: on.

The atom transformer attends within a sliding window. Upstream assembles each window's key set by
multiplying the atom sequence with a one-hot selection matrix. That selection is a contiguous run of
atoms, so tt-bio shifts the sequence once and slices the runs straight out of it.

**Accuracy: identical.** Fifty-six timed folds across two trees wrote one structure per tree, byte
for byte, in both arms, pLDDT equal to six places. Off the device, the slice reproduces the reference
gather under `torch.equal` at every window count the fold produces, padded and unpadded, and refuses
four negative controls (`perf/b2z2_elision/test_shift_equiv.py`).

**Speed: 1.01985x on the fold** (19.8545 s to 19.4680 s, 20 folds per arm, all 10 paired reps
positive), 1.07562x on the sampling stage, which is where the whole gain is. Re-measured at
2f5072d8, the tree the benchmark cell was published from then: 19.7275 s to 19.324 s, eight folds
per arm interleaved ABBA, every pair positive, median 1.02138x against an A/A floor of 1.00886x
worst case.

tt-bio decides on the selection matrix, not on the shape of it: it reconstructs the matrix a centred
sliding window would produce and compares entry by entry, once per fold, for 2.6 ms of host time at
512 residues. A model whose atom attention selects a different key set, or pads its matrix
differently, fails that comparison and keeps the matrix multiply. No model name appears in the
condition.

Reading the matrix rather than its shape is the whole safety argument. The atom axis is padded out
to a bucket, and the selection matrix is empty in the windows past the real atom count, so that
matrix and a same-shaped one that selects real atoms there are two different transforms with
identical dimensions. An earlier version of this optimization looked at the shapes only, got those
windows wrong, and still wrote the identical structure, because the attention mask is built from the
same matrix and discards exactly the entries the selection got wrong. Nothing the model outputs
distinguishes the two, at any size. The matrix comparison does.

## `TT_BIO_ATOM_SUPERSET_WINDOW`

Default: on.

The Protenix atom transformer attends each block of 32 atoms over a 128-atom key window that starts
48 atoms to its left, so the window never lines up with the device's 32-row tiles. tt-bio built
every window by copying rows one at a time (a gather in bf16, a loop of slices in fp32), and that
copy was most of the module's time. With this flag each block attends over the five whole tiles
around it instead (160 keys). The 32 extra keys and every key the window mask excludes get a bias
of -1e9, folded once per fold into the per-block pair bias, so they carry zero weight and the
attention is the same function. The windows become five aligned slices and one concat.

**Accuracy: the same attention up to rounding.** In float64 the superset output matches the
128-key window to 1e-12 at 33 to 5,919 atoms and one to five samples
(`tests/test_atom_superset_window.py`). On the device the softmax and attn@v reduce over 160 keys
instead of 128, so the result is not bit-exact. It was graded together with
`TT_BIO_ATOM_TILE_HEADS` and the triangle-attention flags `TT_BIO_SDPA_FUSED_PADDED` and
`TT_BIO_TRIATT_QK_MASK_PRELOAD` on Protenix-v2, 11 complexes x 4 seeds: the same-seed top-pose deviation
has a median of 0.31 A on Wormhole and 0.10 A on Blackhole against the 0.60 A bar, while re-running
with another seed moves it 0.8 A. Docking success and every confidence score are unchanged within
their confidence intervals.

**Speed** of that set on the Protenix-v2 730-token fold: 505.85 to 454.98 s on a Wormhole Galaxy
chip at 1000 MHz (1.11x), 247.98 to 213.89 s on a Blackhole p150a at 1350 MHz (1.16x). The atom
attention module alone goes from 29.1 to 9.9 ms per call on Wormhole and 15.4 to 4.7 ms on Blackhole.
`TT_BIO_ATOM_SUPERSET_WINDOW=0` restores the windowed path.

## `TT_BIO_ATOM_TILE_HEADS`, `TT_BIO_ATOM_KV_WINDOW`

Default: on. Both write the identical structure.

`TT_BIO_ATOM_TILE_HEADS` splits the superset path's query, key and value heads and merges the
output without leaving tile layout, instead of a row-major pad and permute each way.
`TT_BIO_ATOM_KV_WINDOW` acts when the atom attention runs as one fused kernel (`TT_BIO_ATOM_SDPA32`
in normal mode, `atom_sdpa` under `--fast`): the kernel reads each block's keys, values and queries
straight from the head tensor as sliding windows, so the five slices and the concat are gone.
On a Wormhole chip it halves the fused call: 6.28 to 3.03 ms in fp32 and 3.08 to 1.46 ms in bf16.

## `TT_BIO_ATOM_SDPA32`

Default: on.

Normal mode runs the atom attention in fp32 as four device calls: scores, scale plus bias, softmax,
and the weighted sum of values. This flag runs all four as one fused fp32 attention kernel, the same
recipe the diffusion transformer uses. It is inert in bf16, where `--fast` already fuses this site.

**Accuracy: moves, closer to float64.** Against a float64 evaluation of the same operands the fused
kernel reads rel_rms 0.0217, the four calls 0.0265. On Protenix-v2, 11 complexes x 4 seeds on
Wormhole, the same-seed top-pose deviation has a median of 0.05 A against the 0.60 A bar and every
paired metric's confidence interval covers zero.

**Speed: 1.03x on the fold**, 450.58 to 436.39 s on the Protenix-v2 730-token fold on a Wormhole
Galaxy chip at 1000 MHz (n=3 each, same chip). The atom attention call goes from 9.93 to 3.03 ms
with `TT_BIO_ATOM_KV_WINDOW`. `TT_BIO_ATOM_SDPA32=0` restores the four calls.

## `TT_BIO_BH_DRAM_READ_SPLIT`

Default: on, Blackhole only.

A device kernel's DRAM read larger than 2 KiB is issued as reads of at most 2 KiB. On Blackhole a
large DRAM read next to other cores' DRAM writes can lose its response and hang the chip; newer
tt-metal releases work around it the same way (tenstorrent/tt-metal#59622), and the ttnn wheels
tt-bio runs on predate the fix. tt-bio applies it by compiling kernels from a private copy of ttnn's
headers with that one function patched, so the installed package is never edited.

**Accuracy: identical.** The same bytes arrive, only in smaller pieces.

`TT_BIO_BH_DRAM_READ_SPLIT=0` compiles against the stock headers. A `TT_METAL_RUNTIME_ROOT` you set
yourself is respected and the patch is not applied.

## `TT_BIO_BH_ETH_DISPATCH`

Default: on when the installed ttnn supports it, Blackhole only. Stock ttnn does not, so after a plain
`pip install` this flag does nothing.

A Blackhole chip normally gives one column of Tensix cores to the command queue, so tt-bio computes on
11x10 cores. With Ethernet dispatch the command queue runs on idle Ethernet cores and tt-bio gets all
12 columns, 120 cores (+9.1 %). Stock ttnn 0.68.0 cannot fit the dispatch kernels in an Ethernet core's
firmware region. A ttnn built with [`scripts/ttnn_bh_eth/build_wheel.sh`](../scripts/ttnn_bh_eth/build_wheel.sh)
can: it builds tt-metal v0.68.0 with one patch, using the release wheel's toolchain, and labels the
wheel `0.68.0+bh.eth2`, which tt-bio's `ttnn==0.68.0` pin accepts. At startup tt-bio reads the
installed ttnn's Ethernet-kernel limit and uses Ethernet dispatch only when it is large enough.

**Accuracy: identical.** Protenix-v2 at 730 tokens writes the same structure byte for byte with either
dispatch, in normal and fast mode, on 4 seeds each.

**Speed: 1.036x normal, 1.045x fast** on the Protenix-v2 730-token fold on a Blackhole p150a at
1350 MHz: 130.46 to 125.89 s normal, 109.68 to 104.96 s fast (n=3 warm each, same chip). Wormhole is
unaffected: a single Wormhole chip already dispatches on Ethernet.

`TT_BIO_BH_ETH_DISPATCH=0` keeps Tensix dispatch.

## `TT_BIO_DEVICE_CONDITIONING`

Default: on, Boltz-2 only.

Boltz-2's diffusion conditioning reads the trunk's pair tensor three times, and upstream does all
three on the host: the pairwise conditioner, the 24-layer token bias stack, and the atom encoder's
`z_to_p_trans`. Each one is a channel map applied independently at every (i, j), so this flag runs
them on the card, where the trunk has just left the tensor. Only the `[n, n, 16]` projection the
atom encoder actually consumes comes back, and the token bias never leaves the device at all: it
goes straight into the diffusion cache, which used to upload the host's copy of it.

**Accuracy: not identical.** The device does this in bf16 where the host did it in fp32, and it
uses the fused bias stack, so the structure moves. It is scored against the experimental structure
1HCL rather than against the previous coordinates, four seeds per arm, both arms in one process.
Native CA-lDDT goes up on both pseudo-domains at 512 residues, 0.93732 to 0.94036 and 0.91573 to
0.91860 as a mean of four seeds, and is flat at 298 residues, 0.96742 against 0.96741. Native
CA-RMSD moves the same way. Three of the four seeds move 0.15 to 0.29 Å per pseudo-domain at 512
residues and the fourth moves 1.26 to 1.51 Å, against a seed floor of 0.97 to 1.87 Å over the same
pairs. That fourth seed is a basin, not a loss: it is the worst fold either arm produced, native
CA-lDDT 0.92021 and 0.89702 on the host path where every other host seed is above 0.9397 and
0.9136, and the device arm puts it back with the rest at 0.93585 and 0.91317. At 298 residues the
move is 0.19 to 0.39 Å against a 0.79 to 1.25 Å floor.

**Speed: 17.989 s against 18.773 s, a 512-residue fold.** Both arms in one process on one card,
interleaved as base / device / base inside every rep so drift cannot land on one arm, cold fold
discarded, 8 device folds against 16 host folds, under benchlock on an idle box. qb2, one Blackhole
processor of a p300c board, physical card 0, ttnn 0.68.0, 3 recycles, 200 sampling steps, one
sample, seed 0, templates off. Median ratio 1.04358x against an A/A floor of 0.99824x drawn from
the two host folds that bracket each device fold, and every device fold in the run is faster than
every host fold in it. Spread is 0.90 % on the device arm
(`perf/b2z2_cond/out/timing_qb2c0.json`).

`TT_BIO_DEVICE_CONDITIONING=0` restores the host path and the previous coordinates.

BoltzGen shares Boltz-2's `TrunkModule` but does not get this flag: it never asks the trunk to keep
the pair tensor on the device, so it takes the same deallocate it always did. No other model
reaches the pair track at all.

## `TT_BIO_DEVICE_CONFIDENCE`, `TT_BIO_DEVICE_CONF_HEADS`

Default: both on, Boltz-2 only.

Boltz-2 scores the structure it just predicted with a confidence head, and upstream builds that
head's input on the host: it normalises the trunk's pair tensor, adds the relative-position
encoding, the token bonds and the contact conditioning, broadcasts the single representation into
it and embeds the distogram. Every one of those is a channel map at each (i, j), and the trunk has
just left the pair tensor on the card, so `TT_BIO_DEVICE_CONFIDENCE` does the assembly there. It
also deletes the upload that used to feed the head's own pairformer: 134 MB of fp32 at 512
residues, replaced by an index map.

`TT_BIO_DEVICE_CONF_HEADS` continues the same idea past the pairformer. The pae and pde
projections and the bin contractions behind them run on the card, and only the three aggregated
numbers per token pair come down instead of a tile's worth of bin logits: 67.1 MB to 2.097 MB at
512 residues, 32.0x fewer bytes. It is built on top of `TT_BIO_DEVICE_CONFIDENCE` and is measured
with it, so set them together or not at all.

The win is the deleted host work, not the smaller download, and on Blackhole the download is
deliberately not the smaller one. Narrowing the readback needs a row-major layout, and a row-major
readback on Blackhole runs at a flat 159 MB/s: 0.524 MB through 8.389 MB, 256 to 1024 residues,
every rung within 2 MB/s of that, against 1.8 to 3.9 GB/s for the tiled read. An 8x byte saving
does not cover an 11 to 23x rate penalty, so the narrow shape loses at every rung of the size
ladder, 3.72 ms against 1.13 at 256 residues, 14.40 against 4.88 at 512 and 57.50 against 42.29 at
1024 (`perf/b2z2_confptm/download_shape_*_qb2_c0.json`, 20 interleaved reps per rung with the
three variants asserted bit-identical first). Wormhole ranks them the other way round, 17.500 ms
narrow against 32.644 tiled, and that is the machine the choice was originally fitted on, so the
head picks its download shape by `is_wormhole()`: slice on the card there, slice on the host here.
Scored inside a real fold with both arms interleaved in one process, picking the right one is
worth 9.61 ms of download plus the 0.64 ms untilize it no longer does at 512 residues, and 3.15
plus 0.56 ms at 298, with the head's output and the written CIF asserted identical on every fold
(`perf/b2z2_confptm/readback_ab_298_qb2_c0.json`, `readback_ab_512_qb2_c0.json`). It is a tenth of
a percent of the fold either way; it is not why the flag is on.

**Accuracy: the coordinates cannot move, and they do not.** The confidence head runs after the
sampler and its outputs are scores, so at one diffusion sample nothing it produces feeds back into
a coordinate. The claim is therefore an equality rather than an Ångström bar, and it holds: every
atom is bit-identical at 298 and 512 residues, max 0.000000 Å, against a same-arm control that is
also exactly zero. What does move is the confidence itself, in bf16 where the host used fp32:
per-atom pLDDT by at most 0.362 at 512 residues and 0.185 at 298, on a 0–100 scale, mean 0.032 and
0.022.

The scores in `results.json` move too, and by less than the model moves them itself. Taking
`TT_BIO_DEVICE_CONF_HEADS` on its own at 512 residues on Blackhole, pTM shifts 0.0031 of its
0.6334 and `complex_pde` 0.0028, pAE by 0.149 Å mean absolute on a 14.76 Å mean and pDE by
0.055 Å on 5.77 Å, against a same-arm control that is exactly 0.000000 on all eight scalars. Fold
the same target with four diffusion seeds instead and pTM spans 0.0758, `complex_pde` 0.0538, pAE
1.04 Å and pDE 1.72 Å, 7x to 31x more than the flag moves them
(`perf/b2z2_confptm/seedscatter512_qb2_c0.json`). Per-residue pLDDT is untouched by this half of
the pair, exactly 0.000000, because `to_plddt_logits` reads the single representation and stays in
torch.

A CIF sha256 is the wrong instrument for this flag, and the run reports both readings for that
reason: `write_result` puts pLDDT in the B-factor column, so the file hash changes with every
coordinate identical. `perf/b2z2_confhead/score_conf.py` reports the coordinate delta and the
pLDDT delta separately.

**Speed: 16.537 s against 17.285 s, a 512-residue fold.** Both arms in one process on one card,
`base device device base` inside every rep so the order reverses within the rep, one cold fold per
arm discarded, 8 folds per arm under benchlock. qb2, one Blackhole processor of a p300c board,
physical card 0, ttnn 0.68.0, 3 recycles, 200 sampling steps, one sample, seed 0, templates off.
Median of 8 paired ratios **1.04743x**, all 8 positive, against an A/A floor of 1.00104x median
drawn from this session's own same-arm adjacent pairs
(`perf/b2z2_confship/cell_512_qb2_c0.json`). The earlier Wormhole reading for the first of the two
flags was 1.0128x; the Blackhole fold is less than half as long, so the same block of deleted host
work is a larger fraction of it.

Setting either to `0` restores the host path for that half. No other model reaches the Boltz-2
confidence head.

## `TT_BIO_DEVICE_TILIZE`

Default: on. Protenix-v2, and OpenDDE and PXDesign through the modules they share with it.

A tensor between 4 MiB and 1 GiB that goes to or comes from the chip once per fold (the template
and MSA features, the trunk pair, the confidence head's logits) is converted between row-major and
tile layout on the chip instead of on the host. ttnn's host conversion is single-threaded and
holds Python's GIL, so the chip used to sit idle through it. The conversion is a permutation either
way, so the output is identical byte for byte. The chip briefly holds a second copy of the tensor,
which is why larger tensors keep the host path.

Measured together with `TT_BIO_HOST_LANE`; see that section for the numbers.
`TT_BIO_DEVICE_TILIZE=0` converts every tensor on the host.

## `TT_BIO_DEVICE_ZINIT`

Default: on, Boltz-2 only.

Boltz-2 starts the trunk from `z_init`, a `[1, n, n, token_z]` pair tensor built by summing five
per-`(i, j)` terms: two broadcasts of a `[1, n, c]` projection, the relative-position tables,
`token_bonds`, the bond-type embedding and `ContactConditioning`. Upstream builds all of it in
torch and then uploads the result to the trunk. At 512 tokens that tensor is 134 MB and every
intermediate sum is another one. This flag builds the same sum on the card, where the resident
trunk consumes it, so the adds run in the trunk's own memory and the upload never happens. What
crosses the bus instead is an index map and one packed feature tensor.

It is the same construction the confidence head does with different weights, so both call sites go
through one `PairAssemblyDevice` and there is no second copy of the assembly.

The flag declines rather than diverging: the device path needs the resident trunk to be the
consumer, and it needs nothing else on the host to want the relative-position encoding it skips.
BoltzGen's token-distance recycle does want it, so BoltzGen takes the host path and says so on
stdout. Boltz-2 has no such module and templates do not read it.

**Accuracy: not identical.** The device sums in bf16 where the host summed in fp32, and it sums in
a different order. `cdk2x2_298` moves 0.264577 Å all-atom (CA 0.107924) against a 0.000000 Å A/A
floor and a 0.35 Å bar, both arms in one process. At 512 residues it is scored against the
experimental structure 1HCL rather than against the previous coordinates, eight seeds per arm:
native CA-lDDT goes 0.93357 to 0.93238 on one pseudo-domain and 0.91382 to 0.91288 on the other,
paired differences of -0.00119 and -0.00093 with t of -0.49 and -0.34, against a per-arm spread of
0.0096 to 0.0116. Four and five of the eight seeds move up. That is flat, not a loss.

Per-pseudo-domain RMSD cannot carry a 0.60 Å bar at 512 residues on this fixture, and the reason
is measured rather than argued: `cdk2x2_512` is a chimera whose two halves hinge, so re-running the
same build with nothing changed but the seed moves the worst pseudo-domain up to 2.79626 Å all-atom
across eight seeds. The bar sits below the column's own noise. Against that floor the flag moves
1.6372 Å, and every other column reads the same way: domain 2 1.26886 Å against a 1.74577 Å floor,
hinge-free 1.49445 Å against 2.38987 Å, CA-only 1.32167 Å against 2.37177 Å, whole molecule
12.0147 Å against 21.8416 Å. Every column moves less than a seed change does, which is why the
reading above is taken against the experimental structure instead.

**Speed: 1.01955x on a 512-residue Blackhole fold, ten of ten paired reps positive.** One p300c
processor, ten `base,on,base` brackets in one process under the box benchlock, 17.523 s against
17.195 s. The A/A floor taken from the same brackets is 1.00361x and its widest pair is 1.00992x,
below the slowest of the ten lever pairs (1.01566x), so every pair clears the floor. Wormhole read
1.01090x on an n300, six of six positive. `TT_BIO_DEVICE_ZINIT=0` restores the host path and the
previous coordinates.

## `TT_BIO_DIT_COND_HOIST`

Default: on.

Every layer of the token diffusion transformer reads the same conditioning vector, each through
six projections of its own, and the layer norm in front of those projections is a per-layer scale
over a shared normalised input. This flag folds each layer's scale into its own weight block, so
one parameter-free layer norm and two concatenated matmuls replace 144 matmuls and 48 layer norms
per sampling step. The dot products and the FLOP count do not change. What changes is how the
launches are grouped, and the order of one bf16 rounding.

The folded weights are built when the model loads rather than on the first fold, so a process that
folds once is not left paying for them.

**Speed: +0.2052 s at 512 aa** (14.588 s to 14.392 s at a forced 1350 MHz, five interleaved reps,
95 % CI [+0.1561, +0.2543], against the same session's paired A/A floor of +0.0324 s +/- 0.1087).
That is this flag on its own, which is what ships: the same session also carried
`TT_BIO_UNFUSED_SILU` and read +0.4798 s for the pair, but that flag is off by default and the
stack figure describes no shipped configuration. Timed at the block rather than at the fold, this
lever reads 1.071x and 1.085x across two sessions on this card. The win decays with size: +0.2809 s
at 298 aa, +0.2415 s at 512 aa and +0.1811 s at 768 aa, each above its own interleaved A/A floor.

**Accuracy: not bit-identical, and scored on this flag alone.** At 298 aa, the size the
0.35/0.60 A band is written against, it moves the structure 0.24726 A all-atom (worst of two
seeds). That is inside the pass band and 0.309x that fixture's own base-against-base seed floor of
0.79984 A, with a same-seed A/A control at 0.00000 A and identical digests.

At 512 aa whole-molecule RMSD reads 11.81553 A, and that figure is a hinge rotation rather than a
larger error. `cdk2x2_512`'s hinge is unconstrained, its own base-against-base seed floor runs
1.3735 to 17.5024 A over ten seed pairs, and RMSD superposes the whole molecule, so it cannot tell
a rotated lobe from a worse fold. CA-lDDT against the experimental structure is the instrument that
can: it is superposition-free and local, so a rigid inter-lobe rotation barely moves it. Against
1HCL domain 1 (252 CA) it reads 0.91585 with the flag off and 0.92767 with it on, +0.01182 in the
flag's favour, with the A/A control at exactly 0.00000. pLDDT moves the same way, +0.017089. Two
limits worth knowing: that reading is one seed, and domain 2 did not resolve against 1HCL. RF3's
token DiT inherits this default and has not been scored for it.

Scope: RF3's token DiT builds this same block, so the default applies to RF3 as well as Boltz-2.
The atom-level transformers take a different path and do not read the flag.

## `TT_BIO_DIT_SHARED_COND`

Default: on. Boltz-2 and BoltzGen.

All diffusion samples of a fold share the noise level and the trunk output, so the token conditioning
and every projection the diffusion transformer derives from it are one row repeated per sample. With
equal times and one trunk row the step computes them once and broadcasts. **Accuracy: identical** (fold
digest unchanged on 4 seeds). **Speed:** 158.0 to 152.1 s on the Boltz-2 730-token fold, Wormhole,
1000 MHz. `0` restores the per-sample path.

## `TT_BIO_FUSE_BIAS_STACKS`

Default: on, Boltz-2 only.

Boltz-2's diffusion conditioning builds a per-layer bias stack with one call per layer. This flag
builds all of them in one pass instead.

**Accuracy: not identical.** Unlike every other flag on this page, this one ships on a measured
control rather than on an algebraic argument. `cdk2x2_298` moves 0.218 Å all-atom against a 0.000 Å
A/A floor and a 0.35 Å bar, and pLDDT moves 0.0026. `TT_BIO_FUSE_BIAS_STACKS=0` restores the
per-layer stack and the previous coordinates.

**Speed: 1.0085x on the fold.**

The one thing that would have made this Boltz-2-only claim wrong is BoltzGen, which shares Boltz-2's
diffusion module. It does not call this path: BoltzGen builds its conditioning from
`tt_bio/boltzgen/model/modules/diffusion_conditioning.py`, which inlines the per-layer loop rather
than reusing `_fuse_bias_stack`, so a whole design run makes zero calls to it against a Boltz-2
fold's three. A model that reuses the shared stack builder gets this flag too; one that inlines its
own loop, like BoltzGen, does not.

## `TT_BIO_FUSE_MASK_ADD`

Default: on.

The gated-residual write-back was a multiply followed by an add. `ttnn.addcmul` is the same arithmetic in one dispatch.

**Accuracy: bit for bit** at fold level.

**Speed: 1504 dispatches deleted** per 298-residue OpenFold3 fold. The saving is dispatch count, not arithmetic.

## `TT_BIO_FUSE_NORM_RESIDUAL`

Default: on.

When an add feeds nothing but a layer norm, the norm can take it as its residual input and the separate add disappears.

**Accuracy: bit for bit.**

**Speed: 1.880x on a [1,512,512,128] norm.** Its one call site is Protenix-v2's pde confidence branch, which the 298-residue protocol never reaches, so the fold-level saving is unpriced rather than measured. The op-level ratio is the only number claimed for it.

## `TT_BIO_FUSE_SCALE_ADD`

Default: on, fp32 operands only.

Attention scales its scores and then adds a bias. `ttnn.addalpha` does both in one call.

**Accuracy: bit-identical at every call shape**, measured. At fp32 the fused and unfused arms round the same way, which is why the flag is restricted to fp32 operands: 1500 of the 1503 calls a 298-residue fold makes are fp32, and the three bf16 calls keep the multiply-then-add chain. Fusing those three alone moved an OpenFold3 structure 1.475 Å, so they stay out.

**Speed: not claimed as a wall-clock ratio.** Summing measured per-call times predicts 67.0 ms of a 6.92 s Protenix-v2 fold and 73.9 ms of a 9.93 s OpenFold3 fold. Both are below what a fold A/B on this box can resolve.

## `TT_BIO_GATE_GRANULARITY`

Default: 2.

The reblock-permute kernel that Blackhole's channel-gating path uses (mirroring `binary_ng`'s own
structure: a sigmoid then a multiply, done in one gated kernel instead of two ops) acquires its
destination register tile by tile. This flag sets how many tiles it acquires per DST acquire, from 1
(the previous behaviour) to 4.

**Accuracy: identical at every value.** The three stages run in the same order through the same two
bf16 circular buffers regardless of granularity, so no rounding point moves; pinned by `torch.equal`
against the two-op sequence, per shape, on both architectures.

**Speed: 2 is the value that ships, not the fastest one measured.** Wormhole reads 1.0420x at
granularity 2 and 1.0759x at 4; Blackhole reads 1.0149x at 2 and 1.0011x at 4. 4 is a wash on the
architecture the published cell is measured on, so 2 is the setting that wins on one architecture
without losing much on the other. Worth 0.014 s on a 512 aa Blackhole fold, under the fold's own A/A
floor. It ships because it is free and bit-exact, not because the fold moves. Capped at 4: above
that the kernel's multiply stage would need more DST slots than a 16-bit DST has to give it.

## `TT_BIO_HOST_LANE`

Default: on, Protenix-v2.

Moves the host work the chip used to wait for onto one background thread, so the main thread keeps
the chip fed: the trunk's host-only inputs (padding, template and MSA features, masks) are built
while the chip runs the input embedder, the pair tensor the confidence head starts from is built
during the diffusion, and each confidence sample's post-processing (expected PAE and PDE, pLDDT,
pTM and ipTM) runs while the chip computes the next sample. The arithmetic and its order do not
change, so the output is identical.

**Speed, with `TT_BIO_DEVICE_TILIZE`:** a 730-token complex, warm folds, base and stack interleaved
on one chip.

| | normal | `--fast` |
|---|---|---|
| Wormhole, AICLK 1000 MHz | 274.5 s to 271.5 s (6 pairs) | 226.1 s to 220.0 s (4 pairs) |
| Blackhole p150a, AICLK 1350 MHz | 135.2 s to 131.5 s (4 pairs) | 113.9 s to 110.5 s (4 pairs) |

Outputs were compared fold by fold against the same tree with both flags off: eleven inputs at four
seeds on Wormhole and twelve inputs on Blackhole. Every completed fold wrote the same structure. JapanFold serves one diffusion sample, which keeps the
one-off savings (trunk inputs, confidence entry) and loses most of the per-sample overlap, so a
served fold gains a little less than the table.

`TT_BIO_HOST_LANE=0` runs every job inline, where it used to run.

## `TT_BIO_HOST_LEVERS`

Default: on, Boltz-2 only.

Not an optimization of its own. It gates `TT_BIO_FUSE_BIAS_STACKS` and `TT_BIO_HOST_BLOCK_PAIRWISE` together, so `0` takes the pre-lever host path for both in one variable. Each flag still answers to its own name; this one is the AND in front of them.

Bisecting a host-side result is what it is for: turn the group off, confirm the result moves, then put the members back one at a time.

## `TT_BIO_LEVERS`

Default: the graded set for the run's mode. Protenix-v2 and Boltz-2, each graded on its own.

Protenix-v2 runs a few of its kernels with cheaper numerics than its reference path: the triangle
multiplication's contraction in one block with its residual folded into the epilogue, the trunk's
matmuls at HiFi3 instead of HiFi4, the diffusion transformer's fp32 attention as one fused kernel, and
the transitions' SiLU as a shorter fp32 kernel accurate to a few float32 ulp, fused into the matmul,
triangle attention's gate, output projection and residual add as one kernel, and on Wormhole the MSA
outer product mean's contraction in K blocks of up to 6 tiles.
Each was graded on its own and then all together, and only the combination that passed is on.
`--fast` uses a larger set. `TT_BIO_LEVERS=none` runs the reference numerics; a comma list of names
picks a set by hand (the names are listed in `tt_bio/tenstorrent.py`).

**Accuracy: moves, inside the seed-to-seed spread.** On 11 post-cutoff complexes the same-seed top-pose
deviation from the reference path has a median of 0.58 A on Wormhole (20 seeds, 220 pairs) and 0.55 A on
Blackhole (4 seeds, 44 pairs), against a 0.60 A bar and a 0.75 to 0.76 A median between two seeds of the
reference path itself. DockQ, CA-lDDT, TM, lRMSD, iRMSD, pLDDT and ipTM have confidence intervals
reaching zero on both, and docking success goes from 165 to 169 of 220 on Wormhole and 32 to 35 of 44 on
Blackhole.

**Speed: 2.07x on the fold**, 501.4 to 241.8 s on the Protenix-v2 730-token fold (deep MSA, 5 samples,
10 recycles) on a Wormhole Galaxy chip at 1000 MHz, warm folds; 121.4 s on a Blackhole p150a at
1350 MHz. `--fast` takes 194.2 s on Wormhole and 99.1 s on Blackhole. These figures also contain
the lossless changes shipped alongside the set.

**Boltz-2** runs the same five normal-mode levers (triangle multiplication in one block with its lean
epilogue, HiFi3 trunk matmuls, fused fp32 diffusion attention, the fp32 SiLU kernel), graded on its own
11 complexes: on Wormhole the same-seed top pose moves 0.90 A median (4 seeds, 44 pairs) against a 1.78 A
median between two seeds of the reference path, every paired confidence interval reaches zero and
docking success is 24 of 44 on both. On a Blackhole p150a the same grade moves the top pose 1.20 A
median against a 1.80 A seed floor, again with every interval reaching zero and docking success 24 of
44 on both. On the 730-token fold at 1000 MHz it takes 158.0 to 147.2 s.
A lever graded only on Protenix-v2 does not reach Boltz-2.

Boltz-2's `--fast` is the shared fast set run on the normal path. The older block-fp8 fast path is off
for Boltz-2 because it was slower than normal mode (159.7 s against 135.5 s on the 730-token fold). The
fast set folds the same input in 123.4 s, and 168.3 s instead of 181.6 s at 1024 tokens (Wormhole,
1000 MHz). Graded against normal mode on the same 11 complexes and 4 seeds: CA-lDDT +0.0018, pLDDT
-0.0002, DockQ -0.004 with its interval reaching zero, docking success 24 of 44 in both modes, and a
median same-seed top-pose deviation of 1.29 A against 1.70 A between two seeds of normal mode.
On a Blackhole p150a at 1350 MHz the 730-token fold takes 98.4 s on the reference path, 68.5 s in
normal mode and 64.1 s in fast mode. At 256 tokens fast mode is no faster than normal there (18.0 s
against 17.3 s).

## `TT_BIO_LNBW_FUSED`

Default: off, training only.

Sends the input gradient of `autograd.layer_norm` through one kernel (`tt_bio/lnbw.py`) instead of
the composed path's ~22 ttnn calls, about 14 of them full passes over the activation. It runs
when only dx is asked for, which is every norm in a BindCraft 2 sequence gradient: gamma and beta
are frozen weights there. A norm whose weights need a gradient keeps the composed path, which
already computes the `norm` those gradients read.

On a BindCraft 2 gradient round at 288 tokens it serves 11 of the 12 norms in every Evoformer
block's backward. The one it declines is the [288,2,256] MSA norm, whose 2-row axis is not a
tile. Op level, 0.315 against 1.986 ms at [1,288,288,128]. Eight checkpointed blocks' backward
goes from 0.5553 to 0.4421 s, alternated in one sitting at AICLK 1350 (qb1 p150a).

Off by default because it has not been through a release gate, except inside
`bindcraft2.predictor(exact=False)`, which turns it on through `fast_round` along with the other
gradient kernels its round is measured with. The gradient is closer to float64 than the composed
path's: dx rel L2 1.81e-3 against 3.84e-3 on the same bf16 operands, and nearer float64 on every
block of the teacher-forced float64 VJP (`perf/bcx_afgrad/vjp_n288_bcp_lnbw_{off,on}.json`). A
float32 cotangent declines, because there the composed path is exact float32 (1.5e-4) and the
kernel's FPU stages read TF32 (1.3e-3). On Wormhole it declines (reason `arch`): it was graded
and device-tested on Blackhole only. `TT_BIO_LNBW_FUSED=0` is the way back. At the BindCraft 2
round it is worth 5.969 to 5.186 s on a qb2 p300c chip (1.151x, six arms alternated, AICLK 1350).

## `TT_BIO_MM_LAYOUT`

Default: off, training only.

Gives a core grid to the batched matmuls that call `ttnn.matmul` with no plan at all: no program
config and no core grid. ttnn's default spreads a batched operand badly, and any explicit grid
fixes it. On a BindCraft 2 gradient round at 288 tokens, 696 calls a round across seven shape
classes, 3.04x to 14.34x each, PCC 1.000000 against the call it replaces. The round pays
1.0336x on device seconds (paired median over 24 rounds, pc card 0 at 1350 MHz).

Off by default because it has not been through a release gate, except inside
`bindcraft2.predictor(exact=False)`, which turns it on for its own duration along with the other
gradient kernels its round is measured with (`fast=False` opts out). It changes the order a matmul
accumulates its partial products, so results move by up to 2.0e-3 in bf16. The structure that
comes out has now been scored, as part of the composed stack rather than alone: 0.5376 A on the
confident core against a 0.60 A kill bar, unchanged from the same stack without it.

Batch-1 matmuls are deliberately out of range: there ttnn's planner is already within 4 % of the
best grid available, and forcing one costs up to 26 %. `TT_BIO_MM_LAYOUT_GRID` sets the grid's
side length, clamped to the device's own compute grid; 8 is the measured best and changing it is
worth under half a per cent.

## `TT_BIO_MSA_LADDER`

Default: on, Boltz-2 and BoltzGen.

The MSA depth axis used to pad to a single 1024, so a 35-row alignment cost exactly what a 1000-row
one did. This flag pads it instead to the smallest rung of (64, 128, 256, 512, 1024) that holds the
depth, and to multiples of 1024 above that, so nothing deeper than 1024 rows changes shape. Three
units read the depth axis and between them they move 68.4 % of the MSA block's bytes. The reference
512 aa fixture carries 35 real rows, which is 29.3x padding.

**Accuracy: not identical, and the difference is displacement rather than error.** Poison the padded
rows with garbage instead of zeros and the MSA module's output is bit-identical
(`perf/roof_msa_ladder/z_parity_512.json`), so the padded region provably cannot reach a real token
and a shorter rung only reassociates the same terms. It reassociates them at the front of the trunk
though, ahead of 3 recycles, 64 pairformer blocks and 200 sampling steps, so the structure moves:
0.295 Å and 0.267 Å CA per pseudo-domain at 512 residues, 0.475 Å and 0.420 Å all-atom, against an
A/A floor of 0.000 Å over twenty-two folds. The whole-molecule figure is 0.904 Å CA, and the extra
comes from a 6.6° hinge between the two copies of this chimeric fixture rather than from either
copy.

Scored against the experimental structure 1HCL instead of against the other arm, the ladder is
closer on both pseudo-domains: native CA-lDDT 0.91822 to 0.92249 and 0.89503 to 0.90295, per-residue
paired means +0.00430 and +0.00743 with bootstrap CI95 [+0.00139, +0.00642] and [+0.00374,
+0.01063], both clear of zero. Native CA-RMSD moves the same way, 1.700 Å to 1.565 Å and 1.528 Å to
1.413 Å. Displacing the same arm's own CA atoms incoherently by that same 0.904 Å costs 0.22 lDDT,
dropping it to 0.776, so the metric can see a move this big and this move is not one.

At 298 residues, where the fixture is a single copy and the arms sit 0.111 Å apart in CA, the ladder
arm reads 0.00344 CA-lDDT lower on seed 0. The per-residue bootstrap excludes zero, but that
bootstrap resamples residues inside one structure, not seeds: the figure is 5.5x smaller than the
0.019-wide seed spread the same scorer measured across seeds, and smaller than the 0.00516 an
incoherent displacement of that same 0.111 Å costs. It is a one-seed reading below the floor that
would make it readable, not a resolved loss, and 298 residues is a size where the lever's own speed
win is smallest.

**Speed: 15.469 s against 16.361 s, a 512-residue fold.** 1.0577x, a paired median of 0.865 s over
ten pairs, arms alternating inside a pair with the pair order flipped every block, one process and
one card. All ten pairs favour the ladder and the two arms' ranges do not overlap: the slowest
ladder fold, 15.932 s, beats the fastest 1024 fold, 16.076 s. The A/A null on the same harness and
host reads 0.9981x. qb2, one Blackhole processor of a p300c, ttnn 0.68.0. A p150a reads 0.795 s on
the same fixture. Rung 64 costs 0.131 s to compile, once
(`perf/roof_msa_ladder/ab_512_qb2_rebased.json`).

The win tracks how shallow the alignment is: 0.433 s with 300 rows, and exactly nothing above 512
rows, where both settings pad to the same 1024. A real ColabFold search usually lands above that, so
this flag is worth more on the reference fixture than on a deep-MSA target.

`TT_BIO_MSA_LADDER=0` restores the single 1024 rung and the previous coordinates.

Boltz-2's MSA module and trunk read the ladder, and BoltzGen reaches it through the trunk it shares.
Protenix-v2, OpenFold3 and RF3 have their own MSA modules and do not read it.

## MSA module flags

`TT_BIO_OPM_JOIN_PARTS`, `TT_BIO_OPM_PROJ_BATCH`, `TT_BIO_PWA_FUSED_HEADS`, `TT_BIO_PWA_UNPADDED`. Default: on.

These four make the MSA module do the same arithmetic in fewer, larger device calls.

* `TT_BIO_OPM_JOIN_PARTS`: when the MSA arrives in depth chunks, the outer product mean used to
  contract each chunk separately and add the partial pair tensors in bf16. It now joins the chunks'
  projections and runs one contraction over the full depth, accumulated in fp32. If the joined
  projections do not fit in device memory it falls back to the per-chunk sum.
* `TT_BIO_OPM_PROJ_BATCH`: the outer product mean's output projection runs as a batch of row blocks
  instead of one matmul with several hundred thousand rows. Same bytes out.
* `TT_BIO_PWA_FUSED_HEADS`: pair-weighted averaging projects, averages and gates all heads at once
  and sums the heads inside the output projection, instead of looping over heads and adding their
  outputs in bf16.
* `TT_BIO_PWA_UNPADDED`: the fused heads without padding each head to a full 32-wide tile, which
  matters for narrow heads (Protenix-v2 uses 8 heads of width 8).

**Accuracy: moves, inside the normal-mode bar.** The join and the fused heads round the sums
differently (once in fp32 rather than repeatedly in bf16), so the structure is not byte-identical.
The projection batch alone is byte-identical. Graded on Protenix-v2 on a Wormhole Galaxy chip
against the exact build: 11 complexes with deposited structures, 4 seeds each, 43 of 43 folds
finite. Median top-pose deviation from the exact build 0.209 A, against a 0.60 A bar and 0.81 A
between two exact runs at different seeds. DockQ, lDDT, TM-score, pLDDT and ipTM differences all
have confidence intervals covering zero; 32 of 43 folds dock in each build. One complex (9W89)
lands in either of two binding poses that are both wrong (DockQ under 0.07); over eight seeds the
exact build picks the far one 2 times and the flagged build 4 times (Fisher p = 0.61), which is
what moves its ligand RMSD.

**Speed, Protenix-v2, 730 tokens, MSA depth 9,947, warm folds:**

| | flags off | flags on | |
|---|---|---|---|
| Wormhole Galaxy chip, 1000 MHz | 506.91 s | 462.31 s | 1.096x |
| Blackhole p150a, 1350 MHz | 242.92 s | 231.19 s | 1.051x |

Wormhole gains more because its MSA module is a larger share of the fold. Boltz-2 and OpenFold3
build the same pair-weighted averaging and outer product mean and reach these flags; their structures
move by the same kind of rounding but have not been graded separately here.

## `TT_BIO_PWA_FULL_HEADS_FUSED`

Default: on. Boltz-2.

The 32-wide-head case of the pair-weighted averaging above, Boltz-2's 8 heads of 32. There is
no padding to drop, so the fused path moves the same bytes with a tile transpose where the unpadded
path pays two general permutes. Its row blocks sit on whole tiles. Bit for bit; together with
`TT_BIO_DIT_SHARED_COND` the Boltz-2 730-token fold goes from 158.0 to 146.0 s on a Wormhole Galaxy
chip at 1000 MHz, and the whole-tile row blocks took it from 174.9 to 158.0 s before that.

## `TT_BIO_OPM_LEGACY_LAYOUT`

Default: off.

`OuterProductMean` averages the MSA depth and projects the result into the pair tensor. Two things
about how it finished that job cost a full pass over the pair tensor on every call. The `1/depth`
mean is a scalar, so it belongs on the smallest tensor in the chain, and this path had it on the
largest: it multiplied the assembled pair rows, 536,870,912 B at 512 residues, where the per-row
MSA tensor is 2,097,152 B. The output projection was also issued once per token row with the core
grid pinned, which makes it re-read the whole contraction for every 32x32 it writes; one matmul
over the flattened rows reads each operand once. Set this flag to get both of the old behaviours
back.

**Accuracy: not bit-identical, and inside the sampler's own seed spread.** Against a float64
reference built from the same bf16 inputs and driven through the real module both ways, the two
arms read the same max error, 5.3899e-3, and mean error 6.18831e-4 legacy against 6.18829e-4
default. They differ from each other by at most 1.953e-3, which is one bf16 step at that
magnitude: left unpinned the projection picks its own contraction blocking, so the sum lands in a
different order. Folding the scale earlier is exact whenever the MSA depth is a power of two, and
off a power of two it moves the same last bit.

At the fold, read this as a sampler basin question rather than a geometry one. The 512-residue
reference fixture is two CDK2 copies on a hinge, and the hinge swings under any bf16 perturbation,
including a change of seed with neither arm touched. Both arms were folded at four seeds in one
process on one card, arms interleaved:

| worst pseudo-domain, all-atom, flag on vs off | seed 0 | seed 1 | seed 2 | seed 3 |
|---|---|---|---|---|
| | 1.59156 A | 0.30342 A | 0.30570 A | 0.29390 A |

The floor to read that against is the legacy arm against itself at a different seed, six pairs:
1.08885 to 1.42169 A, median 1.18521. Three of the four seeds land at a quarter of that floor and
the worst one lands inside it. lDDT-CA, which no superposition can flatter, agrees: 0.93294 at
seed 0 rising to 0.99892, against 0.90622 to 0.93923 for the legacy arm against its own seeds.
Against the crystal (1HCL), four seeds per arm, nothing separates the two: lDDT-CA 0.93908 mean
legacy against 0.93657 default on the first copy, 0.91616 against 0.91539 on the second. Scoring a
structure against itself reads 0.000000 A, so that is the instrument and not the result.

**Speed: 1.0111x on the fold**, 14.3923 s down to 14.2346 s at 512 residues, worth 0.1577 s. Six
reps per arm interleaved inside one process on one Blackhole processor of a p300c, 10x11 grid,
cold fold discarded, under the bench lock. The session's own floor is a third arm repeating the
legacy one: it reads 14.4262 s, so the floor is 0.0339 s, a fifth of the effect. The clock was
forced and sampled during the folds rather than before, 133,876 samples with minimum and maximum
both 1350 MHz.

The gain lands where the code is. Splitting the fold at its stage boundaries puts 0.1535 s of the
0.1577 s in prepare-and-trunk and -0.0347 s in the sampler, and `OuterProductMean` runs inside
`MSALayer`. A sampler-side gain here would have meant something else was using the box.

Every model that builds the shared `OuterProductMean` reaches this: the Boltz-2 and BoltzGen
trunk through `MSALayer`, Protenix, OpenFold3's MSA embedder, RF3's MSA stack and AF2. ESMFold2
has its own `OuterProductMean` in `tt_bio/esmfold2.py` and does not.

## `TT_BIO_PAIR_FFN_L1_FC1`

Default: on, ESMFold2 only.

ESMFold2's trunk runs its pair transition in 32-row blocks. Inside a block the first matmul is
split into two halves whose product the SiLU multiply consumes immediately, and both halves used
to write their result to DRAM for that multiply to read straight back. This flag gives them an L1
destination instead, so 2.15 GB per call never leaves the chip.

The matmul had to be told how to drain. Its default schedule spends 1,212,416 B of a 1,461,760 B
bank on the two input buffers, which leaves no room for an L1 output, so the allocator refused one
at every row height and both halves fell back to DRAM without a word. Naming the output block
width at 16 fits the destination into what is left. The contraction block stays at 1, because that
is the accumulation order the DRAM path used: of 80 configs swept at this shape, all 20 with a
contraction block of 1 are bit-exact against the shipped call and all 60 above it differ by one
bf16 ULP.

**Accuracy: identical, bit for bit.** Only a destination moves, so this is not a precision trade.
Both arms in one process on one card, arms alternating, three rounds at 512 residues: the same
CIF, byte for byte, and the same pLDDT to four places
(`perf/ttx_b3/fold_ab_esm512_c0.json`). A second card says the same at 298, 512, 768 and 1024
residues (`perf/esm3p4close/fold_ab_*_c1.json`).

**Speed: 27.780 s against 30.222 s, a 512-residue fold.** 1.0879x, three folds per arm after a
discarded cold fold, arms alternating, every fold in the fast arm ahead of every fold in the slow
one on a 0.056 s A/A spread. qb2, one Blackhole processor of a p300c, ttnn 0.68.0, 11x10 grid, 10
recycles and 100 sampling steps, one fold at a time under the bench lock.

Where it pays depends on the size, so the four sizes were folded end to end rather than inferred:

| residues | fold, flag off | fold, flag on | |
|---|---|---|---|
| 298 | 18.613 s | 18.188 s | 1.0234x |
| 512 | 30.222 s | 27.780 s | 1.0879x |
| 768 | 66.601 s | 66.615 s | 0.9998x, inside a 0.126 s floor |
| 1024 | 147.230 s | 147.241 s | 0.9999x, inside a 0.108 s floor |

298 and 1024 are card 1, 512 and 768 card 0, so read each row against itself and not across rows.
At 512 residues the destination is not merely requested but served: the device's own latch census
counts 17216 served, 0 refused, 0 blocked and 0 declined with the flag on, against nothing at all
with it off (`perf/ttx_b3/fold_ab_esm512_latch_c1.json`). That is the number to read, because the
per-call counter next to it counts requests and is incremented before the call that can still
fall back.

The token axis pads to a multiple of 32, which is why 298 residues gets the lever at all: it runs
10 row blocks, 10/16 of what 512 runs, and the gated-call census is exactly 10/16 of it. Below
that nothing is row-blocked. At 768 and 1024 the block's other L1 residents leave too little room,
the device refuses the first call and the class retires to DRAM for the rest of the fold, so the
flag neither costs nor saves anything there.

No other model reaches it, by construction rather than by luck: the gated path needs a
`SwiGLUFFN` built with `fuse_swiglu=True`, and ESMFold2's `PairUpdateBlock`
(`tt_bio/esmfold2.py::PairUpdateBlock`) is the only place in the engine that builds one. Boltz-2, BoltzGen,
Protenix-v2, OpenDDE, RF3 and OpenFold3's MSA stack use the shared `Transition`, whose two
matmuls already write to L1, so the round trip this removes does not exist for them. OpenFold3's
diffusion track and AF2-IG have their own transitions again. Measured rather than assumed for the
three that were folded: Boltz-2, Protenix-v2 and OpenDDE at 512 residues all report zero gated
calls in both arms (`perf/ttx_b3/fold_ab_b2_512_c1.json`, `fold_ab_px2_512_c0.json`,
`perf/esm3p4close/fold_ab_odde512_c1.json`).

`TT_BIO_PAIR_FFN_L1_FC1=0` restores the DRAM output and the same coordinates.

## `TT_BIO_PAIR_INPLACE`, `TT_BIO_TRIMUL_INPROJ_ROWBLOCK_NORM`

Default: both on.

The pair operations process a big pair tensor in row blocks and then join the blocks into the
result. When the pair tensor is more than an eighth of the card's DRAM (1.5 GiB on a 12 GiB
Wormhole chip) that join ran on the host, because the result is a second pair-sized tensor and the
card cannot hold two. At OpenDDE's 1536-residue refiner the pair tensor is 6.95 GB (2987
structural tokens padded to 3008, 384 channels), and every pair operation sent it to the host and
back, the triangle multiplication six times. Over PCIe that was about 950 of the 1201 seconds the
refiner took.

Every block of these operations reads only its own rows of the pair tensor. So `TT_BIO_PAIR_INPLACE`
writes each finished block of triangle attention, the pair transition and the triangle
multiplication's output back into the pair tensor on the card. `TT_BIO_TRIMUL_INPROJ_ROWBLOCK_NORM`
does the same for the triangle multiplication's input projection above 3 GiB: each row block
normalises its own rows and is gated straight into the two operands, so the projection is never
joined anywhere. What still goes to the host is the triangle multiplication's hidden tensor, once
down and once up, because the pair tensor and a 384-channel hidden tensor together do not fit in 12
GiB.

**Accuracy: bit-exact.** Layer norm, the projections and the gates are all row-local, so the blocks
compute exactly what the join did. The refiner's output digests at 1536 residues are identical with
both flags on and both off.

**Speed: the refiner's first two blocks at 1536 residues go from 649.6 s to 232.5 s** (two host
threads, 1000 MHz). The triangle multiplication drops from about 110 s to 24 s per call, triangle
attention from 22 s to 6 s, the transition from 17 s to 2 s. A whole OpenDDE fold of 1536
residues on one Wormhole chip at 1000 MHz takes 2826 s, down from 4221 s before these flags
(OpenDDE-AbAg 2748 s, down from 4223 s); at 1024 residues it is 769 s with them and 846 s with
both set to `0`.

**Reach:** sizes only. Anything whose pair tensor stays under the threshold runs the same
operations as before. On a 12 GiB chip that is OpenDDE and OpenDDE-AbAg above about 760 residues
(the input-projection change above about 1080). Models with a 128-channel pair track cross it only
past about 2500 tokens: Nesso-1 affinity at 3072 tokens runs the same speed with the flags as without,
and at 3584 tokens it takes 741 s instead of 5166 s, with the same affinity. `0` on either flag
restores the host join for that part.

## `TT_BIO_PAIR_TRANSPOSE_FUSED`

Default: on. Writes the identical structure.

Swapping the i and j axes of a pair tensor, which the ending-node triangle ops do, went through a
row-major round trip. This flag does it as one move kernel. On Protenix-v2's [736, 736, 256] bf16
pair it takes 9.66 to 3.16 ms on a Wormhole chip at 1000 MHz and 5.16 to 1.82 ms on a Blackhole
p150a at 1350 MHz, and saves 7.7 s on the 730-token fold on Wormhole (458.21 to 450.52 s, same
chip, same digests). `TT_BIO_PAIR_TRANSPOSE_SPLIT` sets how many output tiles per unit the reader
rearranges (default 8, the fastest on both architectures; 0 is the unsplit kernel).
`TT_BIO_PAIR_TRANSPOSE_FUSED=0` restores the round trip.

## `TT_BIO_PWA_BATCH_HEAD_WEIGHTS`

Default: on.

The MSA track weights each row of the alignment by a softmax over the token axis, one softmax per
attention head. Upstream builds each head's weights separately, which means projecting the whole
pair tensor through one column of a weight matrix, once per head. tt-bio projects through the whole
matrix once and slices the heads out of the result.

**Accuracy: identical.** The columns pad into the same tile either way, so the batched projection is
the same arithmetic in the same order, not an approximation of it. Off the device it matches the
per-head loop under `torch.equal` on both outputs, max absolute difference 0, and a negative control
fails the same comparison. On the device the three models that share this code all fold to the same
structure byte for byte at 512 residues: Boltz-2 `a91aa44441f0d9c5`, Protenix-v2 `15772214c5b9e990`,
OpenFold3 `6ee6ac7a3e730688`, each in both arms, pLDDT equal to six places. Boltz-2's digest is
the one it wrote then. `TT_BIO_DEVICE_CONDITIONING` shipped after this was measured and the default
now writes `2bc758a1fb24ef30`; the other two are unchanged, because neither model reaches that
flag.

BoltzGen reaches the same code on its design path and takes the batched projection 192 times in one four-binder design. Its designs are drawn unseeded, so two runs of the same build do not produce the same binders and there is no structure to compare between arms; what is comparable is the designability gate, which passes in every run of either arm.

**Speed: 1.01606x on the MSA track, which is too small to read on the fold.** On one Blackhole
processor the track goes 1.8678 s to 1.8371 s, median of twelve folds, all six paired reps positive
and the two arms fully rank-separated, against an A/A floor of 1.00859 from the same run. That is
about 0.03 s of a Boltz-2 fold, so this page does not quote a fold ratio for it: the fold wall
cannot resolve a win that size and a number that survives only in one session's noise is not a
measurement. On Wormhole the same track costs twice as much and the saving is correspondingly
bigger, 3.8337 s to 3.7054 s, or 1.03672x on a single MSA layer.

tt-bio batches the heads only while they all fit the one 32-wide tile a single head's projection
already pays for, which is a property of the shape and holds at every head count these models use.
Above 32 heads the batched output would be wider than the per-head one and the saving would have to
be re-measured, so it keeps the per-head loop there. No model name appears in the condition.

That condition is also why the structures above are quoted with a call count beside them. A declined
call and an engaged one write the same structure, so an identical digest on its own is equally
consistent with the optimization never having run. Every fold is recorded with the number of batched
and per-head projections it actually made: 16 on Boltz-2, 30 on Protenix-v2, 12 on OpenFold3, and
zero in every arm that had the flag off.

## `TT_BIO_REBLOCK_PERMUTE_GATED`

Default: on.

A triangle multiplication moves its pair tensor into a channel-blocked layout, then runs three eltwise passes over the result: the chunk and two sigmoid gates. The fused reader does all three inside the move.

**Accuracy: bit for bit.**

Every triangle multiplication opts in rather than being switched on by model name, so a model whose shapes the fused reader cannot address keeps the separate ops instead of failing.

## `TT_BIO_RESIDUAL_L1`

Default: on.

A Pairformer layer adds each sub-layer's output back into the pair tensor. Two of those updates
used to be written to DRAM and read straight back by the very next op: the starting triangle
attention's output projection, and the pair transition's assembled result. This flag has both
producers write into L1 instead, so the update never crosses DRAM in either direction.

It is a gate, not a placement. Each site asks whether the update fits across the grid's banks at
the live grid size, with the consuming matmul's per-core buffers reserved underneath it, and
leaves the result in DRAM when it does not. At 512 aa and below it fits; at 768 aa and above it
does not, and the site falls back. The large-target outcome is "no win", never "no fold".

**Accuracy: identical.** A memory config decides which banks a tile lands in, not what is in it.
Off the device the projection and the concat reproduce their DRAM output under `torch.equal`, max
|delta| exactly 0.0, at every size where the gate engages, and the same comparison refuses a
perturbed control (`perf/k10_binaryng_land/test_l1_equal.py`). At the fold, sixteen timed folds at
512 aa wrote one CIF sha256 and one plDDT, `0.864509`, in both arms, and 298, 512, 768 and 1024 aa
each wrote a single digest across both arms.

**Speed: measured with `TT_BIO_TRIMUL_MASK_L1`, not separately.** The two flags touch three
different sites in the same block and the pair was measured as a pair. The number is under
[`TT_BIO_TRIMUL_MASK_L1`](#tt_bio_trimul_mask_l1).

## `TT_BIO_SDPA_ADD_GRANULARITY`

Default: auto.

The fused SDPA kernel folds three additions into its main loop: the running-sum/max update and the
mask add. Both do one tile at a time by default; this sets how many tiles they batch per pass,
sized automatically from the query chunk unless you override it.

**Accuracy: identical at every granularity.** Bit-exact against the per-tile loop, because the same
adds happen in the same order; only how many run per pass changes.

**Speed: 1.0317x on the fused SDPA on Blackhole** (2.8299 to 2.7430 ms at 512x512) and **1.0267x on
Wormhole**. `TT_BIO_SDPA_ADD_GRANULARITY=1` restores the per-tile loop.

## `TT_BIO_SDPA_BAND_DIV_K`

Default: on, Blackhole only.

Triangle attention splits its key axis into chunks, and in the 256-384 residue band the chunk was
64 for every length. 64 divides those lengths, so the fused kernel already served there; it is just
not the best divisor. This flag picks the largest 32-aligned divisor instead, which is 160 at a
padded 320 and 192 at a padded 384, so the online softmax makes one pass over the key axis where it
used to make several.

It fires at exactly two padded lengths, 320 and 384, which covers sequences of 289-320 and 353-384
tokens. Padded 288 and 352 keep 64 because no 32-aligned divisor clears the kernel's floor, and
every length outside the band returns the same pick it returns today, byte for byte. Nothing else
moves.

**Speed: 1.01473x at 298 residues**, 9.061 s to 8.930 s, +0.1315 s. Three blocks with the arms
interleaved, two timed folds each after a discarded warmup, on a p300c at an AICLK of 1350 MHz
sampled during every fold, against an A/A floor of 0.337 %. The arms do not overlap: every off fold
is 9.035-9.101 s and every on fold is 8.905-8.967 s. All 560 triangle-attention calls a fold serve
at the wider chunk, none decline.

**Accuracy: not bit-exact, and the structure moves toward the deposited one.** The chunk width sets
the order the softmax reduces in, so the mmCIF digest changes. Scored instead: against PDB 1HCL,
which is what this target is, CA-RMSD goes 0.808517 to 0.763417 Å and TM-score 0.985599 to
0.987320 over the same 294 residues. pLDDT goes 0.910271 to 0.911447. The move away from the
previous route is 0.2266 Å CA and 0.4312 Å all-atom, inside the 0.60 Å bar. That is one seed at one
size, so read it as no measurable accuracy cost rather than as a gain.

**Wormhole ships off, on purpose.** The only Wormhole evidence is an op-level screen, and this same
band already burned that: its q half looked like a clear win on a Wormhole op screen and the fold
came back 6.7 % slower. `TT_BIO_SDPA_BAND_DIV_K=1` prices it on a Galaxy; what it needs before the
default moves there is one interleaved fold A/B at 320 or 384 residues.

`TT_BIO_SDPA_BAND_DIV_K=0` is the way back on any card.

## `TT_BIO_SDPA_FUSED_LARGE_S`

Default: on.

Triangle attention re-reads the same pair bias once per row of the pair tensor. The fused kernel
reads it once per head instead and holds it, and it needs a narrow query chunk against a wide key
chunk to fit. The chunk ladder that picks those two numbers walks query chunks widest first and
takes the first pair that runs at all, so above 1024 tokens it settled on a wide query the fused
kernel cannot take and handed the call to the stock attention. Every length from 1024 to 2592
tokens fell off the fast path this way. This flag lets the ladder try the fused pair first above
1024, ordered by the work each pair puts on a core, and serves 36 of the 50 lengths in that range
on an 11x10 grid at 4 heads. The other 14 have no 32-aligned divisor between 32 and themselves,
which the kernel declines by construction; that is a property of the token count, not of L1.

Nothing at or below 1024 tokens changes. There the ladder already lands on a fused pair, 560 of 560
calls at both 512 and 1024 residues, and those digests are bit-exact and shipped.

**Accuracy: not bit-exact above 1024 tokens, and there is no shipped digest to break**, because no
length above 1024 served this kernel before. The key chunk sets the online-softmax reduction order, so a
wider key means fewer rescales of the accumulator. Against an fp32 evaluation of the same bf16
operands at 1536 tokens the fused pair is marginally closer than the ladder it replaces, 0.402555
against 0.402814. At the fold, a 1536-residue structure moves 1.007 Å all-atom and pLDDT goes from
0.786016 to 0.789493. That is above the 0.60 Å bar, which was set at 512 residues where re-running
with a different seed moves the structure 1.84 Å; at 1536 residues the same seed change moves it
36.6 Å, so the flag sits 36x inside the variation this size already carries, and it moves pLDDT the
favourable way. Two folds of the same arm in two processes are byte-identical.

An L1 refusal costs speed and not correctness. With every fused pair refused the call falls back to
the same stock attention the flag-off arm takes, bit-identical, max absolute difference 0.0.

**Speed: 4.23x on the attention op at 1536 tokens** (136.143 to 32.184 ms on a Blackhole
p300c), 2.678x at 1920 and 1.865x at 2208. The win survives changing operands: timed per call with
a fresh bias every call it is still 2.71x, and rotating whole operand sets 4.06x. At the fold it
saves **27.3 s of trunk time** at 1536 residues, which at the shipped 200 sampling steps is
**1.1856x** (174.178 to 146.915 s, off/on/off interleaved, one process per fold, against a 1.21 %
same-session A/A floor). The saving is a fixed trunk saving, so it barely dilutes with step count:
a 20-step fold of the same target read 1.1973x. `TT_BIO_SDPA_FUSED_LARGE_S=0` restores the stock
ladder.

**It is on by default because the win is a shipped-configuration number and the cap bounds the
risk.** The earlier 1.1973x came off a 20-step fold, which gives the trunk a larger share than the
default 200 steps does, so it could not carry the default on its own. At 200 steps the ratio is
1.1856x, 15.3x the same-session A/A floor, and the 27.3 s is trunk time rather than per-step time.
Below the cap the route is unreachable by construction, and that is measured and not just argued:
off/on/off in one process per fold gives one CIF digest per size across all three arms at 298, 512
and 1024 residues -- including 1024, the cap boundary itself, which is the last length where the
flip has to change nothing. The chunk pick is also unchanged in every arm at every one of those
sizes, so the stock ladder still serves the call.

**Reach depends on the head count and the grid, not on the model.** The fused pair needs one query
chunk per core, so a card with fewer cores, or a model with more heads on the same card, serves
fewer lengths: 36 of 50 at 4 heads on 110 cores, 25 at 8 heads, 18 at 12. Boltz-2, BoltzGen and
Nesso-1 run their trunk at 4 heads and get the full reach. Sites that run triangle attention in
fp32 (`Fp32TriangleAttention`, and the `fp32_softmax` branch that reaches `_tri_att_sdpa_hifi`)
never consult this flag.

## `TT_BIO_SDPA_FUSED_PADDED`

Default: on.

Some token counts have no chunk size the fused triangle-attention kernel can split them into. 736
padded tokens is 23 tiles, a prime, so every dividing chunk is either one tile or the whole
sequence, and neither fits. Those calls fell to the stock attention, which re-reads the pair bias
once per row of the pair tensor. The fused kernel can also run a chunk that leaves a padded tail,
filling the tail with -inf exactly as the stock op does, and then reads the bias once per core.
This flag offers that pair once per call, right before the first stock rung, so a length that is
served fused today never reaches it.

**Speed: 2.16x on the attention op** at Protenix-v2's 730-token call on a Wormhole chip at
1000 MHz, 41.72 to 19.36 ms, measured per op. With the mask preload and one even key chunk the call
takes 15.45 ms on Wormhole and 6.91 ms on Blackhole (stock 21.42 ms). The fold-level number is under
`TT_BIO_ATOM_SUPERSET_WINDOW`, which was measured and graded together with this flag.

**Accuracy: not bit-exact**, because the chunking sets the online-softmax order. Against an fp32
evaluation of the same bf16 operands the padded pair reads rel_rms 0.0226 against the stock op's
0.0223. The fold-level grade is under `TT_BIO_ATOM_SUPERSET_WINDOW` (passed on both
architectures). `TT_BIO_SDPA_FUSED_PADDED=0` restores the stock rungs.

## `TT_BIO_SDPA_GRID_Q_CHUNK`

Default: on.

Scaled dot-product attention is computed in chunks of query rows, and ttnn hands one chunk to one
core. The chunk size tt-bio shipped was a fixed cap with no term for the card: an attention with few
heads produced fewer chunks than the card has cores and left most of the grid idle for the whole op.
This picks the widest chunk whose work still fills a single pass of the compute grid. The core count
comes from the device and the head count from the tensor, so no card and no model is named.

**Accuracy: identical.** The query axis partitions independent rows: each chunk computes its own
rows and nothing is combined across chunks. The softmax reduction order lives in the key axis, which
this does not touch. `torch.equal` and max abs 0.0 at every call site, and one CIF digest across all
84 timed folds of the three sessions below, at equal pLDDT.

**Speed: 1.0048x on the 512 aa Blackhole fold.** Two independent sessions on the shipped tree read
1.00496x (95 % CI [1.00048, 1.00635], same-session A/A floor [1.00075, 1.00393]) and 1.00479x (95 %
CI [1.00322, 1.00636], floor [0.99845, 1.00262]), 28 folds each, paired median over ABBA reps. At
the one call site it moves the op is **1.13x** (118 to 104 us, 20 paired reps a session). An earlier
session measured 1.00355x on a tree without the triangle fusions, where the same fold took 19.324 s
instead of 18.645 s.

**It moves one call site of two, and that is a property of the shapes.** On the 512 aa reference the
diffusion step's token attention goes from 256 query rows to 128 (64 work units on 110 cores where
the fixed cap gave 32) on all 4800 of its calls a fold. The atom attention's 32 query rows are a
single tile, so there is nothing to split, and all 1200 of its calls keep the shipped chunk. The
ratio is a Blackhole number and does not transport: the same rule is 1.3058x at the op on Wormhole,
because the win is occupancy and occupancy depends on the grid. Narrower chunks also re-read the
keys and values once more per chunk, 20.1 GB more traffic over the fold, which the idle cores more
than pay for here but would not on every card.

## `TT_BIO_SDPA_WIDE_K`

Default: on.

Triangle attention picks its SDPA `k_chunk` by searching downward from a 256 cap. The fused kernel
refuses any call whose `k_chunk` does not divide the padded sequence, so at a padded length whose
32-aligned divisors all sit above that cap, the kernel declines every call and the fold falls back
to the stock op on a mask padded out again. This flag offers the wider dividing chunks too, widest
first, with today's pick last.

**It only touches twenty padded lengths**, the ones whose shipped `k_chunk` fails to divide them:
288, 352, 416, 544, 608, 704, 736, 832, 864, 928, 992, 1056, 1088, 1184, 1216, 1248, 1312, 1376,
1472, 1504. Everywhere else the candidate list has one entry and the path is byte for byte the
default. Every model buckets its token axis to a multiple of 32 (`TOKEN_BUCKET`, with no per-model
exception), so any model whose triangle attention reaches this kernel can present all twenty.
OpenFold3, Boltz-2's affinity trunk, ESMFold2 and RFD3 never reach this SDPA and are unaffected
either way.

**Accuracy: not bit-exact, and this is the one flag where that is visible.** The wider chunk changes
the online-softmax reduction order. This path reproduces bit-exactly at a fixed seed, so unlike
`--fast` the change does not hide inside a nondeterminism floor: on a 686-residue chain it moves the
structure 0.060-0.146 A, against a 3.69-7.28 A spread between seeds of the same input, and pLDDT by
0.0001 against a seed-to-seed 0.0041. `TT_BIO_SDPA_WIDE_K=0` restores the old pick exactly, byte for
byte. Full envelope in [sdpa-wide-k-parity.md](sdpa-wide-k-parity.md).

**Speed: 1.27x-4.39x at the op where it fires**, measured on a Blackhole p150a with the arms
interleaved, and the ratio tracks the padded length rather than the model: 704 reads 3.41x at 4
heads, 2.45x at 8 and 3.51x at 12. The one fold-level arm that exists, Protenix-v2's trunk stage at
686 tokens, moved 120.0 s to 106.3 s, and it is deliberately not quoted as a figure: it recorded no
clock and no board class, and the parts run the same fold 1.19x apart. **It changes nothing at a
padded length the 256 cap already divides**, so 512, 768 and 1024 folds are byte for byte the
default with the flag on or off. Eight measured legs are `torch.equal` between arms, so their
0.955x-1.013x spread is the instrument's floor and every win above it is real.

Padded 1248 perturbs numerics for 1.0090x, inside that floor. It is left in rather than allow-listed
out: a hard-coded length list would be calibrated on one core grid, which is how an earlier layout
lever became a 0.62x loss on the other part.

## `TT_BIO_SOFTMAX_BW_FP32`

Default: on, training only.

Runs the softmax backward in fp32. On by default, and it only ever reaches a training tape:
inference never builds one, so a `predict` run is unaffected whatever this is set to.
`TT_BIO_SOFTMAX_BW_FP32=0` turns it off. `bindcraft2.predictor(exact=False)` also turns it off
for its own duration, because the BindCraft 2 gradient round was measured and graded with the
bf16 backward (`docs/bindcraft2.md`); setting the variable still wins there.

`dx = y (g - sum(g y) / sum(y))` is a cancellation. At a converged row `sum(g y) / sum(y)`
approaches `g`, the subtraction keeps only the low bits of two bf16 numbers, and the reduction
feeding it was the one reduction in `tt_bio/autograd.py` carrying no compute kernel config. With
this on, both operands are cast to fp32, the same expression runs through the same function, and
the result is cast back. The fp32 copies are freed at the end of each call.

It is what makes on-device OpenFold3 training accurate. The model-frame gradient reads 0.946x
the pre-registered accuracy bar with it on and 1.067x with it off on a p150a, and 0.997x against
2.168x on a p300c (`perf/of3t_p10default`, `perf/of3t_p10exact`). The fp32 copies do not
set the memory limit: 576 aa runs out of memory at the same allocation with it on and off.

`perf/of3t_p10exact/smbw32_off_is_main.py` asserts that the off path reaches `ttnn.sum` with
exactly the arguments the pre-flag backward used, so `=0` is the old behaviour and not an
approximation of it.

## `TT_BIO_TOKEN_BUCKET`

Default: on.

Every model's token bucket answers to this, and the legacy per-model flags are ANDed with it, so `0` turns all of them off at once and the fold runs at the exact token count.

Off-lattice counts are slower. They are also what `docs/size-generality.md` asks for when checking that a size claim is a property of the model and not an artifact of the bucket lattice: a ceiling measured only at multiples of 32 cannot tell the two apart.

## `TT_BIO_TRANSITION_L1_ROWS`

Default: on, Blackhole only.

Every transition block splits its input into row blocks so the SwiGLU's intermediates fit in L1.
The height of that block was 16 rows everywhere, a number fitted on Wormhole. Blackhole has more L1
per core and more cores, and nothing ever revisited the height for it, so a 512-residue pair tensor
was cut into 32 blocks when its own budget allows 11.

The flag makes the height a function of the shape and the card instead of a constant: the tallest
block whose live bytes (the normalized input plus the SwiGLU's two halves, tile-padded) fit the
card's measured per-core L1 budget, floored at the height that ships today so nothing gets shorter.
On Blackhole the budget is 514,756 B per core. At the pair channel that works out to
`height x width = 24576`, which is 48 rows at 512 residues, 32 at 768 and 24 at 1024, and 16 at
1536, where the old constant already sat at the budget. Above that the flag does nothing at all.

It is one expression, not a table: the MSA track has a quarter of the pair track's channel and gets
its own taller block from the same budget, and a model with a wider channel gets a shorter one.
A fixed height cannot do this. 768 residues refuses 48 rows and 1024 refuses 28, so every constant
between 24 and 48 clashes on some size.

**Accuracy: identical on the shapes it was measured on, and inside the noise elsewhere.**
Byte-identical CIF at 298, 512, 768 and 1024 residues, both arms, on an 11x10 Blackhole p300c
(`perf/roof_transition_chunk_bh_ship/`). Digests `a8c6fd65f70f4418`, `45781db716ebf020`,
`9e1a1fdd392e0b4c`, `9a049ee58a5a0f7e`. Also byte-identical with the MSA depth axis at its full
1024 rows, where the rule gives the MSA track a 96-row block instead of 16.

Bit-exactness is not a property of every shape, though, and the page should not claim it is. On the
release gate's no-MSA prot leg the two arms differ: 7.2161 Å from the fp32 reference with the flag
on against 7.0511 Å with it off, a 0.165 Å move on a target whose arms both already sit 7 Å out.
That leg is a documented bf16 floor and its verdict does not change with the flag.

**Speed: 1.023-1.035x at 512 residues, 1.030x at 768, 1.013x at 1024.** Four paired reps per
measurement, the off arm run on both sides of the on arm so the session's own A/A floor comes out of
the same folds, one process and one card:

| size | off | on | saved | A/A floor |
|---|---|---|---|---|
| 512 | 15.270 s | 14.750 s | 0.519 s | 0.005 s |
| 512, second session | 15.217 s | 14.875 s | 0.342 s | 0.030 s |
| 768 | 31.733 s | 30.806 s | 0.927 s | 0.120 s |
| 1024 | 56.573 s | 55.825 s | 0.748 s | 0.078 s |

512 is quoted as a range on purpose. Both sessions clear their own floor by an order of magnitude
and they still disagree by 0.177 s, because an A/A floor bounds contention inside a session, not
where the session itself sits.

The saving does not shrink as the win per block does: 512 residues goes from 32 row blocks to 11 and
1024 from 64 to 43, but 1024 has four times as many blocks to remove.

**Which models it reaches.** Boltz-2, BoltzGen and OpenFold3, whose pair track is 128 channels
wide and whose MSA track is 64. Protenix-v2 (256) and OpenDDE (384) share the same transition block
but keep the height they ship today, because the budget was fitted at 128 and over-predicts above
it: unbounded, it kills every seed of OpenDDE's structure and abag legs and the release gate's
capacity leg with an L1 circular-buffer clash, all of which pass with the flag off. Getting the
wider channels in needs a budget measured at that channel, not this one extrapolated.

AF2-IG's transition is a different block (ReLU, not SwiGLU) and OpenFold3's diffusion conditioning
has its own unchunked copy; neither changes. Wormhole is untouched: there the same budget only ever
shortens the block, which is what it already did.

## `TT_BIO_TRIATT_B8`

Default: off.

Triangle attention's interior in `bfloat8_b`: the fused qkv+gate matmul writes the block format and
the fused SDPA reads it. The pair representation, the stored weights and every residual update stay
bf16, so the region rounds once on the way out and the accumulator never sees block float.

Enabling it is worth **+0.2020 s a fold at 512 residues** (1.01422x, on a p300c at a pinned
1350 MHz, measured against the rest of the shipping default) and costs **0.37848 Å against the
0.60 Å bar**, which is well inside the variation the sampler already produces between seeds.

**It is off because the output depends on the core grid.** Block float shares one exponent across a
block of values, and the block boundaries follow how the work is split across cores, so the same
input folded on two different grids gives two different structures. The release gate's `l1-budget`
arm fails on exactly that with the flag on and passes with it off. Turn it on per run if you want
the second and your results do not need to match across parts; `TT_BIO_TRIATT_B8=0` is the default
and the way back.

Unmeasured: 298 residues, and any combination with `TT_BIO_TRIATT_BIAS_B8`. The measurements and
the grid-dependence evidence are in `perf/c14_bfp8/compose_result.md`.

## `TT_BIO_TRIATT_BW_FUSED`

Default: on inside a BindCraft 2 round, off elsewhere.

Sends the BACKWARD of `autograd.triangle_attention` through a fused kernel that keeps the
attention scores in L1 and never writes them: 238.88 MB a call against the chunked-recompute
path's 9172.90 at 288 tokens. On a BindCraft 2 gradient round at 288 tokens it is served 108 times
a round and cuts the round's device seconds by 1.1398x.

Above 288 tokens the whole `[N, N]` float32 bias gradient no longer fits one core's L1, so on
Blackhole the kernel walks the query axis in chunks and keeps one chunk's rows of it. That serves
every call from 288 to 864 tokens (216 of 216 at 864, 432 of 432 at 544), bit-identical to the
whole-query kernel at 288, inside the gradient bar at every size. Measured on a p300c at AICLK
1350 against the chunked recompute: 1.17-1.32x a round at 544, 1.62-1.67x at 768, 1.44x at 800,
1.00x at 288. It moves no size ceiling. On Wormhole the chunked form has not been graded, so it
declines there above the size the whole query fits and the chunked recompute runs.

It only reaches a route that enters `autograd.triangle_attention` at all, which means the fused
SDPA or HiFi route. On the materialised path it fires zero times. Anything outside its shape gate
falls through to the chunked recompute rather than approximating, so the answer is the same
function at every shape it declines.

Outside a BindCraft 2 round it stays off. Graded at 1.0133-1.046x against a 1.1-1.2x bar on
dq/dk/dv/dbias, and 1.0511x on the composed round's float64 VJP against a 1.2x bar.
`TT_BIO_TRIATT_BW_FUSED=0` turns it off inside the round too.

## BindCraft 2 `memory=`

Not an environment variable: an argument of `bindcraft2.predictor` and
`bindcraft2.campaign_predictor`. Default `'auto'`.

It picks, per token axis, the cheapest of three ways to hold the Evoformer's activations between
the forward and the backward: `fast` (checkpoint each block), `lean` (also each residual step
inside a block, so every block's forward runs once more a round) and `offload` (`lean`, with the
pinned block inputs in host memory between forward and backward). `auto` keeps every fold that
fits in `fast` there and moves to a slower mode only where `fast` would refuse, from a measured
ladder per board: on a p150a it runs `fast` up to 672 tokens and `lean` from 704 to 864, and on
one Wormhole Galaxy chip it is what carries 544 to 864 tokens. The modes compute the same values: `lean` and `offload` are bit-identical to each
other and 0.0035 rel L2 from `fast`. On Wormhole, `lean` costs 10.6 % a round at a fixed axis and
`offload` rounds run 3.3-4.4x the 512-token `fast` round at 768 to 864 tokens, AICLK 1000. 896
refuses before the first round.

`memory='fast'` turns the slower modes off: a fold too large for `fast` then refuses instead of
running slower. Full measurements:
[docs/bindcraft2.md](bindcraft2.md#large-complexes-the-memory-modes).

## `TT_BIO_TRIATT_DIVIDING_K`

Default: on.

The fused HiFi triangle attention builds its k ladder from chunk widths that divide the padded
length. When none of the shipped widths divides it, the route declines every rung and the fold
falls back to the materialised fp32 softmax. This flag derives the ladder from the divisors
instead, so a legal k exists at those lengths.

It only ever adds a rung where the route serves nothing today, so no length that folds on the
fused path now can have its pick moved. Replayed over all 48 tile-aligned lengths from 32 to 1536
at `openfold3.trunk`, ten serve nothing and this opens exactly one of them: 832. OpenFold3 now
buckets its pair axis to 32, so the other 32 * p lengths are reachable too;
`TT_BIO_TRIATT_HIFI_PAD_UP` serves those.

**Speed: 1.6351x on OpenFold3 at 832 tokens**, +50.999 s. Arms interleaved in one process on a
p300c, AICLK sampled during every leg at 1350 MHz, against an A/A floor of 1.306 s. The effect is
39x that floor.

**Accuracy at 832**, on tiled CDK2 at MSA depth 513, where OpenFold3 is confident (pLDDT 0.806).
CA RMSD, Kabsch, float64, over 832 CA:

| arm | CA RMSD |
| --- | --- |
| control, same arm rerun | 0.000000 A |
| this flag | 0.450148 A |
| a different seed | 1.974757 A |
| flag and seed together | 1.948630 A |

The move is inside the 0.60 A bar and 4.4x smaller than re-seeding, and both confidence heads
move the favourable way (+0.000551 pLDDT, +0.000647 pTM). At 288, the other length anyone has run,
pair rel_l2 against a float64 reference improves 0.021702 to 0.018661.

`TT_BIO_TRIATT_DIVIDING_K=0` is the way back, read live rather than at import so one process can
A/B both arms.

## `TT_BIO_TRIATT_FUSED_QKVG`

Default: on.

A triangle attention's query, key, value and gate projections all read the same normed pair tensor,
and the two matrix multiplies that produce them each read all of it: 67.1 MB at 512 residues, twice
per attention. Concatenating the gate's weight onto the qkv weight makes the four results four
output chunks of one pass, and the second read never happens.

**Accuracy: identical.** Every output tile is its own contraction over the whole contraction axis in
both forms, so which buffer a tile lands in cannot change its value. Measured rather than argued:
`torch.equal` at max abs 0.0 on the block outputs (`perf/b2z2_byte_round2/probe_opclass.py`), with
negative controls that move the block by 1.74 and 0.49. The 512, 640, 1024 and 1536 residue folds of
`perf/b2z2_size_ladder/` write byte-identical structures with the flag on and off.

**Speed:** this flag, `TT_BIO_TRIATT_FUSED_QKVGB` and `TT_BIO_TRIMUL_FUSED_GOUT` together are **1.01573x on the Blackhole benchmark cell**
(19.336 s to 19.0375 s, eight folds per arm interleaved ABBA, all eight pairs positive, worst-case
A/A floor 1.00805x, `perf/b2z2_trunk_ship/cell_512_qb2_c0.json`) and 1.02648x on a Wormhole fold. A
second session on the same box and fixture read 1.01947x over twelve folds against a floor of
1.01075x (`perf/b2z2_trunk_ship/cell_512_guard_benchlock_clean.json`); the number quoted here is the
lower of the two.
On the pairformer block alone, where the reads are, they are 1.05106x. This flag is the smallest of
the three on its own and does not separate from the floor of a single fold.

tt-bio asks for the fused pass only where it declines to exactly what the separate calls would have
declined to: one tile per head, no zero padding in the head channels, the same weight dtype, and no
bias on the gate or the output projection. A model that biases either keeps the separate
projections. No model name appears in the condition.

## `TT_BIO_TRIATT_FUSED_QKVGB`

Default: on.

The pair-bias projection is the third reader of that same normed tensor, one tile wide against the
other four's four. This flag puts it in the pass as well, so the tensor is read once per triangle
attention instead of three times. It needs `TT_BIO_TRIATT_FUSED_QKVG`; with that off it does
nothing.

**Accuracy: identical.** One detail matters for reproducing the shipped numbers: the bias
weight is taken back off the device rather than rebuilt, because it has already been scaled there in
bfloat16, and scaling in float32 and converting afterwards rounds differently. bfloat16 to float and
back is exact, so the fused form carries the same bits.

**One shape is declined to keep it that way: chains that fit in a single tile.** The tile-level
argument the other two flags rest on does not carry this one all the way down. The bias projection is
one tile wide against the other four's four, so adding it widens the fused result, and on a small
enough target that changes how the multiply is split across cores and therefore the order its partial
sums are added. Measured, not argued. Synthetic chains folded with this flag as the only difference
(`perf/b2z2_trunk_ship/qkvgb_boundary.json`) are byte-identical at 48, 64, 96 and 112 residues and
differ at 32; `trpcage_no_msa` at 20 residues differs too, and
`perf/b2z2_trunk_ship/trpcage_pairs.json` puts that change on this flag alone, the other two
reproducing the flags-off structure exactly. `prot_no_msa` at 117, `hsa_no_msa` at 585 and the 512 to
1536 residue ladder are all clean. A chain of 32 residues or fewer occupies one tile on the token
axis and 48 pads to two, so the fused pass declines whenever either axis of the pair tensor is a
single tile, and the bias projection runs where it always ran.

With that guard the flag is bit-identical at every length: 20, 32, 48 and 64 residues all write
byte-identical structures with the three flags on and off
(`perf/b2z2_trunk_ship/qkvgb_guard_verify.json`). It costs nothing, because a chain that short is not
a performance case. It declines nothing at the sizes that matter: 560 fused calls served per fold at
512, 1024 and 1536 residues, with the same structure digests the tree wrote before the guard existed
(`perf/b2z2_trunk_ship/cell_512_guard_qb2_c0.json`,
`perf/b2z2_size_ladder/out/ladder_guard_bh_c0.json`).

**Speed:** the largest of the three. 1.02491x on the pairformer block by itself.

## `TT_BIO_TRIATT_QK_MASK_PRELOAD`

Default: on.

The fused triangle-attention kernel adds the pair bias to the scores in a separate pass over the
score block, after the QK^T matmul has written it. With this flag the kernel copies the bias tiles
into the destination registers first and lets the matmul accumulate onto them, so the scores are
written once with the bias already in. That pass cost 1.17 ms of a 6.55 ms op in an ablation. Only
the persistent-mask kernel uses it, and only where the add would run on every key chunk.

**Accuracy: not bit-exact.** The score plus bias is rounded once instead of twice. Graded on the
fold together with the other attention flags, see `TT_BIO_ATOM_SUPERSET_WINDOW`.

**Speed:** 8-11 % on the triangle-attention op on Wormhole at every length measured (736 tokens:
17.17 to 15.85 ms at 1000 MHz); nothing measurable on Blackhole (6.85 vs 6.83 ms).

## `TT_BIO_TRIMUL_FUSED_GOUT`

Default: on.

A triangle multiplication reads its normed input twice: once for the four-way input projection and
once for the output gate at the tail, with no write in between. Concatenating the gate's weight onto
the input projection's makes the gate a second destination of one pass.

**Accuracy: identical.** Same tile-level argument as `TT_BIO_TRIATT_FUSED_QKVG`, and the same `torch.equal` at max abs 0.0
at the production shape. The op class does change, so it was measured and not assumed.

**Speed:** 1.01080x on the pairformer block by itself.

**It switches itself off on large targets, and that is not a failure.** Above roughly 1024 residues
the input projection runs a multi-iteration channel loop, which would recompute the gate once per
iteration, so the fused form declines and the tail runs the projection it always ran. The structure
is byte-identical either way, verified at 1536 residues where the gate declines all 560 calls. It
also declines under `--fast`, on row-blocked norms, on an L1 channel path, and where the output gate
carries a bias.

### What the three fused-read flags cost in memory

A fused pass holds a wider weight and a wider result, so peak device memory rises. Measured with all
three on against all three off, one Blackhole processor of a p300c, `cdk2x2` at four sizes
(`perf/b2z2_size_ladder/out/ladder_bh_c2.json`):

| residues | peak, off | peak, on | delta |
|---|---|---|---|
| 512 | 1.678 GiB | 1.875 GiB | +11.7 % |
| 640 | 2.090 GiB | 2.264 GiB | +8.3 % |
| 1024 | 3.473 GiB | 3.918 GiB | +12.8 % |
| 1536 | 5.694 GiB | 7.259 GiB | +27.5 % |

The 1536 row is the two triangle-attention flags on their own, because the trimul gate has already
declined every call by then. The 1024 and 1536 sizes were re-run after the single-tile guard landed
(`perf/b2z2_size_ladder/out/ladder_guard_bh_c0.json`): same digests, same engagement, peak within
0.8 percentage points of the table. No size refused an allocation: 7.26 GiB is 22.8 % of the 31.87 GiB
part, and the smallest largest-contiguous free block per bank at the high-water mark is 3013 MiB
with the flags on against 3157 MiB without them.

## `TT_BIO_TRIATT_GATE_EPILOGUE`

Default: off.

Triangle attention ends with `o * sigmoid(g)`, and that multiply is its own op: it re-reads the
attention output and the gate from DRAM and writes the product back, 268.4 MB a Pairformer block,
3.34 % of the block's 8046.8 MB. Both operands first exist together inside the fused SDPA kernel, so
this flag folds the sigmoid and the multiply into that kernel's pack stage and deletes the op. The
gate's own producer cannot host it: `TT_BIO_TRIATT_FUSED_QKVG` writes q, k, v and the gate out of one
matmul, so an epilogue there would multiply by a tensor its own output is an input to.

**Accuracy: bit-exact either way.** `torch.equal` at max absolute difference 0.0 at the op, and over
a whole Pairformer block with its zero-initialised output projections refilled
(`perf/roof_gate_epilogue/block_ab_gated_512_qb2_c2.json`). At the fold, Boltz-2 at 298 and 512
residues writes byte-identical mmCIF with the flag on and off, 560 of 560 triangle-attention calls
served gated at each size with zero declines, and all-atom Kabsch reads 0.000000 Å against the 0.60 Å
bar. A configuration the gated kernel's CB set does not fit is declined and falls back to the
separate multiply, which is the same arithmetic in the same order.

**Speed: 0.99737x on the Pairformer block on Blackhole, which is why it is off.** One Blackhole
processor of a p300c, 512 residues, 11x10 grid, arms interleaved, 7 reps: 50.634 ms base against
50.768 ms gated, on a same-arm floor of 0.110 %. The op-level pair reads the other way and does not
carry: SDPA plus the separate multiply is 3.0202 ms against 3.0130 ms fused, a 0.24 % gain on a
0.089 % floor (`op_ab_gated_512_qb2_c2.json`). The block is the number that decides, and it is a
small loss.

**The bytes were not the cost.** The multiply this deletes already runs at 348.0 GB/s, 82 % of the
424.7 GB/s measured Blackhole DRAM roof, so there is little bandwidth left to recover. The sigmoid is
not free wherever it goes: bolted onto the multiply it costs 0.286 ms, and moved into the SDPA
kernel's reader it costs 0.851 ms against the 0.906 ms the deleted multiply was worth
(`sig_cost_512_qb2_c2.json`). A byte-count prediction scaled from the Wormhole ablation said 1.0132x
on Blackhole; the measurement came back below 1.0.

**Wormhole is not measured, and the arithmetic points the other way there.** Eliding the same two
multiplies on a Wormhole Galaxy card is worth 1.03200x a block, 1.0211x once the host kernel is
charged the 67.1 MB read it gains (`block_ablate_512_whglx_c2.json`). Wormhole's measured DRAM roof
is 227.5 GB/s against Blackhole's 424.7, so the deleted bytes are worth roughly 1.9x more there. The
block A/B has not been run on Wormhole with the kernel built, so the flag ships off on every card.
`perf/roof_gate_epilogue/FINDINGS.md` has the full record.

## `TT_BIO_TRIATT_HIFI_PAD_UP`

Default: on, with 2 tiles of headroom.

The fused HiFi triangle attention needs a q chunk and a k chunk that both divide the padded
sequence length and both fit L1. At a padded length of 32 x p for a prime p there is no such pair,
so the route declines every call and the fold falls back to the materialised fp32 softmax, which
holds the whole score tensor. On a Blackhole p150a that is the difference between a 544-token
BindCraft 2 round at 13.00 GB and 45.33 s and the same round at 25.75 GB and 67.27 s, and at 608
tokens the fallback is refused outright with 4.1 GB free.

This flag pads the sequence axis up to the next length that does serve, masks the added keys with
the same bias the ragged tail already uses and slices the rows back. It is the same attention over
the same keys: the added keys enter the softmax at the mask value and contribute nothing. It fires
only where the native ladder declines, only where the axis is already a tile multiple, and only
with a real bias to mask with. The value is how many 32-wide tiles it may add, so `2` allows
608 -> 640 and `0` turns it off.

**It is not BindCraft 2 only.** The function it sits in is the triangle attention every model in
the repo shares, and OpenFold3's trunk takes the fused HiFi route by default, so the pad-up serves
OpenFold3 too. Counted in the process that folds: at 544 and at 608 residues all **384** of
OpenFold3's trunk calls serve through the pad-up, and all 384 decline with the flag at 0.

**Boltz-2 is untouched, and structurally rather than luckily.** It reaches the fused arm only
behind `BOLTZ2_FP32_SOFTMAX`, which is off by default, and its counters read served 0 and declined
0 in the worker as well as in the parent at both 544 and 608, so the arm is never offered a call
at all rather than being offered one and refusing. On OpenFold3's trunk shape the route
changes at 9 lengths from 64 to 1536: 544, 608, 736, 928, 992, 1184, 1312, 1376 and 1504, the
32 x p lengths with p prime from 17 to 47. Every other length takes the same config either way.

**Accuracy, graded twice.** Against a torch float64 reference forward and VJP at heads 4 and
head_dim 32, which is the per-call shape both BindCraft 2 and OpenFold3 present, a padded rung's
relative L2 equals its natively-serving neighbour's to four decimals: 544 padded to 576 reads
0.02162 against 576 native's 0.02162, and 608 padded to 640 reads 0.02189 against 640 native's
0.02198. What is left is the kernel's own bf16 error, not the pad.

And on the structure OpenFold3 delivers, on human serum albumin at 585 residues, a 608 axis and an
MSA 1000 sequences deep where the model is confident (pLDDT 0.909). Superposed CA-RMSD over all
585 CA:

| pair | CA RMSD |
| --- | --- |
| this flag on against off, seed 1 | 0.029800 A |
| this flag on against off, seed 2 | 0.031200 A |
| a different seed, same arm | 0.832600 to 1.159100 A |

The arm swap is 20x inside the 0.60 A bar and 28x smaller than the smallest move re-seeding makes
on the same input, and it reproduces at a second seed to within 0.0014 A. pLDDT moves 0.909254 to
0.909316 between the arms, against 0.909254 to 0.913107 across three seeds of one arm.

**Speed: 1.30x on OpenFold3 at 608 tokens**, 66.5 and 68.4 s on against 86.6 and 88.7 s off, warm,
one fold at a time on a Blackhole p150a with the AICLK sampled from sysfs during every fold at a
median and maximum of 1350 MHz. The trunk phase alone is 21 s against 27 to 29 s. On BindCraft 2 it
is what raises the supported ceiling from 576 to 864 tokens; see
[docs/bindcraft2.md](bindcraft2.md). `perf/b2p_ship/hsa_ab/README.md` carries the OpenFold3 record,
one JSON per pid.

On a p300c chip the same comparison reads 1.548x at 544 (40.06 s against 25.88 s) and 1.642x at
608 (54.75 s against 33.34 s), AICLK median 1350 MHz on every leg, with the structure moving
0.0236 A at seed 0; `perf/b2p_padup/` carries it.

**One chip of a Wormhole Galaxy is not helped.** The forward serves 544 and 608 there with the
pad-up on, and 544 then refuses in the Evoformer backward instead, where the L1 gate is a size gate
rather than a divisor gate, so padding asks for more L1 and not less.

`TT_BIO_TRIATT_HIFI_PAD_UP=0` is the way back.

## `TT_BIO_TRIATT_NARROW_Q_FALLBACK`

Default: on.

Triangle attention's fused kernel hoists its mask fill, and one precondition is that the q_chunk
divides the padded length. When it does not, the call does not merely pay for the padding: it
declines the fused path and drops to the stock op. The production pick is a fixed 256, so any
padded length that is not a multiple of 256 is exposed. This flag offers the dividing chunks below
256 before falling back to the one that pads.

It is bounded to chunks of 128 and up, and that bound is the reason it ships on. The kernel
re-reads all of K and V once per q chunk, so a narrow chunk trades re-reads for keeping the fused
path. Every narrower candidate fits L1, so the widest offered one always wins, and how wide that
is comes down to the arithmetic of the padded length. Across the 37 tile-aligned lengths from 256
to 1536, twenty of them have no divisor near 256 and would fall to a chunk re-reading K and V
2.7x to 8x more. A padded 1184 is 32x37, whose only narrower divisor is 32. The bound declines all
of those, so they behave exactly as they did before, and it admits the lengths whose fallback sits
close to the shipped pick.

**Speed: 1.1005x on RoseTTAFold3 at 896 residues**, 104.00 s to 94.50 s, +9.5000 s. Two
interleaved pairs on a p300c, both arm orders, board-pair sibling card verified idle, AICLK
sampled during every leg at 1200 MHz or above, against an A/A floor of 0.635 %. The effect is
15.83x that floor and the arms do not overlap.

**Accuracy.** On RoseTTAFold3 it is bit for bit: one mmCIF digest across 12 legs at 896 residues
and 6 at 1088, both arms. The chunk width splits output rows and the online softmax reduces over
the key axis, so the attention itself rounds the same. On OpenBind it is not, because the
narrower chunk leaves enough L1 for one more call to take the persistent-mask kernel, which rounds
differently. On aminopeptidase N (PDB 3B34, 891 residues, padded to 896) at default settings over
three seeds, the flag moves the structure 0.061 to 0.081 A, under the 0.60 A bar and 11x under
the smallest seed-to-seed distance (0.82 A). Distance to the deposited structure changes by
-0.0001 A on average (0.60 to 0.79 A either way). `perf/land_standing/narrowq_openbind_pepn.md`
has the table.

Two models change output at 896 residues: RoseTTAFold3, which gets the speedup, and OpenBind,
which gets the precision change above. A size-ladder census with only this flag flipped leaves
Protenix-v2 at 896 and OpenBind at 640 identical. At 1088 the bound makes the flag a no-op.

`TT_BIO_TRIATT_NARROW_Q_FALLBACK=0` is the way back.

## `TT_BIO_TRIATT_SDPA_HIFI_AB`

Default: on for `openfold3.trunk`, off at every other site.

Triangle attention has two routes on Blackhole: a materialised chain that takes its softmax in
fp32, and the fused SDPA kernel. This one picks the fused kernel at HiFi4 with `math_approx` off
and `fp32_dest_acc` on, spanning the whole key length in a single k chunk so each row reduces in
the order the torch reference uses.

Per site rather than global, because the four OpenFold3 sites do not agree. The trunk alone scores
0.396 A closer to the deposited structure than the incumbent route; all four together land 0.676 A
further out. So the trunk ships on and msa, template and confidence ship off. Boltz-2
(`boltz2.trunk`) and RoseTTAFold3 (`rf3.tri_att`) build the same block and are untouched.

**Speed.** OpenFold3 at 512 residues: 34.138 s to 22.574 s, 1.5123x, on a p300c with the AICLK
sampled during every leg at a median of 1350 MHz, six legs interleaved in one process on one device
open, against A/A floors of 0.888 and 0.950 s. The win grows with length, 57.946 s to 33.501 s at
640 residues. Firing was counted rather than assumed on every leg: 384 calls served, 0 declined, 0
below the length floor.

**Accuracy.** Not bit-exact, and scored as a distribution against a distribution, because a single
pair of folds cannot separate a route change from a seed. At 298 residues over three matched seeds
the structure moves 2.5805 / 7.7510 / 8.0075 A CA against a six-pair main-against-main seed floor
of 5.2283 / 7.7276 / 9.7768 A: the median is exactly 1.00x the floor median and the worst case is
0.82x its maximum, so the route stays inside the variation re-seeding already produces. pLDDT moves
the same way, 0.50319 to 0.56667. The reduction order is what clears that bar. The same route
without it sits at 1.61x the floor median, which is why the two ship together.

**Where it does nothing.** The kernel declines what it cannot fit. At 1088 residues on a p300c it
declines all seven attempts, five on fill preconditions and two on the L1 budget, and the fold is
byte-identical to the incumbent route down to the CIF sha256. At 1024 it serves 384 and declines 5.
Both sizes fold with the backbone intact, 0 breaks and a CA-CA median of 3.77-3.81 A. Below 256
residues the kernel declines every call, so short targets are unaffected, including the release
gate 117-residue fixture: that arm is a no-regression result and not evidence for this route.

## `TT_BIO_TRIMUL_GP_BANK_SPLIT`

Default: on.

A triangle multiplication projects its gates and its values in one matmul, four column quarters
wide. The channel move that consumes that projection reads a value slice and its own gate slice back
to back and waits on both, and the slice offset is what picks the DRAM bank a read lands on.
Ordering the quarters by role, both gates and then both values, puts the two reads of every such
pair 8 tiles apart at the production width, and on Blackhole's 8 DRAM banks 8 tiles apart is the
same bank twice, so the pair serialises. This flag interleaves the roles instead and the pair lands
4 banks apart.

**Accuracy: identical.** The flag permutes which column a value is written to and read from and
changes no arithmetic, so the bar here is equality and not a band. `perf/k10_b1_permute/b1_equiv.py`
gets `torch.equal` on both triangle multiplications at 298, 320, 512 and 640 residues and at both
slice widths, 8 cells of 8, with a negative control that reads the new layout at the old offsets and
is required to differ. At the fold, one CIF digest across all sixteen folds of both arms at 512
residues (`2bc758a1fb24ef30`, pLDDT 0.864509) and one across all sixteen at 298 residues
(`de5d77b220e32a7a`, pLDDT 0.90916).

**Speed: 1.02287x on the fold** at 512 residues (17.7365 s to 17.3400 s, eight folds per arm
interleaved ABBA, all eight pairs positive), 1.5146x on the channel move itself.

It costs nothing at runtime: the weight is laid out once at load, and the reader takes the slice
offsets as arguments it was already taking.

**The gain is Blackhole's.** Wormhole has 12 DRAM banks, so the pair was never congruent there and
there is nothing to recover; the reorder is free on both. The same is true wherever the projection
runs a narrow slice, at 298 residues among others: the pair is two tiles apart in either order, and
those sizes read flat.

## `TT_BIO_TRIMUL_MASK_AFTER_MOVE`

Default: on.

Masking before the channel move puts a masked tensor in front of the fused reader, which cannot address it. Masking after the move is the same arithmetic and leaves the reader a shape it can take, so this is what lets `TT_BIO_REBLOCK_PERMUTE_GATED` reach a masked trimul at all.

**Accuracy: bit for bit.**

## `TT_BIO_TRIMUL_MASK_L1`

Default: on.

The triangle multiplication masks its pair input before the contraction. The mask is `[1, 1, L, L]`
against a `[1, C, L, L]` chunk, so the multiply broadcasts it along the channel axis, and a
broadcast operand is read once per channel block rather than once. This flag puts the mask in L1,
where those re-reads cost no DRAM traffic.

It is the best ratio of the two: the mask is 0.52 MB, and moving that much on chip removes a whole
pair tensor's worth of DRAM reads. It fits at every size tested, 298 through 1024 aa, so unlike
`TT_BIO_RESIDUAL_L1` it does not go dark on large targets.

**Accuracy: identical**, on the same evidence as [`TT_BIO_RESIDUAL_L1`](#tt_bio_residual_l1): `torch.equal` at the op with a
control that fires, and one digest per size at the fold.

**Speed: 1.01492x on the trunk, taken with `TT_BIO_RESIDUAL_L1`.** The two flags place three
tensors in the same block, so they were measured together rather than multiplied together. Eight
folds an arm at 512 aa, interleaved ABBA in one process on an idle box: the Pairformer block wall
goes 9.5479 s to 9.4087 s, all eight paired ratios positive, against a same-arm floor of 1.0042x.

On the whole fold that is **17.736 s to 17.6325 s, 0.1035 s, 1.00748x**. Read the block number
rather than this one. The levers act only in the trunk, the fold wall is dominated by 200
diffusion steps they never touch, and the fold wall's own same-arm floor at this size is 1.01483x,
which is wider than the effect. All eight paired folds still came out positive, so the direction
is not in doubt; the size of it is better read where it happens.

## `TT_BIO_TRIMUL_MM_TRANSPOSE`

Default: on.

A triangle multiplication moves its channels to the batch axis before the per-channel matmul, and
exactly one of the two operands wants the sequence axes the other way round. That cost a separate
transpose of a whole moved chunk, 67.1 MB at 512 residues, once per call. `ttnn.matmul` takes the
transpose itself through `transpose_a` / `transpose_b`, so the flag hands it over and the separate
op disappears.

**Accuracy: identical.** The structure digest is unchanged at 298 and 512 residues, and the matmul
is `torch.equal` at max abs 0.0 against transpose-then-matmul at the production shape on both
operands.

**Speed:** 1.00613x on the fold at 512 residues. The matmul itself gets slightly slower, because it
re-reads the operand tile-transposed; what goes is the whole extra program and 134.7 MB of
allocation per pairformer block.

**Smaller targets take it too, and there it is an op-level win.** A target small enough to keep the
moved chunk in L1 never had a separate transpose to delete: the channel move and the sequence swap
were one permute. Handing the swap to the matmul leaves only the channel move, which the
hand-written reblock kernel can do. At the 298-residue shape that is 0.3824 to 0.3486 ms with a
transposed second operand and 0.3684 to 0.3463 ms with a transposed first one, median of seven warm
calls, `torch.equal` at max abs 0.0 on both (`perf/util_op_deletes/mm_transpose_l1.json`). Across
the 2240 calls a 298-residue fold makes that is about 0.063 s, under what a fold A/B can resolve, so
it is quoted at the op and not at the wall. The structure does not move: `0cf1b879dca3c0d5` at
pLDDT 0.913091 in both arms.

**It switches itself off under `--fast`, and that is not a failure.** `--fast` hands the matmul
bfloat8_b operands, and a block-float tile shares one exponent per face, so transposing inside the
matmul re-quantises where transposing beforehand does not. It is worth 1.0868x there, but it stops
being an exact relayout, so it is declined rather than taken.

## `TT_BIO_TRIMUL_TAIL_F1`

Default: on.

The tail of a triangle multiplication is three ops over the same tensor: an output projection, a gate projection and the gate multiply. One kernel does all three.

**Accuracy: bit for bit**, `torch.equal` at 14 shapes from 32 to 1024 residues.

**Speed: 512 residues in 31.994 s against 32.329 s.** ESMFold2 and Protenix-v2 fire it. Boltz-2, OpenDDE and OpenFold3 never reach it, because their triangle multiplications are not 8 tiles deep and the kernel declines rather than running a shape it was not built for.

## `TT_BIO_TRIMUL_TAIL_F1_L1_OUT`

Default: on.

`TT_BIO_TRIMUL_TAIL_F1` above writes a product the gate multiply reads straight back. With that product in DRAM the fused tail adds a round trip at a 128-channel pair track instead of deleting one, so the fusion only pays once its output stays on chip.

**Accuracy: bit for bit.**

## `TT_BIO_UNFUSED_SILU`

Default: off.

`Transition` is the engine's shared SwiGLU block. Its first matmul can apply silu as a fused
activation, and on Blackhole that costs more than running silu as a separate op afterwards: 174.0
us per call against 83.7 us at the 298 aa pair shape. The fused path runs silu at half the rate the
standalone op reaches, so unfusing pays a full extra round trip through L1 and still wins. The
penalty is specific to silu. A fused relu costs 2.4 us more than its standalone form and a fused
gelu 141.3 us more, and the gap holds across eight matmul program configs.

**This one is off, and it is held off on accuracy rather than on speed.** Turning it on costs
Protenix-v2 0.05088 and 0.07210 CA-lDDT per domain against the experimental 1HCL on cdk2x2_512,
with the two arms fully rank-separated over four seeds: the worst flag-off fold scores 0.03905 and
0.05107 above the best flag-on one. Protenix-v2 also loses 1.11 A and 1.19 A of CA-RMSD against
1HCL. Boltz-2 (-0.00091 / -0.00265 over four seeds) and OpenFold3 (-0.01238 / -0.00251 over two)
are clean on the same fixture, so a Boltz-2-only screen clears this lever and Protenix-v2 does not.
Nor does a structural seed-floor reading see it: Protenix-v2 scatters widely on that fixture while
landing at the same quality every time, so only the comparison against the experimental answer
separates the arms. Do not reopen without a Protenix-v2 CA-lDDT-vs-1HCL re-score on the
configuration you want to ship.

**Speed, if you turn it on anyway: +0.2336 s at 512 aa** (95 % CI [+0.1024, +0.3648]) paired over
five interleaved reps at a forced 1350 MHz, against the same session's paired A/A floor of
+0.0324 s +/- 0.1087.

**It is also not bit-identical.** Unfusing applies silu to the bf16-packed matmul output instead of
to the fp32 accumulator, so a structure moves. On Boltz-2 that movement is small: with
`TT_BIO_DIT_COND_HOIST` the pair deviates 0.25705 A all-atom at 298 aa, inside that fixture's
0.35 A band and 0.321x its own seed floor. That reading is exactly the screen the Protenix-v2
result above proves blind, which is why it does not clear the flag.

Scope, because the flag's name does not say it: every model that builds `Transition` inherits this
default. That is Boltz-2, Protenix, OpenFold3's MSA embedder, and the pairformer and MSA stacks.
OpenFold3's diffusion stack has its own SwiGLU transition and AF2 its own ReLU transition, and
neither reads this flag.

## `TT_PROTENIX_CONF_DEVICE`

Default: on. `TT_PROTENIX_CONF_DEVICE=0` runs the heads on the host.

The sample-invariant pair base is built once on device and only the pae/pde/pLDDT logits come back, so per sample the host uploads coordinates rather than a pair tensor. On a many-sample run that is most of the confidence head's host time.

It matches the host path: on a 730-token Protenix-v2 fold on Wormhole every sample's pLDDT agrees to 1e-4 and pTM and ipTM to 2e-4, and a warm fold is 16 s shorter (368 against 384 s at 1000 MHz). On the 11-complex benchmark set at four seeds each, per-complex pLDDT moves by at most 0.003 and docking success is unchanged (33 of 44 both ways). Coordinates do not change either way. OpenDDE graded the same way: coordinates unchanged, CA-lDDT +0.0009 and docking success 25 to 26 of 44, and a warm 256-residue fold is 3.3 s shorter (81.5 against 84.8 s).

It feature-detects the ops it needs and stays off on a ttnn that lacks one, so setting it on an older runtime is a no-op rather than a crash.

## Idle host threads when a box is full

Not a flag of ours. `OMP_WAIT_POLICY`, `GOMP_SPINCOUNT` and `KMP_BLOCKTIME` are OpenMP's own, and
tt-bio fills them in for the per-card workers it spawns when, and only when, each worker is down to
two host threads. Set any of the three yourself and tt-bio leaves all three alone: they are one
setting spelled three ways, and `GOMP_SPINCOUNT=0` beside your `OMP_WAIT_POLICY=ACTIVE` would undo
it through the back door.

This covers every path that spawns one worker per card: `predict` across queued targets, ESMC
embeddings, and a BoltzGen design fanned out over a box of chips. They all take their worker
environment from one place, so the rule and the thread cap arrive together or not at all.

A fold's host work is small but constant, roughly 1.85 cores at 512 aa whether one fold is running
or thirty-two. Those threads spend most of their time waiting on the device, and OpenMP waits by
spinning. Spinning costs nothing while cores are spare and costs a core once they are not, so the
rule keys on the share each worker gets rather than on how many cards are busy: a small host with
four folds on eight threads is in the same regime as a full server with thirty-two on sixty-four.

Measured on a 32-chip Wormhole Galaxy (AMD EPYC 9354P, 32 cores / 64 threads), Boltz-2 512 aa, 200
sampling steps, 3 recycles, full MSA, one independent fold per chip, three timed folds per chip
after a barrier. Both arms of a row run back to back in one session, and each row is at the thread
share tt-bio itself hands that width:

| concurrent folds | threads each | spinning | parked | |
|---|---|---|---|---|
| 16 | 4 | 1131.6 folds/h | 1123.7 folds/h | 0.993x |
| 20 | 3 | 1344.8 folds/h | 1331.6 folds/h | 0.990x |
| 24 | 2 | 1534.7 folds/h | 1541.0 folds/h | 1.004x |
| 32 | 2 | 1789.4 folds/h | 1813.9 folds/h | **1.014x** |

Repeating the 32-fold row reads 1783.2 against 1811.4, so the same-configuration floor is 0.35 %
and the win at that width is real but small. The first three rows are why it is a rule and not a
default: with three threads a worker still has a spare core to absorb the spinning, and parking
costs more in wake latency than it saves. Every row is bit-exact, both arms writing one CIF digest.

A much larger number is easy to measure here and it belongs to the line above, not to this one. Give
each of 32 workers a fixed 8 threads on the same box and spinning collapses to 1311.2 folds/h while
parking holds 1839.5, a 1.40x. That is 256 threads of demand on 64, and the fix for it is the thread
cap tt-bio already applies by default, which on its own takes that configuration from 1311.2 to
1789.4. Parking is the last 1.4 %.
