# p2-alignment — the unaligned contracted axis: blast radius, mechanism, and what a fix costs

Phase 2, EXPERIMENT. Protenix-v2, trunk only, 298 aa (token axis 298 logical, padded to 320; c_z=256).
qb2 card 0, board 007 chip 0, ttnn 0.68.0. Ratios only, never campaign absolutes (charter §4.8).
Board-mate chip 1 (`protenix-trunk--p2-msa-structural`) carried **no live python process** at the
start of this pass or between runs (`ps aux` checked before the first probe and again between the
`sites` and `fill` runs), so nothing here is contaminated by a busy host path.

**Probes only. No production change: nothing under `tt_bio/` is touched by this leg.** Everything new
is in `perf/align/` on `wk/protenix-trunk--p2-alignment`. The block harness is taken unmodified from
`wk/protenix-trunk--trimul-rescore` (`--tokens 298` default, dropped rows recorded as `null`); I did
not rebuild it.

## Predictions (before measuring)

Committed and pushed in `4f46696d` before the device was opened. Instruments named per prediction.

**P1 — the four-arm A/B reproduces on my chip.** At a fixed padded shape `[1, 32, 320, 320]` with the
fold's own `_triangle_mul_program_config(10)` supplied on every arm, I predict 72.8 us for logical
320x320, 72.1 us for 298x320, 112.1 us for 320x298 and 113.5 us for 298x298, each within 5 % of
B2's figure, and a contracted-axis ratio of 1.56x. **Wrong if** the ratio lands outside 1.45x-1.65x,
or if the M-unaligned arm (298x320) is more than 3 % against the aligned arm's own figure.

**P2 — the blast radius is small, and the contraction is essentially all of it.** From the live-block
op capture the only contracting sites whose contracted axis is logically 298 are the triangle
contraction `matmul` (16 calls/block) and the `attn@v` `matmul` inside `AttentionPairBias`'s
fp32-softmax attention (1 call/block). Every other matmul and linear contracts over c=256, 384, 1024
or 1536 and over head_dim=32 — all tile-aligned, so I predict **no penalty, within 3 % against the aligned arm's own figure**, on the
K=256 pair-track projections. The one site the shape scan cannot see is
`ttnn.transformer.scaled_dot_product_attention`, which contracts over the key axis internally; that
axis is logically 298. I predict SDPA pays **1.00x-1.15x**, not 1.56x, because its flash kernel is
built around explicit per-chunk masks and `ceil(298/64) = ceil(320/64) = 5` k-chunks either way.
**Total blast radius predicted: 360 ms/fold.** **Wrong if** the measured total exceeds 500 ms/fold or
falls under 300 ms/fold, and badly wrong if SDPA's logical-298 arm is more than 15 % slower against its
logical-320 arm at a fixed padded shape.

**P3 — the mechanism is the reader, and the penalty is per output block, not per tile.** Both arms
compile the same reader/writer/compute kernel *names*; what differs is the compiled program, so an
empty-cache compile of the two arms yields the same kernel file names and at least one differing
binary. If the cost is the NCRISC in0/in1 reader masking and zero-filling the tail sub-tile of the
last K block per output block, it is one masked transaction per output block and therefore roughly
**constant in absolute us** as `in0_block_w` goes 10 -> 1 while the arm's total time rises ~2.4x: I
predict the penalty stays 35-45 us/call across that ladder. **Wrong if** the penalty tracks total
time as a constant *fraction* across the `in0_block_w` ladder (that puts the cost inside the compute
kernel), or if the arms compile different kernel names (that makes it program selection).
`TT_METAL_WATCHER` is not used: T4 rejected it (0 of 600 worker lines caught a live program across 10
dumps, and it inflates op time 9.9x).

**P4 — the aligned fill is bit-exact.** A zero tail contributes exactly 0.0 to an fp32 accumulator and
the K-tile count is 10 in both arms, so the accumulation order is unchanged. I predict `torch.equal`
returns **True** on the 298x298 valid region. **Wrong if** it returns False, in which case I report
RMSD and max abs deviation.

**P5 — the fill is not free, because the tail is not already zero.** `layer_norm` writes its bias into
the padded token rows, so the operands carry a non-zero tail at the point of the contraction and a
metadata-only relabel is not exact on its own. I predict the cheapest exact fill costs
**15-35 us/call** against **41 us/call** saved, so per-call placement is marginal-to-negative and the
fill has to live once, upstream. **Wrong if** a metadata-only reshape is both bit-exact and under
5 us/call. I also predict the alignment does **not** survive the block.

## Roofs, measured on this card

I measured every roof below myself, on this card, qb2 card 0, this pass, and did not inherit any of them (charter §4.1) —
`perf/align/a_probe.py roofs`, in the same process style as the arms scored against them. Grid
reported by the device at open: **11x10 = 110 cores**.

| roof | measured here this pass | how |
|---|---:|---|
| square compute, HiFi4 bf16, `fp32_dest_acc_en` + `packer_l1_acc`, DRAM out | **105.37 TFLOP/s** | 4096^3. Nothing below is scored against it — it needs a wide output and a large K (charter §4.6) |
| **K=320, output width nt=10, L1 in and L1 out** — the contraction's own class | **52.65 TFLOP/s** | 10240x320 @ 320x320, best of a 12-point search over grid x `in0_block_w`; 10x10 with `in0_block_w=2` |
| K=320, nt=10, DRAM out | 35.52 TFLOP/s | same shape, DRAM output — the same class costs 1.48x with the output in DRAM |
| DRAM write, unary writer | **251.1 GB/s** at 32 MB | L1 -> DRAM clone |
| DRAM combined read+write | **355.8 GB/s** at 32 MB | DRAM -> DRAM clone |
| machine balance, square | **296.1 FLOP/byte** | 105.37 / 355.8 |
| **machine balance, K=320-corrected, L1 out** | **148.0 FLOP/byte** | 52.65 / 355.8 |

A DRAM-read-only roof did not measure: a 32/16/8 MB L1 destination is refused by the bank manager
under this program's circular buffers, so the read leg is only available inside the 355.8 GB/s
combined figure. Nothing in this leg is scored against a read roof, so it does not bind the result.

The contraction moves **19.66 MB of L1** (3 x 6.554) for **2.097 GFLOP**, so its arithmetic intensity
is **106.7 FLOP/byte** on padded bytes. That is below the K=320-corrected machine balance of 148.0,
but the traffic is L1-resident, not DRAM, so neither DRAM roof binds it: **the binding roof is
compute at 52.65 TFLOP/s.** The aligned arm achieves 30.08 TFLOP/s = **57.1 % of that compute roof**;
the production, logically-298 arm achieves 18.98 TFLOP/s = **36.0 % of the same roof**.

**Cores engaged: 100 of 110, measured, on both arms.** `per_core_M = per_core_N = 1` at Mt=Nt=10, so
the 11th column receives no work: grid 11x10 and grid 10x10 measure 69.48 / 68.58 us aligned and
109.19 / 109.30 us unaligned, i.e. 1.3 % against the aligned arm's own figure and 0.1 % against the
unaligned one. Smaller grids (8x8, 5x5, 4x4) raise `per_core_M` and the circular buffers then exceed
L1, so the ladder cannot be extended downward at this shape. **The alignment penalty is not an
occupancy effect: both arms engage the same 100 cores.**

**Overlap.** The contraction is nearer **`compute + comm` than `max(compute, comm)`**: B2 measured
113.94 us in a real block against 113.51 us standalone, 0.4 % apart against the in-block figure, and my standalone arms reproduce
that regime (110.51 us for the same configuration). An op whose in-block time equals its standalone
time is not hiding behind a neighbour. Inside the op the operands are L1-resident, so there is no
DRAM leg to hide; the penalty this leg is about is an NCRISC-side cost that the compute engine waits
on, which is additive by construction.

## Experiments and verdicts

### E1 — the four-arm A/B, re-taken on my own chip (`perf/align/ab4_qb2c0.json`)

Padded shape held at `[1, 32, 320, 320]` on every arm, the fold's own program config
(`in0_block_w=10`, `per_core_M = per_core_N = 1`, grid 11x10) supplied on every arm, so the bytes,
tiles, grid and program config are identical and only the logical metadata differs.

| arm (logical M x K) | B2, qb2 card 0 | this pass | ratio vs aligned |
|---|---:|---:|---:|
| 320 x 320 | 72.78 us | **69.72 us** | 1.000 |
| 298 x 320 | 72.09 us | **69.86 us** | 1.002 |
| 320 x 298 | 112.14 us | **110.08 us** | 1.579 |
| 298 x 298 | 113.51 us | **110.51 us** | 1.585 |
| 298 x 298 x 298 (the fold's own output width too) | — | 109.08 us | 1.565 |

**P1 CONFIRMED.** The contracted-axis ratio is **1.585x** against a predicted 1.56x and a stated
band of 1.45x-1.65x. The M-unaligned arm is **+0.2 %** against the aligned arm, inside the 3 % against that figure I said
would falsify it. My absolutes run 3-4 % against B2's figures across all four arms, which is run-to-run
allocator state, not a disagreement: the ratio — the only thing this leg uses — agrees to 1.7 %.
Making the *output* width logically 298 as well changes nothing (109.08 against 110.51, 1.3 %).
**Only the logical length of the contracted axis matters.**

### E2 — the blast radius (`perf/align/a_scan.py`, `perf/align/sites_qb2c0.json`)

`a_scan.py` reads the contracted axis's logical length out of the live block capture rather than out
of the source. Of the 23 distinct contracting operand signatures in one real Pairformer block, **two**
have a logically-298 contracted axis:

| site | calls/block | contracted axis | ms/block |
|---|---:|---|---:|
| `matmul` triangle contraction @ `tenstorrent.py:1355` (ledger @1043) | 16 | **K = 298 logical, 320 padded** | 1.823 |
| `matmul` attn@v @ `tenstorrent.py:378` (`AttentionPairBias`, fp32-softmax path) | 1 | **K = 298 logical, 320 padded** | 0.026 |
| everything else — pair-track projections, transition linears, qkv, q@k | 200+ | K = 32 / 256 / 384 / 1024 / 1536, all aligned | — |

Plus one site the shape scan cannot see, because its contraction is internal:
`ttnn.transformer.scaled_dot_product_attention` @ `tenstorrent.py:1629` contracts over the key axis,
which is logically 298. A/B at a fixed padded `[298, 8, 320, 32]`, same program config both arms
(`q_chunk = k_chunk = 64`):

| arm | us/call | verdict |
|---|---:|---|
| SDPA, key axis logical 298 | 1643.93 | **CONFIRMED but small: 1.025x**, 40.38 us/call, and the aligned arm is doing 7.4 % more real attention work against the 298 arm |
| SDPA, key axis logical 320 | 1603.55 | |
| `attn@v` @378, K logical 298 | 94.98 | **CONFIRMED: 1.333x**, 23.74 us/call |
| `attn@v` @378, K logical 320 | 71.24 | |
| pair-track projection, K=256, rows logical 298 | 383.00 | **KILLED as a site: 1.003x.** An aligned contracted axis pays nothing even with an unaligned row axis — the control for E1 |
| pair-track projection, K=256, rows logical 320 | 381.96 | |
| `softmax` over a logically-298 last axis | 27.99 | CONFIRMED as a *class*: 1.18x, 4.26 us/call. A reduction over an unaligned axis pays too, not only a contraction |
| `softmax` over a logically-320 last axis | 23.73 | |

**P2 CONFIRMED on the total, wrong on one part.** I predicted 360 ms/fold and SDPA at 1.00x-1.15x.
SDPA's ratio is 1.025x, inside the predicted band — but SDPA is the largest row in the block, so a
2.5 % ratio against its own aligned arm is 42 ms/fold, which I had priced at ~0. The total is **399 ms/fold**, inside my stated
300-500 band.

### E3 — the mechanism: the NCRISC in0 reader, confirmed by a cold compile

Both arms compiled against a private, empty `TT_METAL_CACHE`, one call each, then the generated
kernel trees diffed (`/tmp/kc2_aligned` vs `/tmp/kc2_unaligned`).

- **11 of 11 program hashes are shared** between the arms. Same op, same program.
- Of the 13 compiled kernels, **12 are byte-identical (same hash)** — the in1 sender/receiver, the
  writer, and all three compute TRISCs.
- **Exactly one kernel differs**, and it is `reader_bmm_tile_layout_in0_sender_padding`, compiled on
  **NCRISC**: hash `18395651649201600694` aligned against `12710934434160243660` unaligned. The
  unaligned arm's `ncrisc.elf` is **382 116 B against 371 700 B, +2.8 %** — the padding path is
  compiled *in*, not selected at runtime.

**H1 (NCRISC in0 reader tail-fill) CONFIRMED at the compile level.** The kernel *name* is identical
across arms, as predicted, and the one binary that changes is the in0 reader on NCRISC. **H2 (the
cost is inside the compute kernel) is KILLED**: all three TRISC kernels are byte-identical between
arms, so the compute kernel cannot be the site. **H3 (ttnn selects a different program for an
unaligned K) is KILLED**: the program hashes are identical, and in any case the fold supplies an
explicit `MatmulMultiCoreReuseMultiCastProgramConfig`, so there is no selection to make.

**H4 — is the extra cost per output block, or per tile?** Two ladders, batch and `in0_block_w`:

| `in0_block_w` (K blocks) | aligned us | unaligned us | penalty us | ratio |
|---:|---:|---:|---:|---:|
| 1 (10 K blocks) | 193.79 | 263.68 | 69.89 | 1.361 |
| 2 (5) | 120.41 | 184.41 | 63.99 | 1.531 |
| 5 (2) | 81.98 | 127.07 | 45.09 | 1.550 |
| 10 (1, the fold's own) | 68.91 | 109.49 | **40.57** | 1.589 |

| batch (output block passes) | aligned us | unaligned us | penalty us | ratio |
|---:|---:|---:|---:|---:|
| 8 | 19.19 | 29.31 | 10.12 | 1.527 |
| 16 | 35.33 | 56.68 | 21.35 | 1.604 |
| 32 (the fold's own) | 69.39 | 109.44 | 40.05 | 1.577 |

**H4 is CONFIRMED in its per-output-block half and KILLED in its strict form.** The penalty scales
almost exactly linearly with the number of output block passes — 10.12 / 21.35 / 40.05 us across
batch 8/16/32 is 1.00 : 2.11 : 3.96 against an ideal 1 : 2 : 4 — so it is charged once per pass over
the K axis, not once per output tile and not once per op. But it is **not** constant in
`in0_block_w`: it rises from 40.57 us at one K block to 69.89 us at ten, a 1.72x range, where the
strict "only the last K block carries the tail" account predicts flat. Total time over the same
ladder rises 2.81x, so the penalty is sublinear in K-block count rather than proportional: the tail
sub-tile costs something in **every** K block, not only the last, and the marginal cost of each
extra block is about a third of the first one's. **My P3 prediction of 35-45 us across the ladder is
therefore WRONG at `in0_block_w = 1` (69.89 us)** and right at the fold's own configuration. The
correct statement for Phase 3: the cost is one masked-transaction penalty per K block per output
block pass, discounted after the first block.


### The four mechanism hypotheses, and the test that settles each

| # | mechanism, at RISC / CB / transaction level | kill test | verdict |
|---|---|---|---|
| H1 | the NCRISC in0/in1 reader masks and zero-fills the tail sub-tile of the last K block per output block, so the transaction count per output tile rises and the compute engine waits on the circular buffer | kill test: compile both arms cold and compare kernel names and per-RISC binaries. Killed if the compute TRISC kernels differ or the NCRISC reader does not | **CONFIRMED** -- 1 of 13 kernels differs and it is `reader_bmm_tile_layout_in0_sender_padding` on NCRISC, `ncrisc.elf` 382 116 B against 371 700 B, +2.8 % against the aligned arm's binary |
| H2 | the cost is inside the compute kernel: an unaligned K forces a masked unpack / dest-register path on every tile | kill test: compare the three compute TRISC binaries between arms. Killed if they are byte-identical | **KILLED** -- all 3 TRISC kernels share a hash across arms, while the penalty is 40.57 us/call at the fold's own config |
| H3 | ttnn dispatches a different program or a different matmul variant when the logical K is unaligned, so this is program selection and not padding-awareness | kill test: compare program hashes across arms with an explicit program config supplied. Killed if the 11 program hashes are identical | **KILLED** -- 11 of 11 program hashes shared, and the fold supplies an explicit `MatmulMultiCoreReuseMultiCastProgramConfig`, so there is no selection left to make |
| H4 | the masked transaction is charged once per output block pass and only in the last K block, so the penalty is flat in `in0_block_w` | kill test: the `in0_block_w` and batch ladders. Killed in its strict form if the penalty moves with `in0_block_w` | **CONFIRMED per output block pass, KILLED as last-block-only** -- linear in batch (10.12 / 21.35 / 40.05 us over 8 / 16 / 32) but 40.57 -> 69.89 us over `in0_block_w` 10 -> 1 |


### E6 — the widen is unreachable from Python in ttnn 0.68, and the guard has an address

Three routes, all at `[1, 32, 298, 298]` L1 with padding the tensor already owns:

| route | result |
|---|---|
| `ttnn.experimental.view(x, (1, 32, 320, 320))` — documented as "a 0 cost view operation" | `TT_FATAL ... new_volume == old_volume` |
| `ttnn.reshape(x, (1, 32, 320, 320), pad_value=0.0)` | same `TT_FATAL`, same line |
| `ttnn.reshape(x, (1, 32, 320, 320))` | same `TT_FATAL`, same line |

All three land on **one guard**: `ttnn/cpp/ttnn/operations/data_movement/reshape_view/reshape_common.cpp:50`.
`view` advertises a zero-cost path but shares the validation, and its own documented conditions rule
this case out anyway — the last dimension is exactly what has to change. There is also no escape
hatch below the op layer: a device `ttnn.Tensor` in this wheel exposes no `buffer`, `storage` or
`device_buffer` attribute, `ttnn.TensorSpec` has no Python-reachable constructor signature, and
`x.spec` carries no writable layout, so a tensor cannot be re-wrapped around its own allocation with
a wider logical shape. **The lever is blocked on a C++ change, not on a cost.**

### E7 — the NCRISC reader really is zero-filling, measured

If the reader masks the tail, a logical-298 contraction must give the same answer whatever sits in
the padding. Polluted arm: both operands passed through `layer_norm` with `beta = 7.0`, which is how
the fold pollutes them — the norm of an all-zero padded row is `beta`, not 0. Zeroed arm: the same
polluted operands multiplied by a mask whose own padding is 0, which an eltwise op propagates into
the tail because it works on whole tiles.

```
logical-298 result, polluted tail (beta = 7.0) vs zeroed tail:
torch.equal = True    max_abs = 0.000e+00    0 of 2 841 728 elements differ
```

**CONFIRMED, and it closes the mechanism loop.** The result is independent of the tail, so the
padding reader is zero-filling it on **every call**. The 40.79 us/call the alignment costs is the
hardware doing, 8384 times per fold, the same zero-fill that one mask multiply does once per call.
That is the Phase-3 argument in one line: the work is not needed, it is re-done.

### E8 — the fill floor, and the net prize

The tail only has to be zero on **one** operand: the contraction sums over k, so a zero factor kills
the term whatever the other operand holds.

| fill | us/call |
|---|---:|
| `ttnn.multiply_` in place, one operand, 2-D mask, L1 | **22.51** |
| `ttnn.multiply` out of place, one operand, L1 | 22.60 |
| `ttnn.multiply_` in place, rank-1 mask | 27.87 |
| `ttnn.pad` on one operand (E5) | 55.17 |

**22.51 us/call is the floor**, against **40.79 us/call** saved. So with a free widen the contraction
row nets **18.28 us/call = 153.3 ms/fold, bit-exact**. Production already has the slot: the trimul
calls `ttnn.multiply_(a_chunk, mask_u)` at `tenstorrent.py:1347` and simply never reaches it, because
the trunk passes no mask (E9).

### E9 — the open item from last pass, settled: the fold passes no mask

`self.PF(s, z3)` at `protenix.py:2223` is called with two arguments, so `mask` defaults to `None`
through `Pairformer` and `PairformerLayer` into `TriangleMultiplication.__call__`, and the mask
multiply at `tenstorrent.py:1347` never executes. The live block capture agrees: **0 records at site
1347** across 272 ops. **KILLED**, and it is the unfavourable answer — the tail is not zeroed for
free, so the 22.51 us/call in E8 is a real cost the fix has to pay.

### E4 — parity (`perf/align/exact_qb2c0.json`)

Production arm: operands built logically 298 from the fold's own data, `[1, 32, 298, 298]`. Aligned
arm: the same data zero-filled to `[1, 32, 320, 320]` on the host, so the tail is exactly 0.0. Same
program config, same L1 in and L1 out, output compared over the 298x298 valid region.

```
torch.equal = True    max_abs = 0.000e+00    rmsd = 0.000e+00    0 of 2 841 728 elements differ
```

**P4 CONFIRMED.** The aligned fill is **bit-exact**, not "should be bit-exact". A zero tail adds
exactly 0.0 into the fp32 dest accumulator and the K-tile count is 10 either way, so the accumulation
order is unchanged. This is the property that puts a ~400 ms/fold lever in the safe pile.

### E5 — where the fill can live, and what it costs (`perf/align/fill2_qb2c0.json`)

| placement | measured cost | verdict |
|---|---:|---|
| **metadata-only relabel**, `ttnn.reshape` to the padded shape | **not available**: `TT_FATAL ... new_volume == old_volume` (`reshape_common.cpp:50`) | ttnn 0.68 has no zero-cost logical-shape widen. This is the one placement that would win and it needs a Phase-3 API |
| `ttnn.pad` on one contraction operand, L1, 6.55 MB | **55.17 us/call** | **KILLED.** It costs more than the 40.79 us/call it saves. End to end: production 109.53 us against pad-both-then-contract 176.77 us = **1.614x slower** |
| `ttnn.pad` on the pair tensor `z` once per trimul call, DRAM 48.8 MB | **332.16 us/call** | **KILLED.** 2 trimul calls/block = 664 us/block against 653 us/block saved by 16 aligned contractions |
| `z` built logically 320 upstream, so no per-block fill at all | no fill cost, but dim1 becomes 320 *real* rows: the pair-track projection goes **382.53 -> 397.79 us, +4.0 %** | **KILLED as stated.** +4.0 % on the ~13.1 s/fold of pair-shaped trunk work is ~525 ms/fold against 399 saved |

**P5 CONFIRMED in its conclusion and wrong in its number.** I predicted the cheapest exact fill at
15-35 us/call; the cheapest one that exists is 55.17 us/call, so the per-call placement is not
marginal, it is straightforwardly negative. And the metadata relabel I said would be the win is not
merely non-free — **it does not exist in ttnn 0.68 at all**, which is a sharper result than "it costs
too much".

**Does the alignment survive the block? No.** Traced through `TriangleMultiplication.__call__`: the
contraction's output is transposed twice and fed to `_trimul_out_proj`, which writes back into `z` at
logical 298, and the next block's `layer_norm` re-enters at 298. So a fill applied inside one trimul
call is gone by the next one and has to be re-applied at each of the 2 trimul calls per block, which
is exactly the placement the pricing above kills.

**Settled this pass, and the answer is the unfavourable one.** Exactness needs a zero tail on only
*one* of the two operands. The trimul already runs `ttnn.multiply_(a_chunk, mask_u)` when a mask is
present, so if the fold passed a mask the tail would be zero for free and the fix would collapse to
the relabel. It does not: `self.PF(s, z3)` passes no mask (E9), and the live block capture records 0
calls at that site. Phase 3 needs **both** a zeroing op (22.51 us/call, E8) and the widen (E6).

## ms/fold at stake, after this pass

Conversion **x524** per charter §4.9 for sites shared with the MSA stack (480 `pf_stack` + 40
`trunk_msa` + 4 confidence). Call counts are the ones I counted in the live block capture, stated
beside each row.

| site | calls/block | calls/fold (x524) | penalty us/call, measured | ms/fold |
|---|---:|---:|---:|---:|
| `matmul` triangle contraction @1355 (ledger @1043) | 16 | 8384 | 40.79 | **342.0** |
| `scaled_dot_product_attention` @1629 | 2 | 1048 | 40.38 | **42.3** |
| `matmul` attn@v @378 (`AttentionPairBias`) | 1 | 524 | 23.74 | **12.4** |
| `softmax` over an unaligned axis | 1 | 524 | 4.26 | **2.2** |
| **blast radius, total** | | | | **398.9 ms/fold** |

**The answer to deliverable 1 is 399 ms/fold**, against the 345 ms/fold the org was carrying for the
contraction alone. The contraction is **85.7 % of the blast-radius total**; the other three sites together
are 56.9 ms/fold. The upper bound on the SDPA row is 69.6 ms/fold if the 1.025x ratio is applied to
the in-block figure of 5.408 ms/block instead of using the directly measured 40.38 us/call delta,
which would put the total at 426 ms/fold; I report the directly measured delta as the headline and
the ratio-scaled figure as the bound, because my SDPA arms ran without the attention bias the fold
carries.

**Net of the fill, the contraction row is worth 153.3 ms/fold** — 40.79 us/call saved against the
22.51 us/call floor for zeroing one operand's tail (E8), 18.28 us/call at x524. The gross 342.0 is
the ceiling, reachable only if the zeroing is folded into an eltwise pass the trimul already makes.

Ranked for Phase 3: **one lever, 399 ms/fold gross and 153.3 ms/fold net at today's fill cost,
bit-exact, blocked on a single guard.** Every placement ttnn 0.68 can express either costs more than
it saves (E5) or is refused outright (E6). The ask is narrow and it is not a cost problem: relax
`new_volume == old_volume` at `reshape_view/reshape_common.cpp:50` for the case where the new logical
shape fits inside the padded shape the tensor already owns, so a widen becomes the metadata change it
physically is.

## Parity

`torch.equal` on the contraction's output, aligned-fill arm against production arm, at the fold's own
shape: **True**, 0 of 2 841 728 elements differ, max abs deviation 0.000e+00, RMSD 0.000e+00. The
change is arithmetically inert, as the zero-tail argument says it should be, and now measured rather
than argued. No other lever in this leg changes the arithmetic. Nothing was run against production
weights and no production code path was altered.

## Corrections to the inherited record

1. **The blast radius is 399 ms/fold, not 345.** The contraction is 85.7 % of that total. Three further
   sites pay the same defect: SDPA @1629 (42.3), `attn@v` @378 (12.4) and `softmax` (2.2 ms/fold).
2. **It is not a matmul-only defect.** `softmax` over a logically-298 axis pays 1.18x at a fixed
   padded shape. Any reduction over an unaligned axis is a candidate, which widens what Phase 3 has
   to look at even though it does not move the ranking much.
3. **The mechanism is settled and it is the reader.** `reader_bmm_tile_layout_in0_sender_padding` on
   NCRISC is the only one of 13 kernels that differs between the arms; the three compute TRISC
   kernels are byte-identical. The "it might be inside the compute kernel" branch that B2 left open
   is closed, and the Phase-3 design does not change.
4. **B2's kill test needed one correction.** The penalty is not one masked transaction per output
   block: it is charged in every K block, discounted after the first (40.57 us at 1 K block, 69.89 at
   10). It scales linearly with output block passes (batch 8/16/32 -> 1.00 : 2.11 : 3.96).
5. **My K=320 / nt=10 / L1-out roof is 52.65 TFLOP/s where B2 measured 59.28 on the same chip.** My
   search was 12 points (3 grids x 4 `in0_block_w`) and 8 of them were refused by the L1 circular
   buffer budget, so B2's wider search is the better number and the CTO should carry it. On my own
   roof the production contraction is at **36.0 % of the compute roof** and the aligned arm at
   57.1 %; on B2's roof it is 31.0 % of that roof. The percentage moves, the 1.585x does not, and only the ratio is
   load-bearing here.
6. **The obvious fix is not affordable in ttnn 0.68 and the reason is an API gap, not a cost.**
   `ttnn.reshape` refuses to widen a logical shape into padding the tensor already owns
   (`new_volume == old_volume`), so every expressible placement moves bytes: 55.17 us/call on an
   operand, 332.16 us/call on `z`, or +4.0 % on every pair-shaped op if `z` is built at 320 rows.
   Phase 3's task is the relabel, not the arithmetic.
7. **The open item from the first pass is closed, negatively.** The trunk passes no mask
   (`protenix.py:2223`), so the trimul's own tail-zeroing multiply never runs and the tail is not
   free. The fix needs a zeroing op as well as the widen.
8. **The reader is measurably zero-filling.** A logical-298 contraction returns a bit-identical
   result with `beta = 7.0` in the padded tail and with zeros there, so the 1.585x buys a zero-fill
   the model could do once instead of 8384 times per fold.
9. **The widen is refused by one guard, not by three APIs.** `ttnn.experimental.view`,
   `ttnn.reshape` and `ttnn.reshape(..., pad_value=)` all fail at
   `reshape_view/reshape_common.cpp:50`, and no Python-level buffer or `TensorSpec` reinterpret is
   exposed to work around it.
10. **Generalisation, recorded and not chased** (charter §1): this will hit any model contracting or
   reducing over a non-tile-multiple axis, including OpenDDE and OpenFold3 at their own token counts.
   One line, out of scope for this org.
