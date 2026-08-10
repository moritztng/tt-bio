# P1 — p2-attention, Phase 2 (EXPERIMENT)

Protenix-v2, trunk only, 298 aa. Pair tensor `[298, 320, 256]`, 8 heads x head_dim 32. qb1 card 2
(`TT_VISIBLE_DEVICES=2`), ttnn 0.67.4. Org: `state/orgs/protenix-trunk/`. Slug
`protenix-trunk--p2-attention`. Branch `wk/protenix-trunk--p2-attention`.

**No production change. Probes only** — everything below runs from `perf/p2_attention/`, which
re-issues the block's ops standalone at the shapes a live 298 aa fold issues them at, plus one real
`PairformerLayer` for the parity arm. Nothing under `tt_bio/` was touched.

Inherited from T1 (its Phase-1 doc, same card, same ttnn): the SDPA at 2993.6 ms/fold decomposes
into a saturated bias leg and a chunk-starved core leg, `nlp_create_qkv_heads` @1319 at 780.1
ms/fold and 95.9 % of the DRAM copy roof, the qkv projection at 906.8 ms/fold with a disputed write
denominator (Q12). This pass re-measures every one of those on this card before using it.

---

## Predictions (before measuring)

Committed and pushed to `wk/protenix-trunk--p2-attention` **before the device was opened**. Each
line says what would count as having been wrong. Conversion throughout is charter §4.9 **x524**
(the SDPA, the head split and the qkv projection are all shared with the MSA stack); I recount the
per-fold call number myself and report what I counted.

| # | prediction | number and unit | wrong if |
|---|---|---:|---|
| R1 | this card's DRAM read roof reproduces T1's | 400.8 GB/s +/- 5 % | outside 380-421 GB/s |
| R2 | this card's DRAM copy roof reproduces T1's | 410.2 GB/s +/- 5 % | outside 390-431 GB/s |
| R3 | the matmul-writer write roof with L1 operands reproduces B1's | 197.7 GB/s +/- 8 % | outside 182-214 GB/s |
| C1 | SDPA calls in a live 298 aa fold | exactly 1048 calls, 2 per block x 524 | any other integer |
| B1 | SDPA at the fold's true shape, production chunk 64, bias present | 2796.7 us/call +/- 10 % | outside 2517-3076 us |
| B2 | SDPA at chunk 320, bias present | 1728.9 us/call +/- 10 % | outside 1556-1902 us |
| B3 | the bias leg (bias on minus bias off) is flat in chunk size | 1157-1177 us at both chunk 64 and chunk 320, spread under 8 % | the two differ by more than 8 % |
| **L1a** | free L1 per core at the SDPA call **inside a real block** leaves room for W9's 204.8 KB resident bias slice | largest free L1 block >= 1.20 MB/core of the 1 532 448 B unreserved, i.e. 204.8 KB is <= 17 % of it | free L1 per core < 400 KB, which would forbid the design outright |
| **L1b** | the bias re-read is a DRAM round trip per (batch, head), not a kernel-internal broadcast, so moving the 1.638 MB mask into **interleaved L1** collapses the bias leg without any kernel being written | bias leg falls by >= 3x, to <= 390 us | the bias leg stays within 20 % of its DRAM value, or SDPA refuses an L1 mask |
| **L1c** | the bias-once prize equals the measured bias leg and no more | 1212.5 ms/fold +/- 10 % at x524 | the measured leg x my counted calls lands outside 1091-1334 ms/fold |
| **L2a** | chunk 320 is worth what T1 measured | 1119.1 ms/fold +/- 12 % | outside 985-1253 ms/fold |
| **L2b** | chunk 320 changes the online-softmax reduction order, so it is **not** bit-exact, but the block-level damage is small against bf16 activations | relative RMSD of the pair tensor after one `PairformerLayer` between 1e-4 and 5e-3 of its own rms | relative RMSD > 1e-2, or exactly 0.0 (which would mean my A/B never took effect) |
| **L3a** | permuting the **output columns** of the qkv weight reorders nothing inside a dot product | `torch.equal` True on the device output, 0 differing elements | one element differs |
| **L3b** | the head split cannot be removed by a weight permutation, because ttnn tile layout carries the head axis as tile **columns** while the SDPA wants it as a **batch** axis, and no column permutation moves data across the row axis | a reshape-only path does **not** reproduce `nlp_create_qkv_heads`; max abs difference O(1), not 0 | a reshape of the projection output matches `nlp_create_qkv_heads` bit-exactly, which would remove 780.1 ms/fold for free |
| **L3c** | the one layout that *does* emit `[298, 8, 320, 32]` from a matmul is a batched matmul at output width nt=1, and B1's qb1 table prices nt=1 at 7.95 TFLOP/s against 43.07 at nt=8, so it must lose | batched form >= 3x the cost of (projection + split) together, i.e. >= 4600 us/call | it comes within 1.2x, which would make L3 a GO by a route nobody proposed |
| **L3d** | the qkv projection's own baseline on this card, re-measured, reproduces T1's | 865.3 us/call +/- 10 % | outside 779-952 us |
| U1 | the SDPA engages most of the grid: T1's core-equivalent count from a grid ladder | 58.2 core-equivalents of 110, +/- 15 % | outside 49-67 |
| O1 | the SDPA's two legs are both DRAM-bound and therefore **additive**, not overlapped | (bias on) - (bias off) reproduces the bias leg to within 8 % at both chunk sizes, i.e. total ~ compute + comm not max() | the bias leg measured by subtraction differs from the standalone bias traffic time by more than 25 %, which would mean the legs partly hide behind each other |

The +/- 5 %, 8 %, 10 %, 12 % and 15 % margins above are of the predicted figure in that row; L1a's
17 % is of the per-core L1 budget and O1's 25 % is of the standalone bias figure.

**Falsifiable mechanism hypotheses** (each names the mechanism at the level the hardware works at,
and the measurement that kills it):

- **HA — NOC/DRAM transaction locality.** The 488.24 MB bias read is the same 1.638 MB buffer
  fetched over the NOC from DRAM once per (batch, head) because the mask reader's circular buffer is
  refilled per chunk-pair and holds no cross-batch state. **Falsified if** placing the identical mask
  in interleaved L1 leaves the bias leg within 20 % of its DRAM figure — that would mean the cost is
  transaction issue rate or reader occupancy, not the DRAM path.

- **HB — compute kernel phase, not circular-buffer depth.** The chunk-64 penalty is the
  online-softmax rescale executed once per k-chunk plus 25 chunk-pair loop iterations per
  (batch, head) occupying the compute kernel phase. **Killed if** the core leg (bias absent) does not
  fall by at least 2.5x between chunk 64 and chunk 320, or if it falls by the same factor with the
  bias present (which would make the two legs coupled, not independent).

- **HC — a strided tile gather.** `nlp_create_qkv_heads` is 10 tile transactions per (batch, head) at
  a stride of 24 tiles through the projection output's tile grid; it computes nothing and it is at
  the DRAM copy roof, so it is removable but not tunable. **Ruled out by** any reshape-only path
  whose output equals `nlp_create_qkv_heads`'s bit for bit.

- **HD — bandwidth, not issue rate.** If HA holds, the L1-mask win should track the L1/DRAM
  bandwidth ratio rather than saturating at some fixed reader occupancy. **Falsified if** the bias
  leg with an L1 mask lands at a fixed floor independent of how many bytes it re-reads — testable by
  halving the head count and checking the leg halves.

---

## Roofs, measured on this card

**qb1 card 2, `TT_VISIBLE_DEVICES=2`, opened as PCIe id 3, KMD 2.8.0, firmware 19.8.1,
`compute_with_storage_grid_size = 13x10`, `CORE_GRID_MAIN = 11x10`. Every roof below I re-measured
on this card this pass; none of them is inherited** (charter §4.1, and T1's figures are on this same
card one pass earlier, which is why they are quoted beside mine rather than used).

| roof | how I took it | this pass | T1 / B1, same card, last pass |
|---|---|---:|---:|
| DRAM read | `clone` DRAM -> L1, 128 MB, both-side sync | **396.3 GB/s** | 400.8 |
| DRAM copy (read + write) | `clone` DRAM -> DRAM, 128 MB | **407.7 GB/s** | 410.2 |
| `clone` write | `clone` L1 -> DRAM, 128 MB | **268.3 GB/s** | — |
| matmul-writer write | `matmul` K=32, **both operands in L1**, output DRAM (B1's rule) | **201.2 GB/s** | 197.7 |
| compute, square 4096, DRAM out | HiFi4, `fp32_dest_acc_en`, `packer_l1_acc` | **128.08 TFLOP/s** | — |
| machine balance | compute roof / read roof | **323.2 FLOP/byte** | 337.9 |
| L1 per bank | from the allocator's own refusal message | **1 461 760 B**, 130 banks = 190.0 MB | 1 532 448 unreserved |

Read and copy land 1.1 % and 0.6 % low against T1's figures for the same quantity, and the matmul
writer 1.8 % high against B1's. Nothing in the inherited roof set needs correcting.

**Where the three ops sit on the balance.** The SDPA does 31.25 GFLOP against 683.53 MB moved =
**45.7 FLOP/byte**, far on the memory side of this card's 323.2. `nlp_create_qkv_heads` computes
nothing at all: **0 FLOP/byte**, pure layout. The qkv projection does 75.0 GFLOP against 195.29 MB =
**384.0 FLOP/byte**, nominally on the compute side — but it reaches 43.09 TFLOP/s, which is 33.6 %
of the square compute roof above and **85.2 % of the K-corrected writer ceiling** that B1's
operational rule gives for a DRAM-output matmul below K ~ 550 (201.2 GB/s x K/1000 = 51.5 TFLOP/s at
K=256). Its binding roof is the matmul writer's DRAM write path, not compute.

---

## Experiments and verdicts

### L1 — the bias-once SDPA

**The bias leg reproduces, and it is bandwidth on the DRAM path.** Three independent measurements of
(bias present) - (bias absent) at the fold's true shape: **1123.2 us** at chunk 64, **1187.9 us** at
chunk 320, **1156.7 us** at chunk 64 on a second run. Mean 1155.9 us, spread 2.8 % of that mean figure.
Prediction B3 said flat within 8 % — **CONFIRMED**, and T1's 1157-1177 us band reproduces.

The head sweep is what turns "flat" into a mechanism. Halving and quartering the head count moves
the leg with the bytes and not with anything else:

| heads | bias bytes re-read | bias leg | rate | verdict |
|---:|---:|---:|---:|---|
| 2 | 122.1 MB | 307.5 us | 397.0 GB/s | scales |
| 4 | 244.1 MB | 555.5 us | 439.4 GB/s | scales |
| 8 (production) | 488.2 MB | 1156.7 us | 422.1 GB/s | **HD CONFIRMED** — 3.76x the leg for 4.0x the bytes |

At 397.0-439.4 GB/s the leg is at **100.2-110.9 % of this card's own read roof (396.3 GB/s)**, so it is
saturated; a pure-read stream beating a clone-derived read roof by ~7 % is the instrument, not a
violated limit (the clone that sets the roof also writes). **HA is CONFIRMED by byte-scaling rather
than by the route I planned**: a fixed reader-occupancy or transaction-issue floor would not have
quartered when the bytes quartered.

**L1b is WRONG, and the way it is wrong is the finding of this pass.** I predicted that moving the
1.638 MB mask into interleaved L1 would collapse the leg with no kernel written. The op refuses:

```
TT_FATAL: When mask is provided to SDPA, it must be in DRAM   (assert.hpp:104)
```

at both chunk 64 and chunk 320. **There is no memory-config route to bias-once.** Any version of
W9's design has to be a kernel. That is a hard call-signature constraint and it was not in the org's
record.

**L1a — capacity, measured inside a real block. CONFIRMED, with room to spare.** This ttnn's
`MeshDevice` exposes no allocator-statistics accessor at all (`get_max_worker_l1_unreserved_size`,
`l1_size_per_core` and every `alloc`/`mem` attribute are absent), so I measured the live buffers by
allocating against them: at each `scaled_dot_product_attention` entry **inside a real
`PairformerLayer`**, the largest interleaved-L1 buffer that still allocates is **187 957 248 B =
1 445 824 B = 1411.9 kB per core**, against a bank size of 1 461 760 B. That is **98.9 % of the per-core L1 budget, free at the moment the SDPA runs** and it is identical to the empty-allocator control — the block
holds essentially nothing in L1 at that point, because the whole pair track lives in DRAM. W9's
204.8 kB resident bias slice is **14.2 % of the per-core L1 budget measured free there**. Capacity does not forbid
the design. Predicted >= 1.20 MB/core; measured 1.41.

**The reduction order, stated plainly.** W9's kernel replaces the streaming online softmax (five
sequential k-chunk passes per q-chunk, each rescaling the accumulator) with a two-pass row softmax
over the full 320-wide row. At `q_chunk = k_chunk = 320` there is exactly one k-chunk, the running
max is already the row max and every rescale factor is 1, so a full-row two-pass softmax and the
chunk-320 online softmax have the **same reduction order**. The consequence for Phase 3 is that
**L1 and L2 share one parity cost rather than stacking two**: whatever chunk 320 costs in accuracy,
bias-once on top of it costs nothing further, because it changes which core holds the bias, not the
order the row is summed in. This is an argument from the kernel structure, not a measurement, and it
is only testable once the kernel exists.

**L1c — the prize. CONFIRMED at 1207.9 ms/fold.** Removing the redundant read leaves one 1.638 MB
read per call, 4.1 us at the roof above, so the net is 1156.7 - 4.1 = **1152.6 us/call**.

### L2 — SDPA chunk 64 -> 320

| arm | bias present | bias absent (core leg) | bias leg |
|---|---:|---:|---:|
| chunk 64 (production) | **2775.5 us** | 1652.3 us | 1123.2 us |
| chunk 320 | **1758.9 us** | 571.0 us | 1187.9 us |

**B1 and B2 CONFIRMED** (predicted 2796.7 and 1728.9 us, both within 10 %). **HB CONFIRMED**: the
core leg falls **2.89x**, past the 2.5x that would have killed it, while the bias leg does not move —
so the chunk-64 penalty is the online-softmax rescale and the chunk-pair loop occupying the compute
kernel phase, and the two legs are independent. With the bias present the same change is only 1.58x,
which is exactly what two independent additive legs predict and a coupled pair would not.

**In a real block, not just standalone.** One `PairformerLayer` at the fold's true shape, three timed
repeats each side: **35.899 ms at chunk 64, 33.741 ms at chunk 320, a 2.158 ms saving = 6.0 % of the
chunk-64 block wall.** The standalone delta predicts 2 x 1016.6 us = 2.033 ms; the block delivers
2.158, agreeing to 6.1 % of the standalone figure. The saving is **additive in the block wall**,
which is the overlap answer for this arm.

**Grid ladder, production chunk and production bias** (U1): 1x1 163 719.9 us, 2x2 41 367.0, 4x4
10 488.4, 6x6 5009.7, 8x8 3203.2, 11x10 2812.8 us. **58.2 core-equivalents of the 110-core grid**,
reproducing T1's 58.2 exactly on an independent run. The knee is at ~36 cores, where DRAM read
saturates.

**The code has not moved under me where it matters, and has where it does not.**
`_tri_att_sdpa_program_config` is byte-identical to what T1 measured: the `256 < q_len <= 384` branch
still returns 64, and its M7 sweep comment still stops at 256 and never tested 320. What *has* moved
since T1's branch point is `_PAIR_PROJ_BW = 16` (`bbb5d85b`, now on main), which changes
`_pair_proj_linear` and therefore the `triangle_bias` row — **not** the qkv projection, which goes
through `minimal_matmul`. Site line numbers have all shifted (T1's SDPA @1333 and its own blockcount's
@1604 are @1631 in my checkout), so a site number from Phase 1 no longer resolves; the call signature
does.

### L3 — the head split

**KILLED, by three measurements.**

**L3a CONFIRMED, and it is the only part of the proposal that survives.** Permuting the qkv weight's
output columns and applying the same permutation to the unpermuted output gives **`torch.equal` =
True, 0 differing elements of 73 236 480, max abs difference 0.0**. A permutation of output columns
reorders nothing inside a dot product, measured rather than argued.

**L3b CONFIRMED — the split is a genuine axis transposition, so no weight permutation can absorb
it.** Taking the q third of the projection output `[298, 320, 256]` and reshaping it to
`[298, 8, 320, 32]` does **not** reproduce `nlp_create_qkv_heads`: max abs difference **127**.
`permute(0, 2, 1, 3)` after the reshape reproduces it **exactly**. In tile layout the head axis is a
tile-*column* axis of the projection output and the SDPA wants it as a *batch* axis; the transform is
10 tile transactions per (batch, head) at a stride of 24 tiles, which is HC, and **HC is CONFIRMED**.
A weight permutation can only reorder columns, and columns are the axis that is already right.

**L3c CONFIRMED — the one layout that would emit `[298, 8, 320, 32]` from a matmul is refused, and
its fallback loses by 3.32x.** The batched form (in0 `[298, 1, 320, 256]` broadcast over the head
axis, in1 `[1, 8, 256, 32]`) is rejected by ttnn's own signature:

```
TT_FATAL @ matmul_device_operation.cpp:195: a_shape[i] == b_shape[i]
bmm (non-bcast matmul) expects input tensors of shapes BCMK*BCKN=BCMN or equivalent
```

The fallback that does run — 24 narrow matmuls, one per (tensor, head), at output width nt=1 —
costs **226.5 us each, 5436.0 us for the set, at 6.9 TFLOP/s**, against **1636.8 us** for the
projection plus the split it was meant to replace. **3.32x worse**, and that is before the concat
back into a single batch axis that it still needs. Predicted >= 3x; measured 3.32x.

**L3d CONFIRMED.** The projection's own baseline on this card is **870.2 us/call** (predicted 865.3),
43.09 TFLOP/s, writing at **168.3 GB/s = 83.6 % of this card's 201.2 GB/s matmul-writer roof**. The
split itself is **766.6 us/call at 382.1 GB/s = 93.7 % of this card's 407.7 GB/s copy roof**, so it
is at its roof and has no tuning headroom — only removal, and removal is what L3c and L3b kill.
Neither op exposes a core-grid knob; both are argued from roof proximity, the exception the org
already signed for this pair. Both are single DRAM-bound streams, so **compute + comm is not a
meaningful split for them: they are all comm**.

**A different route to the same row, found on the way and not proposed by anyone.** The split is
2.15-2.54x faster when its operands stay in L1:

| chunk | L1 -> L1 | DRAM -> DRAM | speedup | full-size extrapolation, L1 |
|---:|---:|---:|---:|---:|
| 16 rows | 35.1 us, 447.8 GB/s | 61.4 us, 256.3 GB/s | 1.75x | 654.2 us |
| 32 rows | 39.0 us, 806.4 GB/s | 99.2 us, 317.0 GB/s | **2.54x** | **363.3 us** |
| 64 rows | 81.7 us, 769.8 GB/s | 175.8 us, 358.0 GB/s | 2.15x | 380.5 us |

The full 146.5 MB qkv output cannot be L1-resident (its q/k/v outputs need another 146.5 MB against
190.0 MB of L1 on this card), but a 32-row chunk needs 15.7 MB in and 15.7 MB out and fits with room.
Bounded at **338-405 ms/fold** depending on how many of T4's 6.40 us launch floors the chunking
costs. This is a bound from an extrapolation, not a delivered number, and it needs its own experiment
because it changes the qkv projection's output buffer type and therefore that op's own binding roof.

---

## ms/fold at stake, after this pass

Conversion is charter §4.9's **x524**, and **I counted it twice in my own runs, not inherited**. In a
real block: **2 `scaled_dot_product_attention` calls per `PairformerLayer`** (12 calls over 6 layer
invocations, at `[298, 8, 320, 32]` with a `[1, 8, 320, 320]` DRAM mask). In a live 298 aa
protenix-v2 fold on this card, production config, counted at the call site: **exactly 1048 SDPA
calls** at `298x8x320x32 + 1x8x320x320`, alongside 1048 `nlp_create_qkv_heads` and 1048 qkv
`minimal_matmul` at the same sites — 2 x 524, C1 **CONFIRMED to the integer**. The same fold issues
**160** SDPA calls at the template shape (2 heads, `298x2x320x32`), which is the x80 row and stays
out of every figure here. x524 **already contains** T5's 40 `trunk_msa` executions (480 + 40 + 4), so
the extra MSA leverage is inside these numbers and must not be added a second time.

| lever | brief said | this pass | verdict |
|---|---:|---:|---|
| **L1 — bias-once SDPA** | 1212.5 | **1207.9 ms/fold** | **CONFIRMED**, capacity clear at 14.2 % of free L1/core, but kernel-only: the L1-mask shortcut is refused |
| **L2 — chunk 64 -> 320** | 1119.1 | **1065.4 ms/fold** standalone, **1130.8 ms/fold** from the block wall | **CONFIRMED**, parity cost measured below |
| **L1 + L2 together** | 2331.6 | **2306.0 ms/fold** (2775.5 -> 575.1 us/call, 79.3 % of the SDPA ms/fold row) | near-additive, as two independent legs predict |
| **L3 — head-split fusion** | 780.1 | **0 ms/fold** | **KILLED** — the split is a transposition, not a reshape; the direct-emit matmul is refused; the narrow fallback is 3.32x worse |
| *new* — split at L1 residency | — | **338-405 ms/fold, bounded** | needs its own experiment; changes the projection's output buffer type |

The three rows this pass re-priced on my own card, for the ledger: SDPA **2908.7 ms/fold** (T1:
2993.6, -2.8 %), `nlp_create_qkv_heads` **803.4 ms/fold** (T1: 780.1, +3.0 %), qkv projection
**911.9 ms/fold** (T1: 906.8, +0.6 %) — each percentage against T1's own figure for that op.

**Overlap, per arm.** The SDPA's two legs are additive, not overlapped: the leg recovered by
subtraction runs at 100-111 % of the read roof, and a leg that were partly hidden behind the core
leg would come out *below* its own traffic time, not at it. The block-level A/B is the same answer
from the other side — the 2.033 ms the standalone delta predicts arrives as 2.158 ms of block wall.
The split and the projection are single DRAM streams at 93.7 % and 83.6 % of their roofs, so neither
has a compute term to overlap with.

---

## Parity

**L3a is bit-exact and measured: `torch.equal` returned True, 0 differing elements.** That is the
only bit-exact claim in this document.

**L2 is not bit-exact, and here is the number Moritz's ask 4285 turns on.** One real
`PairformerLayer` at the fold's true shape (pair tensor logical `[1, 298, 298, 256]`, which is
`[1, 298, 320, 256]` padded — the shape the fold runs, not the square-harness shape), identical
inputs, chunk 64 vs chunk 320:

| tensor | rms of the reference | RMSD | **relative RMSD** | max abs deviation | max abs of the reference | PCC | `torch.equal` |
|---|---:|---:|---:|---:|---:|---:|---|
| pair track `z` | 34.635 | 0.7957 | **2.30 %** | 86.0 | 776.0 | 0.99971 | False |
| single track `s` | 4.134 | 0.00346 | **0.084 %** | 0.5 | 124.5 | 0.9999996 | False |

Both relative figures, 2.30 % and 0.084 %, are the RMSD taken against the rms of that same tensor
under chunk 64.

**Prediction L2b was WRONG.** I predicted a relative RMSD between 1e-4 and 5e-3 and said I would be
wrong above 1e-2. It is 2.3e-2, four to five times my upper bound. The mechanism is not mysterious —
bf16 accumulation over a reordered 320-wide softmax reduction — but the size of it is larger than the
`_tri_att_sdpa_program_config` comment's "PCC 0.9999 vs the 256 config" would lead anyone to expect,
and my own 0.99971 is below that too.

**What this number is and is not.** It is one block of 524, on N(0,1) inputs rather than the fold's
own activation distribution, and per-block deviation does not compose linearly over a recycled stack.
**I did not get to the end of the trunk inside this pass and I am not extrapolating to it**; a
fold-level figure needs a full fold under both configs and that is the next experiment, not this one.
The "shipped 0.0185-0.0217 band" in the record is a **structural RMSD in angstroms at the end of a
fold** — a different quantity from a relative tensor deviation after one block, and I am not placing
one against the other. The CTO and Moritz place it; I report it.

**And L1 inherits this parity cost, it does not add a second one.** Per the reduction-order argument
above, a full-row two-pass bias-once softmax has the same order as chunk 320, so the pair
(L1 + L2) should carry one 2.30 %-class deviation, not two. Unmeasured until the kernel exists.

---

## Corrections to the inherited record

1. **New, and it kills the cheap version of L1: `TT_FATAL: When mask is provided to SDPA, it must be
   in DRAM`.** The bias cannot be made L1-resident by a memory-config change at either chunk size.
   W9's design is the only route and it is a kernel. This constraint was nowhere in STATUS.md.
2. **L3 is dead and the org's 3111 ms/fold slice is really 2306.** `nlp_create_qkv_heads` is a
   `permute(0, 2, 1, 3)`, measured (a reshape alone is off by max abs 127), and ttnn's matmul refuses
   the batch-broadcast that would emit the SDPA's layout directly. T1's "the row is only recoverable
   by not doing it" stands; this pass adds that it also cannot be not-done by a weight permutation.
3. **T1's roofs and its 58.2 core-equivalents reproduce on this card one pass later** — read 396.3 vs
   400.8 GB/s, copy 407.7 vs 410.2, core-equivalents 58.2 vs 58.2. Nothing to correct, and the
   agreement is worth recording because it is the first independent replication of that grid ladder.
4. **A fourth figure for Q12, taken on qb1 card 2 with B1's L1-operand method: 201.2 GB/s.** The qkv
   projection writes at 168.3 GB/s = **83.6 %** of it. Against the org's three existing figures the
   same op reads 96.8 % (174.9), 85.6 % (197.7) and over 100 % (156.6), each of the write roof named
   beside it in GB/s. Mine is closest to B1's and nothing now exceeds its own roof. P2 owns the adjudication; this is one more data point on its own card,
   not a ruling.
5. **The bias leg exceeding the read roof is an instrument artefact, not a violation.** A pure-read
   stream reaches 100-111 % of a roof set by a `clone` that also writes. T1's 103.5 % of the read
   roof was the same effect. Anyone scoring a read-only leg on this card should expect ~7 % of headroom above the
   clone-derived number.
6. **Code drift since T1 measured.** `_PAIR_PROJ_BW = 16` (`bbb5d85b`) landed on main after T1's
   branch point and changes `_pair_proj_linear`, so T1's `triangle_bias` row is on an older
   `_pair_proj_linear`; the qkv projection is untouched by it because it goes through
   `minimal_matmul`. `_tri_att_sdpa_program_config` itself is unchanged, comment included — the M7
   sweep behind it still stops at 256. All site line numbers have shifted by ~27 lines.
7. **Out of scope, recorded not chased (charter §1):** the SDPA bias re-read is a property of ttnn's
   SDPA mask handling, so it almost certainly costs OpenFold3 and Boltz the same way. Not measured,
   not pursued, generalisation noted and dropped.

---

## What Phase 3 should be handed

1. **L2, chunk 320, 1065-1131 ms/fold, one constant** — blocked on Moritz's parity call (ask 4285)
   and now with a number under it. The next experiment it needs is a **fold-level** deviation, not
   another block-level one.
2. **L1, bias-once, 1207.9 ms/fold** — capacity clear (204.8 kB is 14.2 % of the 1411.9 kB/core
   measured free at the call), reduction order the same as chunk 320's, and no shortcut: it is a
   `generic_op` kernel or it is nothing. Take it *after* L2, since L2 is one constant and this is a
   kernel, and the two are near-additive.
3. **L3 is closed.** Do not spend Phase 3 time on the head split by weight permutation.
4. **The L1-resident chunked split, 338-405 ms/fold, is the replacement candidate** and needs a
   Phase-2 experiment of its own first.

**No production change was made this pass. Nothing under `tt_bio/` was touched** — the chunk override
is a probe-side monkeypatch inside `perf/p2_attention/block_probe.py`, and the probes are
`perf/p2_attention/attn_probe.py` and `perf/p2_attention/block_probe.py` with raw results in
`attn_probe_c2.json` and `block_probe_c2.json`, committed to
`wk/protenix-trunk--p2-attention`. Nothing is merged and nothing is proposed for merge here.
