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

*(filled in below, after the predictions above were committed)*

## Experiments and verdicts

## ms/fold at stake, after this pass

## Parity

## Corrections to the inherited record
