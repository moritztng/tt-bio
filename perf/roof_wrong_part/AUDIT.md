# The 48 envelope constants in `tt_bio/tenstorrent.py`, and which of them bind the wrong part

Reproduce the list with the scan the brief specifies (match `^[A-Z_0-9]+ *=`, look back six lines
for `wormhole|galaxy|8x9|72 core|p150|p300|clash|verified envelope|measured on`):

    python3 perf/roof_wrong_part/scan_envelopes.py

It returns 48 at `3df8e8cd4`, the same 48 the brief counted.

A constant is a **candidate** only when both hold:

1. its comment documents an envelope that was MEASURED on one part, not derived from a resource the
   code reads back off the live device; and
2. its use site is not behind `_IS_SMALL_GRID`, a grid-size test, an `arch()` test, or a live
   allocator/fit test that can refuse.

Criterion 2's last clause matters and the brief's phrasing does not name it. Nine of these
constants are numbers fitted on one part whose use site ends in a device allocation that either
succeeds or throws, and the throw is caught and narrowed. Those are not "a Wormhole measurement
capping a Blackhole part": the measurement only picks where to *start*, and the part gets the last
word. `OPM_Z_BUDGET_BYTES`'s own comment writes the rule out — "the device gets the last word
instead: a refusal narrows the block for that shape class and the call is retried". They are
scored NOT-CANDIDATE with that reason named, not with a hand-wave.

## The table

`s above roof` is that constant's unit from `perf/roof_budget/ROOF_BUDGET.md` at the 17.340 s cell;
blank means the constant gates nothing in the Boltz-2 512 aa budget.

| # | constant | line | envelope its comment claims | use site | gated at the use site? | s above roof | verdict |
|---|---|---|---|---|---|---|---|
| 1 | `OPM_Z_BUDGET_BYTES` | 80 | 0.25 GiB; 9i3p dies on exactly 520093696 B at 992 padded tokens, verified on a Wormhole Galaxy | `row_block(...)` in `OuterProductMean.__call__`, L9773 | no arch gate, but reached only at `I > SEQ_LEN_MORE_CHUNKING` (1536 on Blackhole) and backed by `_OPM_DRAM_ROW_CAP`, a refusal-narrowed cap | 0.349 | NOT — device has the last word, and the path is dark below 1536 tokens on Blackhole |
| 2 | `CONCAT_HOST_BYTES_BASE` | 142 | 1.5 GiB; od_9i3p refused 2.61 GiB at 7.2 GiB used on a 12 GiB Wormhole | `_concat_host_budget`, L4917 | yes — `max(base, dram_total // 8)`, so it is a fraction of the part's own DRAM | | NOT — part-relative by construction |
| 3 | `_CONCAT_HOST_BYTES` | 143 | memo of #2 | `concat_host_bytes()` | n/a | | NOT — cache slot |
| 4 | `ATOM_PAIR_BUDGET_FRACTION` | 151 | a quarter of the part; RFD3 died at 4558 atoms with 9.5 of 12 GiB standing | `atom_pair_budget_bytes`, L4890 | yes — `total // 4` off the live DRAM total | | NOT — a fraction, not a byte count |
| 5 | `L1_TOTAL_BYTES_WORMHOLE` | 155 | 72 banks x 1395424 B, a 12 GiB Galaxy chip | `l1_resident_budget_bytes`, L4871 | yes — `_l1_total_bytes() or L1_TOTAL_BYTES_WORMHOLE`; the constant is the no-device fallback only | | NOT — explicitly the tighter class as a fallback |
| 6 | `TRANSITION_W_CHUNK_SIZE` | 156 | 1024 Blackhole baseline | `Transition.__call__`, L7863 | yes — `_apply_grid_thresholds` re-snaps it to 512-scaled under `_IS_SMALL_GRID` | | NOT |
| 7 | `SEQ_LEN_MORE_CHUNKING` | 157 | 1536 Blackhole; re-derived to 1088 on a 12 GiB Galaxy from a 1024 aa footprint measurement (3.461 GiB of 12.0) | 29 sites | yes — `_apply_grid_thresholds` re-derives it from `_dram_total_bytes(device)` | | NOT — and it is the model of how to do this |
| 8 | `PAIR_ROW_BLOCK` | 161 | **none.** "One number so the input and output projections cannot drift apart; a tile multiple" | `_in_proj_rows` L5881, trimul L6388 | **no gate at all** — no arch, no grid, no L1 term, no refusal path | 1.458 | scan false positive on criterion 1 (it inherits the neighbouring Wormhole comment), but it fails criterion 2 outright. **Candidate by gap**: fitted on no part rather than the wrong one. MEASURED, see below |
| 9 | `TRIMUL_IN_NORM_ROWBLOCK_BYTES` | 176 | 3 GiB; 9i3p's 2.59 GiB pair tensor folds, 9j4c's 3.19 GiB is refused — a 12 GiB Wormhole | `TriangleMultiplication.__call__`, L5996 | **no** — a bare byte compare on the tensor | 1.458 | **CANDIDATE**, wrong direction only in size: a 12 GiB part's refusal boundary applied to a 32 GiB p150a. Costs 43 %/call when it fires. Does not fire in 298-1024 aa: 3 GiB of pair tensor is L~2048 at c_z=384 and L~3550 at c_z=128. Out of the ladder's reach, so unpriceable here |
| 10 | `TRANSITION_H_CHUNK_SIZE_BIG` | 220 | 32; "verified envelope W<=384", W=512 clashes in-block L1 | `Transition.__call__`, L7842 | **no** | 2.251 | **CANDIDATE, and the proven one** — owned by `roof-transition-chunk-bh`, not touched here |
| 11 | `TRANSITION_H_CHUNK_BIG_MAX_W` | 225 | 384, validated against the c=256 clash | L7841 | **no** | 2.251 | same row, same owner |
| 12 | `_TRANSITION_L1_CHUNK_BYTES_BASE` | 232 | 393216 B/core; 14 points over 6 widths on UF-EV-A13-GWH02, fits at <=384 KiB, static-CB clash at >=400 KiB | L4322 | yes — inside the `_IS_SMALL_GRID` branch, rescaled by live per-core L1 | | NOT |
| 13 | `_WH_MEASURED_L1_PER_CORE` | 233 | 1466080 B, the L1 #12 was measured at | L4322 | yes — it is #12's denominator | | NOT |
| 14 | `TRANSITION_L1_CHUNK_BYTES_PER_CORE` | 234 | #12 after rescale | L7907 | yes — `return ... if _IS_SMALL_GRID else h` at L7913 | | NOT. This is the careful derivation the brief contrasts against #10, twenty lines below it |
| 15 | `_TRIMUL_MASK_L1` | 370 | 0.8566 -> 0.6238 ms/call, measured on Wormhole at the production shape | L6078 | yes — `_TRIMUL_MASK_L1 and _l1_fits(...)`; the fit test is live | 1.458 | NOT — bool, and the device decides |
| 16 | `_PWA_L1_NORM` | 406 | 3572.2 -> 991.0 us on the eight-head region, qb2 (Blackhole p300c) | L9243 | yes — `_l1_layer_norm(z, 1.0, _PAIR_L1_CONSUMER_RESERVE, ...)` falls back when the reserve is not free | 0.244 | NOT — bool; the fitted number is the reserve, and that has its own row |
| 17 | `_PWA_BATCH_HEAD_WEIGHTS` | 415 | 8.436 ms and 536.9 MB of reads against a 67.1 MB source, one MSALayer at 512 tokens on Wormhole | L9279 | yes — `and self.n_heads <= 32`, a shape property | 0.244 | NOT |
| 18 | `PWA_BATCH_HEAD_STATS` | 416 | census | L9280 | n/a | | NOT |
| 19 | `TRIANGLE_MULT_L1_CHUNK_BUDGET` | 511 | 64*320*320; measured on a 13x10 Blackhole grid, chunk 64 at seq 320 fits and at seq 352 clashes | `_trimul_chunk_size`, L994 | yes — `* gx * gy / (COMPUTE_GRID_X_13 * 10)`, scaled by live core count | 1.458 | NOT, with a caveat: the core-count scaling itself has never been measured on 8x9. `SMALL_GRID_TRIMUL_BUDGET_SCALE` exists precisely to re-fit it and still reads 1.0 |
| 20 | `_IS_SMALL_GRID` | 514 | the gate | 16 sites | n/a | | NOT |
| 21 | `_SUB_TILE_SLICE_WEDGES` | 530 | four shapes, p150a card 2, 130-200 s watchdogs; the same two shapes are byte-identical on Wormhole | L646 | yes — `device.arch() != ttnn.Arch.WORMHOLE_B0` at L4381 | | NOT — the exemplar of an explicit arch gate, and it defaults to the wedging side |
| 22 | `_TRIMUL_CHUNK_CAP` | 551 | test-only hook | L1011 | n/a — unset in production | | NOT |
| 23 | `SMALL_GRID_TRIMUL_L1_MAX_SEQ` | 661 | 0 = keep the derived value; re-fit hook for 8x9 | L930 | yes — `if _IS_SMALL_GRID and ...` | | NOT |
| 24 | `SMALL_GRID_TRIMUL_BUDGET_SCALE` | 662 | 1.0 = keep the derived value | L996 | yes — inside `if _IS_SMALL_GRID` | | NOT |
| 25 | `SDPA_CHUNK_TILE` | 664 | 32 | 9 sites | n/a — the tile height, an architecture invariant on both parts | | NOT |
| 26 | `SDPA_CHUNK_MAX` | 665 | 256, with the comment stating outright that `_capped_sdpa_chunk_size` "has no grid term at all: it returns 256 whatever card it is on" | `_capped_sdpa_chunk_size`, L1041 | **split.** The q side is grid-corrected by `_grid_q_chunk` (`_SDPA_GRID_Q_CHUNK`, on by default, takes `COMPUTE_GRID_MAIN`). The k side, `_dividing_sdpa_chunk_size`, has no grid term | 1.284 | **NOT PAYABLE as a chunking lever**: k_chunk sets the online-softmax reduction order, so it is not bit-exact and by the brief's own parity rule it is a different kind of lever. Named, not measured |
| 27 | `_WH_FULL_L1_PER_CORE` | 689 | 1.5 MiB, the calibration ceiling | L4298/4299/4327 | yes — only inside the `_IS_SMALL_GRID` branch | | NOT |
| 28 | `_MIN_L1_SCALE` | 690 | 0.7 floor | L4299 | yes — same branch | | NOT |
| 29 | `_B2_ADALN_S_MEMO` | 1319 | -1.565 s at 512 aa, bit-exact | L8912 | n/a — a memoisation bool, no envelope | 0.453 | NOT |
| 30 | `_BATCHED_MATMUL_SATURATION_BLOCKS` | 2053 | **32 output blocks. "Output blocks at which a batched matmul reaches the DRAM roof on a p150a. Measured on qb1 card 0"** — three op classes, 80/32/16 blocks | `_batched_matmul_search`, L2168 | **no.** The function already holds `cores = gx * gy` two lines up and uses it in the legality filter; the saturation target alone is an absolute block count with no grid term | 1.448 + 0.44 | **PRIME CANDIDATE** — the one occupancy constant fitted on one part's core count and applied to every core count. MEASURED, see below |
| 31 | `_CLASH_RE` | 2284 | regex over a TT_THROW | L2306 | n/a | | NOT |
| 32 | `_CB_OVERFLOW_RE` | 2287 | regex | L2307/2387 | n/a | | NOT |
| 33 | `_FP32_SOFTMAX_L1_GRID` | 2595 | **`(8, 8)  # 8x8 = 64; this p150a refuses more than 110 shards`** | 5 sites, L2696-L3130 | **no arch gate**, but `_FP32_SOFTMAX_L1_FLOAT_CORES` (default on) lets the actual core count float off it, and every plan ends in a shard the allocator can refuse | | **CANDIDATE, and the inverted direction the brief asks to be flagged**: a Blackhole shard-count ceiling used as the tuned grid on a 72-core Wormhole, where 64 of 72 is a different occupancy from 64 of 130. Unpriceable from this host; the path is opt-in per attention instance and off the Boltz-2 512 aa budget entirely |
| 34 | `FP32_SOFTMAX_STATS` | 2597 | census | 16 sites | n/a | | NOT |
| 35 | `_FP32_SOFTMAX_L1_PADDED` | 2828 | default OFF, self-flagged unmeasured | L2839 | n/a | | NOT |
| 36 | `TRANSPOSE_L1_RESERVE_PER_CORE` | 3455 | 128 KiB; validated where 1.25x was, protenix-v2's 298 aa 52.4 MB pair tensor, 1208 transposes, no refusal. The window it changes is "118-147 MB **on an 11x10 Blackhole grid**" | `_pair_transpose_impl`, L3457 | partially — opt-in per Pairformer instance (`transpose_l1_reserve`), and the allocation is still the real test | | **CANDIDATE, weak**: the 118-147 MB window is an 11x10 Blackhole computation and the Wormhole window is a different interval nobody has computed. Only RF3 768 aa lands in it, so it is off the Boltz-2 budget |
| 37 | `_TRANSPOSE_L1_RESERVE_PER_CORE` | 3456 | env resolution of #36 | — | n/a | | NOT |
| 38 | `_PAIR_BIAS_LN_RETRY` | 3517 | negative-control switch | L6643 | n/a | | NOT |
| 39 | `_PAIR_BIAS_TRACE` | 3521 | trace | L6605 | n/a | | NOT |
| 40 | `_PAIR_FFN_FC1_BW` | 4053 | 1; of 80 configs swept, all 20 at bw=1 are `torch.equal` and all 60 above differ by one bf16 ulp | `esmc.py:630` | n/a — bw=1 is the bit-exactness condition, not an envelope | | NOT |
| 41 | `_PAIR_FFN_FC1_BLOCK_W` | 4054 | 16; "MEASURED on qb2 card 2" (Blackhole p300c, 11x10), 17.918 -> 14.662 ms, obw=32 clashes in the chain, obw=8 costs 2.45 ms/call | `esmc.py:630` | **no** | | **CANDIDATE**, and note the shape of it: a p300c-fitted out_block_w reaching ESMFold2's pair FFN on every part, with a documented clash one rung up. Off the Boltz-2 budget (ESMFold2 only), so it is not measurable against this campaign's ranking |
| 42 | `WORMHOLE_MSA_AREA` | 4494 | 640*8192; L=640 depth 8192 folds in 230.4 s, L=788 fails on exactly 788*8192*128*2 | `msa_depth_cap`, L4510 | yes — `if ... or not is_wormhole(): return max_sequences` | | NOT — exemplary |
| 43 | `BH_PAIR_SINGLE_PASS_MAX` | 4529 | 1024, the top of the ladder walked on this silicon | `pair_row_tile`, L4552 | yes — only reached when `SMALL_GRID_PAIR_TILE_AREA` is 0, i.e. off a small grid | | NOT |
| 44 | `BH_PAIR_TILE_AREA` | 4530 | 524288; esmfold2 at 1536 is refused 4831838208 B, qb2 card 1 and qb1 card 0 agree to within 14336 B | `pair_row_tile`, L4554 | yes — same | | NOT |
| 45 | `TRIMUL_GP_BANK_SPLIT` | 5566 | 1.5034x on `reblock_permute_gated` and 1.0212x on the 512 aa fold, Blackhole p150a, 8 DRAM banks | `set_trimul_inproj_rowblock`, L5569 | **no** | 1.458 | NOT PAYABLE, but **FLAG**: the comment argues from bank arithmetic ("Wormhole has 12 banks, so 256 % 12 = 4 and the serialisation this removes cannot happen there; the reorder is free on both") and never measured it. A reasoned neutrality on the part JapanFold serves. Cheap to close: one paired op run on whglx |
| 46 | `_GP_ROLES_SPLIT` | 5567 | the reordered layout | L5574 | — | | same lever as #45 |
| 47 | `_GP_ROLES_MAJOR` | 5568 | the old layout | L5574 | — | | same lever as #45 |
| 48 | `_TRIMUL_GP_BANK_SPLIT` | 5569 | env resolution of #45 | L5574/5589 | — | | same lever as #45 |

## What the table says

**44 of 48 are correctly handled, and most of them for a better reason than "there is an
`_IS_SMALL_GRID` around it".** Three distinct correct mechanisms show up, and they are worth naming
because the four defects are exactly the constants that use none of them:

1. **Express the budget as a fraction of a resource the code reads off the live part.**
   `#2 #4 #5 #7` and `SINGLE_SHOT_DRAM_NUM/DEN`. `SEQ_LEN_MORE_CHUNKING` is the strongest instance:
   it was re-derived from a measured 1024 aa footprint against the part's own DRAM total.
2. **Ask the part, then narrow on the refusal.** `#1 #15 #16 #36` and `row_block_after_refusal`.
   A fitted number that only picks a starting point cannot cap a bigger part, because the bigger
   part never refuses and never narrows.
3. **Read the architecture.** `#21 #42`. `_SUB_TILE_SLICE_WEDGES` additionally defaults to the
   wedging side, so an unmeasured part gets the safe route rather than a spin.

**The four that use none of them** are `#10/#11` (owned elsewhere), `#30`, `#33` and `#41`, plus
`#8` and `#45` as gaps of a different shape. Three of the four are **Blackhole measurements applied
to Wormhole**, which is the inversion the brief asks to be flagged and the direction that costs the
Galaxy JapanFold serves users from:

- `#30 _BATCHED_MATMUL_SATURATION_BLOCKS = 32`, measured on qb1's 130-core p150a, applied to a
  72-core Galaxy where 32 blocks is 44 % of the grid instead of 25 %.
- `#33 _FP32_SOFTMAX_L1_GRID = (8, 8)`, whose comment names the part in the same breath as the
  number: "this p150a refuses more than 110 shards".
- `#41 _PAIR_FFN_FC1_BLOCK_W = 16`, measured on qb2's p300c with a clash documented one rung up.

Only one of the four sits on a unit in the Boltz-2 512 aa budget, and it is `#30`.
