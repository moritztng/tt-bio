# b2z2-trunk-byte-size-ladder — pre-registered prediction

Written and committed BEFORE the first fold of this row. Nothing below is measured.

## What is being gated

Three bit-exact trunk byte levers, all default OFF, on `wk/b2z2-trunk-byte-round2` @ `67c2e432`:

- `triatt_qkv.qkvg_heads` / `TT_BIO_TRIATT_FUSED_QKVG` — q, k, v and the gate from one pass over
  the normed pair tensor, four N chunks of one matmul.
- `TT_BIO_TRIATT_FUSED_QKVGB` — the pair-bias projection rides the same pass as a fifth, one-tile
  chunk.
- `TT_BIO_TRIMUL_FUSED_GOUT` — the trimul's output gate rides its in-projection, which holds the
  gate live across the channel loop instead of computing it after.

They are bit-exact on Wormhole (21 folds) and Blackhole (66 folds). Parity is settled. The open
question is peak memory and allocation behaviour at large targets, and the standing memory
`of3-1024aa-oom-allocation-count-not-size` says the thing that OOMs this class is how many
allocations are co-live, not how big one of them is.

## Measured facts this prediction is built on, read off the branch before predicting

- `SEQ_LEN_MORE_CHUNKING` on qb2 card 2 (Blackhole p300c, 11x10) reads **1536**, live from the
  device, not the 1536 literal at `tt_bio/tenstorrent.py:156` — `_grid_scale` snaps 640 and then
  raises it against L1, capped at 1536.
- `TriangleMultiplication._gout_eligible` (`tt_bio/tenstorrent.py:5414`) declines on seven named
  reasons, two of which are size-dependent: `row_blocked_tail` at `H > SEQ_LEN_MORE_CHUNKING`, and
  `multi_chunk_channel_loop` at `n_pairs // group != 1`, where `n_pairs = hidden //
  _trimul_chunk_size(H, hidden, batch)` and the chunk size shrinks as H grows.
- Both concatenated weights (`qkvg_weight`, `qkvgb_weight`, `tt_bio/tenstorrent.py:6404`) are built
  at weight load **unconditionally of the env flags**. Their DRAM cost is therefore already paid in
  the arm-OFF arm and a default flip adds none of it. It is a cost of merging the branch, not of
  flipping the defaults, and the arm-off/arm-on delta cannot see it.

## P1 — completion

All six folds (640 / 1024 / 1536 aa x {off, on}) complete. No OOM in either arm at any size.
Confidence high: the branch's runtime allocations are the same buffers in a different producer,
and the one genuinely new residency is the trimul gate.

## P2 — the decline, and it is the interesting half

**`TT_BIO_TRIMUL_FUSED_GOUT` declines at 1024 aa and at 1536 aa on `multi_chunk_channel_loop`,
and serves at 640 aa.** The channel loop is single-iteration only while `_trimul_chunk_size` still
returns the full hidden width; it shrinks with H, and 1024 is where I expect it to have shrunk.

`row_blocked_tail` does **not** fire at 1536: 1536 is not greater than 1536. It fires at the first
rung above, so the lever's real ceiling on this grid is 1536 aa exactly and nothing on this ladder
will show it. Predicted `row_blocked_tail` count: 0 at all three sizes.

## P3 — the two triangle-attention levers engage at every size

`qkvg_heads` and `qkvgb_heads` decline on shape and config, not on token count: `head_dim == 32`,
the concatenated weight's `_mm_block_for` entry (keyed on the WEIGHT, which is size-invariant by
construction), `_PAIR_PROJ_MM`, the L1-out leg, and `pad[0]*pad[-2] > N`. None of those is a
function of H in a way that changes between 640 and 1536. Predicted: both serve 100 % of the
triangle attentions at all three sizes, `QKVG_STATS[1] == 0` and `QKVGB_STATS[1] == 0`.

## P4 — peak memory

Peak device DRAM, arm-on minus arm-off, is **positive and under +1.0 %** at every size, and the
whole fold's DRAM high-water at 1536 aa stays under **12 GiB of the part's 31.9 GiB**. The gate the
trimul lever holds live is one `[1, H, H, hidden]` bf16 tensor; at 1536 aa that is bounded by a few
hundred MB, which is under 3 % of the part. If `TT_BIO_TRIMUL_FUSED_GOUT` declines as P2 predicts,
the delta at 1024 and 1536 collapses to ~0 and P4 is trivially satisfied there — so P4 is only a
real test at whatever sizes the gate actually serves.

L1 high-water is unchanged by all three levers: they move DRAM reads, not per-core residency.

## P5 — bit-exactness at every size

At each size, the arm-off and arm-on CIF are byte-identical. At 512 aa both arms write
`a91aa44441f0d9c5`. Bit-exactness is not re-argued here, it is checked, and a mismatch at any size
is a stop.

## Falsifier

**If all three levers run clean to 1536 aa with the gates engaging throughout, the flip has no
remaining blocker**, and this row says so in those words.

The prediction that would make this row interesting instead is P2: a lever that silently declines
reads as "it worked". This wave has already been caught by that three times (`bond_type_feature`,
`use_templates`, `_reblock.eligible_gated`), which is why every arm prints which branch each site
took and the harness asserts on the counter rather than on the wall clock.
