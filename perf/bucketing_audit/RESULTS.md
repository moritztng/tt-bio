# Which ttnn primitives read a ragged tile tail — measured, ttnn 0.68.0, qb2 card 0

`ttnn.TILE_LAYOUT` pads physically to 32 while the logical shape stays ragged. Whether that is a
bug depends entirely on the primitive, so it has to be measured per primitive rather than assumed
per model. Run from the repo root:

    TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 python3 perf/bucketing_audit/tile_tail_probe.py
    TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 python3 perf/bucketing_audit/tile_tail_arms.py

## tile_tail_probe.py — does `ttnn.softmax` mask its own tail?

Logical `[1,1,32,33]`, padded `[1,1,32,64]`, every entry -5.0. If the 31 padded columns entered the
softmax they would enter at `exp(0) = 1` against a real `exp(-5)`, which is unmissable.

    measured first element   0.0301307
    masked expectation       0.0303030   (1/33)
    unmasked expectation     0.0002158

It masks. Mechanism in tt-metal: `softmax_program_factory_attention_optimized.cpp:37-43` sets
`mask_padded_data = true` whenever `padded_shape[-1] > logical_shape[-1]`; the general-W factories
derive `mask_w` from `logical_shape()[-1]` the same way.

## tile_tail_arms.py — do matmul / sum / max read the tail?

No way to poison a tail directly: `ttnn.reshape` refuses to shrink a logical volume
(`reshape_common.cpp:50`). So the RFD3 p23 method instead — three arms with identical logical
values and different upstream histories, at logical W=33 / padded 64:

    matmul over ragged K (lhs dim -1)   8.25 / 8.25 / 8.25    logical expect 8.25
    matmul over ragged K (rhs dim -2)   8.25 / 8.25 / 8.25    logical expect 8.25
    softmax over ragged W               0.030131 x3           logical expect 0.030303
    sum over ragged W                   8.25 x3               logical expect 8.25
    max over ragged W                   0.25 x3               logical expect 0.25

No measurable exposure through these four ops. The claim is bounded: all three arms may have had
zero tails, since the eltwise ops that build them start from a tilize-zeroed pad. `max` reading
0.25 rather than 7.0 on the add-arm is the strongest single point against a garbage tail.

## The hole is elsewhere, and the mechanism is different

`ttnn.transformer.scaled_dot_product_attention` does not read undefined DRAM. It extends the key
axis to a tile multiple with a defined *zero* bias while the caller's additive bias covers only the
logical length, so padded key columns enter at score 0 and beat real scores that sit below 0.
71-76x the fp64 reference at any ragged length, ~1.4x at every aligned one
(`origin/wk/fused-sdpa-fold-level-root-cause`, e13f00cc). The sibling case is an op that leaves its
output tile padding unwritten feeding a reduce: `ttnn.scatter` does, which is RFD3 p23
(`tt_bio/rfd3/model.py:1087-1104`).

## program_cache_probe.py — the program cache keys on the LOGICAL shape

Measured on qb2 card 0, blackhole p300, cold `TT_METAL_CACHE`. Three softmax calls at logical W =
128 / 98 / 100 all pad physically to 128, and each one builds its own program:

    softmax W=128 (aligned, 1st)   entries 0->1 (+1)
    softmax W=98  (same pad 128)   entries 1->2 (+1)
    softmax W=100 (same pad 128)   entries 2->3 (+1)
    softmax W=128 again            entries 3->3 (+0)      <- a repeat IS a hit
    matmul  K=98 / K=100           +1 each

`MeshDevice.num_program_cache_entries()` is the instrument; a repeat of an already-seen logical
width reads +0, so the counter is alive and the +1s are real.

A cold-cache build costs **0.231 s** on this box. Sixteen ragged widths 98..113 built 28 programs in
6.472 s; the same sixteen calls at one logical width build 0 and take 0.008 s.

**This is what picks the bucket multiple, and it picks 32.** `TILE_LAYOUT` already pads physically to
32, so raising the logical length of a token axis from N to `ceil(N/32)*32`:

  * changes no padded shape, therefore no tile count, therefore no arithmetic — the padded tiles were
    already being computed, with a zero tail;
  * changes no fused-SDPA gate decision either, because `_padded_sdpa_len` (`tenstorrent.py:655`) is
    itself `ceil(len/32)*32`, so `_capped_sdpa_chunk_size` and the K3 dividing pick see the identical
    number before and after. A 32-bucket cannot turn a fused kernel off;
  * collapses 32 logical lengths onto one program, at 0.231 s per removed build.

Every multiple above 32 buys further variant reduction with real compute. 64 halves the variant count
again and pads up to 63 extra tokens, and the triangle ops are O(S^3): at N=98 a 64-bucket runs 128
tokens, `(128/98)^3` = 2.23x the triangle work, against 1.00x for the 32-bucket which also lands on
128 physically. A geometric ladder (128/256/384/...) is worse still — N=130 to 256 is `(256/130)^3` =
7.6x — so the ladder is rejected on this arithmetic, not on taste.

The legacy 64s (`PAIRFORMER_PAD_MULTIPLE`, `protenix.TOKEN_PAD_MULTIPLE`, `esmc.BUCKET`) are
gate-green and stay; whether they should come down to 32 is the open question already registered in
`state/protenix-opendde-token-bucket-flip-measure.md`, and this measurement is evidence for it.
