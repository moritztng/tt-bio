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
