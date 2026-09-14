# The trimul output chain: the bytes are real, the fusion that deletes them is not free

`perf/roof_orchestrator/FUSION_PAIRS.md` ranks the tail's two links at 268.4 MB each, 536.8 MB of
one Pairformer block. Both numbers are right. Both links are refused anyway, on time.

## What the trace says that the names do not

At 512 aa the tail runs three programs and only ONE of them round-trips:

    layer_norm      -> #3 DRAM  67.109 MB
    p_out  = linear(#3,  Wp)  l1_out=True  SERVED   -> L1
    g_out  = linear(norm_in, Wg)  l1_out=True  REFUSED -> #4 DRAM  67.109 MB
    multiply_(p_out, g_out, SIGMOID on b)           -> in place into p_out's L1

One 67.109 MB pair tensor is the single L1 slot an 8x9 Wormhole grid has at 512 aa. `p_out` takes
it, so `g_out`'s own `l1_out=True` is refused. The epilogue link is `g_out`, and no choice of
destination can buy it: it needs one kernel over both projections.

## Measured, whglx card 1, 8x9, 512 aa, c_z=128, two interleaved sessions

| arm | s1 | s2 | vs A |
|---|---|---|---|
| A  three ops, shipped | 4.3719 ms | 4.4339 ms | 1.0 |
| B  F1 (4,4), L1 product | 5.1360 ms | 5.1747 ms | 0.8512x / 0.8568x |
| C  F1 (4,4), DRAM product | 5.1684 ms | 5.1998 ms | 0.8459x / 0.8527x |
| A2 | 4.3790 ms | 4.4260 ms | A/A floor +0.162 % / -0.179 % |

B moves 134.283 MB where A moves 268.437 MB. The 268.4 MB/block the table promised is deleted in
full and `torch.equal` holds at 33 554 432 elements, in both destinations.

## The mechanism

L1 vs DRAM for the product is worth 0.5 %. The descriptor is worth -15 %. One matmul pass through
the hand-written `minimal_matmul` fork costs ~0.70 ms/trimul more than the `ttnn.linear` it
replaces at `[1,512,512,128] x [128,128]`, and a full 134.2 MB round trip at this site is worth
0.648 ms measured by a bare `ttnn.clone` (207.06 GB/s achieved). Link 1 pays two passes for one
round trip. Link 2 would pay one pass for one round trip and land at a wash, which is why its
kernel was priced (`prologue_ceiling.py`) instead of written: the norm program's whole marginal
cost is 0.8782 ms, of which only the 0.648 ms round trip is deletable.

## Carry this forward

Deleted bytes are worth the stream rate only when the program change that deletes them is free.
Here it costs about twice the prize.
