# ROOF Phase A build, qkv -> SDPA FULL arm: what I expect before the device runs

Written and committed before anything on this branch touched a card. The screen
(`state/roof-fuse-qkv-sdpa.md`) is not re-derived here; only the build's own open numbers are.

Machine for every number below: **Blackhole p150a on `pc`, card 2, 13x10 = 130 cores, ttnn 0.68.0**,
warm, interleaved arms in one process, own-session A/A floor. KIND **arithmetic**, k = 0.623, so a
Wormhole screen of the same lever would read `1 + (r-1)/0.623` and must not be quoted here.

## The structural check the row said to run first, answered from the sources

The kill criterion was "kill if the fused route needs a second matmul pass rather than extending the
existing reader". It does not, and here is the reading that says so.

* The qkv projection in the traced 512 aa block is row 64 of
  `perf/b2z2_byte_floor/out/trace_512_wh_c10.json.gz`: `512x512x128 @ 128x384` through
  `mm_generic.py:359:generic_minimal_matmul`, writing three `512x4x512x32` buffers. Its block config
  is `_MM_BLOCK[(4, 12)] = (4, 4, 1, 4, 1)` — **K_block 4 = the whole contraction, one block,
  subblock 4x1**.
* `head_dim = 32`, so `DHt = 1`. Each head's slice of that weight is `Ct x DHt = 4 x 1` tiles, and
  the fold's projection is `(Sqt x Ct) @ (Ct x 1)` — the same single K block, same order.
* The consumer already runs that primitive. `compute_common.hpp:1225 matmul_blocks` and
  `compute_streaming.hpp:80 blocked_matmul_and_pack` are `matmul_block` loops over `in0_block_w`,
  and the SDPA calls them twice per k chunk already. The fold adds `matmul_block` calls inside the
  program that is already there. No second program, no new descriptor, no `minimal_matmul` pass.
* The reader change is one `read_chunk_with_padding` call. `x` is `[B, S, C]`, tile grid
  `(B, Sqt, Ct)` contiguous per batch row, which is the shape that routine already reads. K's
  `transpose=true` is a **tile-order** transpose only, and at `DHt = 1` it is the identity, so the
  projected k tiles land in the layout the qk matmul expects without any extra work.

So this is not `roof-fuse-trimul-out`'s situation. That row paid ~0.70 ms/trimul to stand up a
hand-written matmul *program*; here the matmul is 192 tile-MACs added to a kernel already issuing
512 of them.

## PREDICTED, the number

Prior from the screen: **1.124x - 1.505x on the pair**, low end if none of the 25.77 GFLOP hides in
the SDPA's 22.5 % input stall, high end if all of it does.

**I predict 1.15x - 1.40x, point estimate 1.24x**, and I predict the *low* end of the screen's range
is not safe — I put real weight on landing under 1.124x. Three reasons, in the order they matter.

1. **The fold makes the SDPA read MORE, and added bytes cost roof rate.** Per `(batch, head)` the
   SDPA reads `q + k + v` = 48 tiles today and `x[b]` = 64 tiles folded: 201.3 -> 268.4 MB, +67.1 MB
   on the consumer. At the screen's least-squares 2.072 us/MB that is +0.139 ms, and the screen's own
   asymmetry says I may not price it any cheaper. The whole win is the projection's 0.873 ms, of
   which 0.139 ms is handed straight back before any arithmetic moves.
2. **The arithmetic is 37.5 % more matmul, and only part of it can hide.** Tile-MACs per
   `(batch, head)`: qk `16x16x1` = 256, pv `2 x 16x1x8` = 256, total 512. The projection adds
   `3 x 16x1x4` = 192, so **+37.5 % tile-MACs**. Against that, the fold's MAC:pack ratio is 4:1 where
   the qk matmul at `DHt = 1` runs 1:1, so the projection should NOT be as pack-starved as the
   program's 52.3 TFLOP/s average suggests, and I expect it faster than the 0.493 ms that average
   implies. The stall it can hide in is 22.5 % of 1.452 ms = 0.327 ms, and a stall fraction is not a
   recoverable fraction — the reader is now reading 20 % more bytes into the same window.
3. **Against both: the projection program disappears whole**, not just its bytes. 0.873 ms including
   its own dispatch, its own `cb_reserve`/`push` cycle and its three output writes.

Arithmetic of the estimate: folded SDPA 1.452 ms on bytes, plus 192/512 of the SDPA's matmul time
with roughly half of it hiding in the stall, lands at 1.70 - 1.80 ms against today's 2.186 ms pair.
That is where 1.24x comes from. 1.40x needs nearly all of it to hide; 1.15x is what I get if the
projection matmul runs at the program's own average rate and nothing hides.

**Fold-level, not op-level: I predict this arm does NOT reach the pair ratio at the fold.** The pair
is ~32.8 % of GenericOp time, so 1.24x on the pair is ~6.1 % of GenericOp, and GenericOp is not the
whole step.

## PREDICTED, parity

The screen predicted bit-exact. **I predict it is NOT bit-exact, for a reason the screen did not
have in front of it.** The qkv projection runs under the trunk's compute kernel config, which sets
`fp32_dest_acc_en = True` (`tenstorrent.py:4486`); the fused SDPA runs under the op's own default
`(HiFi2, approx, fp32_dest_acc = False)`. Same K block, same order, **different accumulator width**:
four bf16 tile-MACs summed in an fp32 DST against four summed in an fp16_b DST. That is one or two
bf16 ULPs on q, k and v, and the softmax is exponential in it.

So I predict `torch.equal` fails and I will have to quote Angstrom. Two arms get measured:
`fp32_dest_acc = False` (the op default, fastest, not bit-exact) and `fp32_dest_acc = True` (dst_size
drops 8 -> 4, which re-picks every subblock in the SDPA itself and so is not bit-exact against today
either, and costs time). I predict the deviation at 512 aa is **well under 0.05 A** against the
0.105 A the campaign has left, because the perturbation is sub-ULP on the projection's inputs to a
softmax that the trunk's residual+LayerNorm path damps — but that is a prediction, and the number is
what decides it.

## What kills it

* Op ratio under 1.05x on two interleaved sessions against an own-session A/A floor.
* L1 refusing at 298 aa as well as 512 aa. Priced: the fold needs `x` double-buffered
  (2 x 64 x 2048 = 262,144 B) plus the head's three weight slices (12 x 2048 = 24,576 B) = 286,720 B
  against the 310,784 B `sdpa_generic.cb_bytes` leaves free at the shipped 512 aa config. That is
  23 KB of headroom and it is the thing most likely to make me drop to a single `x` buffer, which
  serializes the reader behind the compute. At 320 padded tokens the same need is 188,416 B against
  781,824 B free.
* Not bit-exact AND the Angstrom cost not worth it.

## The variant I am NOT building first, and why it is written down

The screen's ledger has a fourth row: multicast `x[b]` across the four head-cores that share a batch
row, which reads `x` once per batch instead of once per head — 67.1 MB instead of 268.4 MB, i.e.
**-402.7 MB per attention instead of -201.3**. The reader already carries the mcast machinery it
would need (it is the K-forwarding chain, dead here at 0/0 chains). That is a strictly larger win on
the same fold and it is the follow-on if the simple form clears.
