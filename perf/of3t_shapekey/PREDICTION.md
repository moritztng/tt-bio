# of3t-shapekey — pre-registration

Written before the first device arm. `perf/of3t_shapekey/PICKS.json` is the only thing read first,
and it is not an outcome: every function it calls is an `lru_cache`d pure function of shapes, so it
records the pick the fold will make without running one. The prediction below is built on it.

Tree `wk/of3t-shapekey` at `5d07cbb4c` (branch point `wk/of3t`). Card 0, p300c Blackhole, qb2.
Padded width is on every figure. Frame and floor host are named on every ratio.

## What PICKS.json already settles

`_PF_DIMS = (32, 4, 24, 16)` in `openfold3_trunk.py`, so the two tracks have different shapes:

| | leading rows | heads | head_dim | q shape |
|---|---|---|---|---|
| single, `AttentionPairBias` | 1 | 16 | 24 | `[1, 16, N, 24]` |
| pair, `TriangleAttention` | N | 4 | 32 | `[N, 4, N, 32]` |

**The single track's leading dim is 1.** `_fp32_softmax_attention`'s row block is a block over
`q.shape[0]`, so at both widths `run()` takes the `rows <= 1` branch, calls `shard_for(rows)` with
`rows = 1`, and asks for one block. The plan's row count never reaches an op. What is left of the
plan is the core count, and that only picks a shard:

    N=64    height_per_row 1024  per_row 262144 B  plan (rows 330, cores 110)  _fp32_softmax_shard(1, 1024, 64, 110)
    N=384   height_per_row 6144  per_row 9437184 B plan (rows   9, cores 108)  _fp32_softmax_shard(1, 6144, 384, 108)

`_fp32_softmax_shard` refuses unless `rows * height_per_row % (cores * 32) == 0`. At 64 that is
`1024 % 3520 = 1024`; at 384 it is `6144 % 3456 = 2688`. **Both refuse.** So the single track runs
the interleaved tail at both widths and the shard is never built at either. On top of that the
shipped code states the shard's arithmetic status itself, in `_fp32_softmax_attention_block`: the
interleaved fallback after a shard refusal is "the same ops on the same dtypes and therefore the
same bits".

**The pair track's shipped pick at 384 is already ONE k chunk, not six.** `openfold3.trunk` passes
`tri_att_sdpa_hifi=True` (`openfold3_trunk.py:193`), which forwards `tri_att_one_k_chunk=True`, and
`_tri_att_sdpa_hifi_inner` then offers `padded_k = 384` ahead of the shipped `_sdpa_chunks_shipped`
pick of 64. `_sdpa_chunks_shipped(384, 384)` does return `(64, 64)`, which is where the brief's six
chunks come from, but that value is overridden before it reaches the kernel.

**The pair track's real width asymmetry is a ROUTE change, not a chunk count.**
`_TRIATT_FUSED_HIFI_MIN_S = 128` and `_TRIATT_HIFI_MIN_S_PADDED` is False, so the gate reads the
tensor's own length:

    N=64    min(64, 64) = 64 < 128    -> too_short, declines, falls to _fp32_softmax_attention (materialised)
    N=384   min(384, 384) >= 128      -> serves, fused HiFi4 SDPA, q_chunk 384 / k_chunk 384

So at 64 the pair track runs the materialised fp32-softmax path and at 384 it runs a different
kernel. Setting `tri_att_sdpa_hifi` at both widths does not match the route: the flag is *inert* at
64 because MIN_S refuses first. That is a second, tighter confound than the one the brief names, and
it is unclosed in the banked pair too: `perf/of3t_apbgrad/DEV_SCOPE_RENORM_c64.json` records no
`tri_att_sdpa_hifi` key at all, `perf/of3t_modelboundary/DEV_RENORM_n384_nocaptures.json` records
`true`.

## The arms

A. **c64 and n384 re-taken, flag recorded at both.** `TT_BIO_TRIATT_SDPA_HIFI_AB` left at the
   shipped default so `openfold3.trunk` resolves True at both widths, `TT_BIO_SOFTMAX_BW_RENORM=1`,
   `--lever none`, `--arm flipped`, no captures at either width. Captures are already established
   bit-neutral (`of3t-modelboundary`: 0 of 2,736 tensors moved between
   `DEV_CTRL_c64_captures` and `DEV_CTRL_c64_nocaptures`), so the banked c64 arm's captures are not
   a live confound and this row does not re-take that control.

B. **n384 with the single-track L1 plan pinned to the geometry the 64 case selects.**
   `_fp32_softmax_l1_plan` returns `(330, 110)` instead of `(9, 108)` for the single track's shape
   class only. Harness-side monkeypatch in `perf/of3t_shapekey/`, no shipped default moves.

C. **n384 with the pair track pinned to the 64 case's route**, `TT_BIO_TRIATT_SDPA_HIFI_AB=-openfold3.trunk`,
   so both widths run the materialised `_fp32_softmax_attention` at triangle attention. This is arm
   C re-specified: the literal "pin the chunking to one chunk" is a no-op at 384 because one chunk
   is what ships there, and the census in arm A's sidecar is the evidence for that, not a device
   arm. Pinning the route is the informative version of the same question and it goes the safe
   direction, turning a lever off rather than on.

Every arm carries a reach counter, because a pin that never fires and a pin that fires and is inert
are different results and no output comparison separates them (D121). `FP32_SOFTMAX_STATS`,
`TRIATT_FUSED_HIFI_STATS`, `TRIATT_FUSED_HIFI_PICKS` and the pin's own firing count go into a
sidecar per arm.

Scored with `of3t-widthattr`'s `widthgrowth.py` unchanged, floor and float64 reference unchanged:
both were built on **qb1 (tt-quietbox), CPU only, EPYC 8124P, torch 2.8.0+cpu / python 3.10.12**
(D189: a c64 floor off a different box reads 0.4007237405 against 0.3739375769, 7.2 % apart).

## Predictions

**A reproduces.** `ours_384 / ours_64` lands in **2.10–2.26** against the banked 2.1794882627817005,
and `ours_384/floor_384` in 2.19–2.28 against 2.2341. The c64 re-take should be bit-identical to
`dev_scope_RENORM_c64.pt` over all 2,736 tensors, because the only thing this row changes at 64 is a
flag MIN_S makes unreachable. If A does not reproduce, that is the finding and the row stops there.

**B is bit-identical to A's n384.** Max absdiff exactly 0.0 over 2,736 tensors, growth difference
0.0 mw². The pin will fire (the plan is queried once per shape class per process) and be inert,
because at a leading dim of 1 neither plan can build a shard and the shard is bit-neutral anyway.
Falsified by any tensor moving at all: that would mean the plan reaches arithmetic and the reach
analysis above is wrong. **So the single track's 84.52 % of the growth is predicted NOT to be the
L1 shard geometry.** This is a prediction the named site forbids surviving, and it is cheap.

**C leaves the growth above 1.9x.** Band **1.9–2.3x** for `ours_384/ours_64`. The reason is where
the growth is: 93.80 % of it is LayerNorm affine parameters and `of3t-lnaffine` put the arrival at
the cotangent the **AttentionPairBias** backward writes. That is the single track, and the single
track has no fused route at either width — `AttentionPairBias._attention` calls
`_fp32_softmax_attention` unconditionally on the fp32_softmax branch and the file says in so many
words not to reroute it. The pair route reaches the single track only through `z` into the next
block's pair bias, so it is not bounded by the pair track's own 15.48 % share, but it is second
order. Below **1.3x** the route IS the mechanism.

**What I will conclude on a partial collapse**, stated now because it is the likely outcome and the
one a post-hoc story fits most easily: if C lands between 1.3x and 1.9x I will report the route as a
measured contributor, quote its share of the growth numerator by section and leaf family, and
explicitly decline to call it the mechanism. I will not rename the site to fit. A partial collapse
with the LayerNorm affine share still above 80 % of the residual means the growth is still being
*reported* in the same place and the cause is still upstream of both named sites.

## If both pins fail, these are the residual candidates

Named now so none of them is a post-hoc rescue:

1. **The 328 pad columns themselves.** 56 real tokens at both widths, and `boundary_c64.pt` IS
   `boundary_n384.pt` cropped, bit-identical. Whatever the pad rows contribute has to come through
   the mask reaching the bias (D180, D193).
2. **`batched_matmul`'s program config on `attn @ v`.** That contraction reduces over `k_len`: 2
   K-tiles at 64, 12 at 384, bf16 operands with fp32 destination accumulation. Accumulation depth
   grows 6x with width and the pick is shape-keyed. This is arithmetic, it is width-keyed, and it is
   in neither named site.
3. **The fused kernel's online softmax at 384** (running max, rescale), which arm C removes; if C
   collapses the growth this is candidate 3 promoted, not the chunk count.
4. NOT a candidate: LayerNorm's own reduction. It contracts the channel dim (c_s 384 / c_z 128),
   which does not move with the token axis, so the site where the growth is reported cannot be where
   it is created. `of3t-lnaffine` already measured that contraction innocent at 6.06e-02 against
   float64 on its own operands.

## Cost

Fold time per arm with AICLK sampled DURING the run at 4 s intervals, min/median/max reported. A pin
that costs throughput is a perf lever whether or not it was meant as one, and the verdict on whether
either pin could ever be proposed as a fix has to carry its clock.
