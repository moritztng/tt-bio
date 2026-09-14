# The Boltz-2 token diffusion-transformer layer: why it sits at 19.6 % of the cube

whglx `j10glx02`, Wormhole B0 Galaxy, card 2, 8x9 grid, loadavg 5.2. One process, one device,
one session, arm set run whole once per block in a fixed order, minimum over 5 blocks, three
enqueues per synchronize on the calibration arms and 20-40 on the small ones.

Everything below is a **Wormhole** number. qb2 is committed to release gates, so no Blackhole
re-price was available; the transfer constants are applied as predictions and labelled as such.

Code: `dit_layer.py`. Data: `dit_layer_whglx_wh.json`.

## Calibration, this session

    dense bf16 HiFi4 4096-cube      54.78 TFLOP/s      A/A 0.77 %
    starved 8192^2 add             238.5  GB/s
    DiffusionTransformerLayer       1.1424 ms/call     A/A 0.27 %

The layer is the real `tt_bio.tenstorrent.DiffusionTransformerLayer` at dim 768, 16 heads,
512 tokens, with a `[1,16,512,512]` pair bias: the shipped module and the shipped op sequence,
random weights.

## The answer

**The layer's matmuls are fine and its op count is not.** At 12.281 GFLOP/call the layer runs at
10.75 TFLOP/s, **19.6 % of the cube**. Its eleven matmuls alone, at their real shapes and real
residency with every layer_norm, sigmoid, SwiGLU and residual removed, run at **39.0 % of the
cube** — the second-best class in the fold behind OuterProductMean's 61.8 %, and better than the
pair Transition's 28 %. They are **47.0 % of the layer's time**. The other 53.0 % is about
eighteen non-matmul ops on a tensor of 384 tiles.

    layer_ship        1.1424 ms   10.75 TFLOP/s   19.6 % of cube
      AdaLN             0.1589 ms
      AttentionPairBias 0.4224 ms
      transition        0.4686 ms
      the rest          0.0924 ms   out-projection of s, one multiply, two adds
    mm_only           0.5373 ms   21.36 TFLOP/s   39.0 % of cube   <-- the shape-honest roof

## The mechanism: a launch floor, measured

The same `ttnn.layer_norm` at one, two and four times the rows:

    [1, 512,768]   0.0289 ms
    [1,1024,768]   0.0363 ms
    [1,2048,768]   0.0539 ms

which is a straight line: **20.6 us fixed plus 8.3 us per 512x768 block** (the fit predicts the
2x point to 2.6 %). So **71.2 % of a `[1,512,768]` layer_norm is fixed launch cost**, and the
8.3 us that is not moves 1.5 MB at 180.3 GB/s, **75.6 % of this session's 238.5 GB/s DRAM roof**.
The marginal part is already near its roof. The fixed part is the whole problem.

Eighteen such ops at ~20 us is 0.37 ms, **32 % of the layer**. That is the gap between 19.6 %
and 39.0 % of the cube, and nothing in it is a DST barrier, an SFPU head-of-line stall or a
bandwidth wall.

`AttentionPairBias` partitions cleanly (the five parts sum to 1.3 % of the measured whole):

    fused qkv linear      0.1114 ms   26.4 %
    head split            0.0630 ms   14.9 %   nlp_create_qkv_heads, zero FLOPs
    fused SDPA            0.0960 ms   22.7 %
    head re-assembly      0.0761 ms   18.0 %   slice + permute + reshape + permute, zero FLOPs
    gate + out projection 0.0811 ms   19.2 %

**A third of AttentionPairBias moves data between head layouts and computes nothing.**

## Four levers screened, all against a 0.27 % A/A floor

**1. Concatenate the conditioning half across all 24 layers. 1.0022x. Dead.**

Six of the layer's eleven matmuls are a pure function of `s`: two AdaLN scale/shift pairs and two
output projections, 3.624 of the layer's 12.281 GFLOP. Every layer takes the *same* `s`, and
`LayerNorm_gamma_i(s) = gamma_i * s_hat` with `s_hat` shared, so folding `gamma_i` into the i-th
weight block turns 144 `[512,768]x[768,768]` matmuls and 48 layer_norms per sampling step into
one layer_norm and two `[768, 96*768]` matmuls. Same dot products, same FLOPs.

    sterms_step (shipped, 24 layers)     6.4063 ms   24.8 % of cube
    concatenated, no slices              4.8102 ms   1.3318x
    concatenated, with the 144 slices    6.3924 ms   1.0022x

The concatenation is worth 1.33x and **the 144 slices needed to hand the blocks back cost
1.5822 ms, 11.0 us each — the launch floor again, once per slice.** Blocking the sampling steps
eight deep instead (legal: the sigma schedule is fixed, so `s` for future steps does not depend
on the coordinates of the step before) gives 1.4316x on the matmuls and 1.1055x after the same
slice tax, **1.0232x on the layer**.

The general statement, which is the point of this row: **on a 384-tile tensor a ttnn slice costs
what the matmul it feeds costs, so every batched form of this work has to be un-batched again at
exactly the price the batching saved.** `ttnn.matmul` refuses a batch-1 left operand against a
batch-N right one (`a_shape[i] == b_shape[i] || (i == 1 && in0_reuse())`), so carrying the
concatenation on the batch axis, where the slices would be contiguous, is not expressible.

**2. One-op head re-assembly. 1.0401x on the layer, 1.1276x on AttentionPairBias. Alive, under
the bar.**

`ttnn.experimental.nlp_concat_heads` replaces the four-op epilogue: **0.0761 -> 0.0216 ms,
3.53x**. It keeps the 16 padded head lanes, so the gate and the output projection run at width
1024 instead of 768, which costs 3.4 us on each (`att_out_k768` 0.0301 -> `att_out_k1024`
0.0335). That is exact rather than approximate: SDPA's v pad lanes are zero because the fused
qkv weight was zero-padded there, so o's pad lanes are zero, anything multiplies them to zero,
and zeroing the matching rows of `proj_o` drops them.

Measured integrated, not summed: **`layer_L1` 1.0983 ms, 1.0401x.**

**3. `ttnn.mac` for the AdaLN scale-and-shift. 0.9867x. A regression.**

AdaLN applies `sigmoid(s_scale)` and adds `s_bias` as two in-place eltwise ops. `ttnn.mac` does
both in one and the sigmoid moves into the s_scale linear's own activation epilogue, where it is
free. Three ops become two, and the layer gets **slower**: 1.1578 ms. In-place `multiply_` and
`add_` write into a buffer that already exists; `mac` allocates its output. Stacked with lever 2
it drags 1.0401x down to 1.0265x.

**4. The whole conditioning half free. 1.3104x — an upper bound, not a lever.**

`layer_hoisted` 0.8718 ms is what the layer costs with all six s-projections and both s-norms
already in hand. Nothing reaches it: lever 1 is the only arrangement that computes them
elsewhere, and it is a wash.

## What is left, with a prediction

The one thing the floor allows and ttnn does not already express is **a fused AdaLN kernel** —
`layer_norm`, `* sigmoid(s_scale)`, `+ s_bias` in one `ttnn.generic_op`, the E6/K1/K2 pattern,
no tt-metal build. The a-side of AdaLN is 0.1589 - 0.0938 = 0.0651 ms for three ops. One kernel
moving the same 3.0 MB costs 20.6 us of floor plus 16.7 us at the 180.3 GB/s marginal rate =
**37.3 us**, so it saves 27.8 us per AdaLN and 55.6 us per layer: **1.0512x**, and stacked with
lever 2 (sub-additively) **1.07-1.09x**. Predicted here, before building, so it can be scored.

## Transfer

Lever 2 is **layout** (k = 1.329 +/- 0.157): 1.0401x on Wormhole predicts **1.053x on Blackhole,
band [1.046, 1.060]**, which straddles the 1.05x kill line, so it must be re-priced on qb2
rather than ranked off this. Lever 1 and lever 3 are **placement** (k = 1.161 +/- 0.047) and both
are at or below 1.0 on Wormhole, so no multiplier rescues them. The 39.0 %-of-cube arithmetic
figure is a fraction of the part's own cube and barely transfers at all — 18.8/19.4 = 0.969
Blackhole to Wormhole in `roof-shape-honest-roofs` — so it carries as the shape property it is.

## Not verified, and owed before anything lands

The pad-lane argument for lever 2 is an algebra argument, not a measurement: this harness ran it
on random padded weights, so it has no parity number. A float64 check of `nlp_concat_heads` plus
a zero-padded `proj_o` against the shipped four-op epilogue, and the 512 aa / 298 aa structure
arm against the 0.60 A bar, are both owed before it ships.
