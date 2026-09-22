# of3t-trunkceiling PRE-REGISTRATION

Written before the first arm ran. Nothing in this file is read off a result, and the two
falsifiers below are two-sided on purpose: one kills the prediction if the bar IS reachable, the
other kills it if the levers turn out inert.

## The question, stated as a number

The charter's last unmet condition needs the pairformer trunk at

    mass_weighted_rel_l2(ours, upstream's own bf16 autocast trunk gradient)  <=  0.4361680548

over the 2,736 trunk tensors at crop 384 (56 real tokens, 384 padded), scored in THEIR parameter
space against the pinned references `ref_bf16auto_n384.pt` and `ref_f64_n384.pt`. The shipped
arm reads **1.0293953378** (`perf/of3t_frame384/FRAME_N384.json`,
`MATCHED.ours_vs_REF_LOCAL_bf16_n384.mass_weighted_rel_l2`), so the factor to find is 2.360x.

This row measures the BEST the trunk can do with every accuracy lever on at once, ignoring
wall-clock, and compares that ceiling to the bar.

## PREDICTED CEILING: 0.72, with a range of 0.55 to 0.95 -- it MISSES the bar by 1.3x to 2.2x

Point prediction 0.72. Stated before any arm.

Why that number and not a smaller one, from three facts already measured by this campaign:

1. **The single track's softmax has a float64 escape and the pair track's does not.**
   `TT_BIO_HOST_F64_SOFTMAX_AB` is the biggest accuracy lever in the stack -- it takes the
   diffusion module's gradient from 7.426217e+00 to 7.777580e-02 against upstream's own bf16
   (`perf/of3t_f64softmax/`). In the trunk it reaches exactly one class of site:
   `AttentionPairBias` is constructed with `softmax_site="pairformer"`
   (`tt_bio/tenstorrent.py:9302`), and `TriangleAttention` (`:7580`) has no `site_softmax`
   call at all. The pair track is where the trunk's mass is.

2. **On a bf16 operand the softmax levers have a floor and the campaign has measured it.**
   `softmax_precise_site`'s own docstring: at [1,16,384,384] the three arms read 2.029e-02 /
   1.646e-03 / 5.156e-04 in fp32 and **2.480e-02 / 1.266e-02 / 1.266e-02 in bf16** -- the
   precise config and the 5-op chain land on the same number, because bf16 storage is the floor
   and neither lever goes under it.

3. **The in-frame carrier is LayerNorm affine gradient mass, which the LN backward levers do
   address.** `FRAME_N384.json`'s in-frame top-by-error-mass list is
   `blocks.4.attn_pair_bias.layer_norm_a.weight` (rel 7.446, cos -0.0057),
   `blocks.0.single_transition.layer_norm.bias` (3.590) and
   `blocks.44.attn_pair_bias.layer_norm_a.bias` (9.410). `dW = sum_t g_t * xhat_t` over 24,576
   bf16 summands is the formula `--lever all` rebuilds in fp32, so a real move is expected
   there: 1.1x to 1.9x.

Compounding 1 and 3 against the pair track's floor in 2 gives 1.1x-1.9x on a 1.0294 reading,
i.e. 0.54-0.94, and perturbations in this stack are strongly sub-additive, so I take the middle
of that and predict 0.72. 2.360x is above the top of the range.

## FALSIFIERS

**F1, the bar is reachable.** If the all-levers-on arm reads <= 0.4361680548, this prediction is
wrong and the row's answer is that 2.360x exists on this silicon: the gap is engineering, the
carrier-hunting rows are spending a budget that exists, and the verdict is PARTIAL with a priced
work plan, not NO-GO.

**F2, the levers are inert.** If the arm reads > 0.95 -- i.e. the whole stack of levers buys less
than 1.08x on a reading of 1.0294 -- the prediction is also wrong, in the other direction. That
outcome says the levers do not reach the boundary's carrier at all and the ceiling is
approximately the shipped reading, which is a different finding (a reach problem, not a
precision floor) and would have to be reported as one.

Between them: the prediction survives only in 0.4361680549 <= ceiling <= 0.95.

## What the ceiling arm turns on

Every lever the runtime census finds on the pairformer path, at once, with no regard for time:

    TT_BIO_HOST_F64_SOFTMAX_AB=pairformer      host float64 softmax, AttentionPairBias
    TT_BIO_ACCURATE_SOFTMAX_AB=openfold3.trunk 5-op accurate chain, incl. tri_att_accurate_softmax
    TT_BIO_SOFTMAX_PRECISE_AB=pairformer       precise_config() on ttnn.softmax
    TT_BIO_TRIATT_SDPA_HIFI_AB=openfold3.trunk HiFi4 fused SDPA (already the shipped default)
    TT_BIO_SOFTMAX_BW_RENORM=1                 row-sum-corrected softmax backward (shipped default)
    TT_BIO_TRUNK_MATH_FIDELITY=hifi4           already the shipped default, quoted for the record
    --pf-set s_fp32_residual=1                 fp32 single-track residual
    --lever ceiling                            every LayerNorm-backward fp32 island AND the fp32
                                               softmax backward, which `--lever all` does not
                                               include

## Rules this row holds itself to

- The reference is float64, validated by the campaign's own pinned artifacts, quoted by sha256.
  Never another approximation.
- Coverage of every arm is MEASURED off a runtime census with call counts, never asserted from a
  selector's name. A lever that fires and is inert and a lever that never fires are different
  results and the counters separate them.
- Denominator floor declared before any relative error is read: tensors whose reference norm is
  below the floor are reported separately and never allowed to carry a headline.
- The A/A determinism floor is measured and quoted BEFORE any arm is read.
- Worst case per tensor beside every mass-weighted figure.
- If the ceiling misses the bar, the miss is stated as a factor. **The bar does not move.**
