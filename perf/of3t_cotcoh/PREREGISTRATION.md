# of3t-cotcoh — what makes the cotangent error COHERENT

Committed before the first capture exists and before any number this row produces. Two-sided,
with the falsifier written beside every claim. Nothing below is adjusted after a reading.

## Frame, reference, masking

Frame: the MODEL frame. `of3t-modelframe`'s `boundary_model_n384.pt`
(sha256 583bcd7c91ce6ed46aea59844954fb2bf7ddc3d553d7f9d4f238998183db99e2) and
`cot_model_n384.pt` (4e66d1ef45da2eec18fdff9489d68e980df928dc43d10fd4af82d14ec67141ac), reused
and not rebuilt.

Reference: upstream OpenFold3 0.4.3's own stack in **float64** on the same boundary and the same
cotangent, `/home/ttuser/of3t_frame384/of3pkg043`, the producer `perf/of3t_trunkg043/ref_grad.py`
with a capture hook added and no arithmetic changed. **Every ratio in this row names float64 as
its reference** unless it says otherwise, and where upstream's own bf16 autocast is quoted it is
labelled on the same line (R133: an against-fp32 reading and an against-bf16 reading of the same
op are both true and differ by four orders of magnitude).

Masking: 56 real tokens of 384. Family A is masked to those 56 rows; family B to the 56x56 real
pair block. `of3t-apbleaf` measured the cost of not masking: 1.3827 unmasked against 1.7e-03
masked on the same quantity.

## The two site families, chosen by the model frame's own attribution

From `perf/of3t_modelframe/ATTRIBUTION.json`, `by_leaf_module`, against upstream bf16:

    A  attn_pair_bias.layer_norm_a           42.004 % of trunk error mass, rel_l2 2.9189
    B  pair_stack.pair_transition.layer_norm  32.408 %,                    rel_l2 0.7842

74.41 % of the trunk's error mass between them. Our names: `pre_norm_s` and
`transition_z.norm_weight`. Both exist in all 48 blocks, which is what makes the walk-back a
curve and not a pair of points.

## The quantities

At each site, `G_dev` and `G_ref` are the cotangent ARRIVING at the LayerNorm output, masked and
flattened to `[P, C]` (A: P = 56, C = 384; B: P = 3136, C = 128). `E = G_dev - G_ref` in float64.
`Xh` is the device's own `xhat` at the same site, masked the same way. The affine reduction is
`R(M)_c = sum_t M_tc Xh_tc`, evaluated in float64 — `of3t-lnreduce` measured the shipped
`ttnn.sum` to be float64-equivalent (relative error exactly 0.0 on all-ones to K = 147,456), so a
float64 evaluation is the operator, not a stand-in for it.

1. **COH_SPEC** = `sigma_1^2 / sum_i sigma_i^2` of `E`, with the effective rank
   `sum sigma^2 / sigma_1^2`. Control: the same statistic on `E_flip`, `E` with each ROW
   multiplied by an independent random +-1. That preserves every entry's magnitude and every
   row's internal shape and destroys only the alignment ACROSS positions, which is the thing
   being measured. 8 draws, median and range reported.

2. **COH_RED** = `||R(E)|| / median_flip ||R(E_flip)||`. This is COH_SPEC priced in the only
   quantity the leaf sees. `COH_RED = 1` means the error passes the reduction as noise;
   `COH_RED >> 1` means it survives the sum. Ceiling `sqrt(P)`.

3. **AMP_ISO** and **AMP_ACT**, both as ratios of `||R(M)||/||M||` to the same quantity for
   `G_ref`: `AMP_ISO` for `N` iid gaussian matched in Frobenius norm to `E` (8 draws, seed floor
   reported), `AMP_ACT` for `E` itself. `AMP_ACT / AMP_ISO` is the number `of3t-lnreduce`
   inferred as 4.33x from two published rows and never measured directly. It is measured here.

4. **STRUCTURE**, five least-squares fits of `E`, each reported as explained fraction
   `1 - ||E - fit||^2 / ||E||^2`:

       S1  rank-1 outer          s (x) v          one injected direction
       S2  per-channel scale     diag(d_c) G_ref  a shared per-channel scale / quantisation
       S3  per-position scale    diag(d_t) G_ref  a mask or a per-position rescale
       S4  position broadcast    a_t (x) 1_c      a scalar broadcast across channels
       S5  xhat-aligned          b_t (x) Xh_t     the LayerNorm backward's own third term

## Step 2, the walk-back: PRE-REGISTERED as INJECTED

The cotangent entering block 47's backward IS the reference's own float64 cotangent, exact by
construction. Any error at block 47's sites is therefore made inside block 47. I predict the
coherence is **INJECTED at the top of the stack and propagated**, not accumulated block by block.

Grading, on COH_RED per block for each family separately, with `r` the Spearman rank correlation
of COH_RED against depth `47 - b`:

* **INJECTED** if `COH_RED(47) >= 0.5 * max_b COH_RED(b)` AND `r < +0.5`.
* **ACCUMULATED** if `r >= +0.8` AND `COH_RED(47) <= 0.25 * COH_RED(0)`.
* Anything else is **MIXED** and the curve is published without a verdict word.

Why INJECTED and not the other: block 47 holds 13.620 % of the trunk's error mass in this frame
on an exactly correct incoming cotangent, against under 7 % for every block from 46 down except
block 04. An error that accumulated gradually would be near zero at 47. The reading that kills
this is `r >= +0.8` with a small COH_RED at 47, and if that is what the curve does this row says
ACCUMULATED in its first heading.

## Step 3, naming the op: the candidate list, fixed now

The op must be **applied identically across positions**. On the path that produces the cotangent
arriving at these two sites:

    C1  the softmax backward renorm     dx = p*(dy - sum_j p_j dy_j), a per-row scalar broadcast
                                        across the softmax axis. TT_BIO_SOFTMAX_BW_RENORM.
    C2  the LayerNorm backward's two means, broadcast across channels within a position
    C3  a Linear backward dx = dy W^T, the same W for every position
    C4  the attention mask / inf fill
    C5  the pair-bias broadcast over query positions
    C6  a transpose / reshape seam (tile padding folded into a real row)

NAMED requires a break control that MOVES COH_RED by at least 1.5x in either direction at the
site. A candidate that changes the magnitude but leaves COH_RED inside 1.5x is NOT the carrier,
and this row says so rather than promoting a magnitude into a mechanism (three wrong mechanism
calls were made last pass exactly that way).

## Step 4, the price

If the named op were exact, the trunk is rescored through `perf/of3t_modelframe/score.sh` and the
clause read against **0.15210099830945006**. The trunk must fall **2.2349x**; a perfect trunk
reads 0.6752x the bar. Any counterfactual is reported with the composition control
(`framegate/score_the_best_possible_artifact.py`, rel_difference exactly 0.0) run first.

## Floors owed before any ratio is read

* A/A on the device capture: the same arm twice, the masked cotangents compared tensor by tensor.
* The instrument control: the capture arm's trunk gradient against the banked frame-matched arm,
  which must be bit-identical. The wrapper calls the shipped verb and changes no arithmetic.
* The reference hook control: the hooked reference's 2,736 leaf gradients against the unhooked
  reference on the same boundary.
* The pad control: `||E||` on the pad rows, which must be structurally zero where the cotangent
  is zero.
