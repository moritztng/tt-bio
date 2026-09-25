# of3t-ditgap PREDICTION — registered BEFORE the first scoring run

D187: `diffusion_module.diffusion_transformer` reads **2.019x** upstream 0.4.3's own bf16 over
**43.622 %** of the model's gradient mass and has no owner. This row locates it. This file is
committed before any decomposition is computed, and it is the thing the pass is judged against.

## What is already fixed, not predicted

Read off artifacts, before scoring anything:

* the section is **456 tensors = 24 blocks x 19 leaves**, two sub-blocks per block:
  `attention_pair_bias` (10 leaves) and `conditioned_transition` (9 leaves).
* the DiT leg of the model-scope arm is
  `diffusion=/home/ttuser/of3t_f64softmax/device_grads_043all_renorm.pt`, produced by
  `perf/of3t_f64softmax/devgrad_f64.sh renorm` -> `device_gradient.py --structs all --cap
  /home/ttuser/of3t_softgrad/diffcap043 --softmax-bw-renorm`. It is **capture-driven**.
* `diffcap043/diffusion_boundary.pt` is **padded width 384** (`kwargs.token_mask` shape
  `(1,1,384)`) with **56 real tokens** (`token_mask.sum() == 56.0`), `no_samples = 48`,
  `use_conditioning = False`, `use_high_precision_attention = True`, captured
  `loss = 1.2675874205688995`. It is NOT in `perf/of3t_orchestrator/crops/CROPS.json`; this row
  extends that table rather than starting a second one (D193).
* both denominators are **full-model** runs: float64 `of3t_refprec/bundle_ref/grads_f64_043.pt`
  (sha256 `1d4ea922...`), upstream bf16 `pinned_p175/arm4_bf16_autocast/grads_f64.pt`
  (sha256 `ff78d7bc...`). Instrument floor upstream-f32 vs float64 = **7.4177415630308e-05**.
* the capture carries **its own in-frame float64 gradient**, `diffusion_boundary.pt["grad_f64"]`,
  and its own captured cotangent `cot` of shape `(1,48,422,3)`.
* the trunk's located signature, D191 UPDATE pass 342: 93.80 % LayerNorm affine, top two families
  `attn_pair_bias.layer_norm_a.weight` 34.89 % and `attn_pair_bias.layer_norm_a.bias` 27.43 %,
  three of 48 blocks carry 74.58 %, worst tensor `blocks.4.attn_pair_bias.layer_norm_a.weight` at
  `norm_ratio 7.3729`, `cos -0.0057`.

## The three candidate mechanisms, named in advance

**A. CROSS-FRAME, the same defect as D186.** A capture-scope numerator against a model-scope
denominator is unreadable, not merely pessimistic. The DiT leg is capture-driven from a 48-sample
`diffcap043` replay with `use_conditioning = False`; both references are full-model. D186 proved
exactly this for the pairformer leg of the same stitched arm, and `SECTION_ATTRIBUTION.json` says
in its own `why` field that it asks the same question of every other leg.

*Discriminator, and it needs no card:* score the capture's own `grad_f64` against the model-scope
`grads_f64_043.pt` on the 456 DiT tensors. Two float64 arms of the same function must agree.
*Falsified if:* they agree at or below the 7.4177e-05 instrument floor. Then the frame is matched
and the 2.019x is arithmetic, not framing. I put this first because it is the cheapest and because
the campaign has been wrong in this direction three times in seven passes (D186, D187, D189).

**B. The same object as the trunk's: cancellation-limited LayerNorm affine reductions.** Then the
top of the difference sits on the DiT's LayerNorm affine leaves, which are
`attention_pair_bias.layer_norm_a.layer_norm_s.weight`,
`attention_pair_bias.layer_norm_z.weight` and
`conditioned_transition.layer_norm.layer_norm_s.weight`, at `norm_ratio` far from 1 with `cos`
near 0, concentrated in a few of the 24 blocks, and reported at leaves whose own arithmetic is
innocent (`of3t-lnaffine`).

*Falsified if:* the top families are matmul weights (`mha.linear_o.weight`,
`swiglu.linear_a.weight`, `swiglu.linear_b.weight`, `linear_z.weight`) at `cos` near 1.

**C. The AdaLN conditioning gate.** `linear_ada_out.{weight,bias}`, `layer_norm_a.linear_s.weight`
and `layer_norm_a.linear_g.{weight,bias}` are the adaptive-LayerNorm path; `of3t-adaln` and
`of3t-conditioning` both touched it and D56's renorm is live on this arm, so the residual could be
the gate's backward rather than the norm's.

*Falsified if:* `linear_ada_out.*` plus `layer_norm_a.linear_{s,g}.*` together carry under 10 % of
the difference of absolute errors.

## Quantitative pre-registrations

1. **Concentration.** More than 60 % of the difference of absolute errors sits in fewer than 10 %
   of the 456 tensors, so under 46 tensors. The campaign's standing pattern, stated so it can
   miss.
2. **Non-uniformity across blocks.** The 24 blocks are not flat: either a monotone depth trend, or
   three to five blocks carrying over 50 %. Falsified by a profile flat within 2x across blocks.
3. **The DiT is NOT a copy of the trunk, in one specific structural respect.** The trunk's top two
   families are `layer_norm_a.weight` and `layer_norm_a.bias`, 62.32 % combined. The DiT's AdaLN
   has **no bias leaf at all** (only `layer_norm_a.layer_norm_s.weight` exists), so the trunk's
   second-largest family has no counterpart here and cannot recur. I predict SIGNATURE comes back
   PARTIAL rather than "same object" or "different object".
4. **The low floor is part of the 2.019x.** Our DiT absolute error is `0.11670185505910514` and
   the model-wide upstream bf16 floor is `0.10592054683439786`, so we read **1.1018x the
   MODEL-WIDE floor while reading 2.019x this SECTION's floor of 0.057809234623039614** — this
   section's floor is 0.5458x the model-wide one. I predict a material part of "2.019x" is that
   upstream's bf16 is unusually accurate on the DiT rather than that we are unusually bad, and
   that the section-local floor is the correct denominator to keep (A27) but the wrong number to
   read as a defect size on its own.
   *Falsified if:* the decomposition is dominated by tensors with `norm_ratio` far from 1 at `cos`
   near 0. That is a wrong computation, and a low floor does not excuse it.
5. **Direction.** I predict the row's VERDICT is NOT "2.019x is a clean defect in our arithmetic".
   I expect either A (frame) or 4 (floor) to carry most of it. I am recording that expectation so
   that finding a real defect counts as my prediction failing.

## What would make this pass worthless

Scoring the 456 tensors again in a third frame and reporting a fourth ratio. Every number below
names its frame, its scope, its padded width, and the host that produced its floor (A27, D189,
D180). Nothing here needs a card: both references and both arms are gradient dumps on disk, and a
float64-against-float64 comparison is host-independent to 2e-16 (D189).
