# of3t-confpfe: predictions, committed before arm CF384 runs

## What bisection step 1 found

The float64 and bf16 references that scored GO384 (`of3t_denoise/ref384`, `REF_F64.json`,
`REF_BF16.json`) skip `pae` and `pde`: `missing ['pae_logits']`, `missing ['pde_logits']`. Our
step scores both, `pae` 4.0950 and `pde` 1.3915 at weight 1e-4 each (`DEV_GO384.json`
breakdown), as OpenFold3's `initial_training` / `weighted-pdb` table says it should
(`OF3_LOSS_BASE`, `pae_head_enabled = True` in upstream's `model_config.py:46`). The cause is
`perf/of3t_fullstep64/ref_step.py` `trunk_and_heads`, which ran the confidence Pairformer and then
built only `resolved_logits` from it. Both terms read the Pairformer's z output, so the whole
z-path gradient of `aux_heads.pairformer_embedding` existed on the device side and not in either
reference. That is the defect's signature: norm ratio median 12.8, cosine 0.26, worst tensor a
pair-stack LayerNorm bias.

`tt_bio/` is right and does not change. The fix is the reference: `ref_step.py` now builds
`plddt`, `pae` and `pde` logits too, in our layout, from upstream's own heads. Smoke at 64 tokens
(the batch's real tokens fit in 64): `pae_logits` and `pde_logits` now seed the backward, the
confidence gradient mass moves from 1.03e-7 to 2.93e-6, and `resolved` reads
2.716587554675418, every digit of the old 384 f64.

The rebuilt references run on qb2 CPU (pc has 4 GB free), torch 2.8.0+cpu, where the old ones ran
on pc, torch 2.13.0+cpu. Upstream's bf16 bar is host-dependent; float64 should not be.

## Arm CF384

`perf/of3t_go384/chain.sh` and `devarm.sh` with row, scratch and card changed (card 0, p300c),
plus `devstep.py` saving the structure the confidence heads saw (`repr_CF384.pt`). Tree
f4d02be69 plus this row's perf-only commits; `git diff a9dd90e9e -- tt_bio/` is empty, so
`tt_bio/` is GO384's. Scored by `perf/of3t_fullstep64/score.py`, unedited, against
`ref384c/f64/grads_f64.pt` with `ref384c/bf16/grads_bf16.pt` as the bar, then
`section_ratio.py` (a8a062916, unedited).

## Predictions

1. `grad_CF384.pt` sha256 equals GO384's `9a5c0b81...` (probability 0.9). Nothing under
   `tt_bio/` moved and the draws are the same; card 3 to card 0 must not change a byte.
2. The five GRADIENTS conditions hold: unread 0, placed-but-empty 0, multi-placed 0, global rel
   at or below BF16's, mass at or better than bf16 at or above 0.95 (probability 0.95). Global
   rel 0.156, interval 0.150 to 0.165: the confidence head holds ~1e-8 of the mass.
3. `aux_heads.pairformer_embedding` within 3x its bf16 (probability 0.8), rel between 0.2 and
   1.0.
4. `aux_heads.experimentally_resolved` stays past 3x its bf16 (probability 0.8), rel 0.130
   unchanged to three digits. Its gradient is logits-only and the forward it reads did not
   change, so the new float64 reference holds the same resolved gradient as the old one (checked
   directly: max relative difference below 1e-10, probability 0.95).
5. Host control: the diffusion module's float64 gradients, which the confidence path cannot
   reach (the rollout is detached), agree between the new qb2 reference and the old pc one to
   relative 1e-10 or better (probability 0.9).
6. The `resolved` split, offline at 64 tokens: the float64 reference evaluated at the device's
   own rolled-out structure (`repr_CF384.pt`) against the device's resolved gradient. Predicted:
   rel falls within 3x of bf16's (probability 0.75), which would put the 5.7x in the structure
   the device rollout produced rather than in the resolved head.

Predicted verdict: STOP, the five holding, `aux_heads.pairformer_embedding` cleared, first
failing section `aux_heads.experimentally_resolved`. None of these tolerances move after the arm.
