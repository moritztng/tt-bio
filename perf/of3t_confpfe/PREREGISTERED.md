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

## Amendment, committed before the fixed arm runs

Bisection went one step further than the arm above anticipated, and the arm launched as CF384 at
01:05Z runs the tree BEFORE the second fix. Its artifacts are filed as **RF384** (reference fix
only): it measures the pae/pde reference correction alone. Predictions 1 to 6 above are scored on
RF384 as written. The arm the charter reads, **CF384**, is the tree with both fixes.

### What step 3 found

`resolved` sits 0.078 off float64 on the device (2.7917 against 2.7166), and the rollout
structure carries 0.003 of that: float64 at the device's own structure reads 2.7137
(`split_dev64`, prediction 6 refuted). So the head itself. `conf_split.py` runs upstream's float64
head on the device step's own confidence inputs (dumped by `devstep.py --conf-dump`, self-control
2.6e-8 to 3.7e-8 everywhere):

| output | head (device vs f64 at device inputs) | carried (f64 at device inputs vs f64 inputs) |
|---|---|---|
| resolved_logits | 2.28e-01, norm ratio 0.787 | 5.2e-02 |
| plddt_logits | 1.72e-01 | 4.5e-02 |
| pae_logits | 4.3e-03 | 5.8e-02 |
| pde_logits | 3.7e-03 | 2.8e-02 |
| si_conf | 2.4e-02 | 8.4e-03 |

The z-path heads are clean and the two s-path heads are not. The device dumped
`s_trunk` as BFLOAT16. `forward_device` took the caller's dtype; inference casts `si_trunk` to
fp32 before calling it, the training step handed the trunk's bf16 output, so the confidence
Pairformer's s-track ran bf16 on a tensor at absmax 2.28e5 and the LayerNorm the s-heads read lost
its precision before the fp32 cast `_ln` does. Fix 6267fc178: `forward_device` upcasts
`si_trunk_d` to fp32 at entry when it is not already.

### CF384 predictions (fixed tree, same chain, reference ref384c)

7. The five GRADIENTS conditions hold (probability 0.95), global rel within 0.005 of RF384's.
8. `aux_heads.experimentally_resolved` within 3x its bf16 (probability 0.7): the head error
   drops to the 5e-3 auxgrad measured with fp32 inputs, leaving about the 5e-2 carried from the
   trunk.
9. `aux_heads.pairformer_embedding` within 3x its bf16 (probability 0.75).
10. `resolved` loss value within 0.02 of float64's 2.7166 (probability 0.8).
11. `diffusion_module.*` gradients bit-identical to RF384's (probability 0.85): the fix reaches
    only what the confidence backward reaches, the trunk and the heads.

Predicted verdict: GO if 7 to 9 hold. None of these tolerances move after the arm.

## Second amendment, committed before CF384 runs on the real fix

The upcast (6267fc178) was wrong and is reverted (ef11e2409). Its arm is filed as **UC384**:
`resolved` 5.63x its bf16, the same as RF384's 5.65x, and the head probe on the step's inputs read
0.228 whatever the s_trunk dtype, tape or not, pads zeroed or not. Predictions 8 to 10 were made
for a fix that did not touch the defect: 8 (resolved within 3x) and 10 (resolved loss within 0.02
of float64) failed on UC384; 7 and 9 held. The mechanism named in the first amendment was a
retrodiction and is withdrawn.

What the defect is: at of3t-auxgrad's own capture boundary, upstream float64 matches upstream's
captured outputs to 0.0 and the device head reads `resolved_logits` 2.85e-01 on this tree against
5.37e-03 on auxgrad's tree f937e3029. `git bisect` localised it to a merge range where the probe
cannot run, and the diff of the head across that range shows the one argument that changed:
`scale_pair_bias` True -> False in the confidence Pairformer. 5ec179cda set it True ("the pair bias
was arriving at 1/sqrt(24) of the reference"); merge 45100fbce took the other parent's False.
Fix dfc21275d restores True. Probe on the fixed tree: boundary `resolved` 4.2e-3; on the step's
own inputs `resolved` 4.8e-3, `plddt` 4.8e-3, `si_conf` 5.2e-3 (were 0.228, 0.172, 2.4e-2), z-path
unchanged (pae 4.3e-3, pde 3.7e-3). What remains on the step's inputs is the trunk's: resolved
5.3e-2 and pae 6.0e-2 total, carried from `z_trunk` 5.1e-2 off float64.

### CF384 predictions (tree dfc21275d, card 0, reference ref384c, chain.sh unchanged)

12. The five GRADIENTS conditions hold (probability 0.95).
13. `aux_heads.experimentally_resolved` within 3x its bf16 (probability 0.75), rel 0.03 to 0.07.
14. `aux_heads.pairformer_embedding` within 3x (probability 0.85).
15. `aux_heads.pae` stays past 3x its bf16 (probability 0.7): the head's own error is 4.3e-3 and
    the fix cannot reach the 6.0e-2 it carries from `z_trunk`. So the predicted verdict is STOP on
    `aux_heads.pae`, with the cause moved from the head to the trunk's pair output.
16. `resolved` loss within 0.02 of float64's 2.7166 (probability 0.8).
17. OpenFold3 inference byte-identical across the fix (probability 0.9): inference runs the
    host s-path, which does not read this flag.

## Outcome (after CF384; nothing above edited)

| # | prediction | result |
|---|---|---|
| 1 | RF384 gradient identical to GO384 | holds: 3822 of 3822 tensors bit-identical (card 0 vs card 3); the file hash differs only by torch.save's embedded archive name |
| 2 | five conditions on RF384 | hold |
| 3 | pairformer_embedding within 3x after the reference fix | holds: 1.33x (RF384), 1.34x (CF384) |
| 4 | resolved stays past 3x on RF384; f64 resolved gradient unchanged | holds: 5.65x; 1.4e-15 |
| 5 | diffusion f64 gradients qb2 vs pc within 1e-10 | holds: 1.2e-13 |
| 6 | resolved split puts the 5.7x in the structure | refuted: the structure carries 0.003 of a 0.078 gap |
| 7, 9 | (UC384) five hold, pairformer within 3x | hold |
| 8, 10 | (UC384) resolved within 3x, loss within 0.02 | fail; the upcast is withdrawn |
| 12 | five conditions on CF384 | hold: unread 0 of 4158, 0, 0, global 0.1558 vs bf16 0.9482, mass 0.9807 |
| 13 | resolved within 3x | holds: 0.0239 vs bf16 0.0230, 1.04x (interval 0.03 to 0.07 missed low) |
| 14 | pairformer_embedding within 3x | holds: 1.34x |
| 15 | pae stays past 3x | holds: 0.0780 vs bf16 0.0128, 6.07x |
| 16 | resolved loss within 0.02 of float64 | holds: 2.7225 vs 2.7166 |
| 17 | inference byte-identical | holds: INFAB.json, four folds b70e195d... |

The bf16 bar is host-dependent: rebuilt on qb2 (torch 2.8) the global bf16 rel reads 0.9482
against pc's (torch 2.13) 0.6025; per section the two confidence sections move little (resolved
0.0228 -> 0.0230, pairformer 0.312 -> 0.338). The float64 reference agrees across the hosts to
1.2e-13.
