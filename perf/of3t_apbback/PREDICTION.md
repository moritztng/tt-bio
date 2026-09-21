# of3t-apbback — pre-registration

Written and pushed before the first arm of this row ran. Branch `wk/of3t-apbback` off
`wk/of3t` at fd70adde2. Card 0 (p300c) on qb2. Scratch `/tmp/of3t/of3t-apbback/`.

The brief: the trunk's backward is 5.4139x upstream's own bf16 (0.3148) after the width
dependence is removed, the forward is 1.0345x, `attn_pair_bias` and `single_transition` carry
it, and the job is to bisect one block's backward op by op with host float64 substitutions.

## P0 — the reference audit, run before anything else

Two figures on the record disagree by 4.45x and the brief quotes one of them:

* `of3t-apbgrad` SCOPE_c64.json, RENORM arm, 2,736 tensors: **0.38330656678074404**, against a
  float64 reference whose trunk squared gradient norm is **1.8714981803225081**, floor
  0.3739355395070864, so **1.0251x** the floor.
* `of3t-padshape`, width 64, RENORM=1, the same 2,736 tensors: **1.7043040667627918**, against a
  reference whose trunk gradient norm is **0.7740253256713384**, floor 0.3148, so **5.4139x**.

Both are the `flipped` arm (`scale_pair_bias=True`, confirmed in
`perf/of3t_bwdaccum/SCOPE_LEVERS_c64.json`'s `arms.NONE.config`), both are RENORM=1, both are
c64. So the arm axis and the flag axis are excluded on the record and the reference is the only
remaining difference: `ref_grad.py`'s BOUNDARY-LOCAL float64 backward (upstream 0.4.3 run on the
same captured s_in/z_in and the same captured block-47 cotangent) against the pinned MODEL-LEVEL
`grads_f64_043.pt` (upstream's own full-model float64 backward on batch_step003, 5nw3, crop 384).

**PREDICTION P0.** Both figures reproduce in this row's own processes on the same device tensors,
and the 4.45x between them is the reference and not the arm, the flag, the boundary or the
scorer. Refutation: if the same device gradients scored against both references do not give
0.3833 and 1.7043 to within 1e-6 relative, one of the two is not the quantity its row named and
that is the finding instead.

**What P0 decides.** A substitution inside one block can only move the part of the disagreement
that the block's own backward creates. If P0 holds, the 5.4139x contains a component that is a
disagreement between two float64 references rather than a defect in any op, and deliverable 1
must be scored in the frame where a substitution can act (boundary-local) with the model-level
frame quoted beside it. I register now that I will report BOTH and will not quote a recovery
share against a denominator a substitution cannot reach.

## P1 — deliverable 2, the control that makes the rest mean something. Runs FIRST.

Target block: **4** (of3t-lnaffine's worst trunk leaf is `blocks.4.attn_pair_bias`,
`of3t-apbgrad` gives it rel 2.53 at cos -0.149 carrying 20.21 % of the error mass, and block 4
is 13.223 % of the trunk's error mass against block 0's 15.647 % with 1.44 % of the parameter
mass, so it is the highest error-per-mass block that is not the stack's last rung).

Frame: **teacher-forced**. `of3t-lnaffine` measured that injecting the reference cotangent at
every block boundary makes `attn_pair_bias.layer_norm_a.weight` 5.3x WORSE (3.0400 -> 16.0853)
while its own isolation reading is unchanged at 6.06e-02. Under a wrong incoming cotangent the
block-local defect is partly cancelled, so the bisect is only well posed with the reference
cotangent injected: then the block's leaf gradients depend on the block's own forward
activations and its own backward arithmetic and on nothing above it.

**PREDICTION P1.** With every taped op in block 4's `attn_pair_bias` and `single_transition`
backward replaced by a host float64 computation on the operands the card itself produced, the
block's leaf gradients recover **>= 90 %** of their error mass against the float64 reference,
where recovered = 1 - (error mass after / error mass before) over the block's own leaves in the
teacher-forced frame. The residual I expect is the forward activation error, which the brief
puts at 1.0345x on s, and I register that the residual is the CEILING on what any backward-side
repair can buy. If the all-float64 arm recovers less than 50 %, the harness is not reaching the
computation and every row of deliverable 1 is void and will be reported as void.

## P2 — the null substitution

The same interception, the same scope detection, the same node-function replacement, with the
replacement being the op's own device closure re-taped on the same operands.

**PREDICTION P2.** 0 of the block's leaf gradients move: max absolute difference exactly 0.0
against the uninstrumented arm, `torch.equal` true on every one. Anything else means the harness
perturbs the computation it is measuring and the deliverable-1 table is unreadable.

## P3 — deliverable 1, which op

Substituted one at a time, in the teacher-forced frame, at block 4:
`ln` (the LayerNorm backward), `linear` (the four projections' dW/dx/db), `sumlead`
(`_sum_leading`, autograd.py:432/454), `softmax`, `matmul` (q@k^T and probs@v),
`gate` (`multiply(o, g, [SIGMOID])`), `pairbias` (the `linear_z` fold and the bias scale).

**PREDICTION P3: `ln`, and specifically the xhat the backward RECOMPUTES.**
`_taped_layer_norm`'s backward does not keep the forward's normalised activation; it rebuilds
mean, var, rstd and norm from `x` and, on the shipped path, `ttnn.mean(xv, dim=-1)` for the mean
carries no compute_kernel_config at all. `of3t-apbgrad` scored every one of the 30 ops in float64
against the operands the card had and found none of them individually wrong, and
`of3t-lnaffine` measured the dW contraction itself innocent at 6.06e-02 and well conditioned at
KAPPA 9.41. The quantity neither of those two touched is the OPERAND `xhat`, which is neither an
op's arithmetic nor its input: it is recomputed inside the backward, and a recomputed xhat that
is 2^-8 off feeds every one of the four LayerNorm affine leaves that carry 74.044 % of the
trunk's error mass. Numerically: `ln` alone recovers **>= 40 %** of block 4's leaf error mass.

Second choice, and I name it because the cosine says direction rather than magnitude:
`sumlead`, at >= 20 %.

**The compositional branch.** If no single substitution recovers >= 20 % and only the all-float64
arm does, I will report that the defect is in how the sub-module composes correct ops and not in
any one of them, because that is a different fix and the campaign needs to know which it is.

## Standing

AICLK sampled DURING every device arm and the median reported, board class named,
`host_quiet.py` run and its state reported whether green or red. Gradients scored against
`grads_f64_043.pt` sha256 1d4ea9225f854afe06a8adedcb5f8aa1c53656fbf8cefdaa3a451d0327895cc4,
digest checked before loading, and against `ref_grad.py`'s boundary-local float64 arm, with
every figure naming which. A16 zero baseline and an A/A before any number. Artifacts record
branch, commit, host and card. Release-gated: stays on this branch, unmerged.
