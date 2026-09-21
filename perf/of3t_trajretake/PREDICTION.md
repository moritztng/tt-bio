# of3t-trajretake — PREDICTION, registered before the first run

Registered 2026-09-22, on `wk/of3t-trajretake` at `57ff5f66f37b1cbbb4330f32ddb6e6dd0a5bdcc9`
(tree identical to `origin/wk/of3t`), before any run of this row. Nothing below was measured on
this tree when it was written; the only numbers quoted are `of3t-modeltraj`'s, cited as such.

## What is being re-taken and why

`perf/of3t_modeltraj/traj_shipped.json`
(sha256 `c471cd33556c3aa9c19ca9c4a32b80bc9ca6dca4e272cc53865447dfcf2c4171`) was written at
2026-09-21 02:35:02 UTC on commit `a0851ef8a`. The `_PARAMS` re-key that repairs D126 landed at
2026-09-21 02:55:54 UTC on `965c24f52` and is on this tree at `tt_bio/autograd.py:216-245`, in
the `value` SETTER rather than at the three call sites that replace a leaf's handle. The old
artifact is therefore twenty minutes older than the fix for the defect it reports, and its
`flag_reach` also records `_SOFTMAX_BW_RENORM: false`, so it is the pre-D56 arm as well.

## Predictions

**P1 — `tape_resolves_after_step` reads 26 of 26 at every one of the 20 steps, on the `shipped`
arm, with `repin` OFF.** The mechanism: `AdamW.step` assigns through the property at
`tt_bio/train/optim.py:284` (`t.value = to_device(...)`), the setter deletes `_PARAMS[id(old)]`
and writes `_PARAMS[id(new)] = self`, then `Parameters.rebind` puts that same `new` handle into
the model slot, so `parameter_for(raw)` finds the leaf and its `t.value is raw` test holds. If
this reads 0 the fix does not reach this harness and that is the finding — bigger than a stale
artifact — and I stop and report it instead of finishing the run.

**P2 — `grad_norm` is non-zero at all 20 steps**, starting at 0.4935344531218027 (the old
artifact's k=1, which was already a real gradient) and falling to roughly 0.43 by k=20.
`rebound` stays at 26 rather than collapsing to 0 from k=2.

**P3 — `d_ours_norm` > 0 at every k where `d_theirs_norm` > 0**, i.e. every k >= 2. At k=1 both
sides are exactly 0 because lr(1) = 0 exactly under the AF3 warmup, and per PROTOCOL S7a that
agreement is not a pass on either side.

**P4 — the retake lands on `of3t-modeltraj`'s `repin`/`renorm` numbers, not between them and
`shipped`.** Concretely I predict `rel_d` = 8.039271e-02 at k=2 and 4.763338e-02 at k=20, with
the worst tensor at k=20 being `transition_s.1.layer_norm.weight` at about 1.051e-01, and
`||d_20_theirs||` = 5.086941e-01 unchanged (the reference side is CPU float64 and this row does
not touch it). The reasoning: the re-key in the setter should be functionally identical to the
`repin` arm's explicit `ag.parameter(t)` after each step, and `of3t-modeltraj` measured the
`renorm` arm BIT-IDENTICAL to `repin`, so D56 being on should not move these digits at this
boundary. A deviation from 8.039271e-02 / 4.763338e-02 means either the setter re-key is not
equivalent to an explicit re-pin, or D56 does reach this boundary after all — either way a
finding, and I will say which.

**P5 — the growth law is DEFINED and sub-linear:** exponent about -0.2482, r2 about 0.857 over
k = 2..20. The old artifact reads exponent 0.0, intercept 0.0, r2 1.0 and `shape: "sub-linear"`,
which is a fit of the constant 1.0 and carries no shape information at all. After this row's
guard, a stationary arm returns `shape: "undefined"` with the reason, not "sub-linear".

**P6 — `softmax_bw_renorm_asked` = true and `softmax_bw_renorm_live` = true** on the retake arm,
read back off the LOADED `tt_bio.taped_ttnn` module rather than off the environment, because
what the environment asked for and what the imported module holds are two different facts and
D56's own staleness was exactly that gap.

**P7 — the A/A floor is exactly 0.0 at all 20 rungs** and the A16 zero-gradient baseline is
pinned at `rel_d` 1.0 with `d_ours_norm` 0.0 at every rung, reproducing the shape the old
`shipped` artifact had. Both are run before any number above is quoted. If the A/A is not
exactly zero, every magnitude in this row is unreadable and the row reports that instead.

## Scope, stated in advance so it cannot be widened after the fact

The arm covers `diffusion_module.diffusion_conditioning`, 26 of 26 reference tensors,
36.94617946669957 % of the model's squared gradient norm against
`of3t-wholemodel`'s denominator (model squared gradient norm 10.279642678524981). The charter's
scope clause asks for 99.2594 % and this row does NOT reach it and does not try to. I predict
the scope figure is unchanged, because it is a property of the boundary capture and not of the
tree.

## What would falsify each prediction

P1: any step with `tape_resolves_after_step` < 26. P2: any `grad_norm` of exactly 0.0.
P3: any k with `d_theirs_norm` > 0 and `d_ours_norm` == 0. P4: `rel_d` at k=2 or k=20 differing
from the quoted digits. P5: exponent > 1.0 (super-linear fails at any magnitude), or a fit
published on an arm that did not move. P6: `live` false while `asked` true. P7: a non-zero A/A
at any rung.
