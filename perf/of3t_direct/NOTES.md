# of3t-direct — D72's 42.2794 % becomes 3.1097 %

## The question

D72 re-scored every section of OpenFold3 against upstream's own bf16 error instead of against a
float64 ideal upstream never runs, and read **42.2794 %** of the model's squared gradient norm as
at or better than the recipe's own deviation. Both of D72's columns are distances from the **same**
float64 reference, and two distances from one reference do not order each other. So that reading
was limited by construction, and `of3t-trajectory` showed at pass 175 that the limit bites:
`layer_norm_s` reads 2.3809e-02 against float64 and **5.2797e-02 against their actual gradient**,
larger than either distance.

Trajectory's 547-tensor device arm covered two of D72's five AT-OR-BETTER rows. This row measured
the other three — including `diffusion_conditioning`, which is 36.9462 % of the model on its own
and the single largest section of OpenFold3.

## The answer

`D72_AT_OR_BETTER_ROWS_DIRECTLY_TESTED.json`. Every AT-OR-BETTER row in D72's table now has a
direct column.

| section | % of model | vs float64 | vs their bf16 | floor/r | × threshold | |
|---|---|---|---|---|---|---|
| `diffusion_module.diffusion_conditioning` | 36.9462 | 7.865385e-03 | **6.463839e-02** | 6.207150e-02 | **1.0414** | does not survive |
| `aux_heads` | 2.8431 | 2.299551e-03 | 2.360143e-01 | 2.361502e-01 | **0.9994** | survives |
| `msa_module` | 1.2400 | 1.621240e-01 | **2.260014e-01** | 1.686838e-01 | **1.3398** | does not survive |
| `diffusion_module.layer_norm_s` | 0.9835 | 2.380900e-02 | 5.279714e-02 | — | 1.70× floor | does not survive |
| `diffusion_module.layer_norm_a` | 0.2666 | 9.468300e-03 | 3.496811e-02 | — | 0.95× floor | survives |

The last two rows are `of3t-trajectory`'s, carried as published. The first three are this row's,
each from its own arm's gradient tensors.

**D72's 42.2794 % becomes 3.1097 %.** 39.1697 % of the model was read as at or better than
upstream's own recipe and does not agree with what that recipe actually computes.

And the 3.1097 % that survives will not carry a claim either. 2.8431 of it is `aux_heads`, which
survives by 0.06 % — and `aux_heads.distogram.linear.weight` is **100.0000 %** of that section's
mass, so the section's mass-weighted headline is one tensor. On the same scope the median over
tensors is 1.4613, **176 of 176** measurable tensors are past the 5.0e-02 per-tensor bar, the
worst is 7.9738e+00 on
`aux_heads.pairformer_embedding.pairformer_stack.blocks.1.attn_pair_bias.mha.linear_g.weight`,
and the A18 forward discriminator **fails** at 3.6515e-01 on `plddt_logits` against a 5.0e-02
bar. Under A18 a disagreeing forward invalidates the gradient taken at it. The remaining 0.2666 %
is one layer-norm weight.

The per-tensor bar is not a clean read on any of these scopes, in either direction, and the
comparison has to say so: **upstream's own bf16 gradient** is past the same 5.0e-02 bar on 24 of
26 conditioning tensors, 176 of 176 `aux_heads` tensors and 150 of 151 `msa_module` tensors. A
per-tensor count here measures how far these sections sit from a float64 ideal nobody trains in,
which is D70's finding, not ours.

## Why the threshold is `floor / r` and not `floor`

The measured quantity is `rel(device, bf16) = ||d − b|| / ||b||`, normalised by **‖bf16‖**. The
scope's floor is `rel(bf16, f64) = ||b − f|| / ||f||`, normalised by **‖float64‖**. A port that
reproduced float64 *exactly* therefore reads `floor · ‖f‖ / ‖b‖ = floor / r`, not `floor`. On
these three scopes `r` runs 1.0074 to 1.0597, so the correction is 0.7 % to 5.6 % — enough to
decide `aux_heads`, which sits 0.06 % under it. D76 shipped an unreachable threshold by treating
two `rel` figures with different denominators as sharing one; the division is now inside the
instrument.

Attainable range, checked before the thresholds were trusted: 0 when our gradient *is* theirs,
`floor / r` when it is the float64 ideal, 1.0 when it is zero (measured, A16, exactly 1.000000 on
all three scopes), unbounded above. All three pre-registered branches sit inside it.

## The geometry, which the float64 column cannot show

`agreement.py` now measures `cos(e, t)` directly, where `e = d − f` is our error against the ideal
and `t = b − f` is upstream bf16's error against the same ideal. D76 had to infer this from three
norms.

| scope | ‖e‖ | ‖t‖ | ‖e‖/‖t‖ | cos(e, t) |
|---|---|---|---|---|
| `diffusion_conditioning` | 0.015328 | 0.121864 | 0.1258 | **−0.2727** |
| `aux_heads` | 0.001243 | 0.129407 | 0.0096 | +0.0647 |
| `msa_module` | 0.057689 | 0.063609 | 0.9069 | +0.0152 |

This is what the shared subtrahend was hiding, and it is not one story.

- On `diffusion_conditioning` our error is **8× smaller** than upstream's own, which is what
  D72's 0.13× band was seeing and it is true. But the two errors are weakly **anti-aligned**, so
  ours adds to theirs instead of sitting inside it. An error of our size that were merely
  independent would read 1.0079× the threshold; the anti-alignment is what takes it to 1.0414×.
  The threshold is demanding by construction: it asks that our error be smaller than theirs *or
  aligned with it*, not merely small.
- On `msa_module` the two errors are essentially orthogonal and ours is nearly the same size as
  theirs, so the direct reading exceeds both distances — the same shape as `layer_norm_s`.
- On `aux_heads` ours is 1 % of theirs and slightly aligned, which is why it clears the
  threshold at all.

## Controls, per scope

A number with no floor beside it is not a result, so every scope carries all three.

| scope | A16 zero model | scale: arm4 vs arm2 | break: our cotangent | break: their permuted draws |
|---|---|---|---|---|
| `diffusion_conditioning` | 1.000000 | 6.253421e-02 | 2.259775e-01 (3.50×) | 3.620813e-01 (5.79× floor) |
| `aux_heads` | 1.000000 | 2.393689e-01 | 1.127762e+02 (477.8×) | 2.361491e-01 (**0.986× floor**) |
| `msa_module` | 1.000000 | 1.787655e-01 | 1.620385e+01 (71.7×) | 2.131250e-01 (1.19× floor) |

A16 is measured, not asserted, and reads exactly 1.000000 on all three.

**One control is inert, on the one scope that passes, and that has to be said.** Upstream's
permuted-draws arm moves the `aux_heads` reading to 0.986× its floor — it does not move it at
all, and what is left is just the bf16-vs-fp32 precision difference. That is structurally
expected rather than broken: `aux_heads` is scored on the distogram head's weight, whose loss
term does not depend on the diffusion sample draws, so permuting those draws cannot change its
gradient. The consequence stands anyway: on `aux_heads` the permuted-draws control cannot
discriminate, and the pass rests on the cotangent-scramble control alone, which does move, by
477.8×.

The cotangent break controls differ by scope because the arms do.
`diffusion_conditioning` has `--negative-control reverse-cot`, pairing sample k with sample 47−k's
cotangent. The `aux_heads` and `msa_module` arms had no comparison-breaking control at all — only
`--zero-model` and `--perturb`, which break OUR side and cannot test whether the comparison reads
THEIR seed — so both gained `--scramble-cot`: the same cotangent numbers written into the wrong
positions (the flattened cotangent reversed). Norm and shape are preserved exactly and the
forward, the weights and the arithmetic are untouched.

## The references, by digest (A24)

Verified before loading, and recorded in each result:

    arm4 bf16 autocast  ff78d7bc0bf7a4355014a470fbe931592bd4896fe06a8b400c161c08b5607ccb
    arm2 fp32 upstream  09f1217c8ea254d04f1bfdae73585f058cb2aec51557699bfae3a8090dadd548

Both match the pin in the brief. `--expect` stops the run on a mismatch rather than measuring.
`/home/ttuser/of3t_refprec/run/` was NOT read: `of3t-refprec` was re-running four arms into it
while this row worked (D75).

Each scope's capture was also checked against the bundle's own float64 gradient, because if the
reference our arm was scored against is not the bundle's, our gradient and arm4's are not
gradients of the same loss: **max abs diff 0.0 over 26 of 26, 180 of 180 and 154 of 154 tensors**
respectively. A zero compared count is a hard stop, not a pass — hardcoding the key prefix
`diffusion_module.` would have matched nothing on these two scopes and reported `identical` over
an empty set.

## What each arm reproduces

Every arm re-ran and reproduced its published figure before anything new was computed:

- conditioning: **0.007865384774093601**, every digit of the published 7.865385e-03, 26 of 26
  tensors, 100.000 % of the section
- `aux_heads`: median 1.0482e+00, worst 3.7941e+00 on
  `aux_heads.pairformer_embedding.pairformer_stack.blocks.1.attn_pair_bias.mha.linear_g.weight`,
  100.000 % of the section's own norm
- `msa_module`: mass-weighted **1.6211e-01**, median 8.0189e-02, 99.3368 % of the section's own
  norm = 1.2317 % of the model

## Re-running

The device arms, on one Blackhole card. They take 40 to 90 s of wall clock each; no AICLK was
sampled, because these are accuracy arms and nothing here is a perf measurement:

    perf/of3t_direct/condrun.sh real          # and `revcot` for the break control
    perf/of3t_direct/auxrun.sh aux real       # and `scramble`
    perf/of3t_direct/auxrun.sh msa real       # and `scramble`

Then the comparison, a few minutes and no card. Arguments are scope, device `.pt`,
break-control `.pt`, the capture, its gradient key and the prefix to strip:

    perf/of3t_direct/agreerun.sh diffusion_conditioning \
      /home/ttuser/of3t_direct/cond_grads_fp32.pt \
      /home/ttuser/of3t_direct/cond_grads_fp32_revcot.pt \
      /home/ttuser/of3t_cond_cap/cond_boundary.pt grad_f64 \
      diffusion_module.diffusion_conditioning.

    perf/of3t_direct/agreerun.sh aux_heads \
      /home/ttuser/of3t_direct/aux_grads.pt \
      /home/ttuser/of3t_direct/aux_grads_scramcot.pt \
      /home/ttuser/of3t_auxheads/cap043/boundary_aux_heads.pt param_grads ""

    perf/of3t_direct/agreerun.sh msa_module \
      /home/ttuser/of3t_direct/msa_grads.pt \
      /home/ttuser/of3t_direct/msa_grads_scramcot.pt \
      /home/ttuser/of3t_auxheads/cap043b/boundary_msa_module.pt param_grads ""

    python3 perf/of3t_direct/close_d72.py

## What changed in the instruments, and why

Three arms gained a `--dump-grads` path, the same one `device_gradient.py` gained on
`wk/of3t-trajectory`. Without it an arm publishes rel/`r`/cos against the one reference it was run
against and its gradient can never be compared to anything else. That is the fourth time this
campaign has been blocked by a summary-only result file.

`agreement.py` had three things fixed to the one scope it was written for:

1. the threshold was the constant 5.852018e-02, the floor of the 547-tensor diffusion arm. Section
   floors run 3.1012e-02 to 7.0855e-02, so any other scope read against it is scored on another
   set's bar. It is measured per scope now, and divided by `r`.
2. the capture-identity check hardcoded the key `grad_f64` and the prefix `diffusion_module.`. The
   `aux_heads` and `msa_module` captures call it `param_grads` and key by full name. Both are
   arguments now, and a zero compared count raises.
3. A14 was applied against each pair's own reference rather than against the float64 gradient
   the mass weights already come from. That made the excluded set differ row by row inside one
   table: `aux_heads...blocks.3.attn_pair_bias.layer_norm_z.bias` has a float64 norm under the
   1e-12 floor and a bf16 norm of 1.1339e-12 just over it, so it was excluded from the floor row
   and divided by in ours, and turned up as the scope's **worst tensor at rel 1.1108e+05** while
   holding 1.3e-40 % of the model's mass. A worst case that is an artefact of which pair is being
   read gets quoted as a location. A14 now follows float64 for every pair. No headline moved —
   all three are identical to every digit, because the tensors involved carry no mass — and the
   measurable sets now match each arm's own count exactly: 176 on `aux_heads`, 151 on
   `msa_module`.
4. `cos(our error, their error)` is measured rather than inferred from norms.

Re-running trajectory's own command on its own inputs is unaffected on every number: its scope's
measured floor is 5.852018e-02, the same value the constant held.

## What is out of scope

`pairformer_stack` (5.8282 %) needs a different capture and only 7 of 48 blocks are measured at
all. `input_embedder` (0.8007 %) is 92.4879 % host-applied, so there is no device gradient to
compare (`perf/of3t_orchestrator/COVERAGE_CEILING_IS_NOT_100.json`). Neither was read AT OR BETTER
by D72, so neither is part of the headline this row tests.
