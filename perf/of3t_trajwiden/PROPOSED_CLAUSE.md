# TRAJECTORY's scope clause: what it should read, and why 99.2594 % is not it

PROPOSED, not edited. The orchestrator repoints charter conditions; this row does not.

## Where the clause stands

The charter's TRAJECTORY field requires `scope.pct_of_model_sq_grad_norm >= 99.2594` on an
N >= 20 weight trajectory whose two sides both move. The live artifact,
`perf/of3t_trajretake/traj_retake_shipped.json`, reads **36.9462 %**, and that is the figure
`perf/of3t_orchestrator/charter/CHARTER_EVIDENCE.json` records as the miss.

This row's `perf/of3t_trajwiden/traj_widen_shipped.json` reads **88.0819 %** at 20 of 20 rungs,
573 of the 761 reference tensors at the `diffusion_module` boundary, growth sub-linear
(exponent -0.2772, r2 0.9050), the moves clause met at 19 of 19 fitted rungs. That is a
**partial result against a 99.2594 % bar**, not the bar, and it is labelled as one everywhere
it appears.

## Why 99.2594 % cannot be met by a trajectory, and it is not a cost problem

99.2594 % comes from `perf/of3t_orchestrator/COVERAGE_CEILING_IS_NOT_100.json`. It is the
ceiling on **device-measured coverage**: 100 % minus the 0.74055 % of the model's gradient mass
sitting on a weight the shipped path applies on the host after a `ttnn.to_torch`, where no
device gradient exists to compare. It was derived for the STATIC instrument, the single-step
gradient comparison, and for that instrument it is the right target.

A trajectory needs strictly more than a device gradient at a scope. It needs four things, and
the static instrument needs one:

1. a taped device forward **and** backward at that scope,
2. a captured 0.4.3 boundary whose cotangent is complete against the reference gradients,
3. upstream's own module runnable standalone at arbitrary weights, for 20 steps,
4. 20 optimizer steps of both sides, affordably.

So the two instruments have different reach, and a clause that gives them the same number
makes the weaker one unsatisfiable. **D181 applies directly**: coupled accuracy/coverage
clauses must read the SAME scope. Here the COVERAGE clause and the TRAJECTORY clause read the
same *number* while addressing different *scopes*, which is the same defect from the other
direction.

## The arithmetic, from the campaign's own published shares

Every share below is `perf/of3t_orchestrator/SECTION_MASS_MEASURED.json` (exhaustive, 17
sections, 4,170 tensors, 100.0000 %) with the per-section COMPARED fractions from
`perf/of3t_ditmodel/MODEL_d56_retake.json`. Compared, not total: a trajectory over a section
whose device arm reaches 456 of 552 tensors reaches 43.6221 %, not 43.8936 %.

| piece | % of model | has a device gradient today? |
|---|---|---|
| `diffusion_module`, the 573 tensors this row's trajectory reaches | **88.0819** | yes, and as a 20-step trajectory |
| `pairformer_stack` | 5.8282 | yes, `of3t-frame384` dev_RENORM_n384 at crop 384 |
| `aux_heads` | 2.8431 | yes, static only |
| `msa_module`, compared | 1.2317 | yes, static only |
| the 188 `diffusion_module` tensors the device arm does not reach | 1.1286 | **no** |
| `input_embedder`, the measurable part | 0.06015 | **no** |
| `msa_module_embedder` | 0.0612 | **no** |
| `template_embedder` | 0.0103 | **no** |
| `msa_module`, the part its arm did not reach | 0.0083 | **no** |
| top-level `layer_norm_z` / `layer_norm_s` / `linear_z` / `linear_s` | 0.0059 | **no** |
| | **99.2594** | |

The column sums to the bar, which is the check that this decomposition drops and double-counts
nothing.

Reading it two ways:

* **Reachable by a trajectory, with a device gradient arm that already exists:**
  88.0819 + 5.8282 + 2.8431 + 1.2317 = **97.9849 %**.
* **Not reachable at any price, because no device gradient exists anywhere in the campaign:**
  1.1286 + 0.06015 + 0.0612 + 0.0103 + 0.0083 + 0.0059 = **1.2745 %**. Closing it is porting
  work, not measurement. 0.74055 % of that class is already excluded from the bar; this
  1.2745 % is the part that is not.

## The proposed clause

> `scope.pct_of_model_sq_grad_norm >= 97.9849`, where 97.9849 is the model gradient mass for
> which a device gradient arm exists at a scope a trajectory can be driven at, measured from
> `SECTION_MASS_MEASURED.json`'s shares and `MODEL_d56_retake.json`'s per-section compared
> fractions. The 1.2745 % below it has no device gradient at any scope and is a port-coverage
> gap, reported as such rather than folded into the bar.

Three things about that number, because the failure mode here is a clause repair that lets the
campaign declare success:

1. **It is above what this row reached.** 88.0819 against 97.9849 is a 9.90 point miss and the
   clause stays UNMET. A bar set at 88.0819 would be this row grading its own work and is not
   proposed.
2. **It is a measured property of the instrument, not of a result.** It is derived the same way
   99.2594 % was, from published mass shares, and it moves only if a new device arm lands.
   `a-go-clause-must-be-tested-against-the-references-own-artifact` is the standing lesson; this
   respects it.
3. **It is affordable, so the clause is not being lowered to dodge a cost.** `FLOOR.json` prices
   the 9.90 points: the pairformer reference side is 20 x 1629.18 s = 9.05 h at 16 threads on
   qb1, aux_heads and msa_module are far smaller modules. The reason to move the bar is that
   1.2745 % of the mass has no instrument, not that the rest is expensive.

## If the charter would rather keep 99.2594 %

Then TRAJECTORY is permanently UNMET and should say why in the field rather than as a miss
string: 1.2745 % of the model's gradient mass has no device gradient at any scope, so no
trajectory can score it, and the clause is measuring port coverage through an instrument that
cannot see it. Naming that is worth more than a number that can never go green. What should NOT
happen is the clause keeping 99.2594 % while the coverage clause approaches it from the static
instrument's 92.1568 %, because a reader then sees one bar and two different reaches.
