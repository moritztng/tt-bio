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

## AMENDMENT, 2026-09-22 pass 2: the clause needs TWO scopes, not one

The version of this document written in pass 1 proposed a single number, 97.9849 %, built by
ADDING the section shares whose device gradient arms exist. `COUPLED_VS_UNION.json` tested that
addition and it does not hold.

Every boundary the campaign holds is a frozen capture of upstream's own r = 0 step, so a section
driven from one reads those inputs at all 20 steps. `boundary_aux_heads.pt`'s kwargs are `batch`,
`si_input` and `output.{si_trunk, zij_trunk, atom_positions_predicted}` -- the trunk's outputs and
the diffusion module's. At step k an aux_heads trajectory therefore reads upstream's step-0 trunk
and diffusion outputs, never our step-k ones. A union of such runs tests each section's update
rule in isolation and tests none of the coupling between them, and the two are
indistinguishable in a `pct_of_model_sq_grad_norm` field.

That is the D181 failure mode in a new place. D181 forbids coupled clauses reading different
scopes; here a single clause reads one scope that two different instruments reach, one of them
strictly weaker. A union satisfying 97.9849 % would be a pass earned by the weaker instrument.

So the two quantities have to be named apart:

| | what bounds it | today | what it proves |
|---|---|---|---|
| **COUPLED scope** | 89.2106 %, the widest single capture's span (the 0.4.3 diffusion boundary) | **88.0819 %** | step k+1's forward read the weights step k wrote, through a composed forward |
| **UNION scope** | 92.1568 %, the device gradient arms that exist (`MODEL_d56_retake.json`) | 88.0819 % | each section's update rule, separately |

88.0819 % is 98.73 % of what the diffusion boundary spans, so the coupled figure is already
within 1.13 points of the ceiling any existing capture allows.

## The proposed clause

> `scope.pct_of_model_sq_grad_norm >= 89.2106` **with `scope.coupled: true`** -- the span of the
> widest single boundary capture the campaign holds, which is the most a trajectory can cover
> without a capture that spans more than one section. The UNION scope, bounded at 92.1568 % by
> the device arms that exist, is a separate field and may not be substituted for it.

Four things about that number, because the failure mode here is a clause repair that lets the
campaign declare success:

1. **It is above what this row reached.** 88.0819 against 89.2106 leaves TRAJECTORY UNMET by
   1.13 points. No number this row produced is being proposed as its own bar.
2. **It is a measured property of the REFERENCE, not of our port.** 89.2106 % is the share of
   upstream's own float64 gradient mass lying inside `diffusion_module`
   (`SECTION_MASS_MEASURED.json`, summed from `grads_f64_043.pt`). It would be the same number
   if our port did not exist, and it moves only if a wider capture is taken.
3. **It is honest about being a large reduction.** 99.2594 to 89.2106 is 10.05 points. The
   justification is not cost and not difficulty: 99.2594 % is the ceiling on what the STATIC
   single-step instrument could ever measure, and a trajectory is a different instrument whose
   reach is set by how far one capture spans. Giving two instruments one number is what made
   the clause unsatisfiable.
4. **What it would take to raise it is named, not hand-waved.** Above 89.2106 % needs a capture
   spanning more than one section AND a taped whole-model device forward+backward. The
   reference half of that is already available and digest-pinned -- `bundle_min_043` holds
   `batch_step003.pt`, `w0_043.pt` and `draws_recycles0.pt`. The device half does not exist:
   `instrument_full_model.py` places upstream's tensors onto ours by exact bf16 value and its
   own docstring says it deliberately does not compare a gradient magnitude. So the blocker is
   one named piece of porting work.

### Correcting pass 1's arithmetic

Pass 1's decomposition of 99.2594 % into 97.9849 % reachable and 1.2744 % with no device
gradient is still correct as ARITHMETIC over mass shares. What was wrong is the word
"reachable": 97.9849 % is reachable as a UNION, not as a trajectory. The 1.2744 % with no
device gradient at any scope is unchanged and is listed in the table above this amendment.

### If the charter would rather keep 99.2594 %

That remains a defensible choice and it should then be recorded as permanently unmet with the
reason in the field, not as a miss string. Two independent things block it: 1.2744 % of the
model's gradient mass has no device gradient at any scope, and everything above 89.2106 %
needs a capture that spans more than one section. Neither is a measurement this row or any
sibling row can take by running something longer.

What should NOT happen is the clause keeping one number while the coverage clause approaches it
from the static instrument's 92.1568 % and a trajectory clause is scored on a union. A reader
then sees one bar and three different reaches.
