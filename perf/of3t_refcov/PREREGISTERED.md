# of3t-refcov — pre-registration

Written and committed BEFORE `compose.py` ran. Everything below is a prediction or a refusal,
not a reading.

## The route I am taking, and why it is not the brief's

The brief pre-supposes the coverage leg closes from `tt_bio/train/openfold3.py`. It does not,
and the reason is the instrument's own definition rather than anything about the adapter's
code.

`coverage_total.pct_of_model_compared` is produced by `perf/of3t_wholemodel/model_scope.py`
over `perf/of3t_modelboundary/MODEL_withtrunk_n384.json`. Every scope in that union is an arm
**seeded with upstream 0.4.3's own captured cotangent** at that scope's boundary
(`perf/of3t_diffusion/device_gradient.py:4-5`: "seeded with THEIR cotangent"). The float64
reference is one `loss.backward()` of upstream's full `OpenFold3Loss`
(`perf/of3t_reference/bundle_min.py:368,654`).

`perf/of3t_trainfwd/trainfwd_run.py` seeds from tt-bio's own `af3` objective, and on this batch
only two of its eight terms fire (`ARM_full_n384.json`: `terms_that_fired` =
`["distogram","resolved"]`; the other six are skipped for missing `pred_xyz`/`pred_dist`/
`per_atom_lddt` with the denoise arm off). That is a different function of the weights, so a
gradient from it is not a measurement against this reference at any precision.

Registered as **P0** below and measured before anything else, because if P0 is wrong the whole
row changes shape.

## Predictions

**P0 — the training adapter is on a different loss boundary from the pinned reference.**
The five `input_glue` weights are the one place the adapter's arm and the reference name the
same tensors with no renaming ambiguity. Predicted: the per-tensor norms disagree by ORDERS,
not by a precision factor, in both directions (so it cannot be a scale bug). Bar: if any of the
five reads `rel_l2 <= 2.0e-02` against the reference the prediction is refuted and the adapter
route is live. **The finding is the ratio, and the row turns on it.**

**P1 — the bar is reached, from the union, with no shipped-code change and no adapter change.**
Two arms that already exist on the reference's own captured boundary carry all seventeen
HOST_APPLIED tensors and neither is in the published union:

  * `perf/of3t_hostleg/ie_arm.py` -> `/home/ttuser/of3t_hostleg/ie_grads_f64.pt`, 9 tensors,
    keyed by full checkpoint name (`input_embedder.atom_attn_enc.linear_q.0.weight` + the
    input embedder's 8 `ref_atom_feature_embedder` linears).
  * `perf/of3t_diffusion/device_gradient.py --device-refatom` ->
    `/home/ttuser/of3t_hostleg/device_grads_hl_refatom.pt`, whose 8 keys over the shipped arm
    are the diffusion module's `ref_atom_feature_embedder` linears.

Predicted `coverage_total.pct_of_model_compared`:

    97.98499306866148  shipped, published
  +  0.7530394291090192  diffusion_module...ref_atom_feature_embedder (8)
  +  0.7671990550038748  input_embedder leg (9)
  = 99.50523155277438

against the bar **99.2594**, clearing it by **+0.24583155277438**. Tolerance **1e-6**; an
outcome outside it is the finding. `n_compared` 3643 -> **3660**. The key-set diff must be
exactly the 17 names in `perf/of3t_hostleg/SEVENTEEN.json`, no more and no fewer.

**P2 — the denominator reproduces.** Recomputed in-process from
`/home/ttuser/of3t_hostleg/grads_f64_043.pt` (sha256 pinned
`1d4ea9225f854afe06a8adedcb5f8aa1c53656fbf8cefdaa3a451d0327895cc4`), predicted
10.279642678524986 against the published 10.279642678524981, rel diff below 1e-15. And the
published 97.98499306866148 must be reproduced from the sidecar's own 3643 names before this
instrument is allowed to say anything new.

**P3 — the two arms' published per-tensor readings reproduce against the MODEL reference.**
Both were scored against their own capture's `grad_f64`, not against
`grads_f64_043.pt`. Predicted: recomputed `rel_l2` for all 17 matches the published rows to
better than 1e-6 relative. If it does not, the two references differ at these tensors and the
composition is refused. A reference is part of a measurement's identity and these are two
files.

**P4 — accuracy does not get worse, and the 17 are not a free pass.** The 17 carry 1.5202 % of
the model's mass at published readings 0.0017 to 0.1198, well under the set's current
mass-weighted 14.669240503646682 vs float64 (`shipped`) and 0.5327948845677762 (`renorm`).
Predicted: both mass-weighted headlines **fall**, `shipped` to the band 14.4-14.5 and `renorm`
to 0.525-0.533. Per-tensor against the 5.0e-02 bar, predicted **6 of the diffusion 8 over the
bar and 0 of the input-embedder 9 over it**, so `n_over_per_tensor_bar` 3366 -> 3372 on
`shipped_vs_FLOAT64`. A headline that RISES is a finding and stops the composition: a clause
satisfied by a worse artifact is what the gate exists to refuse.

**P5 — inference reach is zero, by construction and not by timing.** Neither arm is shipped
code. `tt_bio/worker.py:1559` and `tt_bio/openfold3_host_prep.py:331` are byte-unchanged on
this branch against `origin/wk/of3t`, and no file under `tt_bio/` is modified at all.
Predicted: `git diff origin/wk/of3t -- tt_bio/` is empty, which is a stronger statement than a
fold digest A/B and makes the A/B a confirmation rather than the evidence.

## What I am refusing

`of3t-covdefault`'s proposal to repoint the clause from 99.2594 to 97.9849 is refused, and so
is any other move of the bar. P1 says the bar is reachable today.
