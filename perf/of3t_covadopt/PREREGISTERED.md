# of3t-covadopt — pre-registration, committed before the arm

One run of `of3t-modelboundary`'s own instrument (`perf/of3t_wholemodel/model_scope.py`) over
`of3t-refcov`'s composed 3,660-tensor set, so GRADIENTS' coverage clause and its A26 accuracy
clause read ONE artifact at ONE scope (D181). No shipped default moves, no code under `tt_bio/`
is touched, and `_of3t_donecheck.py` / `charter_evidence.py` / `CHARTER_EVIDENCE.json` are not
edited — the clause repoint is PROPOSED in the state doc.

## The set, and which arm

`renorm` only. That arm is the shipped configuration: `TT_BIO_SOFTMAX_BW_RENORM` has defaulted
True at `tt_bio/autograd.py:87` since `2de9355d1` (D56), and it is the arm both charter clauses
read. The `shipped` arm of `MODEL_withtrunk_n384.json` is the PRE-D56 configuration, and its
trunk dump `/tmp/of3t/of3t-modelboundary/dev_CTRL_n384_nocaptures.pt` no longer exists on qb1 or
qb2 — it went with a torn-down worktree's `/tmp`. Re-creating it is a device re-run of a
configuration main does not ship, so this artifact carries one arm and says so.

Scopes, all six driven by the model's own `batch_step003` at crop 384:

    diffusion        /tmp/of3t/of3t-covadopt/device_grads_rc_refatom_on.pt     555 keys
    input_embedder   /tmp/of3t/of3t-covadopt/ie_grads_f64_renorm_on.pt           9 keys
    cond             /home/ttuser/of3t_wholemodel/cond_grads_renorm.pt
    aux              /home/ttuser/of3t_wholemodel/aux_grads_renorm.pt
    msa              /home/ttuser/of3t_wholemodel/msa_grads_renorm.pt
    pairformer_stack /home/ttuser/of3t_trunkceiling/dev_RENORM_n384_nocaptures.pt

The trunk dump is the same file the published artifact used: sha256
`095243e7b29f3778f527ea8835a5266783667b638d5111173aa636ce8c17ca4e`, and three copies on two
hosts (`of3t_frame384`, `of3t_shapekey/refs`, `of3t_trunkceiling`) carry that digest.

## The statistic

`agreement.stat`'s headline is a POOLED ratio, not a mass-weighted mean of per-tensor rel_l2:

    mass_weighted_rel_l2 = sqrt( sum_i ||a_i - b_i||^2 / sum_i ||b_i||^2 )

so it composes by sums of squares, which is what makes the float64 leg predictable exactly.
refcov's `delta` block reports the OTHER statistic (a mass-weighted mean); the two differ by
1.2 % here and only `headline_vs_float64` in that file is this one. Mixing them was the first
thing checked and it is why this section exists.

## Predictions (`perf/of3t_covadopt/PREDICTION.json`, written by `predict.py`)

From committed artifacts only: the published per-tensor sidecar behind
`MODEL_withtrunk_n384.json`, refcov's `COVERAGE_COMPOSED.json` and its
`device_gradient_rc_refatom_on_per_tensor.json`, plus the three pinned references (no arm data).

Exact, and they must land:

    union_n_tensors                        3660
    n_compared / n_reference_tensors       3660 / 4170
    absent from the float64 reference      0
    coverage_total.pct_of_model_compared   99.50523155277378        (within 1e-9)
    model squared gradient norm            10.279642678524981       (within 1e-12 relative)
    renorm_vs_FLOAT64 mass_weighted_rel_l2 0.5276985077644339       (within 1e-12 relative)
                      norm ratio           1.100433406767638
                      cos                  0.8780576615307545
                      n_rel_measurable     3605
                      n_over_per_tensor_bar 3314
                      worst                642.7051680533705 at
                                           pairformer_stack.blocks.26.attn_pair_bias.mha.linear_q.weight
    UPSTREAM_BF16_vs_FLOAT64 and UPSTREAM_F32_vs_FLOAT64 over 3,660, and the bars block derived
    from them, are computed in PREDICTION.json from the references before the arm runs.

The float64 leg is predicted by two independent routes that agree to 2.3e-15: summing squares
over the published rows with the 547 replaced and the 17 added, and refcov's own
`headline_vs_float64.after` (0.5276985077644327), computed in a different process from the
tensors themselves.

NOT exactly predictable: `renorm_vs_UPSTREAM_BF16`. No arm was ever scored against upstream's
bf16 step with `--device-refatom` on, so that leg is the one genuinely new measurement.
PREDICTION.json registers a rigorous interval for it — Minkowski over the concatenated vectors,
`| ||A-F64|| - ||B16-F64|| | <= ||A-B16|| <= ||A-F64|| + ||B16-F64||`, divided by `||B16||` —
and a point estimate that holds the published set's geometry (the angle between the two error
vectors) fixed. A reading outside the interval stops the row. A reading inside it but more than
5 % off the point estimate is reportable, not a stop.

The A26 clause is predicted to FAIL at roughly 3.5x its bar. This row moves a scope, not a
verdict, and a repoint that flipped a clause toward passing would be the thing to distrust.

## Controls

* **C1** — the new artifact restricted to the original 3,643 against the published figures. It
  will NOT reproduce them: `--device-refatom` moves `cl0`/`plm0` onto the card and they feed the
  whole DiffusionModule, so all 547 diffusion-scope tensors move (refcov C2). Predicted:
  3,096 rows BIT-IDENTICAL to the published sidecar, 547 rows moved, worst per-tensor
  disagreement 0.740350719969133 at
  `diffusion_module.diffusion_transformer.blocks.3.attention_pair_bias.layer_norm_a.layer_norm_s.weight`,
  mass-weighted 0.13949978615214453, and the restricted headline 0.5317397054265072 against the
  published 0.5327948845677762 (-0.0010551791412690692). An UNPREDICTED disagreement is the
  finding and stops the row.
* **C2** — coverage at 99.50523155277378 within 1e-9. It is a property of which tensors are
  present, so it is the cheap check that the right set was scored.
* **C3** — the three reference digests read back in-process by the instrument's own `--expect`,
  float64 pinned at `1d4ea922...`, and the denominator recomputed to 10.279642678524981 within
  1e-12 relative (A14/D78: that sum has no single correct last bit, and the instrument's own
  rtol is 1e-12).
* **C4** — an A/A: the instrument run twice on the same inputs, artifact and every sidecar
  byte-identical by sha256.
