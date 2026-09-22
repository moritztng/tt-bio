# of3t-modelboundary — pre-registration

Written and committed BEFORE the first device run of this row. Nothing below is adjusted after a
number exists.

## The question

`READABLE_MASS.json` (`perf/of3t_readable_mass/`, sha256 `19ced3e9...`) classifies 2,736 of the
4,170 reference tensors as READ_ON_ANOTHER_BOUNDARY: a per-tensor device gradient exists for all
of them, taken at `of3t-apbgrad`'s captured 64-token boundary rather than on the model's own
batch. They hold 5.82817 % of the model's squared gradient norm. This row re-runs that arm on the
model's batch so the mass can enter the union instead of being carried as a composed term.

## What "the model's batch" is here, stated before the run

`BOUNDARY_c64.json` and `BOUNDARY_n384.json` are cut from the SAME capture. Both name
`/home/ttuser/of3t_gradients/cap/block0_boundary.pt` sha256 `21e10e3d...` and
`block47_boundary.pt` sha256 `a55ef1c4...`. The only difference is the crop: c64 keeps 64 token
slots of the model's 384, n384 keeps all 384. Both hold the same 56 real tokens. So the arm on
the model's batch is `dev_grad.py --crop 0` over
`/home/ttuser/of3t_trunk043ref/boundary_n384.pt` (sha256 `8cb3a586...`), and after it the trunk's
gradient is a gradient of the same input `batch_step003` at crop 384 that drives the other 907
tensors. That is the one thing `of3t-wholemodel` refused the trunk entry to the union for
(`run_model.sh`, `run_trunk.sh` and the RECONCILE section of its state doc).

## The arms

  * **CTRL** — `TT_BIO_SOFTMAX_BW_RENORM=0`. This is the arm the census read: its c64 twin is
    `/home/ttuser/of3t_apbgrad/dev_scope_CTRL_c64.pt`, arm `flipped`, softmax-backward repair off.
  * **RENORM** — the repair on. On `wk/of3t` today `tt_bio/autograd.py:87` is
    `env_flag("TT_BIO_SOFTMAX_BW_RENORM", True)`, so RENORM is what the tree does by default and
    CTRL now needs the flag set to 0 to reach it. `of3t-apbgrad` ran before that default flipped.
  * **BREAK** — `--permute-cot`, the same control `of3t-apbgrad` used.

## Predictions

**P1 — coverage.** `coverage_total.pct_of_model_compared` over 907 + 2,736 = 3,643 tensors:
**97.98499306866147 %**. That is 92.15682156952718 + 5.828171499134286, and it must equal
`READABLE_MASS.json`'s `headline.pct_with_a_per_tensor_device_gradient_somewhere`
(97.98499306866148) to rounding. The bar is 99.2594 %, so this row alone does **not** clear it and
is not expected to; `of3t-hostleg`'s +1.52024 is the other half.

Coverage does not depend on the reading. The 2,736 names are fixed by the census and the n384 run
uses the same checkpoint and the same bijection, so any coverage number other than 97.98499 means
the bijection placed a different set and the difference is the finding.

**P2 — how many of the 2,736 are over the 5.0e-02 per-tensor bar, against float64.**
At c64 the counts are 2,685 (CTRL) and 2,686 (RENORM) of 2,736, and upstream's OWN bf16 step is
over the same bar on 2,665 of the same 2,736 (`perf/of3t_wholemodel/TRUNK_c64.json`). Predicted at
n384: **2,686 for RENORM and 2,685 for CTRL, band 2,600-2,736.**

**P3 — the second GRADIENTS clause gets worse, and by about the count P2 names.**
`CHARTER_EVIDENCE.json` reads `stats.shipped_vs_FLOAT64.n_over_per_tensor_bar` off
`perf/of3t_wholemodel/MODEL_shipped.json` and finds **678** over 907. Adding 2,736 tensors of
which ~2,686 are over the bar predicts **~3,364 over 3,643**. One clause moves toward its bar and
the other moves away from zero. Registered here so the row cannot later present only the half that
looks good.

**P4 — the A/A control is bit-identical.** Re-running the c64 CTRL arm in this worktree, in a
different process on a different day, against `of3t-apbgrad`'s `dev_scope_CTRL_c64.pt`: **0 of
2,736 tensors moved.** The same for RENORM against `dev_scope_RENORM_c64.pt`. If either moves, the
n384 number is not comparable with anything this campaign has published and the row says so before
quoting it.

**P5 — the A16 zero baseline** over the enlarged union reads exactly **1.0** against float64 and
within 1e-6 of 1.0 against upstream's bf16 step, measured through the same scorer, not asserted.

**P6 — the worst tensor stays in the AttentionPairBias LayerNorm affines.** At c64 it is
`pairformer_stack.blocks.4.attn_pair_bias.layer_norm_a.bias` (RENORM, 44.40) and
`blocks.15.attn_pair_bias.layer_norm_a.bias` (CTRL, 10668.64). Predicted at n384: the same family,
some block.

## The risk this row is registering against itself

n384 is **85.4 % pad in the single track and 97.9 % pad in the pair track** (56 real tokens of
384, 3,136 real cells of 147,456) where c64 is 12.5 % and 23.4 %. `SCORE_n384.json` already shows
the forward at n384 reading 0.1062 masked and 1.3603 padded on the single track. A weight gradient
sums over every row including the pad rows, so unlike the forward it cannot be masked afterwards.
**If our pad handling differs from upstream's, the n384 reading will be worse than c64's and the
over-bar count will go to 2,736.** That would be a real finding about what the trunk does on the
model's actual shape, not an instrument artifact, and it is the reason this row exists rather than
the c64 number being reused. It does not touch P1: coverage counts tensors that have a reading,
not tensors that agree.

## Bars, fixed here

Per-tensor 5.0e-02 relative L2 (PROTOCOL 3d). Coverage bar 99.2594 % (`COVERAGE_CEILING_IS_NOT_100`
pass 175, REASON withdrawn at A32, bar kept). A26 sqrt(2) applies against upstream's bf16 and not
against float64 (A26-SCOPE). Denominator `model_squared_gradient_norm` 10.279642678524981 over
4,170 tensors, from `grads_f64_043.pt` sha256 `1d4ea922...`, verified before loading.

## What this row may not do

It may not edit `perf/of3t_readable_mass/` or `perf/of3t_wholemodel/`: concluded rows own those.
If its result supersedes a number in `perf/of3t_wholemodel/MODEL_shipped.json`, which the charter
reads, it says so by path in its conclusion and leaves the file alone (A33).
