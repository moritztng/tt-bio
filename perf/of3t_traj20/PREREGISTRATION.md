# of3t-traj20 — pre-registration, written and pushed before the full-scope run

PROTOCOL §7, the assembled 20-step trajectory. Fixed here: the arms, the bars, what each
outcome would mean, and what I would do next. A shape read off a curve after seeing it is not
a bar (§3e), so this file is committed and pushed before the run that produces the numbers.

Already seen when this was written: a 40-tensor smoke of the `scaled` arm, run as a functional
test of the harness. It is not scope and no bar below is set from it.

## The compared quantity

`d_k = w_k - w_0` per tensor, scored `||d_k_ours - d_k_ref||_2 / (||d_k_ref||_2 + 1e-30)`, at
model scope on the concatenation and per tensor beside it. Comparing `w_k` is vacuous at these
step sizes and `rel_w` is reported at every rung to show that (§7a).

## Scope and drive

All 4,147 OpenFold3 parameters the published presence census declares, real `w_0` from
`w0_r0_rebuild.pt`, real gradient magnitudes from `grads_f64_r0.pt`, both sha256-recorded in
every result file. Sample s of step k drives both sides with the same function

    g_p = a[k][s] * G_ref_p + rho * b_p * (w_p - w0_p)

evaluated at each side's OWN current weights. The feedback term is what makes stale state
visible; a fixed gradient list cannot tell a stack that updates its weights from one that does
not. `rho` is set per arm so the feedback carries a comparable share of the drive despite the
two warmups moving the weights by ~1e-4 and ~1e-2 relative. The share is MEASURED at k = 2, 10
and 20 and published. **A share below 1 % at k = 20 invalidates that arm's stale-state reach**
and is reported as such rather than passed over.

This does NOT compare the two stacks' gradients. That is instrument A (§3), and §7a-bis is why
a trajectory is the wrong instrument for it: Adam cancels a uniform per-tensor gradient
scaling at any N.

## Arms

| arm | warmup | accum | our side |
|---|---|---|---|
| `shipped` | 1000 | 1 | `recipes.py:118` construction, `train_loop` order |
| `scaled` | 20 | 4 | the same, with the accumulation cycle and the per-parameter participation counts engaged |
| `wired` | 20 | 4 | attribution: `weight_decay=0.0` and the per-sample `clip_and_accumulate` path called |
| `miswire` | 20 | 4 | **instrument-can-fail**: D11's off-by-one restored, one line, our side only |
| `stale` | 20 | 4 | **break control**: our step k fed step k-1's gradient, nothing else changed |

Their side is upstream's own `PerSampleGradManager`, `AlphaFoldLRScheduler` and
`torch.optim.Adam`, loaded by path and executed, stepped in `runner.py:449-470`'s order. Not
transcribed: a transcription would put a second implementation into the comparison.

## Bars, fixed now

1. **`d_1` must be exactly 0 on BOTH sides** (A10). `lr(1)` is 0, so a correct implementation
   cannot move. Any non-zero `d_1` is a defect to localise, never a rung to skip, and D11's
   closure by `of3t-updaterule` is what this rung tests end to end.
2. **The bar is the SHAPE** (§7b), fitted over k = 2..20 on the `scaled` arm, which is the run
   §7a added to give the law dynamic range. **Exponent ≤ 1.0 passes; > 1.0 FAILS at any
   magnitude, including magnitudes inside the per-step bars.** No threshold on `d_20`.
3. **The controls must move the thing they read.** `miswire` must break bar 1 (`d_1` non-zero
   on our side against their exact 0). `stale` must produce super-linear growth. If either
   passes quietly the instrument has not been shown capable of failing and bar 2 is not
   evidence (§3e).
4. **Every rung is reported**, k = 1..20, with the fp32 differencing floor beside it. An
   endpoint hides the shape.

## What each outcome would mean, and what I would do next

- **Super-linear on `scaled`**: a different update rule, not a rounding difference. Next step
  is to bisect by arm: re-run with `weight_decay=0`, then with the per-sample path wired, then
  with participation averaging, and report which one restores a sub-linear shape.
- **Sub-linear, magnitude at the floor**: the assembled wiring agrees and §7 is clean.
- **Sub-linear, magnitude far above the floor**: the update rule is not divergent, but the two
  stacks are not applying the same one. The shape bar passes and the magnitude is the finding;
  the `wired` arm is what attributes it. This outcome must NOT be written up as a pass, and
  bar 2 must NOT be moved to cover it.
- **`d_1` non-zero on our side**: D11 is not closed end to end; localise against
  `perf/of3t_updaterule/lr_wiring.py`'s arm B before anything else is believed.

## Read before running (from the code, not from the trajectory)

Three wiring gaps were found by reading the two stacks' shipped call paths, before any number
existed. They are named here so the run tests a prediction rather than explains a result:

1. `recipes.py:118` constructs `AdamW(params, lr=lr, data_parallel=dp, schedule=...)`, taking
   `weight_decay=0.01`. Upstream's `configure_optimizers` builds `torch.optim.Adam` with no
   weight decay at all.
2. `per_sample_clipping: True` is upstream's shipped OF3 default at `clip_val 10.0`.
   `AdamW.clip_and_accumulate` implements it and **has no caller anywhere in `tt_bio/`**;
   `train_loop` clips the batch once instead.
3. `AdamW` records `self.participation` and never reads it. Upstream's
   `_sync_and_average_grads` divides each parameter's accumulated gradient by its own
   participation count.

Prediction: 1 and 3 are small because Adam cancels a per-tensor scaling constant in k, and 2 is
not, because per-sample clipping changes the DIRECTION of the accumulated update. The `wired`
arm is what separates them.
