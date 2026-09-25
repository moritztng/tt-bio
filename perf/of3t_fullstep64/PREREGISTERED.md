# of3t-fullstep64: pre-registration

Committed before any gradient of this row exists (no float64, bf16 or device gradient of the full
step has been produced by this row at the time of this commit). Nothing below is edited after
results exist; misses are reported as misses.

## The step

`perf/of3t_trainfwd/trainfwd_run.py --arm full` as `perf/of3t_stackship/stepcost.py` runs it:
batch `batch_step003.pt` (sha256 `3c32597a20f09bf50769defa561f7df86da4de721da325eb76431a9d80b6285f`),
checkpoint `of3-p2-155k.pt` (sha256 `af09eac4...f4bee4`), 56 real tokens padded to 384, one trunk
cycle, 20 rollout steps (detached, feeds the confidence heads), the `af3` objective at
`of3_loss_weights("initial_training", "weighted-pdb")`, one backward. On this batch only two terms
fire: distogram (w 0.03) and resolved (w 1e-4). Denoise is off, so the diffusion module carries no
gradient.

## Arms

* F64: upstream OpenFold3 0.4.3 modules composed as the step above, float64 throughout
  (`bundle_min.no_autocast`, dropout r = 0), torch autograd, the same objective code evaluated in
  float64 on its outputs. It SAMPLES the rollout draws; every other arm replays them.
* BF16: the same composition under `bundle_min.cast_policy("bf16")` over float32 parameters
  (upstream's own bf16-autocast recipe on CPU). The bar.
* ON: our step, exact softmax + exact LayerNorm (the default), qb2 p300c card 1. Run twice (A/A).
* OFF: our step under `autograd.exact_training(False)`, same card.

## Metrics (fixed now)

Gradients are compared on the bijection between our walked device tensors and upstream parameter
names. For an arm `a` against F64 over a set S of tensors:

* global rel = ||g_a - g_64|| / ||g_64||, all tensors in S concatenated into one vector. This is the
  concatenated definition (A43), so r = ||g_a||/||g_64|| and cos satisfy rel^2 = 1 + r^2 - 2 r cos
  exactly; the residual is published with every cos.
* per head: S = trunk (input embedder, template, MSA module, pairformer stack, everything upstream
  outside `aux_heads` and `diffusion_module`), distogram (`aux_heads.distogram.*`), confidence (the
  rest of `aux_heads`), diffusion (`diffusion_module.*`).
* mass at or better than bf16: the fraction of F64 squared-gradient mass in tensors whose per-tensor
  rel for the arm is <= BF16's per-tensor rel.
* unread mass: F64 squared-gradient mass in tensors the bijection does not score, as a fraction of
  the F64 total. Reported beside every figure.

Tie: ON and OFF are TIED on a set when |rel_ON - rel_OFF| <= 0.05 * min(rel_ON, rel_OFF). A/A: ON
twice must be bit-identical on every tensor, else the ON/OFF comparison is reported against the
A/A spread instead.

## Predictions

| id | claim | P |
|----|-------|---|
| P1 | global: ON nearer F64 than OFF (not tied) | 0.70 |
| P2 | trunk: ON nearer | 0.70 |
| P3 | distogram: ON nearer | 0.65 |
| P4 | confidence (non-distogram): ON nearer | 0.55 |
| P5 | diffusion: exactly zero gradient in F64, ON and OFF (not scored) | 0.95 |
| P6 | loss: |loss_ON - loss_64| < |loss_OFF - loss_64| | 0.65 |
| P7 | squared gradient norm: F64 nearer ON's 15.317 than OFF's 9.786 | 0.60 |
| P8 | global: ON rel <= BF16 rel | 0.35 |
| P9 | global: OFF rel <= BF16 rel | 0.25 |

Genuine disjunction: P1 fails either because OFF is nearer (STOP condition: reported as the result)
or because they tie within 5 %. If P1 holds and a head is worse under ON, the head is reported with
its numbers and no fix is opened here.

## Gating control (A40), before any ratio

1. Draw replay: every replaying arm consumes exactly the F64 run's recorded draws, 0 mismatches.
2. The F64 loss recomputed from a fresh float64 forward equals the recorded one to 1e-12 absolute.
3. Central finite difference of the F64 loss along u = g_64/||g_64|| (the full-gradient direction),
   rollout structure held at the base forward's (the gradient is of the step with the detached
   rollout frozen, so the check is of that function): (L(θ+hu) - L(θ-hu)) / 2h against ||g_64||,
   at h in {1e-3, 1e-4, 1e-5}. Pass: best relative disagreement <= 1e-6.

A failed control stops the row: no ratio is published through a reference that failed it.
