# of3t-wirefix — pre-registration, written and pushed before the fixed run

`of3t-traj20` found four wiring divergences by reading the two stacks' call paths, measured
each one's cost in a harness arm, and returned NO-GO. This row closes them **in the shipped
source**, not in a harness arm, and re-runs §7 against that source. The bars below are fixed
here and this file is pushed before the run.

## The four fixes, as source changes

1. **Weight decay.** `train_loop` gains `weight_decay=0.0` and passes it to `AdamW`.
   Upstream's `configure_optimizers` builds `torch.optim.Adam`, which carries none
   (`runner.py:852-860`). `AdamW`'s own class default stays 0.01: a class called AdamW whose
   default is 0 is a worse lie than a recipe that names its own value.
2. **Per-sample clipping.** `train_loop`'s inner loop becomes a per-sample forward/backward
   over this rank's shard, each sample clipped by `opt.clip_and_accumulate()` before the next,
   and `opt.step()` once at the end. `per_sample_clipping: True` at `clip_val 10.0` is
   upstream's shipped OF3 default (`model_config.py:156-159`) and `clip_and_accumulate` has
   had 0 callers in `tt_bio/` since it was written.
3. **Participation average.** `AdamW.step()` divides each parameter's accumulated gradient by
   its own `self.participation` count before it is used, which is upstream's
   `_sync_and_average_grads` (`grad_manager.py:225-232`). The counter was written and read
   zero times.
4. **Schedule family.** `train_loop` gains `plateau_until=50000` and passes it to `af3_lr`,
   selecting OpenFold3's AF2 schedule instead of Protenix's. `None` still selects Protenix's,
   so this is one argument and not a per-model branch.

## The arms

| arm | both sides | what it is for |
|---|---|---|
| `fixed` | ours (post-fix shipped path) vs upstream | the result |
| `scaled` | ours (pre-fix wiring) vs upstream | the arm that failed, kept so the fix has a before |
| `miswire` | ours + D11's off-by-one vs upstream | instrument-can-fail |
| `aa` | upstream vs upstream | A/A on a path this row did not touch |
| `ulp` | ours vs ours, one side perturbed | the closed loop's own fp32 rounding floor |

All five at warmup 20, 4 samples per step, rho 0.005, seed 20260920, over the same 4,147
tensors. `fixed` and `scaled` therefore differ in exactly the four fixes and nothing else.

`ulp` perturbs one side's ACCUMULATED GRADIENT by a relative `2**-24` per element before
every step, and scores it against the unperturbed same side.

**Amended before the run, and the reason is arithmetic rather than a result.** The first draft
of this file put the perturbation on the master weight. That does not work: `2**-24` relative
is one fp32 unit roundoff, which is exactly half an ulp, so `theta*(1 + 2**-24*eta)` rounds
back to `theta` for every `|eta| < 1` and the control would have injected nothing and read a
floor of zero. The perturbation is therefore formed in float64 and rounded back to float32,
which is precisely the form a differently-associated fp32 expression takes -- it lands on the
same value about half the time and one ulp away the rest. The gradient is the right place for
it because that is where the two stacks demonstrably differ: our clip coefficient is
`min(1, 10/gnorm)` against their `10/max(gnorm, 10)`, computed by different reduction code, and
that coefficient multiplies every gradient in every sample of every step. Both sides run the same
code, so the only thing it measures is how far the closed loop carries a rounding-sized
difference over 20 steps. It is a **lower bound** on the rounding-explained residual and is
reported as one: real arithmetic injects rounding at every operation of every step, not once
per step, and our update `lr*(m/bc1)/(sqrt(v/bc2)+eps)` and torch's
`(lr/bc1)*m/(sqrt(v)/sqrt(bc2)+eps)` are the same algebra in a different association, so they
round apart at every element of every step by construction.

This replaces NOTES.md's proposed float64 discriminator. `AdamW` refuses a non-fp32 master at
construction, by design and for a measured reason, so a float64 arm would require changing the
shipped guard for an experiment. The `ulp` arm answers the same question -- is the residual the
loop amplifying rounding, or a fifth divergence -- without touching shipped code.

## Bars, fixed now

1. **`d_1` exactly 0 on BOTH sides, 4,147 of 4,147 tensors bit-identical**, on every honest
   arm. `lr(1)` is 0 so a correct implementation cannot move. Categorical, not approximate.
   This is D11 staying closed and it is the cheapest regression signal this row has.
2. **The 74 zero-reference-update tensors must be bit-identical** once weight decay is off.
   `lr*wd*theta` moves a tensor whose reference update is exactly zero; with wd on, 0 of 4,147
   were bit-identical at k=2 and with it off, 76. Also categorical.
3. **The SHAPE bar, unchanged from §7b**: fit `log d_k` against `log k` over k = 2..20 on
   `fixed`. **Exponent <= 1.0 passes, > 1.0 FAILS at any magnitude.** Not converted into a
   threshold on `d_20`, not refitted over a different window after seeing the curve.
4. **Every rung k = 1..20 is reported**, with the fp32 differencing floor beside it.
5. **The controls must move what they read.** `miswire` must break bar 1 on our side against
   their exact 0. `aa` must read exactly 0.000000e+00 at every one of the 20 rungs with
   4,147 of 4,147 bit-identical; anything else means the instrument is not deterministic and
   every magnitude above is unreadable.

## What each outcome means, and what I do next

- **`fixed` reproduces `avg`** (k=2 near 8.715e-03, exponent near +1.267): the four fixes
  landed in the source exactly as the harness arm modelled them. §7 stays **NO-GO on shape**,
  and the residual is the finding. I do not move bar 3 to cover it.
- **`fixed` differs from `avg`**: either a fix landed differently in the source than in the
  arm, or fix 4 reaches inside 20 steps. Fix 4 must not: both families share the 0..1000 linear
  warmup and 20 steps is inside it, so a move here is a defect in the fix and I bisect against
  `wd0`/`wired`/`avg` before believing anything.
- **The residual GROWS against `avg`'s 8.715e-03**: a fix is mis-applied. That is the
  unfavourable branch and I report it as one -- bisect fix by fix against the three traj20
  attribution arms, name which fix regressed it, and do not ship the branch.
- **`ulp` reaches the residual's magnitude**: the residual is the closed loop amplifying each
  stack's own fp32 rounding. Not a fifth wiring divergence. The super-linear shape is then a
  property of a chaotic Adam loop at these step sizes, which is a statement about the
  instrument and I say so rather than calling §7 passed.
- **`ulp` lands far below the residual**: rounding at one ulp per step does not explain it, a
  fifth divergence is unlocated, and I report its size and its shape and say plainly that I did
  not find it. Inconclusive is the honest word here, not "clean".

## A fifth divergence, read from the code before the run

`step()` skips a parameter with no accumulated gradient entirely (`g is None -> continue`).
Upstream's `sync_and_average_grads` assigns `param.grad = accumulator.clone()` for **every**
parameter and `_sync_and_average_grads` zeroes the ones with count 0, so torch's Adam still
steps them: momentum decays and a non-zero update is applied from history alone. Ours applies
nothing.

**Prediction: this does not fire in this harness** and cannot be the residual. `disabled_sets`
leaves one of the four samples per step fully enabled, so every parameter has participation
>= 1 at every rung. It is named here so the run tests the prediction, and it is a real defect
for a schedule where a parameter is disabled on all samples of a step.

## AMENDMENT, written before the arm that measures it: a FIFTH divergence, and it is the residual

Found the way the other five were: by reading the call path and then measuring two steps on
40 tensors, not by fitting the 20-step curve. The evidence that forced it, in order:

1. `of3t-traj20`'s `avg` arm reads ||d_ours|| / ||d_theirs|| = **1.008620** at k=2 with a
   per-tensor median of 8.940e-03 and a p90 of 9.010e-03. That is a uniform MAGNITUDE excess
   across essentially every tensor, not a direction error and not a few bad tensors.
2. `AdamW.step()` is **bit-identical** to `torch.optim.Adam` on identical gradients, measured
   over eight elements spanning 1.0 down to 1e-12 for two steps: ratio exactly 1.0 on every
   element, ||o||/||t|| = 1.0000000000.
3. At k=2 the accumulated gradient reaching each optimizer is **bit-identical**
   (rel = 0.000000e+00, summed and participation-divided), the counts are identical, the
   applied rate is identical at 9e-05 -- and `d_2` still reads 9.095550e-03.

Identical inputs, identical formula, a 0.9 % different update. The only thing left is a
constant, and it is **beta2**.

**`recipes.py:118` never passes `betas`, so `AdamW`'s class default `(0.9, 0.999)` ships where
OpenFold3 runs `(0.9, 0.95)`.** Upstream's `configure_optimizers` reads them out of its config
(`runner.py:855-860`), and that config sets `beta1: 0.9, beta2: 0.95`
(`projects/of3_all_atom/config/model_config.py:143-146`). Exactly the class the other four
belong to: a parameter the recipe never passes, so the library default ships instead of the
reference recipe's value.

Passing `betas=(0.9, 0.95)` on the same 40-tensor two-step probe takes `d_2` from
**9.095550e-03 to 4.884661e-06**, a factor of 1862, down to the same order as the `ulp` arm.

### The sixth fix, and the arm

`train_loop` gains `betas=(0.9, 0.95)` and passes it, the same one-argument shape as
`weight_decay` and `plateau_until`. The arm is `fixed5`, run at full scope over the same
4,147 tensors with everything else identical to `fixed`.

**Prediction, fixed before the run.** `fixed5` reads `d_1` exactly 0 with 4,147 of 4,147
bit-identical, `d_2` in the low 1e-06 range, and it tracks the `ulp` arm in both magnitude and
shape from k=2 to k=20, because if beta2 was the whole residual then what is left is the two
stacks' own fp32 rounding and nothing else. **If `fixed5` stays materially above `ulp`, a
sixth divergence is unlocated** and I report its size and shape rather than calling the job
finished. `fixed` is kept and reported beside it: it is the four-fix arm and it is what makes
the fifth one attributable.
