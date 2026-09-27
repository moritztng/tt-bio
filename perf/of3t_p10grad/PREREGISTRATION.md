# of3t-p10grad — the bar, fixed before any number was taken

Written and committed before the harness ran once. Nothing below is allowed to move after the
first result lands; if it has to move, the amendment goes in a separate file beside this one with
its own commit, the way `of3t-p10trainout` did it.

## What is being measured

For each op on OF3T's **shipped** backward path, on the shapes the trunk actually uses: run the
shipped device backward, and compare every gradient it produces against a **float64 reference
computed from the same host values**, where the reference is itself validated by central finite
differences.

The comparison is `rel L2 = ||dev - ref||_2 / ||ref||_2`, per gradient, per dtype (bfloat16 and
float32).

The reference reads the device's own inputs back to the host first, so what is measured is the
op's ARITHMETIC error alone. Rounding the inputs into bf16 is not charged to the op.

## Bar 1 — is the reference trustworthy

The float64 reference is accepted for an op only if a central difference along a random unit
direction `v`,

    (L(x + h v) - L(x - h v)) / (2 h)   against   <dx_ref, v>,   L(x) = <g, f(x)>

agrees to **rel error <= 1e-6**, with `h` swept and the best taken. An op whose reference fails
this gets **no verdict**: the instrument is not calibrated, and a number off an uncalibrated
instrument is not evidence.

## Bar 2 — is the op right (per gradient, per dtype)

| dtype | PASS | MARGINAL | WRONG |
|---|---|---|---|
| float32 | rel L2 <= **1.0e-3** | 1.0e-3 .. 1.0e-1 | rel L2 >= **1.0e-1** |
| bfloat16 | rel L2 <= **3.0e-2** | 3.0e-2 .. 1.0 | rel L2 >= **1.0** |

Where those two numbers come from, so they are not picked to suit an answer:

* bf16 unit roundoff is `2^-8 = 3.91e-3`. An op whose inputs are bf16 cannot do better than
  about that, and a normalisation or softmax Jacobian carries a condition factor. **3.0e-2 is
  ~8x the unit roundoff**, so an op is allowed to amplify its own input rounding eightfold and
  still pass.
* fp32 unit roundoff is `5.96e-8`. With a `sqrt(K)` reduction factor at the longest reduction
  the trunk runs (K=384, so ~20x) an honest fp32 op lands near `1e-6`. **1.0e-3 is ~1000x that**.
* **WRONG** is an error the size of the signal itself: at rel L2 >= 1.0 in bf16 the "gradient"
  carries no more information than a random vector of the same norm, and at >= 1.0e-1 in fp32 the
  op has lost more than a digit that fp32 physically has.

## Bar 3 — bias or noise

Over **M = 32** independent random draws (fresh x, gamma, beta, g each draw), take the
systematic component of the error along the reference gradient,

    s = <dev - ref, ref> / ||ref||^2

which is the relative amount by which the device gradient over- or under-shoots the true one in
the true one's own direction. Let `m` be its sample mean over the 32 draws and `se` the standard
error of that mean.

* **BIAS** if `|m| / se >= 5` **and** `|m| >= 1e-3`. Statistically certain, and large enough that
  compounding it over a training run matters: 1e-3 per application over the 816 layer-norm and
  240 softmax backwards a single OF3T step runs is not something averaging removes.
* **NOISE** otherwise.

Reported beside it, for readability rather than for the verdict: the elementwise error mean
divided by the elementwise RMS of the reference.

## Bar 4 — determinism

The same device backward, run twice on bit-identical inputs in one process, must return
**bit-identical** gradients. `of3t-p10trainout` concluded that the device layer-norm backward is
nondeterministic run-to-run and that this, not a precision gap, is what makes device-only OF3T
training different training. That is a claim about a whole training step; this row tests it at
the op.

Nondeterminism is a defect at **any** magnitude and is reported whether or not bar 2 passes.
A bit-exact repeat is not by itself a pass on bar 2, and a bar-2 pass does not excuse a
non-bit-exact repeat.

## What this row does NOT claim

It does not claim a per-op number predicts a held-out loss. `of3t-p10trainout` already showed the
two-step A/B is bimodal, so the table in this row's brief (softmax owns 7ohe, layer norm owns
7vus) is two faces of one coin rather than two attributions. This row measures ops, and says so.
