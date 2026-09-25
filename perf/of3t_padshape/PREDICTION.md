# of3t-padshape — PREDICTION, registered before the first arm

Written and pushed before any device run of this row. Nothing below was edited after a number
existed. The banked endpoints the prediction is built on are the brief's, not this row's:
trunk gradient norm **12.3912543630** at width 64 (8 pad rows) and **43.2103398400** at width 384
(328 pad rows), ratio **3.487164**.

## What is already fixed before the sweep starts

Checked on host this pass, recorded in `SETUP.json`:

* `boundary_c64.pt` and `boundary_n384.pt` agree **bit-for-bit on the real block**, both tracks
  (absdiff exactly 0.0; s_in real norm 5623.548807821603, z_in real norm 64263.67350116303). The
  c64 capture is literally the first 64 tokens of the n384 capture, pad content included, so the
  four width boundaries can be cut from one file and the 64 arm must reproduce 12.3912543630.
* The block-47 cotangent is **exactly zero on every pad row and every pad column**, both tracks
  (`s_pad_absmax` 0.0, `z_pad_absmax` 0.0, `sum_ALL == sum_REAL` to the last bit).
* Upstream 0.4.3's float64 forward reference is **bit-identical on the real block** at c64 and at
  n384 (absdiff exactly 0.0, s 2178093.2982020588, z 88025.41673887867). One reference therefore
  scores the forward at every width; no new CPU reference is needed at 128, 192 or 256.

## Deliverable 1 — the gradient. Three laws, three different numbers

Write ||g(W)||^2 = A^2 + B^2 f(W): a genuine part that does not see the width, and a spurious part
that does. Fit f on the two banked endpoints and the three candidates disagree at 128 and 256 by
more than any plausible run-to-run floor.

| law | f(W) | A | predicted \|\|g\|\| at 128 | at 192 | at 256 |
|---|---|---|---|---|---|
| spurious mass per PAD ROW | n_pad = W - 56 | 10.521 | **22.28** | 28.13 | **34.37** |
| spurious mass per PAD PAIR | n_pad^2 | 12.352 | **15.33** | 21.31 | **28.11** |
| pure power law in W, p = 0.69718 | W^2p | 0 | **20.10** | 26.42 | **32.59** |

I predict the **pad-row law**, f(W) = W - 56, and I expect the sweep to land near 22.3 and 34.4.
The reasoning is not a fit. The pad ACTIVATIONS are large and grow with width (of3t-ditmodel swung
them by 6.43e+06 at every block and `padleak` moved the pad output by 7.7862e+02 while the real
block stayed bit-identical), so any backward path that puts a nonzero cotangent on a pad ROW
contributes g_pad (x) x_pad to dW once per pad row. That is linear in the row count, and the norm
is therefore the square root of it. The pair-track variant is the same mechanism counted on z, and
it is the one I expect to lose, because a pad row leaking into a weight gradient needs only the
single track to carry it.

What each outcome would mean, per the brief:

* **smooth in W (any of the three above)** -> a reduction or accumulation-order effect, and the
  first place to read is `_sum_leading` (`tt_bio/autograd.py:432`), which reduces 147,456 leading
  coordinates at 384 against 4,096 at 64, plus the matmul backwards that share the property.
* **a step at a tile or power-of-two boundary** -> shape-keyed kernel or config selection. 64, 128
  and 256 are all powers of two and cannot tell a step from a power law on their own, which is why
  **192 is in this sweep and was not asked for**: 192 is 6 tiles and not a power of two, so a law
  fitted on 64/128/256/384 that misses 192 is a step and not a law.
* **flat** -> the 3.487164 is not a width law at all and D175 is retired rather than moved. I do
  not expect this: the two captures are now known to be one file, so a difference beyond the real
  block is no longer available as an explanation.

## Deliverable 1, AMENDMENT 1 — the forward at the same four widths

Registered before running, as asked. **I predict the forward real-block error is FLAT in width
while the gradient grows**, and therefore that the defect is backward-only.

This is a prediction with a number attached and it is nearly already decided by artifacts
committed on `wk/of3t`, which is worth saying plainly rather than claiming as a discovery:
`perf/of3t_trunk043ref/SCORE_c64.json` and `SCORE_n384.json` hold the shipped arm's masked forward
error against the 0.4.3 float64 reference at exactly the two endpoints of this sweep --
s 0.1065338121878499 at 64 against 0.1061988978775229 at 384, z 0.04947045199709192 against
0.04853716018979657. That is 0.997x on s and 0.981x on z where the gradient is 3.487x. So I predict
my own four-width forward curve reproduces those two endpoints and stays inside 1.05x end to end.

If it does: a forward-side kernel-selection story for D175 is dead, and deliverable 2 must bisect
the BACKWARD. If instead the forward grows with the gradient, the bisection starts from the forward
because it is the cheaper half to isolate, and the shape-keyed-dispatch reading comes back.

## Deliverable 2 — which sub-module

Pre-registered so the ablation is a test and not a description: I expect the width dependence to be
**concentrated, not spread**, and I expect it in the **triangle ops** (`tri_mul_in`, `tri_mul_out`,
`tri_att_start`, `tri_att_end`) rather than in the transitions. The transitions are per-token and
carry no cross-token reduction, so a per-row pad leak cannot reach their weight gradients at a rate
that depends on width; the triangle ops reduce over the token axis and are the only place in the
block where a pad row can enter a real output's gradient. If the dependence turns out to sit in
`transition_s` or `transition_z`, this prediction is wrong and I will say so in that word.

## Controls fixed now

A16 zero baseline and an A/A before any number is quoted. A/A at width 64 and at width 256, same
card, same process shape: two runs of one arm must agree bit-for-bit, and if they do not, the
sweep's differences are read against that floor and not against zero. Gradients scored only against
the pinned float64 reference `grads_f64_043.pt`, sha256 verified before loading. One card, card 0
of qb2, every arm.
