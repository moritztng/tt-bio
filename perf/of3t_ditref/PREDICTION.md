# Registered BEFORE the re-price, per the brief

Written at the top of pass 1 of `of3t-ditref`, committed and pushed before any arm of this row
ran on the card. Nothing below was informed by a measurement this row took.

## What I already knew when I wrote this, and it matters

Two figures existed before this pass and I am not going to pretend otherwise, because a
prediction that hides its prior is worthless:

* `perf/of3t_rebase/device_gradient_043all.json` -- scope median rel_l2 **0.16588485538135056**
  over **547** compared tensors, forward median **8.474800850934073e-03**, share of the diffusion
  squared norm **0.5732**. That artifact was already taken at a **0.4.3** diffusion capture.
* `of3t-ditcot`'s three arms -- 0.7055 / 0.2584 / 0.1065 -- every one of them at
  `cap = /home/ttuser/of3t_diffusion_cap`, the **0.5.0** capture, which their own `provenance`
  block records and their state doc does not mention.

So I am not predicting blind that the denominator is wrong. I am predicting **by how much**, in
which direction each downstream figure moves, and what the repair does NOT reach.

## P1. The repair is a CAPTURE swap, not a package build, and it is already on disk

`/home/ttuser/of3t_softgrad/diffcap043` holds `diffusion_boundary.pt` and `sub_boundary.pt` built
by `perf/of3t_softgrad/recap043.sh` on the 0.4.3 tree. I predict `device_gradient.py --cap` at
that directory is the entire Deliverable 0 and that no reference package has to be built. If a
build turns out to be needed, this prediction is a miss.

## P2. The compared-tensor count goes 523 -> ~547, and the +24 are the per-block norms

0.4.3 gives `diffusion_module` **761** parameters where 0.5.0 gives 738, and 761 - 738 = 23 =
24 per-block `attention_pair_bias.layer_norm_z` minus the 1 shared
`diffusion_transformer.layer_norm_z` that 0.4.3 keeps only for the cross-attention atom
transformers. At the 0.5.0 capture our 24 trained per-block norms have no reference counterpart at
all, so they are silently dropped from the comparison. I predict the repaired arm compares
**547 +/- 2** tensors against ditcot's 523, and that the 24 newly-comparable tensors are exactly
the per-block `attention_pair_bias.layer_norm_z`. This is the cleanest falsifier in this file: if
the count does not move, the capture I picked is not the 0.4.3 one.

## P3. SHRINK, and of3t-ditcot's 6.62x OVERSTATES the repair

Predicted scope median on the repaired denominator: **0.13 to 0.20**, i.e. near 0.16588 rather
than 0.7055, for a repair factor of about **4.2x** -- materially LESS than the 6.62x
`of3t-ditcot` priced.

The reasoning, registered in advance: `of3t-ditcot` priced the gap by crippling OUR side. It
deleted the 24 trained per-block norms and installed one all-ones norm, which is the reference's
architecture but also removes 24 trained tensors' worth of real bf16 error from the numerator.
The correct repair goes the other way, fixing the REFERENCE while our port keeps every trained
weight and every bf16 rounding it commits on them. So the two directions must NOT agree, and the
honest repair has to be the smaller of the two. If the repaired reading lands at or below 0.1065
I was wrong about the direction of the asymmetry.

## P4. D129 -- SHRINKS, and more likely than not DISSOLVES

D129 reads 0.693974 against the 0.5.0 bf16 floor 1.5815634233e-01 = **4.388x**, and 3.10x A26's
bar. `of3t-cond043` measured the 0.4.3 floor on the same leaf at **1.5931532097e-01**, 0.73 %
away, so the bar goes to 0.2253059 and the DENOMINATOR of D129 does not move. Everything
therefore rides on our own arm at the 0.4.3 capture, which is what this row takes.

Predicted `conditioned_transition.layer_norm.layer_norm_s.weight` median over its instances:
**0.12 to 0.30**, giving a ratio to its own 0.4.3 bf16 floor of **0.8x to 1.9x**.

Outcome probabilities, fixed now:

* **DISSOLVES** (under 2x its own floor, so it no longer clears A26 as a defect) -- **60 %**
* **SHRINKS but survives** (2x to 4x) -- **30 %**
* **SURVIVES INTACT or INVERTS** (at or above 4.388x) -- **10 %**

And the flat 2.35x cotangent excess `of3t-condtrans` located (the brief calls it 2.28x; the
ledger figure at D129 UPDATE pass 237 is 2.35x, uniform 1.95x-2.59x over 24 blocks) should fall
by roughly the same factor the scope does. If the leaf improves and the 2.35x does NOT, the
excess is real arithmetic and the architecture gap was never its cause.

## P5. D30 -- NO MOVE, because it was never at 0.5.0

D30's 19.6x comes from `device_gradient_043all.json`, which is already a 0.4.3 reading. I predict
this row REPRODUCES it rather than re-prices it: forward median **7e-03 to 1.0e-02**, gradient
median **0.13 to 0.20**, ratio **15x to 24x**. A re-run that lands far outside that says something
in `tt_bio/` moved between that artifact and today, which would be a finding of its own.

## P6. D58 -- OUT OF REACH of this repair, and that is the answer

`msa_module` is trunk-side. The architecture difference is `attention_pair_bias.layer_norm_z` on
the DIFFUSION path only, and `of3t-auxheads` took its boundary from upstream 0.4.3 already
(reference loss 1.267624369070698 reproduced digit for digit, gradients bit-identical to
`grads_f64_043.pt`). I predict the repaired denominator does not touch D58's 19.8x at all, and
that the honest re-price is a scope argument plus a resolved-tree reading, not a new number. If
`msa_module`'s reference turns out to have been the 0.5.0 one after all, I am wrong and D58 is
the bigger finding of the two.

## P7. The ablation, if there is anything left to attribute

If P4 lands where I expect, D129's leaf is no longer the object and a per-op ablation aimed at it
would be attributing a factor that is not there. The residual at scope, about 0.166 against the
2.0e-02 median bar and still 8x out, is what would be worth ablating instead, and I am registering
that switch of target now so it cannot look like a rescue after the fact.

On which op carries it I keep `of3t-ditcot`'s ranking, which was registered before its own stop
and never tested: **not one op**. A dtype boundary at the block edge first, the AdaLN gain/shift
path second, the softmax backward third. The brief ranks softmax backward first and I expect that
to be wrong.
