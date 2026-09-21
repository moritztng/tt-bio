# Registered before measuring, per the brief and AMENDMENT 3

Written at the top of pass 1, before any transport run and before any ablation. No ablation
ran this pass: D141 stopped it, so most of what is here stays UNTESTED and is recorded so a
later pass cannot quietly rewrite it.

## What I predicted about the 2.28x, before looking

**The factor is not one op.** The brief predicts it is, and I expect that to be wrong.

Reasoning registered in advance: a constant that holds to 1.33x spread while the underlying
quantity moves 2.98x across the stack is a property of the structure the transport repeats,
not of one kernel inside it. If a single op carried it, the op's own error would have to track
the cotangent magnitude it is handed, and the ratio would inherit some of that 2.98x span. A
per-block-invariant multiplier is more naturally a dtype boundary crossed once per block, or a
rounding that every block applies identically to a renormalised quantity.

Ranked, most likely first:

1. **A dtype boundary at the block input/output**, not an op interior. bf16 activation storage
   between blocks rounds the cotangent once per block regardless of its size.
2. **The AdaLN gain/shift path**, because it multiplies the whole residual stream by a
   per-block learned scale, which is exactly a multiplicative constant.
3. The softmax backward. Listed first in the brief; I rank it third because D56's renorm
   repair already landed and the arm is measured with it ON.
4. The attention matmuls.
5. The SwiGLU transition.

## What I predicted about D141, before running the probe

I expected the mismatch to be REAL but to cost little, on the reasoning that a LayerNorm whose
weight is stuck at all-ones is still a LayerNorm, so the reference would be a slightly
different function rather than a badly wrong one.

**This half was measurable this pass and the first clause held.** The probe confirms the
mismatch: the reference runs one shared `layer_norm_z` at all-ones, `max|w-1| = 0.0`, while the
checkpoint's 48 trained per-block tensors are dropped as `unexpected_keys`. The trained weights
have mean 0.276 to 0.568 and std 0.130 to 0.224, so they are nowhere near ones and "slightly
different" was optimistic on its face. Whether it costs little is a separate number and the
probe does not settle it.

## What I predicted about D55's four reductions

I expected all four INERT, same as the six before them, on the grounds that the six already
measured were inert and nothing distinguishes these four arithmetically.

**Held, 4 for 4.** Every pull is bit-identical, every LoFi break control on the same sites
moves. The part I did NOT predict, and should have, is that two of the four do not execute in
either model scope at all: T1 is a dead twin of the rule production dispatches, and T3 needs
the fused-SDPA verb. A prediction of "inert" that turns out to rest on "never runs" is a
different answer, and the reach counts are what separate them.
