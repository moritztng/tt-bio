# b2z2-sampler-stall-split — PREDICTED, written before the first number exists

Written 2026-09-12, before any device run. Inputs I had already read: CONTEXT.md (incl. both
corrections), PROFILER-WHGLX.md, `state/b2z2-sampler-ceiling-map.md`,
`state/b2z2-whglx-profiler-build.md`. I have NOT looked at any diffusion-step CB column — the
sampler map verified they are blank in all 1066 rows of every committed capture, which is why
this row exists.

## P1 — the inner split

The step's math thread is input-stalled at a fraction **>= 55 %** of its TRISC1 residency on WH
(the trunk reads 62.0 % on WH, 57.0 % on BH), output-stalled 8-15 %, computing 25-37 %.

Mechanism: the step's top three op codes are Matmul (41.19 %), BinaryNg (16.31 %) and SDPA
(11.38 %) of 22.015 ms. BinaryNg is 85.7 % input-stall on the WH trunk and there is no reason a
broadcast add in the sampler behaves differently. The step's matmuls are M=16 tiles x N=24 tiles
with K the only long axis, so per-core they are thin and operand-starved.

**Falsifier: wait_in < 45 % of TRISC1.** That would mean the sampler's math thread is genuinely
busy inside its kernels, the campaign's floor statement is a trunk property and not a device
property, and no movement lever prices against the step at the trunk's multiplier.

## P2 — the wait is byte-shaped, like the trunk's

Per-site wait correlates with bytes moved at Spearman **>= +0.8**, and the tile-count model scores
worse than the mean (R2 < 0), reproducing `CONTEXT §2-CORRECTION-B`'s sign on a third workload.

**Falsifier: bytes Spearman < +0.6, or the tile model beats bytes.** Then the sampler's wait is a
different mechanism from the trunk's and `b2z2-sharded-sampler`'s ~2x-on-the-term ceiling is void.

## P3 — the 15.3 % that never computes

The 3.369 ms/step of zero-math-residency programs are pure layout moves. A token-axis shard halves
their bytes and so roughly halves them; a fusion of the SDPA head chain deletes most of them
outright. I predict >= 2.2 ms of the 3.369 ms is in the SDPA head plumbing + ReshapeView and is
fusable, and that the remainder (Pad, Copy) is not worth a lever.

## P4 — TRANSFER-VERDICT: NO, and the arithmetic is already visible

Even if P1 holds exactly, the Pairformer **multiplier** cannot transfer, because a multiplier is a
fraction of the WALL and the two blocks hold their math thread resident for very different
fractions of it. Trunk: 32.144 ms resident in a 36.344 ms span, movement-free = 36.344/14.900 =
**2.439x**. Sampler: 15.211 ms resident in a 26.400 ms wall, so even at the trunk's own 72.8 %
stall fraction the step's movement-free bound is 26.400/(26.400 - 0.728*15.211) = **1.72x**, not
2.44x. The step loses 42.4 % of its wall before the CB question is asked.

So I predict `b2z2-final-ceiling`'s 1.977x top rung is **not supported** and comes down toward its
own 1.611x best-supported point. **Falsifier:** the step's resident fraction under the profiler
comes out near the block's 88 % (i.e. the 57.6 % was an artifact of the production wall including
exposed host dispatch that a profiled replay does not have), in which case the multiplier does
transfer and the rung stands.

## What I will not do

No step-cut, no recycle-cut, no MSA-depth cut. 200 steps / 3 recycles / 35 rows stay. The
instrument does not change what the device computes.
