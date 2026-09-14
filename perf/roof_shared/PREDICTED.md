# PREDICTED — written and committed before the first fold ran

The anchor's TT-vs-upstream numbers are unpaired: 0.90077 A at 298 aa and 1.42726 A at 512 aa
(worst per-pseudo-domain all-atom), against a measured seed floor of 0.92565 A / 1.82527 A for the
reference stack compared to itself with only its noise realisation changed. This task folds the TT
side with `TT_BIO_SHARED_DRAW_SEED=0` so the two stacks consume the same noise and the residual is
arithmetic.

What I expect, before looking:

1. **The draws align.** Both stacks reseed the CPU generator at the sampler's first draw and then
   draw `(1, n_atoms, 3)`. `tt_bio/data/pad.py` is a verbatim port of `boltz/data/pad.py` and
   `pad_to_max` on a batch of one returns a stack with no padding, so the atom axis is the real
   atom count on both sides, and the anchor already verified the two CIFs carry the same atom key
   list. First-draw shape and digest are recorded in the run json so this is checked, not assumed.

2. **298 aa, paired, worst per-pseudo-domain all-atom: 0.10-0.35 A.** Upstream's own fp32-vs-bf16
   paired control costs 0.01993 A, and the TT port is a larger arithmetic perturbation than an AMP
   cast: different kernels, different accumulation order, bf16 activations in places fp32 AMP keeps
   wide. I expect roughly an order of magnitude above the bf16 control, not two.

3. **512 aa, paired, worst per-pseudo-domain all-atom: 0.15-0.55 A.** Same reasoning, plus the
   longer chain and the free hinge, which the per-pseudo-domain reading is chosen to exclude.

4. So **at least 60 % of the unpaired figure is the seed floor and dissolves.** If the paired
   number comes back near the unpaired one, either the draws did not align or the port's arithmetic
   really is a full Angstrom away from upstream, and the draw trace tells the two apart.

5. **The CA-lDDT deficit does not move**: within +/-0.005 of the anchor's -0.0102 (298 aa),
   -0.0156 (512 copy 1) and -0.0240 (512 copy 2). The deficit is a systematic property of the
   port's arithmetic, not a sampling accident, so pairing the draws should sharpen it rather than
   shrink it.

6. **Control**: the TT plain arm against the TT shared arm is the same code with a different noise
   realisation, so it should land at seed-floor scale, 0.7-1.3 A at 298 aa and 1.0-2.0 A at 512 aa.

Card: one Wormhole chip on the whglx Galaxy (card 5), not the Blackhole the anchor's TT arm ran on.
pc's p150a is excluded by `pc-card0-512aa-fold-nondeterminism` (it silently miscomputes matmuls at
a low rate) and no qb2 card was free. The paired and unpaired TT arms are both folded here in one
session, so the comparison between them is card-consistent; only the comparison to the anchor's
absolute Blackhole numbers crosses card types.
