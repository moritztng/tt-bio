# of3t-widthattr — pre-registration, written before the first scoring run

Row `of3t-widthattr`, branch `wk/of3t-widthattr` off `wk/of3t` at `fb217fb68` (of3t-frame384's
tip). CPU only: no card taken, no Tenstorrent device opened. Committed before the decomposition
script was run once.

D191: at padded width 384 our trunk arm reads **0.8354121633** against the capture's own float64
and at width 64 it reads **0.3833065668**, while upstream's own bf16 floor is flat to six digits
(0.3739383921 at 384, 0.3739375769 at 64, both built on qb1). Ratio 2.2341x against 1.0251x. The
question is which tensors' ABSOLUTE error grew.

## What I already know before measuring, and it shapes every prediction below

Read off `REF_F64_N384.json` and `C64_F64_plain.json`, which exist and were not produced here:

  * **the content is identical.** `real_tokens` is 56 at BOTH widths. The 384 run is the 64 run
    with 320 more pad rows, not a longer or different sequence.
  * **the loss is identical to every digit**, -0.3073181442478611 at both.
  * **upstream's float64 trunk gradient is width-invariant**: squared norm 1.8714981803225017 at
    384 against 1.8714981803225077 at 64, 3.2e-15 relative. So the DENOMINATOR of the
    mass-weighted rel L2 does not move, and a difference of numerators is a difference of
    absolute errors, not a share moving under a collapsing denominator.
  * **the pads carry content, and most of it.** `z_in_norm` is 16158081.29 at 384 against
    1320432.53 at 64. Squared, 150.0x. The pad-cell count ratio is (384^2-56^2)/(64^2-56^2) =
    144320/960 = 150.33. Those two agreeing to 0.2 % says z_in is essentially uniform-magnitude
    pad filler with a negligible real corner. `s_in_norm` is 11757.13 against 5850.17, squared
    4.04x, against a pad-row ratio of 328/8 = 41x, so the single track's pads are NOT uniform
    filler and its real rows carry real weight.

That combination is the whole frame of this row. Adding exact zeros to a sum changes nothing, so
upstream's width invariance says upstream's pad terms are EXACTLY zero in the gradient path.
Therefore any width dependence in our arm means our pad terms are not exactly zero, or our
arithmetic on the real tokens changed when the shape changed. Those are the two families below.

## Four candidate mechanisms, named in advance

**M1 — pair-axis pad leak.** Pad cells of z leak into parameter gradients through reductions
that run over the pair axes. Predicts growth ordered by how pair-axis-heavy a parameter's
reduction is, and a large exponent: the available pad mass grows 150.3x.

**M2 — single-axis pad leak.** Pad rows of s leak through reductions over the single token axis.
Available pad-row ratio 41x, but s pads are not uniform filler, so the realised exponent should
be smaller than M1's. Separates from M1 by which FAMILIES carry the growth: `layer_norm_a`,
`single_transition` and the q/k/v projections read s; `tri_mul_*`, `tri_att_*`,
`pair_transition` and `linear_z` read z.

**M3 — shape-keyed arithmetic on the real tokens.** Our port selects kernels, program configs or
chunk sizes by shape, so the arithmetic applied to the same 56 real tokens differs at 64 and at
384 with no pad leak involved. `of3t-padshape`'s non-monotone shape-keyed step is the precedent
and D175 is refuted as a monotone law, so this is live. Predicts a roughly UNIFORM multiplier
across families rather than an ordering by axis, and it is the one mechanism that does not
require our pads to be non-zero.

**M4 — depth amplification of a width-seeded perturbation.** A small width-dependent error is
created in the first blocks and amplified through 48 blocks. Predicts per-block growth RISING
with depth.

## Pre-registered predictions, each with what falsifies it

Unit throughout: the numerator is mw^2 = sum_t m_t rel_t^2 = sum_t ||ours_t - ref_t||^2 / ||g_ref||^2,
so GROWTH_t = m_t rel_t^2 at 384 minus the same at 64, and the growths sum exactly to
0.8354121633^2 - 0.3833065668^2 = 0.5509... That additivity is exact, not a model, and I will
report it as a closure check.

  * **P1 — the reference is width-invariant per tensor**, worst case under 1e-10 relative over
    2,736 of 2,736 tensors. Falsified by any tensor over that; if it fails, every number in this
    row is measuring two different references and the row says so instead of reporting a growth.
  * **P2 — `boundary_c64.pt` is exactly `boundary_n384.pt` cropped**: s[:, :64] and
    z[:, :64, :64] bit-identical, max absdiff exactly 0.0. Falsified by any non-zero difference,
    which would mean 64 and 384 are two captures and not one capture at two widths.
  * **P3 — the floor's own growth is under 1 % of ours.** The published floors differ by 2.2e-05
    relative, so I expect the floor's per-tensor growths to cancel rather than to be individually
    zero. Falsified if the floor's summed |growth| is comparable to ours.
  * **P4 — the growth is concentrated, not spread.** I predict the top 20 tensors of 2,736 carry
    **over 50 %** of the growth, and `attn_pair_bias.*` as a family carries **at least 40 %**.
    That follows from of3t-frame384's leaf table, where `attn_pair_bias.layer_norm_a.weight`
    alone takes 31.80 % of the 384 error mass and moves 0.85x -> 2.32x of floor. Falsified by a
    flat family table.
  * **P5 — SOFTMAX: no.** The `of3t-apbback` bisect's scope is 16 tensors, `attn_pair_bias.*`
    and `single_transition.*` of block 47 ALONE. of3t-frame384 measured block 46 as barely
    moving. I predict those 16 tensors carry **under 2 %** of the trunk growth, which makes the
    softmax backward finding and D191 two objects whatever the softmax does at 384. Falsified if
    that scope carries a large share, in which case they are one object and I say so.
  * **P6 — the family growth exponents will separate M1 from M3.** Writing the per-family growth
    as ||e||^2(384)/||e||^2(64) = 6^p: M1 predicts p > 2 for z-reading families, M2 predicts
    0.8 < p < 2 for s-reading families, M3 predicts p roughly equal across all families. I
    pre-register that the SPREAD of p across the top eight families will exceed 0.5 (families
    separate, M1/M2 live) rather than being under 0.2 (uniform, M3 live).
  * **P7 — M4 is the least likely and I predict it fails.** of3t-frame384 already reported block
    0 moving 1.0668 -> 2.5224 and block 46 moving 0.1233 -> 0.1152, which is the opposite of
    rising with depth. I predict the per-block growth will NOT be monotone in depth and that the
    last two blocks will contribute under 5 % of it. Falsified by a monotone rising profile.

## What the brief says that I expect to contradict

The brief states the softmax result as "at crop 64 the softmax backward alone recovers 51.55 %
of block 47's error". `perf/of3t_apbback/devblk.py` line 7 says the arms were "run at crop 384",
`DEV_BLK47_softmax.json`'s probe reads `z_in_norm` 15601002.6, which is the 384 boundary and not
the 64 one, and `BISECT_BLK47.json` carries no crop field at all. I expect to confirm that the
bisect is a crop-384 object, which means there is no crop-64 bisect to difference it against and
the standing hypothesis cannot be tested in the differencing form the brief asks for. If that
holds it is reported to the orchestrator as a record correction and answered by containment
instead: how much of the GROWTH lives inside the 16 tensors the bisect scored.

## Controls, fixed here, all run before any headline number

  * A/A: the 384 float64 reference scored against itself must read exactly 0.
  * A16: an exact-zero gradient against each reference must read exactly 1.0, measured in this
    row's own scorer at BOTH widths, not inherited.
  * REPRODUCTION: this row's scorer must recompute of3t-frame384's four published figures --
    0.8354121633458239, 0.3739383921130369, 0.3833065667807629, 0.3739375768802167 -- to 1e-12
    before any growth number is quoted. A decomposition of a number I cannot reproduce is a
    decomposition of something else.
  * CLOSURE: the per-tensor growths must sum to the difference of the squared headline numbers
    to 1e-12.
  * Every floor names the host that produced it (D189: two c64 floors off different boxes read
    0.4007237405 and 0.3739375769, 7.2 % apart).

## What this row cannot do

No card, so no device arm at any width other than the two already banked, and no re-run of the
block-47 substitution bisect at a second width. Intermediate widths are reachable for the
REFERENCE and the FLOOR only, CPU-side, and if they are run they are labelled as a floor curve
with no device arm beside them.
