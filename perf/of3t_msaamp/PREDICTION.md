# of3t-msaamp: pre-registration

Written and committed before `msa_amp_arm.py` exists. Scored in `state/of3t-msaamp.md`.

## The object

D58 says the backward-over-forward factor belongs to the tape and not to any module, resting on
diffusion and `msa_module` agreeing. `of3t-tapeamp` priced the diffusion leg against upstream
0.4.3's own bf16 recipe (7.666x of our 11.026x). This row does the same at the `msa_module`
boundary: `boundary_msa_module.pt` (sha256 `ca142fe9...d879f`, the float64 capture whose 227
parameter gradients are bit-identical to `grads_f64_043.pt`), 384 tokens padded, 56 real, 2 MSA
rows, cotangent on `z` out.

Arms, all out of one process each, all against that one float64 capture:

  ours      `perf/of3t_auxheads/msa_instrument.py` on qb1 card 1 (p150a), `--dump-grads`
  bf16      upstream 0.4.3 `MSAModuleStack`, fp32 params under `torch.autocast("cpu", bf16)`
  bf16 A/A  the same, second process
  bf16in    the same, with `m` and `z` rounded to bf16 before entry (our arm uploads them bf16)
  fp32      upstream, fp32, no autocast
  f64       upstream, float64: the in-frame check that the arm reproduces the capture
  BREAK     bf16 with the cotangent's token axis rolled by one

## Statistics, fixed now

Forward: rel_l2 of `z_out` on the real 56x56 token block against the capture's output (the
statistic behind our 0.008176476213472778). The full 384x384 rel_l2 is also published for the
upstream arms. Our arm's full-block figure (0.971) is a padding convention and is not used.

Gradient: per-tensor rel_l2 against the capture's `param_grads`, median AND mass-weighted, on
two scopes: upstream's full 227 and the MATCHED scope, the 151 names our arm scores. Both arms
are scored by one function (`amp_arm.score_grads`) over the dumped tensors, so the median
convention and the A14 floor are the same code on both sides.

A fact read off banked artifacts before any arm ran, and the reason both statistics are
mandatory: **D58's 10.903x is mass-weighted (0.08915 / 0.008176); diffusion's 11.026x is a median
(9.344e-02 / 8.475e-03).** On the median our `msa_module` factor is 0.027085 / 0.008176 = **3.31x**.
The two legs D58 rested on were never the same statistic. Every ratio below names its statistic.

## Predictions

P1. Upstream bf16 forward (real block) lands in [4e-03, 3e-02], central 1.2e-02.
    Falsified outside the band.

P2. Upstream bf16 MASS-WEIGHTED ratio on the matched scope lands in `of3t-tapeamp`'s registered
    6x to 13x. Falsified below 4x. Below 2x refutes the conditioning mechanism on this track.
    Upstream bf16 MEDIAN ratio lands in [1.5x, 8x], central 3.5x (ours 3.31x).

P3. Upstream fp32 ratio (both statistics) lands in [2x, 40x], forward in [1e-07, 1e-05].
    Falsified below 2x on both statistics.

P4. bf16 A/A bit-identical on forward, median, mass-weighted and both ratios.

P5. Our device arm re-taken on qb1 card 1 (p150a) reproduces qb2 card 1's (p300c) forward
    0.008176476213472778 and mass-weighted 0.08915269973058039 to every digit. A differing digit
    is output that depends on the board: a hard stop, reported as such, not smoothed.

P6. The f64 arm reproduces the capture: forward rel_l2 <= 1e-12 and gradient median <= 1e-12.
    Otherwise the arm is not the capture's function and nothing below is read.

P7. BREAK: forward bit-identical to bf16, gradient moves by more than 3x.

P8. bf16in moves upstream's forward by less than 1.5x relative to bf16.

## Decision rule, a genuine disjunction

Let G_o, G_u be the MATCHED-scope mass-weighted gradient rel_l2 (ours, upstream bf16), F_o, F_u
the real-block forwards, R = G/F. The same rule is also applied on the median, and a branch is
claimed only if both statistics land in it; a split is reported as a split.

  B1  upstream's own.   R_u >= 0.5 R_o AND G_o <= 1.25 G_u.  D58 closes as not a defect.
  B2  denominator.      R_u <  0.5 R_o AND G_o <= 1.25 G_u.  The excess is our forward being more
                        accurate than upstream's, as on diffusion. Not a defect.
  B3  gradient worse than the reference's own.  G_o > 1.25 G_u. This is NOT yet "a module
                        amplifier". Before that name is used, three other carriers must be
                        excluded, each with its own measurement:
                        (i)  input precision at the boundary: if bf16in's G_u' >= G_o / 1.25,
                             the carrier is bf16 inputs, which is the boundary, not the module;
                        (ii) the tail: if removing the three worst tensors on our side brings
                             G_o to <= 1.25 G_u, it is a per-tensor defect (owned elsewhere),
                             not a module factor;
                        (iii) the forward: if F_o > 1.25 F_u too, the gradient excess may be the
                             forward's error propagated, and the ratio R_o/R_u decides whether
                             the backward adds anything of its own.
                        Only B3 surviving all three names a module-specific backward defect.
  B4  no conditioning.  R_u < 2x on both statistics. `of3t-tapeamp`'s mechanism is refuted on this
                        track whatever B1-B3 say; stated beside the branch that also holds.

What else could carry a difference, and is not tested here: CPU autocast versus upstream's CUDA
autocast (upstream's `autocast("cuda", enabled=False)` islands do nothing on CPU, a caveat
`of3t-tapeamp` carried too); and the 9 `pwa` leaves plus 64 fused uploads our arm does not score
(73 tensors, 0.663 % of the section's squared norm), which no ratio here sees.

The 1.25x margin is chosen now: about 4x the 6.7 % cross-host bf16 scatter
`of3t-bwdaccum` control 7 measured, and it is not moved after the numbers exist.

## Addendum 1: the carrier the disjunction left out (written after the 384 arms, before the crop)

The 384 arms land in B3 on both statistics and survive exclusions (i) and (ii). The list above
missed one carrier: **padding**. 56 of 384 tokens are real, the capture's cotangent is exactly 0
outside the real 56x56 block, the reference's padded `z` has norm 1.6e7 against 6.4e4 real, and
our arm's full-block forward is 0.971 off, so its padded activations are nothing like upstream's.
Upstream masks padded tokens out of every gradient path. If ours leaks any gradient through a
padded position, the weight gradient picks up products with those huge, differing activations,
and that would look exactly like a module amplifier while being a masking defect.

Discriminator: crop the boundary to the first 64 tokens (56 real + 8 padded; the bucket rule
forbids 56). Scored post hoc and labelled as such.

C0. Upstream f64 at 64 reproduces the 384 capture's real-block `z` and all 224 scored parameter
    gradients to rel_l2 <= 1e-10. Otherwise the crop is another function and C1 is not read.
C1. Ours at 64 against the in-frame f64 at 64, beside upstream bf16 at 64.
    Our matched mass-weighted gradient <= 1.25x upstream's: the carrier is padding (the number of
    padded tokens), not the module's backward at real tokens.
    Our excess >= 5x (it is 7.29x at 384): padding is not the carrier and B3 stands as a
    backward defect in the four triangle-op families.
    Between the two: padding carries part of it, and the split is reported as a split.
I expect the second outcome, central 6x, because the per-family excess is 16-21x on all four
triangle families and 1.2-2.8x everywhere else, and a masking leak would not respect op family
that neatly.
