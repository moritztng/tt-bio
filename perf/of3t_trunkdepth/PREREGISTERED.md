# of3t-trunkdepth — pre-registered before the first arm ran

The trunk's backward error tracks DEPTH. `of3t-trunkback` measured three single-block
boundaries, each a complete problem (that block's own captured input in, that block's own
captured output cotangent back, no error arriving from above), and they span 136x:

| scope | ds_in cos | r = err/signal |
|---|---|---|
| block 0 alone | 0.999988 | 0.004899 |
| block 23 alone | 0.999318 | 0.036951 |
| block 47 alone | 0.831806 | 0.667303 |

Every block runs identical code, so what differs at depth is the activations there. This row
puts SEVEN points on that curve instead of three and asks one question with the answer fixed in
advance: **does single-block backward error track activation norm?**

## Depths

Blocks **0, 8, 16, 23, 32, 40, 47**, captured out of ONE float64 CPU replay of the bundle's own
step (`perf/of3t_trunkdepth/capladder.sh`), so the seven boundaries are one function evaluated
once and not seven runs that happen to agree. Blocks 0, 23 and 47 are re-captured on purpose:
`of3t-trunkback` already published those three, so the re-capture is this row's reproducibility
control and costs nothing, because one forward and one backward produce all seven.

## Boundary, reference, bars — carried unchanged, not renegotiated

Boundary: the bundle's own step, `num_recycles` 0, every stochastic draw replayed, dropout at
r = 0, cropped to 64 token positions of which 56 are real (padding fraction 0.125000 single,
0.234375 pair). Manifest pinned at `462c1f52`, sha256 checked before the capture reads a byte.

Reference: upstream **0.4.3**, the revision `of3-p2-155k` binds to, every parameter and every
activation **float64**, checkpoint upcast once at load, no cast on the path (A27 names the
policy, not a width). Per-depth floor: the same upstream 0.4.3 code on the same boundary with
every parameter and activation in **bfloat16** and no autocast anywhere (`--policy bf16pure`).

Bars: per-tensor **5.0e-02**, mass-weighted **2.0e-02**, fixed here before any number exists.
**A26-SCOPE**: against a float64 reference only one side carries error, so the reachable bar is
the threshold itself. No sqrt(2) is quoted and none rescues a tensor. Everything by MASS (A23)
with the norm ratio r and the error cosine beside it (A25/D35). **Mask always**: every
activation figure is restricted to the 56 real token positions (pair to real x real). The padded
figure may appear beside a masked one, never instead of it.

`scale_pair_bias=False` on every device arm. No shipped default moves. Nothing is merged.

## The statistics, per depth k

Dependent (what we are trying to explain), two of them, reported separately:

* **R(k) = E(k) / F(k)** — our single-block mass-weighted rel_l2 against the float64 reference,
  divided by the pure-bf16 upstream floor **at that same depth**. Every row carries its own
  floor; the trunk-wide figure is **13.0x**, and 268x is not the thing being closed.
* **Ds(k) = 1 - cos(dL/ds_in)** — the direction deficit in the cotangent this block hands to
  whatever precedes it. `Dz(k)` for the pair track beside it.

Candidate explanatory variables, six, all fixed here:

* **Ns(k) = ||s_in||**, **Nz(k) = ||z_in||** — masked activation norms at that boundary;
* **Ncs(k) = ||dL/ds_out||**, **Ncz(k) = ||dL/dz_out||** — the incoming cotangent norms;
* **Ms(k)** — the single track's share of that block's float64 reference gradient mass. Named in
  advance because `of3t-rebase`'s weight census measured it at 13.91 % (block 0), 0.0036 %,
  0.0027 %, 0.0022 %, 0.0705 %, 0.0480 % (blocks 8..40) and 39.66 % (block 47) on a different
  bundle. Five orders of magnitude of depth dependence is a candidate whether or not it wins;
* **k** — the depth index itself.

## What counts as positive, and what counts as negative

Test statistic: **Spearman rank correlation** between a dependent variable and a candidate, over
the n = 7 depths.

**POSITIVE — "the error tracks activation norm"** requires all three of:

1. some NORM candidate (Ns, Nz, Ncs, Ncz) reaches **|rho| >= 0.929**. That is the exact
   two-tailed p <= 0.01 critical value at n = 7; six candidates are screened, so at p <= 0.05
   each the family-wise error would be ~0.26 and a rank correlation on seven points is cheap to
   find by accident. 0.929 keeps the family under ~0.06;
2. that candidate **spans >= 10x** across the ladder. This gate is not decoration: `of3t-rebase`
   measured `z_norm` flat to **2.1 %** over these same seven boundaries, so ||z_in|| can reach
   rho = 1.0 while explaining a 136x spread with a 2 % move. A variable that does not move
   cannot be the amplifier;
3. the log-log OLS fit of the dependent on that candidate **reproduces at least half the
   observed log-spread** of the dependent across the ladder.

If that fires, the amplifier is scale-dependent and the candidate class is a normalisation, a
clamp or a reciprocal whose Jacobian conditioning degrades as its input grows. This row names
the class; it does not get to also claim the fix.

**NEGATIVE-DEPTH — "it tracks depth but not norm"**: k reaches 0.929 and no norm candidate
passes all three conditions. Then depth dependence is real and is not a scale effect, and the
report says what it tracks instead.

**NEGATIVE-MASS — "it tracks the single track's mass share"**: Ms passes all three conditions
and no norm candidate does. Reported as its own outcome rather than folded into either of the
above, because Ms is a property of the REFERENCE and not of our port: if it wins, our
single-track error is roughly constant with depth and only becomes visible where the reference's
single track carries mass, which is a different defect statement from an amplifier.

**DIFFUSE — "no single locus"**: nothing reaches 0.929, or the dependent variable's own spread
across the seven depths is under 3x, in which case there is nothing to explain and the
three-point 136x was a sampling artifact.

All four are legitimate results and the unfavourable ones will be reported as prominently as the
favourable one. `of3t-trunkback` named a direction and not a fix, and saying so again with a
seven-point curve behind it is an acceptable answer.

## Controls, every one of them, or the row does not conclude

* **A16 zero model** — a model emitting nothing, scored on the same set. Measured 1.0 on this
  scope; anything the ladder reports above 1.0 is worse than emitting nothing.
* **Instrument floor** — the reference scored against itself, which must read 0.
* **Upstream's own recipe** — 0.4.3 under its shipped autocast (`bf16auto`) beside the pure-bf16
  floor, so the floor is not a strawman of our own construction.
* **A break control that is shown to FAIL** — the captured cotangent permuted across the real
  token positions at one depth (`--permute-cot`). Same cotangent, wrong token positions. If it
  does not read far worse than the unpermuted arm, the instrument cannot tell a right answer
  from a wrong one and no ladder row means anything.
* **Reproducibility** — blocks 0, 23 and 47 re-captured and re-scored against `of3t-trunkback`'s
  published figures. A disagreement there is reported as a finding, not smoothed over.
* **Capture self-check** — forward loss against the manifest, and the block's own parameter
  gradients recomputed here against the bundle's, at every one of the seven boundaries.

## Already refuted upstream of this row. No card is spent re-testing any of it

* **The pair-bias convention.** Pre-scaling makes the gradient worse, 5.367727 -> 8.855764,
  while improving the forward 6.44x. No setting of `scale_pair_bias` satisfies both.
* **Padding.** Zeroing every padded position on both sides removes 99.76 % of the pair input's
  squared mass and changes no reported digit. The NaN-pad probe is not a discriminator: it fires
  on both sides, because `0 x NaN` is NaN while `0 x finite` is 0.
* **`s_fp32_residual`.** 4.30x worse.
* **One leaf op's backward.** Single blocks are 10-130x better than the chain.
* **The bf16 floor as the whole explanation.** Upstream's own all-bf16 run reaches 4.196175e-01
  where we reach 5.351880e+00.
