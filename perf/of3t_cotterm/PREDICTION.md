# of3t-cotterm — pre-registration

Committed before the first capture exists and before any number in this row's artifacts.
Nothing below may be edited once a measurement has been read; a miss is the finding.

## What is being decomposed

`of3t-apbleaf` located the trunk's gradient residue at `attn_pair_bias.layer_norm_a` in the
**cotangent arriving at that site**, and split that cotangent's error exactly, using the
linearity of `dW = sum_t g_t xhat_t` in `g`:

    e = g - g_ref,  e_par along g_ref,  e_perp ACROSS it

    from e_par     ours 7.971e-02    upstream 6.358e-02    residue 1.2537x
    from e_perp    ours 6.408e-01    upstream 2.684e-01    residue 2.3877x
    across share   ours 99.72 %

`e_perp` is this row's object. It is not the total: the total cotangent error (ours 1.4924
rel_l2 against float64, upstream's own bf16 1.2706) is mostly shared with upstream and is the
reference's own difficulty. What is ours is the direction.

## The transfer function, and where the boundary is

The cotangent at the site is the output of the AttentionPairBias backward. Written as a
function of that module's own inputs, with `a` the LayerNorm's OUTPUT (= the device's `s_norm`,
which is APB's first argument):

    g = F(a, z, do, W)        F = the float64 APB forward+backward
                              a  single-track activation at the site   [1, N, 384]
                              z  PAIR-TRACK activation at APB's input  [1, N, N, 128]
                              do inherited cotangent at APB's output   [1, N, 384]
                              W  the module's weights

`z` enters F only through `bias_raw = permute(linear_z(layer_norm_z(z)))`, so the stored form
of the `z` channel is that projection, evaluated in float64 on each side's own `z` with the same
operator. The attention mask is `-1e9 * (1 - single_mask)` on both sides -- upstream builds it
in `_prep_bias`, our harness passes the identical tensor (`dev_grad.py`: `attn = (1.0 - sm) *
-1e9`) -- so it is not an error channel and is held fixed.

Two exact decompositions of `e = g_dev - g_ref` follow.

**TERMS.** `a` reaches the attention through four linears, so F's output is a sum of four
terms and the split is an identity, not an approximation:

    g = T_Q + T_K + T_V + T_GATE
    e = LOCAL + sum_t [ T_t(ours) - T_t(ref) ],   LOCAL = g_dev - F(a_d, z_d, do_d, W_d)

`LOCAL` is every rounding the device's own APB arithmetic makes, forward and backward,
including the bias projection, the softmax and the matmuls. It is the same quantity `L` was at
the leaf, one level up.

**CHANNELS.** One input at a time from the device, the rest from the reference:

    A  = F(a_d, z_r, do_r, W_r)      Z = F(a_r, z_d, do_r, W_r)
    DO = F(a_r, z_r, do_d, W_r)      W = F(a_r, z_r, do_r, W_d)
    ALL = F(a_d, z_d, do_d, W_d)

F is nonlinear in these, so the channel split is first-order and its sum check is a real check
rather than an identity. It is reported as such.

Every figure is **masked to the 56 real rows of 384**. `of3t-apbleaf` measured the forward
activation at 1.3827 over padded 384 and 1.7e-03 to 1.5e-02 over those 56 rows, and
`of3t-trunkact` found the pad region holds 99.9855 % of the forward's squared error and 0.000 %
of the gradient. A padded figure here would be pad junk. Direction is reported beside magnitude
at every site, because the reduction below the site is cancellation-limited and that is exactly
how the object was mis-described as a magnitude deficit for five passes.

## The prediction, two-sided

Let `S_c = ||P(e_c)||` be the across mass a channel or term carries, `P` the projector across
`g_ref`, and the denominator the across mass of the `ALL` arm, pooled over the 48 sites with one
declared denominator.

* **NAMED** if a single channel or term holds `S_c / S_ALL >= 0.60`.
* **SPREAD**, and the hypothesis dies, if `max_c S_c / S_ALL <= 0.40`.
* Anything between is reported as a share and named neither.

**Primary prediction: NAMED on `Z`** -- the across component is concentrated on the pair-track
channel, i.e. a contaminated pair-track contribution folded into the single track. The ground
for it: the single track's own activation at this site is accurate to 0.17 %-1.5 % on the rows
that matter (`of3t-apbleaf`), so `A` has little to give; the pair track carries the trunk's
whole width and its own 48 blocks of triangle ops; and `z` reaches `g` only through the softmax,
which is where a small bias error becomes a new DIRECTION rather than a scale -- the exact
signature `e_perp` has.

**Falsifier, and it is a clean death:** if `max_c S_c / S_ALL <= 0.40` the direction error is a
property of the whole backward rather than one path, and this row says so. A second, sharper
falsifier: if `DO` is NAMED, the across component is INHERITED from below the site and the
object is not in this module at all.

Three secondary predictions, each with its refutation:

1. `LOCAL / ||e|| <= 0.25`, i.e. the device's own APB arithmetic is not the carrier, on the same
   ground `L = 0.005544` held at the leaf. Refuted if `LOCAL` is over 0.25 of the total.
2. `T_GATE` is not the carrier: `S_GATE / S_ALL <= 0.25`. The gate path never touches `z` and
   `sigmoid` is contractive. Refuted by a `T_GATE` share above that.
3. The sum check on the channel axis holds to within a factor of 2:
   `0.5 <= ||sum_c P(e_c)|| / ||P(e_ALL)|| <= 2.0`. A reading outside that band means the
   channels interact strongly and no one-at-a-time attribution is readable; it would be reported
   rather than repaired.

## Instrument constraints, fixed in advance

**`R44` is not used and cannot be.** It is `ds` at rung 44, a single-track metric.
`of3t-blk4544` measured `ds` at exactly 0.0 on all 49 rungs for both pins, and
`of3t-readverbs`' on-path control doubled the cotangent at 960 sites and left all 49 `ds` rungs
bit-identical while 48 of 49 `dz` rungs diverged. A refutation read off `R44` is scoped to `ds`
and says nothing about the pair track. This row reads the track on `dz` -- the cotangent the APB
backward EXPORTS to the pair track, `dbias`, which is the same quantity `dz` is at this site --
and on bit-identity of the instrumented arm against the banked device gradient.

**The float64 reference** is upstream 0.4.3's own `AttentionPairBias` arithmetic in float64,
re-expressed with four separate leaves so the terms are separable. It is validated three ways
before any arm is read: the forward against upstream's own module on the same operands, the
total cotangent against torch's float64 autograd through that module, and against float64
**central finite differences** with `h` SWEPT. `of3t-apbleaf`'s `FDSWEEP` showed the campaign's
1e-9 clause misses at `h = 1e-6` and the residual falls as `1/h`, so the band is quoted rather
than a single point, and the bar is not moved.

**The A/A determinism floor** is read and reported BEFORE any arm, on the quantities this row
consumes -- the captured APB operands, not only the parameter gradients, because a floor on the
outputs does not cover the operands.

**Every artifact carries a `host` field** and the board class, with AICLK sampled DURING the
device arm and its sample count reported (D155/D235).

## The counterfactual

The trunk figure if the NAMED term were exact, in the frame `of3t-apbleaf` and D218 fixed:
ours against upstream's own bf16 autocast at padded 384, where the shipped trunk reads
1.0293953377723410 and the in-frame A26 bar is **0.5268825372815341**. `0.4361680548` is
cross-frame and withdrawn and appears in no artifact of this row. An in-frame number is not
divided into a model-frame threshold (D214/D218).
