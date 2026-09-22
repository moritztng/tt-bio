# of3t-trunkback — pre-registration, written before the first number exists

The predecessor `of3t-trunkgrad` (NO-GO) left one contrast. Against a 0.4.3 float64 reference
over the same captured boundary, the 48-block trunk`s pair-track FORWARD reads 4.947045e-02,
under the 5.0e-02 bar, while its parameter GRADIENT reads 5.367727e+00 with norm ratio 5.412308
and error cosine 0.136779. A model emitting nothing reads exactly 1.0. Upstream`s own
float32+autocast bf16 recipe reads 3.686000e-01. We are 5.4x further from the reference than
silence and 14.6x further than upstream`s own recipe, at a forward that agrees.

This row asks where in the BACKWARD that comes from. Everything below is fixed now.

## Boundary, reference and bars — unchanged from `of3t-trunkgrad`, deliberately

Boundary `of3t_gradients/cap`: block i`s captured input in, block j`s captured output cotangent
back, cropped to the first 64 token positions of which 56 are real (padding fraction 0.125000
single, 0.234375 pair). Reference: upstream 0.4.3, the revision the `of3-p2-155k` checkpoint is
bound to, every parameter and every activation float64, checkpoint upcast once at load, no cast
on the path (A27 names the policy, not a width). Bars: per-tensor 5.0e-02, mass-weighted
2.0e-02. A26-SCOPE: the reference is float64, one side carries error, the reachable bar is the
threshold itself and no sqrt(2) is quoted. Everything is reported by MASS (A23), with the norm
ratio and the error cosine beside every relative L2 (A25/D35).

D1 is NOT flipped. `scale_pair_bias=False` stays the shipped default in every arm that claims to
be the shipped arm; the pre-scaled convention appears only as a named contrast arm. No shipped
default moves in this row.

## I1 — LOCUS by DEPTH. Is the error created in one block`s backward, or accumulated over 48?

A 48-deep stack`s parameter gradient can disagree because one leaf op`s backward is wrong, or
because a per-block error of a few parts in a thousand compounds through 48 chained backwards.
Those are different defects with different fixes and the stack figure cannot tell them apart.

`cap/` holds three captured boundaries: blocks 0, 23 and 47. Each is a complete, exact
single-block problem — that block`s own input in, that block`s own output cotangent back — so
each block`s backward can be scored ALONE, with no chaining at all. Run all three on the device
and in float64, score the same way, and put the three single-block figures beside the 48-stack`s
own per-block figures at the same three blocks.

Branches, named now:

- **B1 ACCUMULATION.** Every single-block arm is at or under the 5.0e-02 per-tensor bar while
  the stack reads 5.367727e+00. Then no leaf op`s backward is wrong and the error is created by
  chaining the cotangent through 48 of them. The locus is the chain, and the fix, if one exists,
  is a precision one on the carried cotangent and not a kernel one.
- **B2 PER-BLOCK OP.** A single-block arm is already order 1. Then one block`s own backward is
  wrong, the per-leaf medians name which leaf, and the stack figure is that defect 48 times.
- **B3 MIXED.** The single-block arms are above the bar but far below 5.37. Then both are real
  and the deliverable is the growth factor from one block to 48 beside the per-block floor.
- **B4 DIFFUSE.** The per-leaf medians are flat across leaves and no single leaf carries the
  error. "There is no single locus" is a legitimate outcome and it will be said plainly.

## I2 — LOCUS by LEAF, by MEDIAN and never by the worst tensor

`worst-tensor-names-the-tail-not-the-locus`: the predecessor`s worst case,
`pairformer_stack.blocks.5.attn_pair_bias.layer_norm_z.bias` at 1.4540e+15, is a denominator
artifact and names nothing. The locus reading is the MEDIAN rel_l2 per leaf op, with that leaf`s
share of the reference mass and of the error mass beside it, so a median over a massless leaf
cannot lead the conclusion. `pair_transition` is a starting point and not an answer.

## I3 — FORWARD_AGREES, in the SAME process that takes the gradient

Every device arm scores its own forward against that same arm`s float64 reference forward, both
tracks, masked, in the process that then takes the gradient (A18, `--forward-reference`). A
forward carried in from another process is not a contrast; `of3t-auxgrad` had to undo exactly
that. At the single-block boundaries this is the load-bearing reading: a block whose forward
passes and whose isolated backward fails is the row`s premise demonstrated inside one process.

Named in advance: the pair track is the track that agrees. The SINGLE track`s forward is
1.064843e-01 under the shipped convention, which is above the bar, and `of3t-trunkcliff` closed
that as the held pair-bias convention. If the gradient error mass sits in the single track`s
leaves, "the forward agrees" is true only of the pair track and this row will say so rather than
inherit the premise.

## I4 — NORM. Magnitude or direction?

For one arm against one reference, with r the norm ratio and c the error cosine,

    rel^2 = 1 + r^2 - 2rc = (r - 1)^2 + 2r(1 - c)

exactly. The first term is the magnitude half, the second the direction half. Reported as shares
of rel^2, overall and per leaf, the way `of3t-trunk043ref` reported 74.88 % magnitude / 25.12 %
direction for the forward.

Beside it, the scale-invariant reading, because the decomposition above is not by itself a
verdict on "is it a scale error": the smallest rel_l2 any rescaling of our gradient could reach
is sqrt(1 - c^2), attained at r = c. If that number is near 1.0, no constant rescaling helps and
the disagreement is a direction disagreement whatever the magnitude share says.

## I5 — CONTROL, re-run in this row`s own process

Carried numbers are not evidence. All four are re-measured here over this row`s own tensor set:
the A16 zero model (expected exactly 1.0, measured anyway), the instrument floor (the reference
re-read from disk and scored on itself, expected 0.0), upstream`s own float32+autocast recipe,
and a break control that permutes the captured cotangent over the 56 real token positions,
leaving weights, forward, masks and arithmetic untouched. A comparison that cannot separate the
break control from the real arm is not measuring agreement with anything.

The predecessor`s own headline is re-derived in this row`s process as a fifth control. If it
does not reproduce 5.367727e+00, that is the finding and the rest of this row is void.

## I6 — FIX_OR_NOT. Three candidates, to be TESTED and not assumed

- **D28, the mask-clean backward.** The campaign`s ledger carries D28 as MEASURED and UNFIXED
  with the note "the backward has never been checked for mask-cleanliness", and its own stated
  cheapest discriminator has never been run. This boundary is crop-64, so it is exposed: 8 pad
  rows in 64, and a LayerNorm WEIGHT gradient sums over every position including the padded
  ones. A mask-clean forward does not imply a mask-clean backward. The discriminator: poison the
  pad rows of the captured input with NaN and count how many parameter gradients come back NaN,
  on OUR side and on the reference`s side recomputed on the same poisoned input. NaN on our side
  and none on theirs says the pad reaches the weights through the backward and every crop-64
  gradient figure in this campaign inherits it. This is registered as the leading candidate
  because the predecessor`s error mass sits 99.42 % in LayerNorm weight and bias gradients,
  which is the signature a pad leak would produce.
- **`tape-recomputes-what-a-fused-kernel-computed`.** The taped forward and the shipped untaped
  forward are compared in-process. If the tape`s own forward output differs from the shipped
  one by more than the shipped forward`s own error against the reference, the tape is
  differentiating a different function and its gradient is that function`s gradient.
- **A missing backward returning something plausible.** `weight_coverage` gives registered and
  with_grad against the total, and the unreached set is reported with its share of the reference
  mass (A20), so a leaf whose backward never ran cannot hide inside a passing denominator.

## What this row will NOT do

It will not flip D1: the predecessor measured that pre-scaling degrades the gradient 1.65x while
improving the forward 6.30x, and the convention is held for Angstrom reasons anyway. It will not
treat an arm that shrinks the gradient by doing less of the model`s work as an improvement. It
will not move a shipped default, and `compose_verify.sh`s assertion that `scale_pair_bias=False`
stays true. It will not widen a bar: the two above were fixed before any number existed and a
miss is the finding.
