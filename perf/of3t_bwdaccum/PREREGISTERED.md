# of3t-bwdaccum — pre-registration, written before any number exists

`of3t-trunkg043` read the trunk's gradient at a forward that passes A18 and found
**9.025172e+00** mass-weighted against upstream 0.4.3 float64, over 2,736 of 2,736 tensors
holding exactly 1.0 of the stack's squared gradient norm, **22.52x** upstream's own bf16 arm on
the same boundary (4.007237e-01) and 9.03x worse than emitting zeros (A16 baseline exactly 1.0).
The per-block profile grows monotonically with distance travelled down the backward: 0.94x to
3.81x the floor over blocks 44-47 where the backward starts, 4.22x to 66.88x over blocks 0-11
where it ends. 92.68 % of the error mass sits on four single-track LayerNorm affine leaves
holding 2.121 % of the gradient mass. Padding is refuted; the disagreement is arithmetic.

This row asks which of two mechanisms that is, and it fixes the discriminator before running it.

## The boundary and the arms, inherited unchanged

Nothing here is rebuilt. `/home/ttuser/of3t_trunk043ref/boundary_c64.pt` sha256
`85015b4a2622c1d3a6ae16f49c45cafd6f575dc0dab144fe08ffa87b83498ffa`, 56 real tokens of 64; the
capture's own block-47 cotangent from `/home/ttuser/of3t_gradients/cap/block47_boundary.pt`
sha256 `a55ef1c4e6c90c87984242f9d1cc8523fbd9470a5f592eac0d0aff7000d29cf5`. The reading arm is
FLIPPED (`scale_pair_bias=True, tri_att_scale_pair_bias=False`), the only arm whose forward
clears A18 on both tracks. REF-F64 and REF-BF16 are `of3t-trunkg043`'s `ref_grad.py` arms, built
by the same script with the same dtype policy, and every ratio names which one it is against
(A27).

## Deliverable 1 — the discriminator, and what each answer means

**Score the cotangent ENTERING each block against the reference's**, per track, at every one of
the 49 block boundaries. The quantity is `ds_k = dL/d(s_in of block k)` and `dz_k` likewise, ours
against REF-F64, with REF-BF16's own cotangent at the same rung as the floor, and with
`norm_ratio` and `cos` beside every `rel_l2` (D35: rel alone cannot tell 18x too small from 2x
too big).

Fixed before the numbers:

- **cotangent at the bf16 floor at every rung** (ours/floor <= 2.5x at all 49 rungs) →
  mechanism **(B)**: a correct cotangent with a wrong LayerNorm-affine backward. The four leaves
  are the whole story and the target is one backward formula.
- **cotangent degrading monotonically with depth** (ours/floor rising from rung 47 to rung 0 with
  no interior maximum, end-to-end growth >= 4x) → mechanism **(A)**: the error is injected per
  block and carried. The leaf gradients are a symptom and the row must name what injects it.
- **neither clean** — in particular flat early and degrading late → a THIRD shape, a
  regime-dependent injection, which is what `of3t-trunkcliff` found in the forward. Say so
  plainly rather than forcing it into (A) or (B).

This is a discriminator, not a bar: all three answers are results and the row reports whichever
fires.

## Deliverable 2 — the LayerNorm affine backward in isolation, against float64

`dW = sum_t(g_t * xhat_t)` and `db = sum_t g_t` over the leading axes. Both are reductions over a
long axis and the campaign already suspects two things here:

- **D55/D56**: `autograd.py:683` and `:1611` form the summands with a bare
  `ttnn.multiply(g, norm)` — no kernel config and no dtype, so 64 x 384 summands are built and
  rounded to bf16 and only THEN summed precisely. Summing bf16 numbers in fp32 does not recover
  the bits lost making them. **Untested at this scope.**
- **D111's shape**: `_sum_leading` passes `precise_config()` but **no output dtype**, and
  `_taped_linear`'s own dW rule documents why that is not enough (`packer_l1_acc` accumulates
  per-K-block partials at the OUTPUT dtype) and passes `dtype=ttnn.float32`. `_sum_leading` does
  not. Both the gain and the bias gradient go through it.

The test captures the REAL operands (`x`, `g`, `gamma`) in situ at the four leaves at several
depths, then replays the op on device against a float64 reference computed from **the same
operands**, so an op-level defect is separated from the chain-level one. Beside every reading,
the site's **cancellation factor** `K = sum_t |term_t| / |sum_t term_t|`, which is what predicts
how far a bf16 summand can throw the sum.

Pre-registered:

- isolated `op_rel` at or below bf16's own unit roundoff times sqrt(K) → the op is as good as
  bf16 allows and mechanism (B) is refuted at the op;
- isolated `op_rel` above that, and a lever that removes it → the lever is the fix and it must
  then be measured at scope;
- a lever that fixes the op and does NOT move the scope reading → the op was never the scope's
  defect, and the row says so.

Controls, fixed now: a deliberately-worse arm (LoFi, `fp32_dest_acc_en` off) must make the same
reading WORSE, or no flag reached the kernel; and a float64 host recomputation of the same
formula from the same operands must reproduce the reference to roundoff.

## Deliverable 3 — a fix, or the arithmetic that says there is not one

Any lever that survives deliverable 2 is measured **at scope**, over the same 2,736 tensors, with
the shipped control reproducing **9.025172e+00** in the same process beside it. A lever measured
only on one op is not measured (`a-lever-can-fire-and-be-inert`).

- lever takes the scope inside **2.0e-02** mass-weighted → the trunk is REPRODUCED and this is
  the campaign's closing result;
- lever moves the scope but not inside the bar → report both the new reading and its multiple of
  upstream's own bf16 floor, 4.007237e-01, which is the denominator that decides whether a bf16
  port can get there at all;
- no lever moves it → the arithmetic that says so, with the isolated op readings as the evidence.

## What this row will not do

No shipped default moves. `compose_verify.sh` asserts `tt_bio/openfold3_trunk.py` reads
`scale_pair_bias=False, tri_att_scale_pair_bias=False` and must still pass against this branch.
Every artifact goes under `perf/of3t_bwdaccum/`. Any precision change to `tt_bio/autograd.py` is
release-gated: it changes training behaviour for every model that tapes, so it stays on this
branch and is flagged, never merged.
