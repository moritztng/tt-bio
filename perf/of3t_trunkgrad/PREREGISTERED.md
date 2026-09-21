# of3t-trunkgrad — pre-registration

Committed before this row's first number exists. Every branch, bar and control is named here so
none of them can be chosen after the reading is known.

Base: `wk/of3t-trunkgrad` cut from `origin/wk/of3t-trunkcliff` at `f4933feef`, which carries the
0.4.3 reference build, the per-block arms and the pre-scaled-bias arm.

## What is inherited and not re-derived

- `of3t-trunk043ref`: rebuilt at upstream 0.4.3, the shipped trunk's PAIR track reads
  4.947045e-02 masked at crop 64 against the 5.0e-02 bar. 92.08 % of the old 2.793661e-01 was
  the 0.5.0 reference. 0.5.0 forced to the 0.4.3 orientation reads exactly 0.0.
- `of3t-trunkcliff`: the remaining SINGLE-track error is a convention, not bf16.
  `AttentionPairBias` folds the pair bias inside its own score scale, so the reference's
  unscaled add means pre-baked by `sqrt(head_dim)` here. Pre-scaled reproduces upstream 0.4.3
  float64 to 4.2e-16; shipped is 78.0 % wrong on the block-45 update against upstream's own
  9.1e-03 bf16 floor.

      arm              crop64 s      crop64 z
      SHIPPED          1.065338e-01  4.947045e-02
      bias pre-scaled  1.655263e-02  4.947045e-02

  The pair track is bit-identical either way.
- D1 is HELD. `of3t-confhead` measured the flip at +0.050 A of best-of-5 and -0.463 A on the
  structure a user receives. `compose_verify.sh` asserts `scale_pair_bias=False`.
  **This row changes no default and proposes no flip.**

## The question

What does the held default cost in GRADIENT terms? The trunk holds 5.8282 % of the model's
squared gradient norm and its gradient has never been scored against a reference built from the
revision this checkpoint is bound to. The 1.061553e+01 on record was scored against a 0.5.0
reference and is superseded; it is not carried in as a number, only as a target the instrument
control must reproduce.

## The boundary

`of3t_gradients/cap`, block 0's captured input in and block 47's captured output cotangent back,
cropped to the first 64 token positions. 56 of the 64 are real: padding fraction 0.125 on the
single axis and 0.234375 on the pair axis. The same slicing `extract_boundary.py` wrote into
`of3t_trunk043ref/boundary_c64.pt`, and the reference arm asserts bit-identity against that file
rather than re-deriving the convention.

## Arms

Reference side, CPU, one upstream tree per process (the two release trees define the same module
names and cannot share an interpreter). A27, every denominator names a POLICY and not a width:

| arm | tree | dtype policy |
|---|---|---|
| `REF043_F64` | 0.4.3 | every parameter and every activation float64, checkpoint upcast once at load, no cast on the path. **The reference.** |
| `REF043_BF16` | 0.4.3 | float32 parameters under `torch.autocast('cpu', bfloat16)` — upstream's own training recipe. **The floor.** |
| `REF050_F64` | 0.5.0 | as `REF043_F64`. Reference-side instrument control. |

Device side, card 0, crop 64, 48 blocks as one taped stack, `instrument_a_bundle.py`:

| arm | `--scale-pair-bias` | what it is |
|---|---|---|
| `DEV_SHIPPED` | `off` | the shipped default, `scale_pair_bias=False`. Never measured in the gradient. |
| `DEV_SCALED` | `on` | the pre-scaled convention. Also reproduces `of3t-pairformer`'s configuration. |
| `DEV_BREAK` | `off` + `--permute-cot` | break control. |

**`of3t-pairformer`'s device arm was NOT the shipped convention.** Its report records
`shipped_config.scale_pair_bias: true` with `scale_pair_bias_arm: "shipped"`, because
`instrument_a_bundle.py` hardcodes `SHIPPED_SPB = True` while `openfold3_trunk.py:166` passes
`False`. So `DEV_SCALED` is the arm that reproduces it, and the shipped convention has no
gradient reading at all today. Registered here before the run, not discovered afterwards.

## Bars, fixed here

- per-tensor **5.0e-02**, mass-weighted **2.0e-02**.
- **A26-SCOPE: `sqrt(2)` does NOT apply.** The reference is float64, so only one side carries
  error and the reachable bar is the threshold itself. The `sqrt(2)` figures are not quoted as a
  bar and may not rescue a tensor.
- The floor `REF043_BF16` vs `REF043_F64` is measured on this same scope and reported beside
  every device figure, because a device arm cannot be asked to beat upstream's own recipe.

## Reporting rules

- **A23, by MASS.** The headline is rel_l2 over the CONCATENATION. Beside it: how much of the
  compared squared reference mass sits OUTSIDE the per-tensor bar, and how many tensors do. When
  the count and the mass disagree the mass decides.
- **The numerator is decomposed too.** A tensor whose reference gradient is near zero and whose
  device gradient is not contributes unbounded `diff_sq` and no `ref_sq`, so it can carry a
  mass-weighted headline on its own. The top contributors to the ERROR mass are named, and the
  headline is reported both over all scored tensors and over the A14-surviving population
  (reference norm >= 1e-8 x the population's median reference norm).
- Norm ratio and error cosine beside every relative L2, on the concatenated set and on the worst
  tensor. Worst tensor located by full parameter name; medians BY LEAF OP beside it, because the
  worst tensor names the tail and not the locus.
- Per-block split with each block's mass share.
- **Reach is over the FULL reference mass** (A20): an unplaceable tensor is UNREACHED, not
  absent from the denominator.

## The reach asymmetry the lever creates, registered in advance

`scale_pair_bias=True` folds `sqrt(head_dim)` into the device's `attn_pair_bias.linear_z.weight`,
so the by-value bijection stops recognising it and that tensor goes UNREACHED in `DEV_SCALED` and
is reached in `DEV_SHIPPED`. The lever changes what the instrument can see. Therefore:

1. The arm-vs-arm comparison is scored over the INTERSECTION of the two reached sets, so the two
   headlines share a denominator.
2. Each arm's own reach and its unreached mass are reported by name.
3. A secondary, separately labelled reading recovers the scaled arm's `linear_z.weight` by
   matching the bijection against the pre-scaled weight and returning the gradient to checkpoint
   coordinates (`dL/dw = sqrt(head_dim) * dL/dw_device`). It is the tensor the convention acts
   on, so dropping it would answer the question by assumption. The uncorrected norm ratio is
   printed beside the corrected one so the factor is visible rather than asserted.

## Branches, named before the run

**T1 — the trunk's gradient CLEARS the mass-weighted bar under the pre-scaled convention.**
`DEV_SCALED` vs `REF043_F64` <= 2.0e-02. The forward explanation is complete and the gradient
cost of the held default is exactly the gap to `DEV_SHIPPED`.

**T2 — pre-scaled misses the raw bar but is inside upstream's own bf16 floor.** Between 2.0e-02
and the measured `REF043_BF16` distance. The residual is what bf16 costs at this depth, not ours.

**T3 — the trunk's gradient FAILS under BOTH conventions.** Above both the bar and the floor
under `DEV_SCALED`. Named here as a real outcome: the forward explanation would be INCOMPLETE,
something reaches the gradient that an agreeing forward cannot see. It would be reported as the
result, and the A18 addendum (an agreeing forward is necessary, never sufficient) is why this is
not a contradiction of `of3t-trunkcliff`.

**T4 — the shipped convention costs the gradient by an order of magnitude.** `DEV_SHIPPED`
reads in the 1e+00–1e+01 class against the 0.4.3 reference. Then the held D1 default has a
gradient cost the Angstrom evidence cannot see, and that is the missing half of the D1 decision.

**T5 — the two conventions are indistinguishable in the gradient.** Predicted in advance as
plausible: the pair track is bit-identical either way, so if the pair track holds the bulk of the
trunk's gradient mass the single-track convention cannot move the headline. Reading that lands
here means D1 costs the gradient nothing measurable and the decision stands on the Angstrom
evidence alone. The discriminator is the per-leaf split: `attn_pair_bias` mass share against the
rest.

**C1 — the device-side instrument control fails.** `DEV_SCALED` scored against the pinned 0.5.0
references must reproduce `of3t-pairformer`'s published 1.047662e+01 (vs float64) and
1.061553e+01 (vs upstream bf16). If it does not reproduce to the digits published, this row's
numbers are not comparable to the record and that is the finding, not a footnote.

**C2 — the reference-side instrument control fails.** `REF050_F64` built here must reproduce
`/home/ttuser/of3t_pairformer/upstream_f64_crop64.pt` bit for bit. A cross-process reproduction
is what made `of3t-auxgrad`'s number trustworthy.

## Controls, all measured before any headline

1. **A16 zero model, measured.** Every device gradient replaced by zeros, scored against
   `REF043_F64`. Expected exactly 1.0 on the mass-weighted headline; if it is not, the
   comparison is not reading our side.
2. **Instrument floor.** `REF043_F64` scored against itself: exactly 0.0, and the same file
   loaded twice from disk, so the scorer's own arithmetic contributes nothing.
3. **Break control.** `DEV_BREAK`: the captured cotangent permuted across the 56 real token
   positions, weights, masks, flags and kernels untouched. The headline must move by orders of
   magnitude. A comparison that cannot tell this apart from the real run is not reading the seed.
4. **Boundary identity.** The sliced crop-64 boundary the reference arm builds must be bit-identical
   to `of3t_trunk043ref/boundary_c64.pt`, whose sha256 is recorded.
5. **A18 in the same process as the gradient.** The device forward is scored against
   `REF043_F64`'s own forward outputs inside the gradient process, both tracks, masked, with the
   padding fraction beside each figure. It must reproduce `of3t-trunkcliff`'s 1.065338e-01 /
   4.947045e-02 (shipped) and 1.655263e-02 / 4.947045e-02 (pre-scaled). A disagreeing forward
   invalidates the gradient taken at it; an agreeing one removes mis-wiring from the list and
   bounds nothing.
6. **A24 same function.** The loss at the boundary and the gradient global norm are recorded for
   every reference arm, and which block each end of the boundary came from.

## Rules carried

- No shipped default moves. `scale_pair_bias=False` and `tri_att_scale_pair_bias=False` stay as
  they are in the tree; the conventions live in harness arguments.
- Explicit `git add <paths>`. Push `wk/of3t-trunkgrad`, verified with `git ls-remote`. No merge
  to main.
- Bit-exactness is not required; accuracy is. No timing is claimed in this row, so no clock is
  quoted — the deliverable is arithmetic, and a clamped clock changes how long it takes, not what
  it computes.
