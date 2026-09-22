# of3t-trunkact — pre-registration, committed before the first arm

Row `of3t-trunkact`. Branch `wk/of3t-trunkact`, continuing `wk/of3t-blk4544`.
Namespace: **`perf/of3t_trunkact/`**. The brief assigns `perf/of3t_trunkfwd/`, but that
namespace belongs to the concluded row `of3t-trunkfwd` and holds its record; writing this row's
arms into it would edit a concluded row's evidence. Same namespace discipline, different
directory, and it is named here so the orchestrator can find the files.

## The frame, fixed before any number exists

Everything is measured at **padded N = 384**, which is the frame the graded gradient artifact
runs in: `perf/of3t_modelboundary/DEV_RENORM_n384_nocaptures.json` is `--crop 0` over
`boundary_n384.pt` (sha256 `8cb3a586...`), 56 real tokens of 384. Masked columns are reported
beside every padded one, never instead of it.

Reference: upstream **0.4.3** (`/home/ttuser/of3t_frame384/of3pkg043`), the revision
`of3-p2-155k` is bound to, every parameter and every activation **float64**, checkpoint upcast
once at load, no cast on the path (A27). Floor beside it: upstream's own **bf16 autocast**
(fp32 parameters under `torch.autocast('cpu', bfloat16)`), upstream's own training recipe.
Both walks run in one process over one boundary so the three arms share a function by
construction.

Host: **qb1 (tt-quietbox), p150a Blackhole, card 0**. The in-frame 384 shop
`/home/ttuser/of3t_frame384` was built on this box (D189) and every figure here stays on it.
AICLK sampled every 4 s DURING each device run, min/median/max reported.

## Controls that must pass before a number is read

1. **Reference identity.** The float64 walk's stack output must reproduce
   `REF_F64_N384.json`'s `s_norm` 2547074.6682744888 and `z_norm` 15590710.480237307. A walk
   that does not is a different function and no per-block reading off it is admissible.
2. **Our own identity.** The device walk's stack output must reproduce
   `perf/of3t_trunkcliff/STACK_scaled_n384.json`'s `s_norm` 5255672.101057217 and `z_norm`
   12554717.658698618 — that arm is `scale_pair_bias=True`, which is what
   `tt_bio/openfold3_trunk.py` has shipped since `701ddcf63`, so it is today's shipped
   configuration and not a lever.
3. **Composition is the model's own.** The per-block dump is taken by wrapping
   `ops.checkpoint_segment`, which is the seam `Pairformer.__call__` itself composes through,
   so block *k*'s recorded output is the stack's own state and not a re-composition.
4. **A/A.** The device walk run twice must be bit-identical on both tracks.

## What is already excluded and must not be re-run

* `of3t-blk4544`: every matmul, softmax and `triangle_attention` backward in blocks 45 and 44
  pinned to its float64 VJP, 518 substitutions at padded 384, R44 2.203113 against 2.201450
  with the refutation line at 1.90. The backward does not carry it. `norm_ratio` at rung 44
  moves 1.722041 -> 1.722830 under the saturating pin.
* `of3t-trunkcliff` + `of3t-d1-pairbias`: the single track's blocks-45-47 forward cliff was the
  pair-bias scale convention, and it is **fixed and shipped** (`701ddcf63`,
  `scale_pair_bias=True`). Any hypothesis that re-names it is already spent.

## H1 — the carrier is the PAD region of the single track

`of3t-trunk043ref` published our padded single track at N=384 as **1.360300e+00** relative
while the masked column reads 1.061989e-01, and read the padded column as unusable because
"85 % of those tokens are padding where neither side is defined". That is the sentence this row
tests, because the **gradient is computed padded**: `dW = sum_t g_t * xhat_t` runs over all 384
token positions, so a pad-row activation that disagrees enters every parameter gradient in the
block unless the cotangent is exactly zero there.

The arithmetic that makes this the primary hypothesis: our stack's padded single-track norm is
5255672.101057217 against upstream float64's 2547074.6682744888, a ratio of **2.0634**. The
graded trunk gradient's `mass_weighted_norm_ratio` is **1.9890** with `mass_weighted_cos`
**0.2250972629419331** (`MODEL_withtrunk_n384.json`, `renorm_vs_UPSTREAM_BF16::pairformer_stack`,
2736 tensors, 5.8281714991342914 % of the model). A gradient 1.99x too large sitting under a
forward activation 2.06x too large is a prediction, not an observation, until the per-block
split is measured.

**Prediction.** Of the composed ABSOLUTE single-track error at block 48 at padded 384,
`|| ours - ref ||` restricted to the 328 pad rows is **>= 80 %** of the total squared error.
**Falsifier: < 50 % refutes H1**, and the row then reports the pad share it measured and moves
to the real-token decomposition.

## H2 — the forward does NOT step at 45 -> 44

Registered before looking, and it is the falsifiable direction. `of3t-trunkcliff` measured the
depth ladder with the pair bias pre-scaled as monotone and smooth across that boundary:
1.4048e-02 at k=44, 1.1922e-02 at 45, 1.2533e-02 at 46, 1.4171e-02 at 47, 1.6553e-02 at 48
(masked, crop 64). The step the ladder had there was the convention, and the convention is
fixed. So:

**Prediction.** In the per-block DIFFERENCE of absolute errors at padded 384 over blocks
40..47, the increment at the 44/45 boundary is **< 2x the median increment** of that window on
both tracks. If that holds, the backward's step at 45 -> 44 is a **different object** from the
forward's growth and is created by something local to the backward's reading of these
activations.

**Falsifier.** An increment **>= 2x** the median of the window at 44/45 refutes H2, and
forward and backward are then one phenomenon, located at one depth.

## H3 — the op, and the order candidates are measured in

A candidate is the **CARRIER** only if replacing it with a float64 recomputation removes
**>= 50 %** of the composed absolute single-track error at block 48. It is **REFUTED** at
**< 20 %**. Anything between is reported as partial with its number and not called either.

Order, and each one is a census before it is an arm:

1. **`Transition` on the single track** (`trans_mask_s`, D174). It is the one op whose masking
   was hoisted out of the per-block loop, it fires 48 times, and a mask applied at a different
   place acts on every pad row at every depth. Predicted carrier.
2. **`AttentionPairBias` token attention.** The attention mask masks KEYS; pad QUERY rows still
   receive a value, and that value is unconstrained.
3. **The single-track layer norms.** Bounded per row, so this predicts a SMALL pad effect; it
   is on the list because it is cheap and because H1 being pad-carried does not by itself say
   which op writes the pad rows.
4. **`triangle_multiplication` start/end** (96 calls). Pair track. On the list because the pair
   track's masked reading is 9.60x upstream's own bf16 (4.947045e-02 against 5.155485e-03) even
   after the convention fix, which is the larger RATIO excess of the two tracks.

## H4 — the counterfactual, and what would make it unevaluable

If a carrier is found and a correction exists that is inert on the masked real-token output,
the counterfactual is one device trunk-gradient arm at n384 plus one CPU re-score through
`perf/of3t_wholemodel/model_scope.py` against the same three pinned references
(`1d4ea9225f854afe...`, `ff78d7bc0bf7a435...`, `09f1217c8ea254d0...`). The number wanted is
`per_section.renorm_vs_UPSTREAM_BF16.pairformer_stack.mass_weighted_rel_l2` against upstream's
own 0.3147698293887927, and `stats.renorm_vs_UPSTREAM_BF16.mass_weighted_rel_l2` against the
A26 bar **0.14735268326440318** (today 0.5201243840984896, 3.53x over).

**Stated before the arm:** if the correction changes the masked real-token forward at all, it is
a lever on accuracy and is release-gated, stays on this branch, and the counterfactual is
reported as a bound rather than as a fix. If no carrier clears H3's 50 %, the counterfactual
cannot be evaluated and this row says so with the exclusion table instead of estimating it.

## What this row will not claim

It will not claim a full training-run reproduction. It measures one forward of one captured
boundary of one step, and the campaign's bound stands: the update rule over N steps, never
stability over a full run.
