# of3t-twoside preregistration, D241

Written and committed before the first arm was run. Host qb1, 2026-09-22.

The frame `of3t-modelframe` built is one-sided. Our trunk is handed the reference's float64
boundary and its float64 incoming cotangent (`runarm.sh:24-25,67`). The bf16 denominator the
clause divides by is `of3t_refprec/pinned_p175/arm4_bf16_autocast/grads_f64.pt`, a full-model
bf16 autocast run whose trunk was driven by its own bf16 boundary and its own bf16 cotangent
(`score.sh:49`). The two sides are different experiments. This row puts them on the same
boundary.

## Arms

| tag | policy | boundary | cotangent | blocks | crop | what it is |
|---|---|---|---|---|---|---|
| CTRL_F64 | f64 | boundary_model_n384.pt | cot_model_n384.pt | 48 | 384 | the gating control |
| INJ_BF16 | bf16auto | boundary_model_n384.pt | cot_model_n384.pt | 48 | 384 | the missing arm |

Producer is `perf/of3t_trunkg043/ref_grad.py` unchanged, no fork, no new argument. Tree
`/home/ttuser/of3t_frame384/of3pkg043`, weights `of3-p2-155k.pt`, `--checkpoint` (the c64
bit-identity control in `of3t-frame384` says the flag is inert). CPU only, no card is opened.

Inputs, digests verified on qb1 before the run:

- boundary 583bcd7c91ce6ed46aea59844954fb2bf7ddc3d553d7f9d4f238998183db99e2
- cotangent 4e66d1ef45da2eec18fdff9489d68e980df928dc43d10fd4af82d14ec67141ac
- float64 model reference grads_f64_043.pt 1d4ea9225f854afe06a8adedcb5f8aa1c53656fbf8cefdaa3a451d0327895cc4

## Step 1 bar: does the injection harness reproduce the run the pair was captured from?

The boundary and the cotangent were both taken from `grads_f64_043.pt`'s own full-model float64
backward, at num_recycles 0 where the stack runs once. So an injected float64 trunk differentiates
the same function over the same point and must return the same gradient up to float64 round-off
and a different summation order.

**PASS requires mass-weighted relative L2 of CTRL_F64's `pairformer_stack.*` against
`grads_f64_043.pt`'s own `pairformer_stack.*` <= 1e-12.**

Why 1e-12 and not tighter: it is PROTOCOL A34's exact-arithmetic tolerance, and
`CAPTURE_model_n384.json` records the capture's own full-model backward reproducing the
published reference at 1.6952505222168705e-14 on this same statistic. 1e-12 is two orders looser
than an already-demonstrated agreement, so a failure is a wiring or semantics difference and not
round-off.

Secondary, reported but NOT gating: worst per-tensor relative L2, and worst per-tensor relative
L2 restricted to tensors carrying at least 1e-6 of the trunk's squared gradient mass. The
unrestricted worst is not a gate because the capture's own witness read 3.8612005062108574 on
`pairformer_stack.blocks.17.attn_pair_bias.layer_norm_z.bias` while the mass-weighted number
was 1.7e-14: layer-norm bias gradients are a cancelling sum and their relative error is
meaningless at that norm.

If step 1 FAILS the row stops there and reports it. The frame of3t-modelframe established would
then not be what it claims and every reading on it since 17:26 on 2026-09-22 is in question.

## Step 2/3: what separates the excess is ours from the excess was the asymmetry

Let

- **U_inj** = INJ_BF16's pairformer_stack against float64 (this row's new number)
- **U_own** = 0.3147698293887927, upstream's un-injected bf16 trunk against float64, the number
  the one-sided clause currently divides by (`CLAUSE.json` frame_matched.trunk
  `upstreams_own_floor_here_vs_float64`)
- **O** = 0.9349175217825587, ours against float64 on this frame, banked, not re-run
  (`CLAUSE.json` frame_matched.trunk `reading_vs_float64`)

The one-sided multiple is O / U_own = 2.970162431380236.
The two-sided multiple is M = O / U_inj.

**Separator, fixed now: M = 2.0.**

- **M >= 2.0 — THE EXCESS IS OURS.** Injecting the exact boundary and the exact cotangent into
  upstream's own bf16 recipe does not close most of the gap. Our trunk carries error upstream's
  own arithmetic does not at the same site, and the campaign has a real defect left to fix.
- **M < 2.0 — THE EXCESS WAS THE ASYMMETRY.** A large part of the 2.9702x was the reference being
  driven by its own drifted cotangent rather than the injected one, and the one-sided reading
  flattered nobody.

2.0 is not chosen to be reachable. It is the midpoint in log between the two outcomes the frame
JSON names as the only two: U_inj reproducing its un-injected 0.3148 (M = 2.970) and U_inj
rising to ours near 0.9 (M near 1.04). sqrt(2.970 * 1.04) = 1.758; 2.0 is set deliberately
ABOVE that midpoint, so calling the excess ours takes more evidence than calling it the
asymmetry. A result between 1.758 and 2.0 therefore reads as the asymmetry.

## Step 5: the re-scored clause

Bar 0.15210099830945006, unchanged. The two-sided denominator is
`arm4_bf16_autocast/grads_f64.pt` with its 2,736 `pairformer_stack.*` tensors replaced by
INJ_BF16's, every other section untouched. Nothing else in the composition moves.

The A/A gate comes first: `clause.py`'s recomposition must reproduce the published headline at
`rel_difference` exactly 0.0, and the re-scored artifact's own recomposition check must too,
before any re-scored number is read.

**The one-sided 1.7814428090278143 is reported beside the two-sided number, never instead of
it.** Both are true readings of different questions.

## What this row may not conclude

Not that the trunk passes unless step 5 says so with the A/A at 0.0. Not that the 1.7814x was
wrong; it is a lower bound and it fails. Nothing about a full training run: this is one taped
backward of one step.
