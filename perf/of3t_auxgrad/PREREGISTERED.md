# of3t-auxgrad — pre-registration, committed before the first number exists

Two deliverables. Every branch below is named here so that whichever one the measurement lands
on, it was a branch and not a story assembled afterwards.

Base: `wk/of3t-auxgrad` cut from `origin/wk/of3t-auxfind` at `2796f6e01`, which carries the mask
fix merged with `wk/of3t` at `8dfb22045` and re-verified by running it.

---

## Deliverable 1 — re-take the `aux_heads` gradient with the mask fix ON

The 2.2996e-03 mass-weighted headline over 2.843136 % of the model's squared gradient norm was
measured by `of3t-auxheads` on the taped `forward_device` path **before the fix existed**.
`forward_device` passed no masks to the confidence Pairformer, so the forward it was taken at is
the unmasked function `of3t-auxfind` showed is wrong. A18's first clause says a disagreeing
forward invalidates the gradient; the addendum (D9) says an agreeing one merely stops
invalidating it. The scope is UNMEASURED, not void and not passing.

**Arms.** Same boundary (`of3t_auxheads/cap043/boundary_aux_heads.pt`, `batch_step003`, 5nw3,
crop 384, 56 real tokens), same float64 0.4.3 reference (`bundle_min_043/grads_f64_043.pt`),
same `grad_device.tape_parameters` bijection, same cotangents. Two arms differing in one
argument pair:

* **arm N** — `forward_device(..., pair_mask_d=None, attn_mask_d=None)`. Reproduces
  `of3t-auxheads`' reading in THIS process. If it does not reproduce, nothing below is
  comparable and that is the finding.
* **arm M** — `forward_device(..., pair_mask_d=token_mask outer token_mask,
  attn_mask_d=(1-token_mask) * -1e9)`, built from the boundary's own `batch["token_mask"]`.

**Bars, fixed here.** Per-tensor 5.0e-02. Mass-weighted 2.0e-02. A26's reachable bar
`sqrt(2) x 5.0e-02 = 7.071068e-02` per-tensor and `sqrt(2) x 2.0e-02 = 2.828427e-02`
mass-weighted, quoted beside them. Caveat stated in advance: A26's sqrt(2) is the geometry of
subtracting two independent error vectors of equal size, and it applies when the reference is
itself imprecise. This reference is float64, so the "equals the ideal" bar is the one that
binds; the sqrt(2) bar is reported because the brief asks for it and because `of3t-auxfind`
scored the forward against it.

**Branches.**

* **G1 — the fix moves the headline.** The mass-weighted figure changes by more than the
  round-trip noise of arm N against the published 2.2996e-03.
* **G2 — the fix does NOT move the headline, and that is predicted, not a failure.** A23
  measured `aux_heads.distogram.linear.weight` at **100.0000 %** of this section's mass. That
  head reads `zij_trunk` directly, before the confidence Pairformer runs: its parameter
  gradient is the cotangent contracted with `zij_trunk`, which contains no mask-dependent term.
  So the prediction registered here is that the mass-weighted headline is **unchanged to within
  bf16 run-to-run noise**, and the mask cannot reach it. Reading that lands here: the gradient
  headline was never the thing the defect broke.
* **G3 — the fix collapses the massless population.** 171 of 176 tensors over the per-tensor
  bar and a worst tensor of 3.7941e+00 at
  `aux_heads.pairformer_embedding.pairformer_stack.blocks.1.attn_pair_bias.mha.linear_g.weight`
  are all Pairformer tensors, which is exactly where the mask lives. G3 predicts the over-bar
  count and the worst value both drop sharply.
* **G4 — the fix makes some tensor worse.** Possible and must be published if it happens.

**What would mean "the fix does not help the gradient".** G2 holding AND G3 refuted: the
over-bar count stays near 171 of 176 and the worst tensor stays near 3.79e+00 with the masks on.
That is a real possible outcome — the gradient error may sit somewhere the mask never touches
(the host/device dtype policy, the tape's own recompute, the atom-head LayerNorm) — and it would
say the mask fix is a forward-only repair at this scope. It would be reported as the result, not
worked around.

**A18 in-process, not carried in.** The same process that takes the gradient also runs the
shipped `OF3ConfidenceHead.forward` on the same boundary with the masks on and must reproduce
`of3t-auxfind`'s **3.865648e-03** worst head. Separately, `forward_device`'s OWN forward is
scored in both arms, because that is the function the gradient is actually taken at and it is a
different function from the shipped host-path `forward` (device s-path vs host s-path). Both are
reported. Carrying a number in across processes is what made this re-take necessary.

**Reporting rules.** Norm ratio and error cosine beside every relative L2, over the concatenated
mass-weighted set and on the worst tensor. Worst tensor LOCATED by full parameter name, per
`worst-tensor-names-the-tail-not-the-locus`; medians by leaf op beside it. Every set statistic
carries the fraction of reference squared norm it holds (A23). A27: every arm's dtype policy
named.

**Controls, registered before the numbers.**

1. **A16 zero-model, measured** — replace every device gradient by zeros and read the
   mass-weighted headline. Expected exactly 1.0; if it is not, the comparison is not reading our
   side.
2. **Break control** — `--scramble-cot`: the same cotangent numbers written into the wrong
   positions, norm and shape preserved. A comparison that cannot tell this from the real run is
   not reading the seed. Registered to move on arm M.
3. **A/A on the shipped default** — `forward_device` with both masks unset against the same
   method at `origin/wk/of3t`, same card, same boundary. Must be bit-identical on all 7 output
   tensors. If a shipped default moves, the work stops.
4. **Same-function control** — the taped `forward_device` re-run is compared against the
   boundary's own captured reference outputs so a broken load shows up as a broken control
   rather than as a result.

---

## Deliverable 2 — does the fix reach users, MEASURED on a real target

`of3t-auxfind` read `tt_bio/_vendor/openfold3/core/data/pipelines/featurization/structure.py:114`
(`features["token_mask"] = torch.ones(len(token_starts))`) and concluded the shipped fold's token
axis carries no padding. That is a source reading. D90 is the campaign escalating a product claim
from a source reading and having to withdraw it.

**Measurement.** Run a real target end to end through the shipped `tt_bio` OpenFold3 fold, twice:
once on the shipped default (`_confidence` passing no `token_mask`, i.e. today's behaviour) and
once with the mask threaded through from the fold's own `token_mask`. Instrument the
`_confidence` call site to record `token_mask.numel()`, `token_mask.min()`, the count of zero
entries and the padding fraction, plus the token count the confidence head actually receives.

**Branches.**

* **E1 — no padding on a shipped path.** `token_mask` is all ones at the `_confidence` call, the
  padding fraction is 0.0000 %, and the two arms agree exactly on pLDDT, PAE, PTM, IPTM and the
  ranking score. This is the reassuring branch and it is reported as prominently as E2: the
  defect is training-path only and no user's fold moves.
* **E2 — padding present.** Then the deltas on pLDDT, PAE, PTM/IPTM and the ranking score are
  quoted against the **same-arm different-seed scatter** (`conf-scatter-bar`), never against
  zero, and the row escalates it as a product issue.
* **E3 — a padded axis reaches the confidence head from somewhere other than `token_mask`.**
  `openfold3_fold.py:376` slices both trunk outputs back to `n_token` before `_confidence`, so
  bucketing should not reach it. Measured rather than assumed: the token count at the confidence
  call is asserted against the real token count.

A multi-chain target is run as well as a single chain, because a complex is where a token axis
would most plausibly carry an unused slot, and because PTM/IPTM only mean something on one.

**Seed scatter.** Two runs of the SAME arm with different diffusion seeds give the scatter that
every confidence delta is scored against. Registered before the deltas exist.

**What E1 costs.** If E1 holds, the two arms are identical by construction and the deltas are
exactly zero, which on its own proves nothing about the instrument. So the padding statistic is
the primary reading in that branch, and a negative control is run: force one token's mask to
zero and confirm the two arms then differ, so "exactly zero" is a fact about the input and not a
dead code path.

---

## Rules carried

Explicit `git add <paths>` only. The fix is release-gated and stays unmerged on
`wk/of3t-auxgrad`. Its A/A control (default path bit-identical on 7 of 7 tensors) is preserved;
if any work here moves a shipped default, it stops. No timing is claimed in this row, so no clock
is quoted.
