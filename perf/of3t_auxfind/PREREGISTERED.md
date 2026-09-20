# of3t-auxfind — pre-registration, committed BEFORE any number of this row exists

Row brief: `aux_heads` fails A18 on four of five heads. `distogram_logits` passes at 3.479065e-03
(real tokens) while `pde` 3.000855e-01, `pae` 5.177082e-01, `experimentally_resolved` 3.331495e-01
and `plddt` 3.651515e-01 fail by 6x to 10x. 2.8431 % of the model's squared gradient norm is void
behind that. Two explanations are already refuted: the reference revision (`of3t-auxheads043`,
NO-GO, the reference was already upstream 0.4.3 and the rebuild reproduces it bit for bit) and
padding dilution (`of3t-maskaudit`, GO, on real tokens the failures get worse).

Three magnitude claims in this campaign were withdrawn for being published without a
pre-registration. This is that pre-registration. Nothing below was written after seeing a result.

## The contrast, read from the source before any run

`head_modules.py:AuxiliaryHeadsAllAtom.forward` computes `distogram_logits = self.distogram(z=zij)`
from the **trunk** pair, **before** the confidence Pairformer runs and with **no LayerNorm**. The
other four heads all read the confidence Pairformer's *output*: `pae`/`pde` read `zij` after
`pairformer_embedding`, `plddt`/`experimentally_resolved` read `si` after it, and all four apply a
`LayerNorm` to that input before their linear. Our port
(`tt_bio/openfold3_confidence.py:_atom_head`, `forward`) has the same wiring: `distogram_logits =
F.linear(zij_trunk, ...)`, everything else off `zij_conf` / `s_single`.

So the four failing heads share exactly two properties `distogram` does not have:

* **(C1)** they depend on the confidence `pairformer_stack` (and on `embed_zij` upstream of it);
* **(C2)** they LayerNorm their input, which strips the dominant residual and turns a small
  relative error on a large-residual tensor into a large one on the logits.

C1 and C2 are confounded by construction. This row separates them.

## The two candidate causes inside C1, neither assumed

* **(K) the mask.** Upstream passes `single_mask=repr_x_mask` and `pair_mask=token_mask outer` into
  `PairformerEmbedding.forward`; our `OF3ConfidenceHead.forward` passes **none**. The crop is 384
  tokens carrying 56 real ones, so 328 padded tokens enter every k-axis reduction in
  TriangleMultiplication, every softmax key axis in TriangleAttention and AttentionPairBias, and
  every transition. If the padded region of `zij_trunk` is not small, the real block is polluted by
  a term upstream never computes. This is a WRONG transform, not an imprecise one.
* **(Q) the arithmetic.** Our z-path runs bf16 on device; the s-path runs host fp32 fed the
  per-block device z. `of3t-auxheads`' own bisect shows the padded-arm disagreement growing
  monotonically 1.7169e-03 (embed_z) -> 1.6908e-02 (block0 z) -> 4.4821e-02 (block3 z), which is
  consistent with either cause and distinguishes neither.

## Arms

**Arm R (reference).** Upstream 0.4.3's `PairformerEmbedding.pairformer_emb` + the five heads, in
float64 on CPU, on the captured boundary `of3t_auxheads/cap043/boundary_aux_heads.pt`, with the
masks upstream actually passes. A27 dtype policy: every parameter and every activation float64, no
cast on the path, `of3-p2-155k` upcast once at load; `torch.amp.autocast("cuda", ...)` is inert on
CPU and torch says so, which is stated here as a limit of the arm before it runs.

**Arm U (unmasked structure).** The identical module, identical inputs, identical float64 policy,
with `single_mask` and `pair_mask` forced to **all ones** — the function our port computes, in exact
arithmetic. R vs U is the mask's whole effect with precision removed.

**Arm P (our port).** The device figures already published by `of3t-auxheads` at this boundary.

**Arm E (host emulation of our dtype policy).** Our port's structure — unmasked, z-path in bf16,
s-path fp32 — written out on host, scored against R. E vs U sizes the arithmetic on top of the
structure, and P vs E says whether the device adds anything beyond the policy.

Every forward figure carries the padding fraction and the scoring scope (A18 as amended at pass
176), the norm ratio `r = ||ours||/||ref||` and the error cosine beside the relative L2 (a relative
L2 alone cannot identify a direction), and per A27 the dtype policy of any arm used as a
denominator. Bars: **5.0e-02 per-tensor, 2.0e-02 mass-weighted**; per A26 the reachable bar for an
independent reimplementation is **sqrt(2) x threshold**. Bit-exactness is not required.

## Pre-registered branches

**On the mask (R vs U, real-token block, worst of the four failing heads):**

* **M1** — >= 5.0e-02 and within 3x of the port's own gap: the missing mask IS the defect. Fix it in
  `OF3ConfidenceHead` and re-measure A18 on device.
* **M2** — >= 5.0e-02 but more than 3x away from the port's gap: the mask is a real defect and an
  incomplete explanation; fix it and report what survives.
* **M3** — in (1e-10, 5.0e-02): the mask contributes below the bar. It is still a defect of the
  port but it is not why A18 fails, and Q owns the rest.
* **M4** — <= 1e-10: masking is inert at this boundary. Our unmasked confidence Pairformer computes
  upstream's function in exact arithmetic and the mask hypothesis is refuted.

**On the arithmetic (E vs R, after the mask question is settled):**

* **Q1** — arm E reproduces arm P to within 2x on all five heads: the device adds nothing to the
  dtype policy, and **the heads are simply imprecise** — there is no further defect to find. This
  is a legitimate outcome, it is named here before the run, and it will be reported in exactly
  those words if that is where it lands.
* **Q2** — P is more than 2x worse than E: there is a second defect in the device path, and the
  per-leaf median below has to locate it.

**On the locus (LOCUS, median per leaf op, never the worst tensor):**

The leaves of the confidence Pairformer block are `tri_mul_out`, `tri_mul_in`, `tri_att_start`,
`tri_att_end`, `transition_z`, `attn_pair_bias`, `single_transition`. Each is measured two ways:
**created** (feed the leaf the reference arm's own captured input, run it in the other arm's
configuration, score only that leaf's output) and **propagated** (the two full runs compared at
that leaf's output). Created is the locus measurement; propagated is reported beside it.

* **L1** — one leaf's created median is >= 3x the next: the disagreement is made in one op.
* **L2** — all leaf created medians within 2x of each other: the disagreement is diffuse and no
  single op owns it; report the block-level accumulation instead.
* **L3** — the created medians are all at float64 round-off while the propagated ones grow: nothing
  is created inside the stack and the disagreement enters at `embed_zij` or at the boundary inputs.

**On the amplification (C2, LayerNorm):**

* **A1** — head rel / input rel >= 3x on the failing heads: the A18 verdict on those heads is
  largely LayerNorm amplifying a smaller error on the Pairformer output, and the Pairformer output
  itself must be quoted beside the head.
* **A2** — < 3x: the head gap is the Pairformer gap and no amplification story is needed.

## Controls, fixed in advance

1. **A16 zero-model baseline**, measured not assumed: replace our arm's output by zeros and record
   the relative L2 it reads. Every figure is quoted against it.
2. **Same-function control**: arm R re-run in a fresh process must reproduce the captured reference
   outputs at **exactly 0.0** on all five heads. If it does not, every arm below it is void.
3. **A control that breaks what the check reads** (`negative-control-must-break-what-check-reads`):
   the leaf instrument is run with one leaf's mask deliberately broken, and it must move at that
   leaf and stay at round-off at the others. A control that cannot break the instrument is not
   evidence.
4. The unmasked arm is built by forcing the mask tensors, never by deleting the mask argument, so
   the two arms differ in one value and not in one code path.

## What this row will NOT do

It will not merge anything. A mask fix inside `OF3ConfidenceHead` changes the confidence head's
output for every OF3 fold that runs on a padded crop, which is release-gated; it stays on
`wk/of3t-auxfind`, flagged, whatever the numbers say.
