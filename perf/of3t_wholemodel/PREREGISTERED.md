# of3t-wholemodel — registered before any model-scope number existed

Committed before the first arm was scored. Nothing here may be edited once a number exists;
corrections go in an amendment block at the bottom with the date and what already existed.

## What is being measured

One mass-weighted `rel_l2` over the OpenFold3 0.4.3 parameter set (4,170 tensors, model squared
gradient norm 10.279642678524985), against two references named per A27:

  * **upstream 0.4.3's own bf16 training step** — `of3t-refprec`'s `arm4_bf16_autocast`, built
    with fp32 parameters under `torch.autocast(bfloat16)`, so LayerNorm and softmax are exempt
    from the cast. sha256 `ff78d7bc0bf7a4355014a470fbe931592bd4896fe06a8b400c161c08b5607ccb`.
    This reference carries error, so A26's `sqrt(2)` applies to it.
  * **float64** — `bundle_ref/grads_f64_043.pt`, the campaign's spine. One side carries error,
    so A26 does **not** widen this bar (A26-SCOPE).

Four arms, all release-gated and all default-off:

  | arm | how |
  |---|---|
  | shipped | no flag |
  | renorm | `TT_BIO_SOFTMAX_BW_RENORM=1` (`of3t-apbgrad`'s two-op softmax backward repair) |
  | renorm + host float64 softmax | the above plus `TT_BIO_HOST_F64_SOFTMAX_AB=all` / `--softmax-site-f64` |
  | zero-gradient baseline | A16, measured through the same scorer, not asserted as 1.0 |

## Thresholds, and the attainable-range check D76 says to do first

Against upstream's bf16 step the quantity is `rel(ours, bf16) = ||g_ours - g_bf16|| / ||g_bf16||`.
Write `floor` for upstream's own bf16-vs-float64 distance on the same tensor set and
`r = ||g_bf16|| / ||g_f64||` on that set. Both are measured here, not borrowed: per-section floors
already on the record span 3.1012e-02 to 2.3937e-01, a 7.7x spread, so a single borrowed floor
would be wrong on most of the model.

  * a port that reproduced float64 exactly reads **`floor / r`**
  * two independent implementations of equal accuracy read **`sqrt(2) x floor / r`** (A26)
  * a model emitting zeros reads **1.0** (A16, measured here)
  * unbounded above

Both thresholds are strictly inside that range, so neither branch is vacuous.

## Pre-registered branches

**B1 — the campaign's central claim.** The best arm's model-scope reading is at or below
`sqrt(2) x floor / r` against upstream's own bf16 step. Then our training step's gradient is as
close to the ideal as an independent implementation as good as upstream's own bf16 recipe would
be, at model scope. To be stated exactly that carefully and no more strongly: it is an agreement
statement about one step's gradient, not about a training run.

**B2 — the cheapest possible close.** B1 holds on the **renorm arm alone**, with no host round
trip. Then the round trip is not needed at model scope at all.

**B3 — outside.** Name the scope that carries the residual and its share of the model's squared
gradient norm, and do not average it away. A scope holding 2 % of the mass at 10x the bar and a
scope holding 40 % at 1.1x are different findings and the mass-weighted headline alone does not
separate them.

B1 and B2 are not exclusive: B2 implies B1.

## Reconciliation

Deliverable 2 is a check on the scopes, not a second headline. Two quantities:

  * **assembled** — the union of per-tensor rows, one row per reference tensor, scored in
    float64 in the model bundle's own denominator.
  * **composed** — `sqrt( sum_s share_s * reading_s^2 / sum_s share_s )` from each scope's
    published headline.

Where a scope was measured on the model's own batch these are the same arithmetic and must agree
to **1e-9 relative**; anything larger is a defect in one of the two. Where a scope was measured
on a **different boundary or a different batch** they cannot be made to agree, and that scope is
reported as a separate composed term with its boundary named, never concatenated into the
assembled vector. Registered before scoring: the pairformer trunk is known to be in the second
category (its arms are driven by a captured 64-token boundary, not by `batch_step003`), so its
5.8282 % is expected to appear as a composed term and the difference between the two quantities
is expected to be **exactly that term**, not a discrepancy.

## Controls

  1. **A16 zero-gradient baseline**, measured by running an all-zeros arm through the same scorer
     over the same tensor set. Reported as a measurement even though the theory says 1.0.
  2. **A break control** that must move the reading: the same device arm with structure `k+1`'s
     cotangent (`device_grads_043all_permcot.pt`, `dev_scope_BREAK_c64.pt`). If it does not move,
     the scorer is not reading what it claims to read.
  3. **The instrument floor**: `upstream_f32` against `float64` on the same set. Upstream's own
     fp32 arm reads 3.117006e-05 on the diffusion scope; at model scope this must stay orders
     below every arm or the scorer itself is the error.

## Coverage

Stated as mass, not as count (A15/A23). The campaign's standing figures are 2.0067 % of the
model's squared gradient norm with no reading, 0.74055 % of that behind one host round-trip line
in `openfold3_host_prep.py:256`. Both are re-measured here rather than quoted.

## What this row does not do

It does not run one process that computes the whole model's gradient end to end. Whether that can
be done at all on this hardware today is a question this row answers with arithmetic and with the
campaign's own memory ladder, and the answer is part of the deliverable either way.
