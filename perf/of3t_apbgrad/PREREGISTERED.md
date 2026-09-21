# of3t-apbgrad — pre-registration

Written and committed BEFORE any number exists. Nothing below is adjusted after a reading.

## What is inherited, not re-derived

`of3t-bwdaccum` (581e10bad) established this, and this row starts from it rather than re-proving it:

  - the trunk scope reads mass-weighted **9.025172e+00** against upstream 0.4.3 in float64 over
    2,736 of 2,736 tensors holding 100 % of the squared gradient norm;
  - the four LayerNorm affine leaves holding 92.68 % of the error mass have a backward correct
    to **1.370e-02** against float64 on their own captured operands, at or under their `u*K`
    bound — innocent;
  - the **AttentionPairBias** backward, fed the reference cotangent at every block, writes a
    cotangent at its own LayerNorm-ed input that is **0.98x to 33.30x** the float64 reference in
    norm at **cos 0.019 to 0.920**;
  - the sibling **Transition** path, same track, same blocks, same tape, reads **1.00 to 1.04 at
    cos 1.000**;
  - upstream's own bf16 arm on this boundary rebuilt in that environment read **3.739355e-01**
    against the **4.007237e-01** on record, 6.7 % apart. This row rebuilds it again and reports
    its own value.

## The instrument

The shipped token-level `AttentionPairBias.__call__` is NOT rewritten. Every op it calls is
recorded where the tape already sees it (`taped_ttnn._taped_verb`), so the arithmetic stays the
shipped arithmetic, and an A/A must show the gradient is bit-identical with the recorder on.

Per taped op inside the module, at chosen blocks, with the reference cotangent injected at every
block boundary (`of3t-bwdaccum`'s teacher forcing, reused unchanged):

  - the forward operand VALUES the device actually had;
  - the cotangent the device handed that op's backward;
  - every `add_grad` contribution that backward made, tagged to the operand it landed on.

Two scorings, both against float64:

  **ISOLATION.** Each op's backward recomputed in float64 from ITS OWN captured operands and
  ITS OWN captured incoming cotangent, compared against what the device wrote. This is the
  method `of3t-bwdaccum` used on the LayerNorm leaves and it asks one question: is THIS op's
  arithmetic wrong.

  **COMPOSED WALK.** The whole AttentionPairBias backward recomputed in float64 from the
  device's captured forward operands and the device's incoming cotangent at the module's
  output. Walking it backwards names the first node at which device and float64 part company.

## Pre-registered bands, fixed here

Per-op isolation `rel_l2` of the device contribution against the float64 recomputation:

  - **<= 3.0e-02** — INNOCENT. This is the band the four LayerNorm leaves read (1.370e-02) and
    it is bf16 unit roundoff times a modest cancellation factor.
  - **>= 3.0e-01** — the NAMED OPERAND. An op at or over this is doing arithmetic its own
    operands do not justify.
  - between — contributing, not the locus. Reported with its share, not promoted to a cause.

Composed walk, at the module's `ds` (the cotangent to the LayerNorm-ed input):

  - if the float64 walk from the DEVICE's forward operands reproduces the **reference's** ds
    inside 5.0e-02, the defect is in the backward ARITHMETIC and one of the ops above owns it;
  - if it reproduces the **device's** ds instead (inside 5.0e-02), the backward arithmetic is
    right for the forward it was given and the defect is in the forward VALUES the module
    computed — a different finding, and it must be said plainly, because the forward passes A18;
  - if it reproduces neither, the error is distributed across the walk, and the walk's per-node
    table is the result.

## The three outcome branches the brief fixes

  1. a named operand accounts for the 33x AND a fix takes the trunk scope inside **2.0e-02**
     mass-weighted over the same 2,736 tensors → the trunk is REPRODUCED;
  2. inside A26's reachable sqrt(2) of that bar but not the bar → REACHABLE, with the residual
     reported against upstream's own bf16 arm REBUILT in this environment, not the 4.007237e-01
     carried in;
  3. no single operand accounts for it → say so plainly. A distributed cause in one module whose
     sibling is exact is a different and harder finding, and it is a real outcome.

## Controls, fixed here

  - **A/A**: the recorder on vs off must give a bit-identical trunk gradient (`torch.equal`) on
    all 2,736 tensors, or every reading below is the instrument's.
  - **A16**: a zero gradient scores 1.000000 at scope; a zero cotangent scores 1.000 at every
    rung. Measured, not asserted.
  - **BREAK**: a control that moves the reading. The per-op isolation is broken by permuting the
    captured cotangent's real token positions before the float64 recomputation; it must move the
    per-op rel_l2 by at least 3x, or the isolation scoring reads nothing.
  - **SIBLING**: the Transition path's output cotangent must still read cos 1.000 after any
    change this row makes, on the same blocks `of3t-bwdaccum` probed.
  - **GATED**: `scale_pair_bias=False, tri_att_scale_pair_bias=False` stays the shipped default;
    `compose_verify.sh`'s named assertion is re-run against this branch's
    `tt_bio/openfold3_trunk.py`. Nothing merges.

## Two traps already on the record, not to be re-walked

  - the token-level branch of `AttentionPairBias.__call__` computes attention INLINE and never
    calls `self._attention`, so `fp32_softmax` is structurally dead there (`of3t-trunkcliff`).
    No lever is priced on that flag at this site.
  - `_accurate_softmax` was reached at this class of site with a bf16 input where `ttnn.max`
    truncation is exact, so D111 does not apply on the bf16 path. The dtype is checked before
    either assumption is made.
