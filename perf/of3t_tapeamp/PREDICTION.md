# of3t-tapeamp — pre-registration, written before this row's first arm

Committed before `amp_arm.py` ran once. Every number below is either quoted from a concluded
row's artifact (with the artifact named) or is a prediction with a band and a falsifier.

## 0. The figure in the brief is the pre-repair figure

The brief opens on forward 0.85 % / gradient 16.6 % -> 19.6x for `diffusion` and 0.82 % / 16.2 %
-> 19.8x for `msa_module`. Both are **superseded**, by two concluded rows, and neither is the
number this row is about:

    scope          forward         gradient        ratio     source
    diffusion      8.474801e-03    9.344246e-02    11.026x   of3t-ditref REPRICE.json, of3t-tapediverge AMPLIFY
    msa_module     8.176476e-03    8.915364e-02    10.903x   of3t-ditref D58_msa_today.json, same AMPLIFY table

`of3t-ditcot` divided by a 0.5.0 reference that drops our 24 trained per-block
`attention_pair_bias.layer_norm_z` tensors; `of3t-ditref` swapped in the 0.4.3 capture
(`diffcap043`) and the diffusion gradient went 0.7055 -> 0.0934. `of3t-tapediverge` separately
found the shipped arm reads 1.250047e-01, not D30's filed 1.6588e-01, with the forward
bit-identical across both runs, so part of the 19.6x was a tree that had already moved.

**So the object is 11.03x and 10.90x, not 19.6x and 19.8x.** The two-significant-figure agreement
the brief calls the reason to treat this as one object survives the re-price at the new value:
1 % apart, two modules, two harnesses, two references. That agreement is the thing to explain.

## 1. Boundary artifact against amplifier — what separates them

**Boundary artifact (the D206 class).** A dtype boundary crossed asymmetrically in the backward:
fp32 weights meeting bf16 cotangents, a cotangent entering a rule at a dtype the forward value did
not have, a typecast a shipped caller performs inline that the tape's own path skips. D206 is this
exactly, in a training forward, and it read eleven orders out.

  * PREDICTS: the amplification is OURS. Upstream's own bf16 recipe, run at the same boundary
    against the same float64 reference, would show a **much smaller** forward-to-gradient ratio,
    because upstream crosses no such boundary.
  * PREDICTS: the factor is a **bf16** phenomenon. An fp32 arm of the same function, scored the
    same way, would not show it.
  * PREDICTS: a census of the tape's backward rules finds at least one site where the incoming
    cotangent's dtype differs from the forward value's.
  * FALSIFIED BY: upstream's own arm showing the same ratio, or an fp32 arm showing the same ratio.

**Amplifier (conditioning).** Reverse mode through this architecture is worse-conditioned than the
forward, for reasons that have nothing to do with dtype: `dW = sum_t g_t xhat_t` is a
cancellation-limited reduction and `of3t-bwdaccum` measured site cancellation factors K up to
5.876e+01 on the trunk, with every leaf reading at or below `u*K` at `u = 2^-8`. A function whose
Jacobian-transpose product has condition number ~10x its own forward loses one more decimal digit
in the gradient at ANY precision.

  * PREDICTS: upstream's own bf16 arm shows the **same ~10x**, because it computes the same
    function's derivative.
  * PREDICTS: the ratio is **precision-independent**. An fp32 arm shows it too.
  * PREDICTS: there is no defect to repair here, and D30/D58 are a property of the model, not of
    tt-bio.
  * FALSIFIED BY: upstream's own ratio landing near 1x, or the fp32 ratio landing near 1x.

The two hypotheses disagree on exactly one measurement that nobody in this campaign has made:
**upstream 0.4.3's own forward accuracy at this boundary.** `perf/of3t_cond043/BARS043.json` has
its gradient (median 1.2940662e-01, mass-weighted 1.8817643e-01, n=761) and
`FLOOR043_bf16auto.json` has its forward *seconds* (24.757 s). Its forward *accuracy* is not in
any artifact in the campaign. That is the first arm, and it is a CPU arm: no card, no lease.

Why the frame is safe: both halves of the ratio come out of ONE process on ONE host, so the
`a-bf16-reference-is-host-dependent-float64-is-not` trap (6.0 % between two x86 boxes on the same
code; `of3t-bwdaccum` control 7 saw 6.7 % on an upstream bf16 arm rebuilt in a new environment)
cannot reach the ratio. The arms on record were measured on qb2; these run on qb1, so the
re-measured gradient is reported beside the record and any difference is named rather than hidden.

## 2. Three candidates, named in advance and ranked

1. **CONDITIONING, not a defect (60 %).** Upstream's own bf16 ratio lands within 30 % of our
   11.03x and the fp32 ratio lands in the same decade. Supporting evidence that already exists:
   `of3t-ditref` measured our gradient at **0.5865x** upstream's own bf16 gradient mass-weighted
   and **0.491x** on the like-for-like subset, i.e. our backward is already BETTER than upstream's
   own recipe. A tape that amplified would have to be worse than the thing it is compared to.
2. **A leaf-rule class, sub-dominant but real (25 %).** `of3t-ditref`'s ablation puts softmax at
   47.71 % of the baseline median and 2.0x the next class, so bounding the softmax verb in float64
   halves the gradient error. That is an ordering of a residual, not a defect, unless the residual
   sits above upstream's own floor — and at scope it does not. If this is the answer, the ratio
   shrinks partway on an op arm and never reaches 1x.
3. **A dtype boundary in the tape (15 %).** The D206 class, in the shipped tape rather than a
   training harness. Named lowest because D206's site was a caller that bypasses
   `OF3SampleDiffusion.__call__`, and `device_gradient.py` goes through the module directly too —
   so if this were live in the diffusion arm the forward would also be wrong, and the forward
   reproduces bit-identically at 8.474801e-03 across four independent runs on two cards.

## 3. Bands, so the readings are results and not targets

  * **P1.** Upstream 0.4.3 bf16 forward median over the 48 structures lands in
    **[6.0e-03, 2.5e-02]**, central estimate 1.2e-02. FALSIFIED below 3.0e-03 or above 6.0e-02.
  * **P2.** Upstream's own bf16 ratio (its gradient median over its forward median, one process)
    lands in **[5.2x, 21.6x]**, central estimate 10.8x. FALSIFIED below 4.0x.
  * **P3.** The fp32 arm's ratio lands in **[3x, 40x]** — the same decade, not near 1x. Its
    gradient median is on record at 1.2533394e-05, so this is a prediction about its forward:
    I expect **[3e-07, 4e-06]**. FALSIFIED if the fp32 ratio lands below 2.0x, which would make
    the factor a bf16 phenomenon and hand it back to hypothesis A.
  * **P4.** The A/A of the bf16 arm is bit-identical on both halves, so the determinism floor of
    this instrument is exactly 0 and the wall-clock floor is read off repeat timings, not asserted.
  * **P5.** My re-measured upstream bf16 gradient on qb1 lands within **10 %** of the 1.2940662e-01
    measured on qb2. If it moves more than that, the bf16 floor is more host-sensitive than the
    6.7 % on record and every `x floor` ratio in the campaign inherits the caveat.
  * **P6.** A call census of `tt_bio/autograd.py` and `tt_bio/taped_ttnn.py` finds **zero** backward
    rules where the incoming cotangent's dtype differs from the forward value's dtype on the
    diffusion scope. Stated as a census to run, not a route to read: D196 is the defect where a
    source-read route with a measured share attached turned out never to execute.

## 4. What I conclude if the factor is only partly explained

If upstream's own ratio is materially LOWER than ours but far above 1x — say upstream 5x against
our 11x — then part of the factor is the function and part is ours, and the honest report is a
**split with both parts priced**, not a mechanism. In that case the deliverable becomes the
per-block cotangent reading (`of3t-bwdaccum`'s discriminator: flat at the bf16 floor means a wrong
leaf backward, degrading with depth means something injects per block) on the OUR-minus-UPSTREAM
difference, and the row reports a NARROWING of D30 and D58, explicitly not a closure. The campaign
has already carried a concluded row as a live owner for days by calling a narrowing a closure; this
row will name which of D30, D58, D129 and D55's forward arm it closes, which it narrows, and which
it leaves untouched, in those words.

If upstream's ratio matches ours, D30 and D58 stop being defects and become a measured property of
the model, and I will say so plainly rather than looking for a smaller amplifier inside a residual
that already sits below upstream's own floor.
