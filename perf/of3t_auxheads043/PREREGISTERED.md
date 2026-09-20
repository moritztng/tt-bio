# of3t-auxheads043 — pre-registration, written and committed BEFORE any number exists

Row brief: `aux_heads` fails A18 at **3.6515e-01** against a reference the brief says was built
on upstream **0.5.0**. Rebuild the reference at **0.4.3** (the revision our port targets and the
revision `of3-p2-155k` belongs to) and measure the forward again. 2.8431 % of the model's squared
gradient norm is carried as VOID on the strength of that failing forward.

The campaign has withdrawn three magnitude estimates published without a pre-registration. This is
that pre-registration. Nothing below was written after seeing a result.

## Arm 0 — WHICH REVISION BUILT THE EXISTING REFERENCE (A24: an input is pinned by digest)

Before rebuilding anything, the brief's premise gets checked: the tree that produced
`boundary_aux_heads.pt` is hashed file by file against the 0.4.3 and the 0.5.0 sdist.

* **P1** — the tree matches **0.5.0**. The brief's premise holds; the rebuild at 0.4.3 is a real
  change of reference and arms 1-3 decide what it is worth.
* **P2** — the tree matches **0.4.3**. The brief's premise is **false**: the reference was already
  at the revision the brief asks for, the rebuild cannot move the forward, and the row's answer is
  that the revision does not explain the failure. This is the branch that keeps the scope VOID and
  it must be reported exactly as loudly as P1.
* **P3** — the tree matches neither. Everything downstream is void and the row stops and says so.

## Arm 1 — THE REFERENCE REBUILT AT 0.4.3, on the captured boundary

Upstream's own `AuxiliaryHeads.forward` from each release tree, run in one process per tree, on the
**identical** captured inputs from `boundary_aux_heads.pt` (batch `batch_step003.pt`
sha256 `3c32597a…`, 5nw3, crop 384 carrying 56 real tokens, `of3-p2-155k` weights), float64.

**Dtype policy, stated per A27, because "float64" alone is not a policy.** Every parameter and
every activation is float64. `torch.amp.autocast(device_type="cuda", …)` — the mechanism BOTH of
the revisions' precision differences act through — is inert on CPU tensors and PyTorch says so out
loud. So this arm measures the two revisions' **structure**, with their precision difference
switched off by the platform. That is a limit of the arm, it is stated here before the run, and
arm 3 is the arm that does not have it.

Read per head, relative L2 of the 0.4.3 output against the 0.5.0 output, plus the norm ratio
`r = ||a||/||b||` and the cosine of the error against the reference, never the relative alone
(A25, and the fleet's standing rule that a relative L2 alone cannot identify a direction).

* **R1** — any of the five heads differs by **>= 5.0e-02**: the revision moves the reference at the
  bar the comparison is judged at. Rebuilding is load-bearing and `aux_heads`' A18 verdict has to
  be re-read against the 0.4.3 arm.
* **R2** — all five differ by **< 5.0e-02** but the largest is **> 1e-12**: the revision moves the
  reference by less than the bar. It contributes, it does not explain, and the failing forward
  stays the port's.
* **R3** — all five differ at **<= 1e-12** (float64 round-off or exactly 0.0): the revision is
  **inert on this boundary in this dtype policy**. A rebuilt 0.4.3 reference is the same reference
  and the A18 failure is untouched by it.

## Arm 2 — WHICH DIFFERENCE FIRES (measured, not read off a diff)

Two differences are live between the revisions on this path:

1. `prediction_heads.py`: 0.4.3 wraps `embed_zij` in `autocast("cuda", float32)` and casts back;
   0.5.0 removes it and instead wraps the confidence pairformer stack in
   `autocast("cuda", pairformer_dtype=float32)`. Two precision changes in **opposite** directions.
2. `head_modules.py`: 0.5.0 feeds the pairformer embedding `single_mask=token_mask` where 0.4.3
   passed `single_mask=repr_x_mask`.

For (2) the knob is forced to match — 0.5.0 patched back to `repr_x_mask` — and the two arms must
then agree **exactly**. **A control that does not read 0.0 means something else on the path is
live and arm 1 is VOID.**

D94 measured `repr_mask_sum = token_mask_sum = 56.0`, 0 differing tokens, on this batch. If that
holds, the forced-knob control is **vacuous** — it cannot break anything — and a vacuous control is
not a control. In that case a second, deliberately breaking control is run: one token's
representative atom is marked unresolved so the two masks genuinely differ, and the arms must then
**diverge**. A control that cannot break what the check reads is not evidence (fleet rule
`negative-control-must-break-what-check-reads`).

## Arm 3 — THE PRECISION DIFFERENCE, WHERE IT IS ACTUALLY LIVE

D94 recorded that a CPU arm cannot size the autocast differences. D93 then showed at pass 176 that
this is true of the **flag** and false of the **arithmetic**: a flag that selects a dtype policy
can have its policy written out by hand. So the same three-arm construction is used here, base
dtype bf16 (training precision, A26), on the same captured inputs:

    A   0.4.3's policy   embed_zij in fp32, confidence pairformer stack at ambient bf16
    B   0.5.0's policy   embed_zij at ambient bf16, confidence pairformer stack in fp32
    C   control          both policies forced identical -> must read exactly 0.0

Read as relative L2 per head with `r` and the cosine beside it, and against **arm 1's float64
answer for the same head**, which is the floor the policy difference sits on.

* **S1** — A vs B is **>= 5.0e-02** on a head that fails A18: the revision's precision difference
  is large enough to matter at the bar, and any future reference built at the wrong revision under
  CUDA autocast is a real hazard even though arm 1 could not see it.
* **S2** — A vs B is **< 5.0e-02**: the precision difference is smaller than the bar even where it
  is live.
* **VOID** — control C does not read exactly 0.0: arm 3 is void and is reported as void.

## What decides VOID_OR_NOT

`aux_heads` leaves this row **A18-void** unless arm 1 lands on **R1** and the 0.4.3 forward then
reads **< 5.0e-02** on the heads that currently fail. Any other combination leaves it void. Both
answers are results; the row does not get to prefer one. The bars are the protocol's and are fixed
here: **5.0e-02 per-tensor, 2.0e-02 mass-weighted**, and per A26 the bar an independent bf16 port
can actually reach is **sqrt(2) x threshold**.

## What this row does NOT do

It holds **no card lease**. Our port's `OF3ConfidenceHead.forward_device` is a ttnn implementation
and cannot run here, so no NEW reading of our port's output tensors is produced by this row. If
the reference does not move, the A18 forward does not move either and the published 3.6515e-01
stands as the reading against 0.4.3 — which is a statement about the reference, not a re-run of
the port. The D95 masking refinement of the port's own reading needs those device tensors and is
named in DOESNOT rather than estimated here.
