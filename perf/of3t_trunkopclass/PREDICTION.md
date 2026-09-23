# of3t-trunkopclass — pre-registration

Committed **before the first arm**. Every bar below is fixed here. A bar that moves after a
number exists is not evidence, and this row may not move one.

Scope: the OpenFold3 0.4.3 pairformer trunk, 48 blocks, 2,736 leaf gradients in upstream's
parameter space, at the campaign's crop-384 frame (`boundary_n384.pt`, sha256
`8cb3a58669eaadc9a6782d927ff4dd4dc7b706f244e5575709b738c5ccb59aa3`) and at the cheap crop-64
control scope.

## The reference, pinned by hash

    ref_f64_n384.pt             2dcd8e70d6b564677e4a568af3ca1ca291e09d027b6dfea1f00f95159f951e1b
    ref_bf16auto_n384.pt        6d537f5f944a8f1649d75ecf3d035b0190454354902686d4b398368d781acc92
    ref_bf16full_n384.pt        ceeb4b4d1df543ab8f9d2a53199041fc3721122cf316dc3a21d6003e3efea5b1
    dev_RENORM_n384_nocaptures  095243e7b29f3778f527ea8835a5266783667b638d5111173aa636ce8c17ca4e
    ref_f64_c64.pt              1e9ad718cfef334508ce8b387004d18c7a0733f66f8f148bfa76c197b73b74d2
    ref_bf16auto_c64.pt         a94ce0d21637ea34622acb32e0a4b1f9677f59defbc09280e1ecbb3f4de0588f
    dev_scope_RENORM_c64.pt     5869fe06e7c088d118169c866199280433762fd186aea3d4760732cb0ffc2bb7

The float64 reference is upstream 0.4.3's own stack in float64, validated by float64 central
finite differences at `num_recycles = 0` per PROTOCOL §3c-bis. Never against another
approximation. The in-frame clause figure this row must move is **1.0293953378** (ours against
upstream's own bf16 autocast over these 2,736 tensors); the clause needs **<= 0.4361680548**,
a factor of **2.360x**.

## The op-class map, fixed before any mass is computed

57 distinct leaves per block, 48 blocks. Every leaf is assigned to exactly one class and no
leaf is unassigned. The brief names four classes; the stack has five families, because the
transition class appears twice per block (pair track and single track). Both splits are
reported: the five-family census is the finer one, and the four-class axis folds the two
transitions together, which is the brief's `pair transition` class.

    TRI_MUL    20 leaves   pair_stack.tri_mul_in.*, pair_stack.tri_mul_out.*
    TRI_ATT    16 leaves   pair_stack.tri_att_start.*, pair_stack.tri_att_end.*
    TRANS      10 leaves   pair_stack.pair_transition.* (5) + single_transition.* (5)
    APB        11 leaves   attn_pair_bias.*  (single attention with pair bias)
                           -- the brief's `single / MSA attention`; the MSA stack is a
                           different section of the model and is not in this scope.

## Denominator floor (A14)

The mass-weighted statistic is `sqrt( sum_k ||arm_k - ref_k||^2 / sum_k ||ref_k||^2 )` over the
class's tensors, the same definition `perf/of3t_wholemodel/agreement.py` uses, so the numbers
compose with the graded artifact. Per-tensor relative L2 is `||a-r|| / (||r|| + 1e-30)` per
PROTOCOL §3d. **Floor: a tensor with `||ref|| < 1e-12` is excluded from per-tensor relative
statistics and its count reported; it is never excluded from a mass sum, where it contributes
its own absolute error to the numerator and its own norm to the denominator.** A relative
figure on a collapsing denominator is how a 1.142e+13 headline got published (A14).

Absolute error mass is reported first and relative shares second, because a share moves when
its denominator moves (`a-share-moves-when-its-denominator-collapses`).

## Hypotheses and their falsifiers

**H1 — the error mass is concentrated in one class.** One of the four classes holds **>= 50 %**
of our trunk's squared absolute gradient error over the 2,736 tensors.
*Refuted* if the largest class share is **< 35 %**, which would say the mass is spread and no
class owns it. Between 35 % and 50 % is neither and is reported as such.

**H2 — the class is APB, not triangle multiplication.** The brief's leading suspect is triangle
multiplication, on the grounds that it is the block's largest reduction. I predict the opposite.
**This is not an independent prediction and is recorded as such:** `of3t-trunkact` published a
LEAF census over this same artifact in which `attn_pair_bias.layer_norm_a.{weight,bias}` — 96 of
2,736 tensors, 0.8583 % of the reference mass — hold 56.1174 % of our error mass. Re-indexing a
published leaf census by class is arithmetic, not a discovery, so **the census is not this row's
evidence; the ablation is.** H2 is stated so that a census disagreeing with trunkact is caught
as an instrument defect rather than read as a finding.

**H3 — the ablation decides the carrier, and a census cannot.** A class is the carrier only if
holding that class at the highest accuracy the stack offers, with the other three shipped,
removes **>= 50 %** of our trunk's excess over the A/A determinism floor, measured as the drop
in the mass-weighted rel L2 against upstream's own bf16 autocast. **Refuted for that class under
20 %.** Between 20 % and 50 % is a contributor, not a carrier.

**H4 — sub-additivity, and the honest exclusion it licenses.** Perturbations of this stack are
strongly sub-additive. If every one of the four single-class arms removes **< 20 %** while the
all-classes arm removes **> 80 %**, then no single class is the carrier and the finding is that
the trunk's gradient error is a **composition across classes** — a measured exclusion of all
four, which is a complete result. This is pre-registered so that an all-four exclusion cannot
later be reported as a failure to find anything.

## Controls, run before any arm is read

1. **A/A determinism floor.** The same device arm run twice, same card, same config, compared
   tensor by tensor. Reported FIRST. Three of the four class differences may sit inside it, and
   a difference inside the floor bought nothing.
2. **Instrument control.** The class census must re-pool to the artifact's own published trunk
   figures: `ours_vs_f64` 0.8354121633458239, `ours_vs_bf16auto` 1.029395337772341,
   `bf16auto_vs_f64` 0.37393839211303687. Relative difference must be under 1e-12 or the
   instrument is wrong and no class number is read.
3. **Coverage is measured, not asserted.** The runtime call count per class comes from counting
   module invocations on the scored batch, not from reading the source. A selector called `all`
   reached 6.88 % once in this campaign (`of3t-blk4544`), and a percentage attached to a
   code-read route has already survived to a brief with zero calls behind it.

## What this row will not do

No shipped default moves. An accuracy lever that exists only as an ablation is a finding; moving
it for inference is a separate decision with its own A/B and it belongs to the orchestrator.
No merge to main.

## Boundary with `of3t-trunkblocks`

That row splits the same error by block. This row has not read its artifacts or its state doc
and will not until this row's own census and ablation are committed. The statement of whether it
was read, and when, goes in the state doc.
