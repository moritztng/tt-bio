# of3t-tapeattn — result

Pre-registration: `PREDICTION.md`, committed at `8826fb445`, before the first device arm.
Everything below was measured after it. The pre-registration is **refuted**, on its own stated
falsification clauses, and the row's premise is refuted with it.

Host for every arm and every floor: **qb1 (tt-quietbox), card 2, p150a Blackhole**, subsystem
device `0x0040`. Every float64 reference and every bf16 floor quoted here was built on that same
box (D189).

## What the row was sent to find, and what is there instead

The brief's object was "a per-block multiplicative factor of about 2.9x on the single track's
cotangent" between padded width 64 and padded width 384, expected to live in one of the tape's
attention verbs in `tt_bio/autograd.py`.

Measured in the tape, at both widths:

1. The cotangent does carry width growth, but **it is not per-block**. It is created in the first
   three blocks the backward touches and then decays over the remaining forty-five.
2. **No verb in the single track's attention chain injects more at 384 than at 64.** softmax is by
   far the largest injector at both widths, and its own per-firing error gets slightly *smaller*
   with width. The 1.78x that the pooled table shows is its reference-norm mass moving, not its
   arithmetic degrading.

## Floors, all taken before any arm was scored

| floor | reading |
| --- | --- |
| A/A determinism, padded 64 | two independent shipped-path runs, 98/98 tensors bit-identical, **exactly 0.0** |
| A/A determinism, padded 384 | same, 98/98 bit-identical, content sha `c70be48e`, **exactly 0.0** (`CMP_A384_A384R.json`) |
| wall clock, padded 384 | 732.0 s and 718.2 s over those two byte-identical runs, **1.92 %** |
| instrument inertness, 64 | census arm reproduces the shipped arm's cotangent, content sha `7ecaa69e` |
| instrument inertness, 384 | 98/98 bit-identical, content sha `c70be48e` (`CMP_A384_C384.json`) |
| reference checkpointing, 64 | plain float64 vs activation-checkpointed float64, 98/98 bit-identical (`CMP_REF_F64_CKPT_C64.json`) |
| reference width-invariance | float64 reference at crop 64 vs padded 384, on the 56 real tokens, **rel_l2 exactly 0.0 at all 49 rungs, both tracks** (`REFWIDTH_F64.json`) |

AICLK sampled every 4 s during each 384 arm: n=122 min=800 median=1350 max=1350 (A384), n=119
min=800 median=1350 max=1350 (A384R). The 800 MHz samples are the head and tail of each run; the
fold itself ran at the 1350 MHz burst.

The last floor is the one that makes the rest readable. The denominator does not move with padded
width **at all** — not to 1e-6, bit-identically — so every bit of the width growth is ours.

## Deliverable 2, part one — the call census

`tapecensus.py` wraps `autograd._tape`, so every verb is counted where it runs, not read off a
constructor (D196). Identical chain and identical counts at both widths, per taped backward, on
the single track (`heads=16`):

    transpose 48, matmul 48, add_ 96, multiply_ 48, softmax 48, matmul 48,
    create_qkv_heads 144, sliced 48, permute 96      624 captured nodes, 0 capture errors

48 firings is one per block. `AttentionPairBias` runs on the tape and nowhere else, which
confirms the one thing the brief carried over from `of3t-shapekey`.

## Deliverable 2, part two — per-verb float64 injection, both widths

Each verb's backward on its own device operands, differenced against the same closure recomputed
in float64 on the same operands.

| verb | rel_l2 @64 | rel_l2 @384 | pooled ratio | cos_min @64 | cos_min @384 |
| --- | --- | --- | --- | --- | --- |
| softmax `[1,16,N,N]` | 0.011735 | 0.020884 | 1.780 | 0.62053 | 0.65174 |
| matmul (four sites) | 0.001672–0.001881 | 0.001653–0.001830 | 0.93–1.01 | 0.99999 | 0.99995 |
| multiply_ | 0.001580 | 0.001664 | 1.053 | 0.99999 | 0.99999 |
| add_, transpose, permute, sliced | 0.0 | 0.0 | — | 1.0 | 1.0 |
| create_qkv_heads | 5.5e-35 | 2.5e-35 | — | 1.0 | 1.0 |

softmax is the only verb above the elementwise bf16 floor (~0.0017) and the only one whose pooled
figure moves with width. Its per-call direction is the tell: cos as low as **0.62053**, against
0.99995 or better everywhere else.

**The pooled ratio is the wrong reading, and it is wrong in the flattering direction.**
`rel_l2_aggregate` is a reference-norm-weighted RMS over the 48 firings, so it moves when the
mass moves. Crossing the two widths' error vectors with the two widths' weights
(`sm_reweight.py`, `SMREWEIGHT_softmax.json`):

| | pooled | over the 64 cell |
| --- | --- | --- |
| 64 errors, 64 weights | 0.011735 | 1.0000x |
| 384 errors, 384 weights | 0.020884 | 1.7797x |
| **384 errors, 64 weights** | 0.010994 | **0.9369x** |
| **64 errors, 384 weights** | 0.024552 | **2.0923x** |

Hold the weights fixed and softmax's own arithmetic gets **better** at 384, by 6.3 %. Hold the
errors fixed and move only the weights and the whole 1.78x appears, with room to spare. Only 10
of 48 firings are worse at 384. The mass is concentrated at the top of the backward — softmax
ordinal 0 alone carries 69.3 % of the reference-norm mass at 64 and 62.3 % at 384, and ordinal 3
goes 1.01 % to 4.01 % while carrying 13x ordinal 0's per-firing error. That reweighting is the
entire effect.

The same decomposition on the other two live verbs finds nothing to decompose: matmul 1.0082x
pooled and 1.0002x with weights held; multiply_ 1.0530x pooled and 1.0569x with weights held,
both at the elementwise bf16 floor.

## Deliverable 2, part three — the cotangent entering each block, both widths

`score_cot.py` against the in-frame float64 reference, masked to the 56 real tokens, with
upstream's own bf16 recipe as the floor at each rung. Backward order, rung 48 is the injected
seed and rung 0 is the stack input.

| rung | ds @64 | ds @384 | x | dz @64 | dz @384 | x |
| --- | --- | --- | --- | --- | --- | --- |
| 48 | 1.5799e-03 | 1.5799e-03 | 1.000 | 5.4699e-01 | 5.4699e-01 | 1.000 |
| 47 | 8.8386e-01 | 8.6770e-01 | 0.982 | 4.5920e-01 | 4.7363e-01 | 1.031 |
| 46 | 7.1347e-01 | 7.4352e-01 | 1.042 | 8.2667e-01 | 8.2772e-01 | 1.001 |
| 45 | 6.1508e-01 | 1.1427e+00 | **1.858** | 7.0619e-01 | 8.1202e-01 | 1.150 |
| 44 | 6.9791e-01 | 1.5364e+00 | **2.201** | 5.9378e-01 | 1.1588e+00 | **1.952** |
| 43 | 6.9835e-01 | 1.5888e+00 | 2.275 | 5.6215e-01 | 1.0830e+00 | 1.926 |
| 32 | 8.7195e-01 | 1.5228e+00 | 1.746 | 4.1202e-01 | 7.6179e-01 | 1.849 |
| 16 | 8.5462e-01 | 1.5335e+00 | 1.794 | 3.6087e-01 | 5.5461e-01 | 1.537 |
| 0 | 8.1204e-01 | 1.4000e+00 | 1.724 | 4.2914e-01 | 6.3703e-01 | 1.484 |

Full ladder in `COTSCAN_64.json` and `COTSCAN_384.json`, 49 rungs each.

The bf16 floor is **exactly width-invariant at every rung**: floor@384 / floor@64 = 1.000 for all
49 rungs on both tracks, which follows from the reference control above.

## Deliverable 2, part four — injection or accumulation

`of3t-bwdaccum`'s discriminator, evaluated mechanically, calls **branch B at both widths**:
over-floor never exceeds 2.5x. Read literally that says the cotangent sits at the floor and the
defect is in the leaf backward. The label is too coarse for what is actually there, and the honest
reading is the ratio column, not the branch name:

- The growth is **neither per-block injection nor accumulation**. It is a **step**, created in the
  backward of blocks 45 and 44 (ds goes 1.042 → 1.858 → 2.201 across rungs 46, 45, 44), then
  carried and slowly diluted over the remaining 44 blocks, ending at 1.724. A per-block factor
  would compound; this one decays monotonically after rung 43.
- ds at rung 44 reads **2.201x**, against the campaign's model-scope D191 growth of **2.1795x**.
  The whole of D191's width growth is already present in the cotangent four blocks into the
  backward, and nothing after block 44 adds to it.
- At 64 our cotangent is at or **below** upstream's own bf16 floor (over-floor 0.72–0.94 on ds).
  At 384 it is **above** it (1.14–1.81 on ds, 1.42–2.19 on dz). We did not get worse in a place
  the floor also got worse. The floor did not move.
- The ladder is saturated: relative error is O(1) from rung 47 down at both widths, and so is the
  bf16 floor's (0.93–1.23). A saturated ladder cannot localise a small per-block injector, which
  is a real limit on what this measurement can say and is why the step location, not a per-block
  share, is the result.

## What this refutes

Scored against `PREDICTION.md`'s own clauses:

- **Candidate 1 (softmax, primary) — REFUTED.** Its clause was "softmax carries the largest single
  share of the per-block injected error at 384, **and its share grows between the widths**". The
  first half holds and the second fails: with weights held fixed the ratio is 0.9369x.
- **Candidate 2 (matmul) — confirmed inert as predicted**, 1.0082x pooled, 0.93–1.01 per site.
- **Candidate 3 (elementwise and transpose) — confirmed inert as predicted.** add_, transpose,
  permute and sliced are exactly 0; multiply_ is at the floor.
- **Candidate 4 (leaf reductions) — excluded by arithmetic, not by an arm**, as pre-registered. A
  leaf-side reduction writes a parameter gradient and does not feed the cotangent chain.
- **The ladder-shape prediction — REFUTED.** It predicted branch C, a sharp rise over the first few
  rungs then flat. The discriminator reads branch B at both widths and the ratio column reads a
  step-then-decay that the pre-registration named no branch for.
- **One falsification clause could not be evaluated as written.** It named
  `dev_cot.py --lever softmax_fp32`. `dev_cot.py`'s levers are all on the LayerNorm backward
  (`dW = sum_t g_t * xhat_t`); there is no softmax lever. That is recorded rather than repaired.

## What the row does not claim

No fix, no shipped default moved, nothing merged. `tapecensus.py` is instrumentation and it is
shown inert by content sha at both widths rather than asserted. A leaf-census extension to
`tapecensus.py` was attempted and its single run failed; it is deliberately left out rather than
committed untested, because candidate 4 is excluded by arithmetic anyway.

The cotangent step at blocks 45 and 44 is **not attributed to a named op**. This row's census sees
only the single track (`heads=16`); the pair track (`heads=4`, TriangleAttention) and the triangle
multiplications run in the same blocks and are outside it. That is the next row's arm, and the
cheapest decisive one is a substitution rather than another census: pin the single track's softmax
backward to its float64 VJP for all 48 firings and re-score the ladder. If the 384 over-floor
plateau does not drop toward the 64 level, the carrier is outside the single track, which is what
these readings already suggest.
