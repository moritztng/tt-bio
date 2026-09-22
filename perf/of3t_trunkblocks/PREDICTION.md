# of3t-trunkblocks — PREDICTION

Committed BEFORE the census script is invoked once. What it fixes: the metric, the decision
rule, four numbered predictions with the reading that refutes each, and the arithmetic threshold
the counterfactual has to clear. Nothing below moves after a number appears.

## The metric

    e_b   = SUM over block b's 57 leaf tensors of || ours - upstream_bf16 ||^2       (ABSOLUTE)
    E     = SUM over all 48 blocks of e_b
    R     = SUM over all 2736 tensors of || upstream_bf16 ||^2
    v     = sqrt(E / R)                                   the trunk's in-frame clause reading
    share_b = e_b / E

Absolute and differenced, not the relative curve. `of3t-tapeamp` established the per-block
cotangent curve is ramp / step / saturation, a CANCELLATION signature, and
`a-difference-of-absolute-errors-locates-the-carrier` is that a share moves when its denominator
collapses — one leaf went 0.150 % -> 28.313 % of an error mass while its absolute error FELL
0.946x. So the census is over e_b, and the relative reading is carried beside it, never instead.

Frame: our banked device arm `dev_RENORM_n384_nocaptures.pt` against upstream 0.4.3's own bf16
autocast step `ref_bf16auto_n384.pt`, both driven from the same capture `boundary_n384.pt`, with
the local float64 `ref_f64_n384.pt` as the reference that validates the floor. This is the frame
in which the live clause figure 1.0293953378 was measured (`perf/of3t_frame384/FRAME_N384.json`,
`MATCHED.ours_vs_REF_LOCAL_bf16_n384`), so v must reproduce it exactly or the arm or the scorer
is not the published one.

## Decision rule (fixed here)

    CARRIER          a block holding >= 10 % of E
    CARRIER SET      the smallest set of blocks holding >= 50 % of E
    the premise under test   "the carrier is in the last two blocks", i.e. blocks 45 and 44,
                             which is what of3t-blk4544, of3t-trunkact and of3t-vjpln assume

## The threshold the counterfactual has to clear, computed here from committed numbers

The GRADIENTS clause needs the trunk at <= 0.4361680548 against upstream's own bf16
(`of3t-orchestrator`, pass 359). In this frame R = 1.9851011361181046 and
E = 1.0293953377723410^2 * R, so

    E_allowed = 0.4361680548^2 * R = 0.3777117
    E         = 1.0293953378^2   * R = 2.1035447
    fraction of E that must be removed = 1 - E_allowed/E = 82.05 %

**A set of blocks can only save the clause if it carries >= 82.05 % of the trunk's absolute
error mass.** That number is arithmetic on two committed readings, not a new measurement, and it
is written down before the census so no k can be chosen after the fact.

## Predictions

**P1 — the carrier is NOT block 45, and blocks 4 and 0 are in the top three.**
`of3t-frame384`'s per-block table, against the LOCAL float64 and not in the clause frame, reads
blocks 44, 4 and 0 at 27.2 %, 22.6 % and 18.4 % of the trunk's error mass, and block 45 at
1.5 %. I predict the clause frame ranks the same three at the top and puts block 45 outside it.
REFUTED IF block 45 appears in the top three by e_b, or holds >= 10 % of E.

**P2 — the top three carry 55 % to 80 % of E.** The float64 frame reads 68.2 %.
REFUTED IF the top three fall outside that band.

**P3 — no set of three or fewer blocks saves the clause.** Since the top three are predicted at
<= 80 % and the threshold is 82.05 %, correcting them leaves model scope above 0.147353.
REFUTED IF model scope reads <= 0.147353 with three or fewer blocks at upstream's own level.

**P4 — the ranking is frame-robust.** The pinned model-frame census, read off
`perf/of3t_modelboundary/sidecar_modelboundary/renorm_vs_UPSTREAM_BF16.json`'s per-tensor
`diff_norm` (the graded artifact's own denominator, pinned float64 sha256 1d4ea922...), will name
the same top-three SET, in any order.
REFUTED IF the two censuses disagree on the set — in which case neither locates a carrier on its
own and that disagreement is the row's finding.

**NULL — the error is spread.** If no block holds >= 10 % of E and the top three carry under
25 %, there is no carrier block, the per-block split is the wrong instrument, and the row says so
with the table rather than naming a block.

## What this census cannot see, stated in advance

Blocks PARTITION the 2736 leaf tensors, so the split of E over blocks is additive by
construction and a sum check on it can only catch a coding error, not a cancellation. What is
NOT additive is ATTRIBUTION: block b's leaf gradients are produced by the cotangent that arrives
at block b, which has already passed through blocks 47..b+1. A block can therefore hold error
mass it inherited rather than generated. The census measures WHERE the error mass sits, and the
ratio e_b against the floor's own per-block mass is the only separation of generated from
inherited this instrument offers. Naming the op inside a carrier block is a different arm and
this row does not claim it.
