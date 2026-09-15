# The quiet re-capture: the trunk pairformer really is above its floor

`roof-true-floor` and `roof-residual-census` both stopped at the same missing measurement. The
per-unit times in the roof budget were taken at loadavg 27-28 against a 24.731 s contended session
and mapped onto the quiet 17.340 s cell of record by one scalar, 0.7011. On that table the trunk
pairformer's floor, 10.123 s, came out **above** its measured time, 9.611 s, 105.3 % of roof. A
floor above measured time is a defect signal, and it had two candidates: the single rescale, or
`roof-shape-honest-roofs`' per-class rates not being upper bounds.

This is that fold, re-taken quiet with every call's own wall kept.

**Verdict: candidate 2.** Removing the rescale makes the crossing wider, not narrower. At least one
class rate is not an upper bound, and `TriangleAttention` is the one, as the row suspected.

## The capture

| | committed | quiet re-capture |
|---|---|---|
| host, card | qb2 card 2 | qb2 card 0 (same p300c Blackhole) |
| head | f072ae02f | f072ae02f |
| session loadavg | 28.11 / 28.60 / 27.72 | 2.22 / 1.34 / 0.59 |
| plain folds | 21.7 % spread | 17.232 / 17.308 s, 0.44 % spread |
| session fold | 24.731 s | 17.270 s |
| cell of record | 17.340 s, imported | 17.270 s, its own |
| **cell scale** | **0.7011** | **1.0000** |
| instrumented fold | 74.159 s | 48.565 s |
| captures | 26 | 26 |

The fold is bit-identical across repetitions and its mmCIF hashes to `2bc758a1fb24ef30`, the same in
both quiet runs taken this session.

Pre-registered control (`predict.txt`, written before the capture): the bytes and FLOPs must not
move. They do not. `fold_TB` is 2.9449 and `fold_TFLOP_executed` is 219.492 in both, so the floor
side is untouched and only the times changed.

## Per-call times measure the instrumented fold, not the fold of record

The raw sum of every call's own measured wall is 46.172 s against a 17.270 s fold. It reconstructs
the *instrumented* fold, 48.565 s, because a `ttnn.graph` capture slows every call nested under it
while it is open, not just the call it is named for.

The contamination is far out and easy to see. The per-call bodies are tight -- `TriangleMultiplication`
spans 6.263 to 6.451 ms across 528 calls -- and the contaminated calls sit at 1130 ms, 170x the
median. 534 of 33683 calls, 1.6 %, carry 29.691 s of the 46.172 s raw sum.

| estimator | total | of the 17.270 s fold |
|---|---|---|
| raw sum of per-call walls | 46.172 s | 267.4 % |
| trimmed sum (a call above 3x its unit's median is replaced by that median) | 16.481 s | 95.4 % |
| median x calls (the published rule) | 16.752 s | 97.0 % |

Trimmed and median agree to 1.6 % and both land within 5 % of the fold. This is why
`roof_budget_table.py --time-stat percall`, which substitutes the median for the *one* call named
`captured_call`, is not usable: the substitution is 1 call where 534 are contaminated. `trimmed` is
the rule to use with per-call data, and it is what the join below runs on.

## The join, re-run

Same scripts, three arms. Floor and prize from `perf/roof_true/true_floor.py`.

| arm | cell | scale | trunk measured | trunk floor | % of roof |
|---|---|---|---|---|---|
| committed, contended, rescaled | 17.340 s | 0.7011 | 9.611 s | 10.123 s | 105.3 % |
| quiet, median | 17.270 s | 1.0000 | 8.709 s | 10.123 s | **116.2 %** |
| quiet, trimmed per-call | 17.270 s | 1.0000 | 8.709 s | 10.123 s | **116.2 %** |

The crossing does not close. It widens from 105.3 % to 116.2 % once the rescale is gone, and the two
quiet estimators agree to the millisecond on the trunk because its 264 calls have only 4
contaminated ones and a tight body.

It cannot be closed by attribution either. The three disjoint top-level captures account for
15.980 s of the 17.270 s fold, leaving 1.290 s unattributed anywhere. Give every one of those
seconds to the trunk and it reaches 9.999 s, still short of its 10.123 s floor. The floor exceeds
the trunk's time under the most generous attribution the fold allows.

## Which rate is wrong

Three classes have a floor above their own measured time, but only one of them is a rate error.

| class | measured | floor, sum of per-op max | sum of arith | sum of traffic |
|---|---|---|---|---|
| TriangleAttention | 2.565 s | 4.553 s, 177.5 % | **4.125 s** | 1.338 s |
| TriangleMultiplication | 3.544 s | 4.440 s, 125.3 % | 3.206 s | 2.120 s |
| ConditionedTransitionBlock | 1.119 s | 1.175 s, 105.1 % | 0.906 s | 0.534 s |

For `TriangleAttention` the arithmetic sum alone, 4.125 s, already exceeds the 2.565 s the fold
demonstrably takes. No construction detail can explain that away: the catalogue rate is too low.

    TriangleAttention, 63.737 TFLOP a fold
      catalogue   15.45 TFLOP/s   14.7 % of the 104.93 TFLOP/s dense cube   -> 4.125 s
      the fold    24.85 TFLOP/s   23.7 % of cube                            -> 2.565 s
      underpriced by 1.61x

    dominated by   TriangleAttention fused SDPA (QK^T and AV)   38.483 TFLOP at 15.67 TFLOP/s, 14.9 %

The other two are not rate errors: for both, the arithmetic sum and the traffic sum each sit *below*
measured, and only the per-op `max(bytes/424.7 GB/s, FLOPs/class rate)` summed rises above it. That
construction charges every op the larger of its two terms, so a class mixing traffic-bound and
arithmetic-bound ops is charged more than either resource's own total. Across the whole fold the gap
between the two constructions is 3.071 s of the published 15.031 s floor.

## Corrected numbers

The floor itself does not move: it is built from bytes and FLOPs, both reproduced exactly, so
`floor_s` is 15.031 s in every arm, bit for bit. What the quiet times correct is the verdict on that
floor.

- The published 15.031 s floor is **not a floor**. It exceeds what the fold achieves for
  `TriangleAttention`, `TriangleMultiplication` and `ConditionedTransitionBlock`.
- Capping each class at the time the fold demonstrably achieves for it gives **12.090 s**, and that
  is an upper bound on the corrected floor, not the corrected floor itself -- a class may well go
  faster than it currently does.
- The prize is therefore **at least 5.180 s** of the 17.270 s fold, 30.0 %, against the 2.239 s that
  the published floor implied.

`roof-true-floor`'s bracket of [15.031, 15.544] s does not hold. The direction is not ambiguous, so
the bracket is not the right cap: the floor is below its low end.

## What to do with this

One note back into the catalogue, not a re-measurement: `roof-shape-honest-roofs`' `TriangleAttention`
rates are standalone-arm rates and the arm is 1.61x slower than the same work inside the fold. A rate
measured on an isolated op is a lower bound on what the fold delivers, not an upper bound, so it
cannot be used to build a floor. The same question is open for every other class in the catalogue;
only `TriangleAttention` is *proven* wrong here because only it crosses on the arithmetic sum alone.

## Reproducing

    perf/roof_quiet/refold.sh                     # CARD=0 for the run of record
    perf/roof_quiet/rejoin.py --attrib perf/roof_quiet/capture/attrib_quiet_512_qb2c0.json \
        --captures perf/roof_quiet/capture/captures --time-stat trimmed --tag quiet

`out_quiet_trimmed/` is the arm of record, `out_quiet_median/` the cross-check, `out_quiet_percall/`
the rejected estimator. Its `residual_census.py` step fails, and the failure is the estimator's own:
under `percall` the trunk `PairformerLayer`'s exclusive time comes out at -1.983 s, because the
one-call substitution removes 3.903 s of capture cost from the parent that its children keep. A
negative exclusive time is what `percall` is worth here.
