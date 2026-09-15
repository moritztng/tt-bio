# The floor was 15.031 s because it priced TriangleAttention by a kernel the fold does not run


> **A third term, `perf/roof_launch/LAUNCH_FLOOR.md`.** The floor below takes
> `max(traffic, arithmetic)` per op and charges nothing for launching it. Measured per (class,
> shape, K) on the part of record under trace replay, the per-op launch floor adds **0.487 s**:
> the floor is **13.193 s** and the prize **4.077 s**, the fold at **76.4 %** of roof. The
> registered prediction of 15.5-17.0 s and 90-100 % is **refuted**, and the campaign's
> "353,384 small calls bind" premise with it: 150,160 of the 465,664 calls are host metadata or
> wrappers whose device child is already counted, and only 105,728 have launch as their binding
> term. Bytes, FLOPs and rates are unchanged. The more useful number from that row is the
> over-reading beside the floor: priced at what each op costs run **alone** on a quiet card the
> total is 17.821 s against a 17.270 s fold, **103.2 %**, so the remaining 4.077 s is in the
> kernels and not in dispatch slack.

`roof-quiet-attrib-refold` re-took the 512 aa attribution fold on a quiet card and found the trunk
pairformer's floor 16.2 % ABOVE its measured time. A floor above measured time is a defect, and it
put the blame on `TriangleAttention`'s catalogue rate: 15.45 TFLOP/s, against 24.85 TFLOP/s for the
same work inside the fold. Its reading of why was that a rate measured on an isolated op is a lower
bound on what the fold reaches when that op is embedded.

That reading is wrong, and the real cause is duller and larger.

**The fold's TriangleAttention issues no matmul at all.** Its capture holds three `ttnn.generic_op`
calls: `triatt_qkv.qkvgb_heads`, the fused SDPA in `tt_bio/kernels/triatt_sdpa/`, and
`triatt_qkv.out_proj`. `perf/roof_shape/`'s three arms for the class are `ttnn.linear`,
`ttnn.transformer.scaled_dot_product_attention` and `ttnn.linear`. The catalogue priced a different
implementation of the same arithmetic.

## Both implementations, one session, both alone

qb2 card 3 (p300c, 11x10), cube 108.54 TFLOP/s, 9 reps after 3 warmup, arms interleaved per rep,
minimum over reps, A/A floor 0.65 %. `perf/roof_triatt_rate/rate_ab.py`.

| | catalogue arm | shipped kernel | |
|---|---|---|---|
| fused SDPA, QK^T and AV | 19.06 TFLOP/s | **46.85** | 2.46x |
| in-projection `[q\|k\|v\|g\|bias]` | 17.67 | **29.22** | 1.65x |
| output projection 128->128 | 16.33 | **18.05** | 1.11x |
| the three, per call | 6.197 ms | **3.192 ms** | 1.94x |

Nothing is embedded in either column. The shipped kernels win standing alone.

The whole shipped unit standalone is 3.787 ms where the fold spends 4.574 ms on the same call, so
the isolated arm is 1.21x FASTER than the fold, not slower. The effect the refold proposed has the
opposite sign to the one it needed.

## The 1.61x, decomposed

    published floor, per call                                            7.366 ms
      / 1.190   the same stock arms measured on qb2 instead of carried from a pc p150a
                                                                         6.197 ms
      / 1.941   the shipped kernels instead of the stock ops             3.192 ms
      + 0.595   layer norm, permutes, the gate multiply, the bias projection -- not matmul
      the whole shipped unit, alone                                      3.787 ms
      x 1.208   what the fold pays over the same unit alone              4.574 ms

    7.366 / 4.574 = 1.610x

The cross-host term is the third candidate `roof-quiet-attrib-refold` named and did not chase. It
is real and it is 1.19x on this class, against `ROOF_SHAPE.md`'s argument that a 4 % per-core
difference would let the fractions carry.

## What it does to the floor

`perf/roof_shape/weigh.py`'s three TriangleAttention classes gained the shipped arm alongside the
stock ones; `class_rates` already takes the max, so a roofs file without those arms is untouched.
**Control: the committed chain re-run with the class table edited reproduces
`out_quiet_trimmed/` byte for byte, all five outputs.** Then the same chain on
`shape_roofs_qb2c3_shipped.json`, which carries the six arms measured here natively and every other
arm at its own pc fraction:

| | published | corrected |
|---|---|---|
| floor | 15.031 s | **12.706 s** |
| prize against the 17.270 s fold | 2.239 s | **4.564 s** |
| trunk pairformer, % of its own floor | 116.2 % | **91.1 %** |
| TriangleAttention, % of its measured time | 177.5 % | **88.8 %** |
| conservative floor, every class capped at measured | 12.090 s | 11.802 s |
| fold TB / matmul TFLOP | 2.8589 / 219.058 | 2.8589 / 219.058 |

Bytes and FLOPs do not move, which is the control: only rates changed. The crossing that the whole
`roof-true-floor` -> `roof-quiet-attrib-refold` chain existed to explain is closed. The trunk
pairformer sits below its floor at 91.1 %, and TriangleAttention is a floor again rather than a
number above the thing it is supposed to bound.

## The lever that was already there, and is already dead

`TT_BIO_TRIATT_FUSE_QKV` folds the qkv projection into the SDPA kernel so the projection program
never runs. `roof-qkv-sdpa-build` built it, merged it, measured 1.4982x / 1.5502x on the pair, and
left it off by default. At this site it declines every call with `qkv_already_fused_with_gate`: 48
of 48, because `_fused_qkvg` already folds the gate in with q, k and v and runs first. Measured
1.0097x on the unit against a 0.62 % A/A floor, which is the flag doing nothing. Turning it on is
not a lever here.

## What closes the rest

The corrected prize is 4.564 s and two classes still price above their own measured time.

`TriangleMultiplication`, 123.9 %, looked like the same defect one class over: its in-fold capture
holds four `ttnn.generic_op` calls beside one `ttnn.matmul` and one `ttnn.linear`, and two of its
three catalogue arms are stock ops. **Measured, it is not.**
`perf/roof_triatt_rate/trimul_rate_ab.py` runs the shipped unit against its three catalogue arms
the same way, and the shipped unit comes out at or above the arms rather than far below:

| | catalogue arm sum | shipped unit | ratio |
|---|---|---|---|
| TriangleAttention | 6.197 ms/call | 3.787 ms | **0.61** |
| TriangleMultiplication, run 1 | 6.684 ms/call | 7.548 ms | 1.13 |
| TriangleMultiplication, run 2 | 6.828 ms/call | 7.001 ms | 1.03 |

Both trimul runs were taken on a busier box than the TriangleAttention session -- A/A floor 9.2 %
and 7.0 %, and the session cube came in at 1.495 and 2.033 ms against 1.266 -- so the absolute
milliseconds are not publishable. The ratio is, because both terms are normalised by the same
session: it is 1.0 within noise, where TriangleAttention's is 0.61. Its arms already price its
shipped kernels. So its 123.9 % is the `sum(max(...))` construction `QUIET_REFOLD.md` identified
and not a rate error, and re-rating it does not close any of the prize.

That leaves the corrected 4.564 s prize with no second mis-rated class behind it. What is left is
what the budget table already says: the fold is bandwidth-shaped almost everywhere, the trunk
pairformer is at 91.1 % of a floor that is now honest, and the remaining distance is real work
rather than a modelling error. The next lever has to come from moving bytes or deleting ops, not
from re-pricing.

`ConditionedTransitionBlock`, 105.0 %, is not a rate error. Its arithmetic sum and its traffic sum
each sit below measured and only the per-op `sum(max(...))` construction rises above, which is the
modelling artifact `QUIET_REFOLD.md` already identified. Leave it.

The 1.208x the fold pays over the same unit run alone is 0.44 s across the class and is the one
part of this that is a real in-fold effect rather than a mismeasurement. It is not attributed here.

## Cross-model

`TriangleAttention` is shared pairformer infrastructure. Boltz-2, Protenix-v2, OpenFold3,
OpenBind-0 and RFD3 all reach these same three kernels, so the corrected rate is the rate for all
of them and no model gets a private number. Nothing in this file changes engine behaviour: it
changes what the roofline believes about kernels that already shipped.

## Reproducing

    TT_VISIBLE_DEVICES=3 python3 perf/roof_triatt_rate/rate_ab.py
    python3 perf/roof_triatt_rate/merge_roofs.py
    python3 perf/roof_quiet/rejoin.py \
        --attrib perf/roof_quiet/capture/attrib_quiet_512_qb2c0.json \
        --captures perf/roof_quiet/capture/captures --time-stat trimmed \
        --tag shipped --roofs shape_roofs_qb2c3_shipped.json

Dropping `--roofs` is the control and reproduces `out_quiet_trimmed/` exactly.
