# The floor was 15.031 s because it priced TriangleAttention by a kernel the fold does not run

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

`TriangleMultiplication`, 123.9 %, is the same defect one class over and is the largest single
item left. Its in-fold capture holds four `ttnn.generic_op` calls beside one `ttnn.matmul` and one
`ttnn.linear`, and its catalogue arms are `trimul_in_flat` (an `ttnn.linear`) and `trimul_einsum`
(an `ttnn.matmul`). The einsum arm is a fair match for the `ttnn.matmul` the fold issues; the
in-projection arm, carrying 1.527 s of the corrected floor's arithmetic, is not. Measuring
`trimul`'s shipped generic ops the way this file measured TriangleAttention's is the next tranche,
and the method is already written -- `rate_ab.py` plus `merge_roofs.py` plus `rejoin.py --roofs`.

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
