# The max, taken per op: the 512 aa fold's one true floor is 15.031 s


> **A third term, `perf/roof_launch/LAUNCH_FLOOR.md`.** The floor below takes
> `max(traffic, arithmetic)` per op and charges nothing for launching it. Measured per (class,
> shape, K) on the part of record under trace replay, the per-op launch floor adds **0.594 s**:
> the floor is **13.300 s** and the prize **3.970 s**, the fold at **77.0 %** of roof. The
> registered prediction of 15.5-17.0 s and 90-100 % is **refuted**, and the campaign's
> "353,384 small calls bind" premise with it: 150,160 of the 465,664 calls are host metadata or
> wrappers whose device child is already counted, and only 103,672 have launch as their binding
> term. Bytes, FLOPs and rates are unchanged. The more useful number from that row is the
> over-reading beside the floor: priced at what each op costs run **alone** on a quiet card the
> total is 17.103 s against a 17.270 s fold, **99.0 %**, so the remaining 3.970 s is in the
> kernels and not in dispatch slack.

> **Re-priced on the kernels the fold runs, `perf/roof_triatt_rate/TRIATT_RATE.md`.** The
> `TriangleAttention` rate above is a stock-op rate: the arms are `ttnn.linear` and
> `ttnn.transformer.scaled_dot_product_attention`, and the fold issues three `ttnn.generic_op`
> kernels instead, each faster than the stock op it was priced by (2.46x, 1.65x, 1.11x, measured
> alone in one session). Not an isolated-arm effect -- the shipped unit run ALONE is 1.21x faster
> than the fold, the opposite sign. With those rates the floor is **12.706 s** and the prize
> **4.564 s**, and the crossing this page's correction note is about is closed: the trunk
> pairformer sits at **91.1 %** of its floor, `TriangleAttention` at 88.8 % of its measured time.
> Bytes and FLOPs are unchanged. `TriangleMultiplication`, still 123.9 %, is the same defect one
> class over and is the next tranche.

> **Corrected by the quiet re-capture, `perf/roof_quiet/QUIET_REFOLD.md`.** 15.031 s is not a floor.
> The per-unit times it is judged against were rescaled off a loadavg-27 session by one scalar,
> 0.7011. Re-taken quiet, with the fold's own wall as the cell and no rescale, the trunk pairformer
> goes from 105.3 % of its floor to **116.2 %**, and it stays above even if all 1.290 s of
> unattributed fold time is given to it. Three classes come out above the time the fold demonstrably
> takes for them. `TriangleAttention` is a genuine rate error, its arithmetic sum alone is 1.61x
> measured; the other two are the `sum(max)` construction this page adopts, worth 3.071 s of the
> 15.031 s. Capping every class at what the fold achieves gives **12.090 s**, an upper bound on the
> corrected floor, so the prize is **at least 5.180 s**, not 2.309 s. The bracket [15.031, 15.544]
> does not hold; the floor is below its low end.

A roofline floor is `max(traffic, compute)` per unit, summed. The campaign measured both terms to a
high standard and never combined them. The floor of record is an arithmetic aggregate, 219.49 TFLOP
at a FLOP-weighted harmonic 18.8 % of the dense cube, **11.134 s**, with a traffic aggregate beside
it, 2.8589 TB at 424.7 GB/s, **6.732 s**. Neither is a roofline.

Taking the max per top-level ttnn op and summing:

    TRUE FLOOR   15.031 s      bracket [15.031, 15.544]
    PRIZE        17.340 - 15.031 = 2.309 s      (the number of record was 6.206 s)

`max(sum) <= sum(max)` by construction and the gap is large: **3.897 s above the arithmetic
aggregate, 8.299 s above the traffic one.** 63 % of the campaign's headline prize was an artifact
of never taking the max.

## What it is taken at

Per top-level ttnn op, inside the three disjoint top-level captures, summed over their calls:
264 `PairformerLayer` + 16 `MSALayer` + 200 `DiffusionModule` is the whole fold. That is the same
op set `shape_census.py` counts FLOPs over and `roof_arb` counts bytes over, so the floor sits on
exactly the byte total and exactly the FLOP census the campaign settled.

    t_traffic = corrected bytes / 424.7 GB/s
    t_arith   = sum over the op's census shape rows of FLOPs_row / that row's class rate
    t         = max(t_traffic, t_arith)

Four instruments, two join keys, no averaging.

| what | file | key |
|---|---|---|
| op list, parent/child ownership | `perf/roof_residual/split_units.py` | capture + op index |
| bytes, L1+PRE+GATE corrections on | `perf/roof_arb/corrected_traffic.py` | capture + op index |
| FLOPs and op kind | `perf/roof_budget/exec_flops.py` | capture + op index |
| shape-honest matmul rates | `perf/roof_shape/shape_roofs_pc_bh.json` | `(batch, M, K, N)` |

The first three run `itemize()` over the same capture, so "capture + op index" is one key, not a
correspondence someone invented. The fourth joins on the census shape tuple `shape_census.py`
builds, which is already the key `weigh.py` uses; this file takes it per op rather than per fold.
Class rates are `roof-shape`'s fraction of ITS cube (pc p150a, 128.70 TFLOP/s) applied to the fold's
own cube (qb2 p300c, 104.93 TFLOP/s), which is what `weigh.py` does with the aggregate, and every
class rate is the fastest arm measured for it, so every rate is an upper bound and the floor under
it is a lower bound.

## The split, which is the answer to "does arithmetic still bind"

Yes, but nothing like the aggregate said.

| | seconds | share | op calls |
|---|---|---|---|
| floor set by arithmetic | 10.808 | 71.9 % | 99 528 |
| floor set by traffic | 4.224 | 28.1 % | 366 136 |
| **true floor** | **15.031** | | 465 664 |

Arithmetic binds 21 % of the op calls and 72 % of the floor. The other 28 % is time the 11.134 s
figure does not contain at all, so the traffic share of the true floor is not 0 as the arithmetic
verdict implied, and it is not 45 % as `6.732 / 15.031` would suggest either. Taking the max per op
is what tells them apart.

| kind | op calls/fold | TB | TFLOP | s traffic | s arith | s MAX |
|---|---|---|---|---|---|---|
| matmul | 112 280 | 1.1549 | 219.06 | 2.719 | 10.868 | **11.019** |
| layout / free | 281 808 | 1.2500 | 0.00 | 2.943 | 0.000 | **2.943** |
| eltwise | 71 576 | 0.4541 | 0.11 | 1.069 | 0.000 | **1.069** |

## The number nobody had: 4.012 s of irreducible non-matmul work

**4.012 s of the 15.031 s floor, 26.7 % of it, is ops that are not matmuls.** 353 384 op calls,
1.7040 TB, 0.108 TFLOP. Every one of them is traffic bound and no matmul lever reaches any of it.
It is absent from the 11.134 s arithmetic floor as a term, and it is the part of the fold that a
faster MAC array cannot touch.

| ttnn op | calls/fold | TB | s |
|---|---|---|---|
| `multiply_` | 42 720 | 0.3268 | 0.770 |
| `generic_op` (reblock/permute, layout only) | 1 680 | 0.3006 | 0.708 |
| `layer_norm` | 35 304 | 0.2823 | 0.665 |
| `add_` | 14 616 | 0.2718 | 0.640 |
| `permute` | 12 432 | 0.1088 | 0.256 |
| `transformer.scaled_dot_product_attention` | 6 000 | 0.0797 | 0.188 |
| `experimental.nlp_create_qkv_heads` | 6 264 | 0.0641 | 0.151 |
| everything else, 22 op names | 234 368 | 0.2699 | 0.635 |

`roof-residual-census` found the trunk pairformer's 7 residual `add_` and 1 `layer_norm` running at
86.3 % of the 424.7 GB/s stream roof and called them not waste. This is the whole class of that,
priced.

**The rule for ops with no matmul class is the traffic term, and it is settled rather than
assumed.** These ops carry 0.108 TFLOP between all 353 384 of them, an arithmetic intensity under
1 FLOP/byte against a machine balance of 247.1. Pricing their FLOPs anyway at the slowest measured
matmul class in the fold, 13.59 TFLOP/s, moves the floor by 0.000 s. At 0.5 TFLOP/s, 0.5 % of the
dense cube, it moves by 0.003 s. Nothing about the floor rests on that choice.

## The two things that are bounded rather than known

**22 matmul shapes, 6.398 TFLOP, 2.92 % of the census, have no measured class rate.** They are
priced at the fastest measured class, 80.12 TFLOP/s, which keeps the number a floor. At the slowest
measured class, 13.59 TFLOP/s, they add 0.471 s instead of 0.080 s. Floor bracket **[15.031,
15.307] s**. Never the cube rate, which is the error this campaign has been unwinding.

**`ttnn.transformer.scaled_dot_product_attention` carries 5.506 TFLOP/fold that nobody has
counted.** `exec_flops.py` has a rule for the hand-written `generic_op` SDPA and none for the stock
ttnn one, so it falls through to "one FLOP per output element" and the fold's token attention
arithmetic is missing from the 219.058 TFLOP census, from the 219.49 TFLOP of record, and from the
11.134 s floor. Bounded, not assumed:

| priced at | TFLOP/s | floor |
|---|---|---|
| fastest measured class | 80.12 | 15.031 s (+0.000, traffic still binds) |
| the fused triangle SDPA class, the closest analogue | 15.67 | 15.218 s (+0.187) |
| slowest measured class | 13.59 | 15.268 s (+0.237) |

Both bounds together put the floor in **[15.031, 15.544] s** and the prize in **[1.796, 2.309] s**.

## Calibration

| | this file | of record | gap |
|---|---|---|---|
| bytes | 2.8589 TB | 2.8589 TB (`roof-byte-arbitration`) | +0.002 % |
| matmul FLOPs | 219.058 TFLOP | 219.058 TFLOP (`shape_census.json`) | +0.000 % |
| effective covered rate | 212.66 TFLOP in 10.788 s = 19.71 TFLOP/s = 18.79 % of the cube | 18.8 % (`roof-shape`) | 0.01 pt |
| op set as time | 24.644 s of session fold, 17.278 s at the cell | 17.340 s cell of record | -0.36 % |
| floor above both aggregates | 15.031 s | 11.134 s and 6.732 s | yes, by construction |

The arithmetic-only aggregate comes out at 10.868 s here against the published 11.134 s, and the
difference is one convention, not one instrument: the published figure divides the whole padded
219.49 TFLOP by the harmonic mean rate, which charges the uncovered 2.9 % at 19.71 TFLOP/s (+0.325 s)
where this file charges it at 80.12 TFLOP/s (+0.080 s). Put the uncovered FLOPs at the covered
effective rate and this file returns 11.113 s, 0.19 % from the number of record.

Second reading, the 26-capture per-unit own-work partition instead of the three top-level captures:
2.9366 TB (+2.72 %) and a floor of 15.124 s (+0.62 %). The byte gap is a known property of the
capture set, not a disagreement: a standalone capture re-reads at its own edge what the enclosing
capture dedupes on buffer address. The `TriangleAttention` region inside the trunk pairformer reads
2030.453 MB per layer where two standalone captures of the same unit read 2231.812 MB. The
three-capture reading is the one that reproduces the settled byte total, so it is the headline.

## Where the floor and the measured time cross, and why the prize has a second bracket

A floor cannot exceed the time the work actually takes. At class granularity, four do.

| class | measured own work at the cell | floor | of which arithmetic | % |
|---|---|---|---|---|
| TriangleAttention | 2.945 s | 4.553 s | 4.125 s | **155 %** |
| TriangleMultiplication | 3.839 s | 4.440 s | 3.206 s | **116 %** |
| ConditionedTransitionBlock | 0.806 s | 1.175 s | 0.906 s | **146 %** |
| MSALayer own work | 0.022 s | 0.030 s | 0.000 s | **139 %** |
| AttentionPairBias | 1.724 s | 1.360 s | 0.661 s | 79 % |
| AdaLN | 0.926 s | 0.807 s | 0.554 s | 87 % |
| PairformerLayer own work | 0.517 s | 0.445 s | 0.000 s | 86 % |
| PairWeightedAveraging | 0.515 s | 0.313 s | 0.073 s | 61 % |
| OuterProductMean | 0.625 s | 0.360 s | 0.124 s | 58 % |
| Transition | 2.765 s | 1.266 s | 1.070 s | 46 % |
| DiffusionTransformerLayer | 0.578 s | 0.261 s | 0.141 s | 45 % |
| Diffusion own work | 0.882 s | 0.019 s | 0.007 s | 2 % |
| DiffusionModule / DiffusionTransformer own work | 1.135 s | 0.000 s | 0.000 s | 0 % |

It is not the class split doing it. At the coarsest granularity there is, the three top-level
captures, with no parent/child heuristic anywhere:

| capture | calls | measured at the cell | floor | % |
|---|---|---|---|---|
| `PairformerLayer\|1x512x384,1x512x512x128` | 264 | 9.611 s | 10.123 s | **105.3 %** |
| `MSALayer\|1x512x512x128,1x1024x512x64` | 16 | 1.900 s | 1.334 s | 70.2 % |
| `DiffusionModule\|` | 200 | 5.767 s | 3.574 s | 62.0 % |

The trunk pairformer's floor is 5.3 % above the time it is recorded as taking. Two candidates, and
this row cannot separate them without a device:

* **The measured column.** The per-unit times come from a bracketed fold of 24.731 s taken under
  loadavg 28 and scaled to the 17.340 s quiet cell by a single factor, 0.7011. Against that
  session's own 13.708 s the pairformer's floor is 74 %, and the crossing disappears entirely. A
  uniform scale from a contended session onto a quiet cell is not defensible at 5 % resolution, and
  it is the only place a per-unit number depends on it.
* **The class rates.** They are p150a micro-benchmarks transferred to a p300c fold by fraction of
  cube. The fold runs its triangle attention at 21.6 TFLOP/s, 20.6 % of the 104.93 TFLOP/s cube,
  where the arms for the same three matmuls measure a FLOP-weighted 14.7 %. `roof-shape` presented
  its rates as upper bounds because each takes the fastest of several arms; for this class the fold
  beats all of them by 1.40x, so for triangle attention they are not upper bounds. One of the SDPA
  arms, `triatt_sdpa_q512`, refused with a `TT_THROW` and was never measured.

So the floor carries a second bracket, on the rates rather than on the arithmetic. Capping every
class at the time it is recorded as taking gives a conservative floor of **12.443 s** and a prize of
**4.897 s**. The honest statement of the prize is therefore:

    2.309 s   at the join as specified
    4.897 s   with every class capped at its own measured time
    6.206 s   the number of record, which is above both

Closing this needs one thing and it is cheap: the attrib fold re-taken quiet, with per-call times
kept, so the measured column stops depending on a single scale factor. `roof-residual-census` owes
the same fold for the same reason. One 25 s fold settles both.

## Reproducing

    git archive origin/wk/roof-budget            perf/roof_budget perf/b2x_difflayer | tar -x -C .scratch
    git archive origin/wk/roof-residual-census   perf/roof_residual | tar -x -C .scratch
    git archive origin/wk/roof-shape-honest-roofs perf/roof_shape   | tar -x -C .scratch
    git archive origin/wk/roof-byte-arbitration  perf/roof_arb      | tar -x -C .scratch
    python3 perf/roof_true/true_floor.py --perf .scratch/perf

No device. Committed captures and committed scripts only. The 424.7 GB/s streaming roof and the
104.93 TFLOP/s dense-cube roof are `roof-budget`'s, measured in one session on qb2 card 2; the
shape-honest fractions are `roof-shape`'s, measured on pc's p150a against its own 128.70 TFLOP/s
cube. Nothing was re-measured here.
