# The shape-honest arithmetic roof, and which roof binds the 512 aa fold

> **Re-measured on the fold's own kernels and its own part,
> `perf/roof_triatt_rate/TRIATT_RATE.md`.** Both halves of the note above are superseded. The gap
> is not an isolated-arm effect: it is that these arms are `ttnn.linear` and
> `ttnn.transformer.scaled_dot_product_attention` while the fold issues three `ttnn.generic_op`
> kernels, and that the fractions here were taken on a pc p150a and carried onto a qb2 p300c. The
> 1.61x is 1.190x cross-host transfer x 1.941x wrong kernel / 1.208x an in-fold penalty. Measured
> alone on qb2 card 3, the shipped kernels run at 46.85 / 29.22 / 18.05 TFLOP/s against these
> 19.06 / 17.67 / 16.33. `weigh.py`'s three TriangleAttention classes now carry the shipped arms
> beside the stock ones; `shape_roofs_qb2c3_shipped.json` is the roofs file that has them.

> **`TriangleAttention`'s rates are not upper bounds.** Measured against the same work inside a quiet
> 512 aa fold (`perf/roof_quiet/QUIET_REFOLD.md`), the class delivers 24.85 TFLOP/s, 23.7 % of the
> dense cube, where these standalone arms give 15.45 TFLOP/s, 14.7 %. Underpriced 1.61x, dominated by
> `TriangleAttention fused SDPA (QK^T and AV)` at 15.67 TFLOP/s against 38.483 TFLOP a fold. A rate
> taken on an isolated op is a lower bound on what the fold delivers, so it cannot be used to build a
> floor. Only `TriangleAttention` is proven wrong so far, because only it crosses its own measured
> time on the arithmetic sum alone; the question is open for every other class here.


`perf/roof_budget/ROOF_BUDGET.md` put the floor at **6.934 s, set by bandwidth** (2.9449 TB at a
measured 424.7 GB/s) and priced the arithmetic term at 2.092 s by dividing 219.49 TFLOP by the
dense-cube rate. No matmul in this fold is a dense cube. This file measures the rate each matmul
class actually reaches at its own shape, and the answer changes which roof binds.

**The arithmetic floor is 11.13 s. It binds, and it is 1.61x above the traffic floor.**

## The table

Two parts, two sessions, two cubes measured in those same sessions. Each row is the class's own
matmuls at the fold's shapes with the residency the fold keeps and nothing that is not a matmul.
Where more than one residency or program config is plausible the harness carries all of them and
the table takes the **fastest**, so every fraction is an upper bound on the rate and the floor
below it is a lower bound on the floor.

pc p150a, 13x10, cube **128.70 TFLOP/s**, A/A floor 0.41 %:

| class | TFLOP/s | % of cube | TFLOP/fold | % of fold | s |
|---|---|---|---|---|---|
| TriangleAttention fused SDPA (QK^T and AV) | 19.22 | 14.9 % | 38.48 | 17.57 % | 2.002 |
| TriangleMultiplication in-proj `[g_a\|g_b\|p_a\|p_b\|g_out]` | 19.32 | 15.0 % | 24.05 | 10.98 % | 1.245 |
| DiffusionTransformer token linear 768x768 | 25.67 | 19.9 % | 23.31 | 10.64 % | 0.908 |
| TriangleAttention in-proj `[q\|k\|v\|g\|bias]` | 19.06 | 14.8 % | 20.44 | 9.33 % | 1.073 |
| TriangleMultiplication triangle product | 17.81 | 13.8 % | 19.24 | 8.78 % | 1.080 |
| pair Transition fc1 / fc2 | 38.52 | 29.9 % | 19.24 | 8.78 % | 0.500 |
| DiffusionTransformer token linear 768x1536 | 42.44 | 33.0 % | 18.36 | 8.38 % | 0.433 |
| DiffusionTransformer token linear 768x3072 | 53.10 | 41.3 % | 11.60 | 5.29 % | 0.218 |
| pair Transition fc3 | 30.07 | 23.4 % | 9.62 | 4.39 % | 0.320 |
| OuterProductMean depth contraction | 98.27 | 76.4 % | 8.80 | 4.02 % | 0.090 |
| DiffusionTransformer token linear 1536x768 | 29.01 | 22.5 % | 6.28 | 2.87 % | 0.217 |
| TriangleMultiplication out-proj | 16.67 | 13.0 % | 4.81 | 2.20 % | 0.289 |
| TriangleAttention out-proj | 16.67 | 13.0 % | 4.81 | 2.20 % | 0.289 |
| PairWeightedAveraging | 42.81 | 33.3 % | 2.20 | 1.00 % | 0.051 |
| atom transformer linear | 17.11 | 13.3 % | 1.41 | 0.64 % | 0.082 |

Coverage 212.66 of the census's 219.06 matmul TFLOP, **97.1 %**. The uncovered 2.9 % is assumed to
run at the weighted mean; at any rate between 5 % and 100 % of cube it moves the floor by under
0.3 s.

whglx Wormhole Galaxy card 1, 8x9, cube **54.28 TFLOP/s**, A/A floor 0.13 %: same construction,
same census, **19.4 %** weighted against Blackhole's **18.8 %**. Two architectures, two cubes, the
same answer to within 0.6 points. The denominator error is architectural.

## The one number

A floor is a sum of times, so the aggregate is the FLOP-weighted **harmonic** mean of the
fractions, not the arithmetic one. The arithmetic mean is 23.0 % and it is 1.22x too optimistic.

    FLOP-weighted mean fraction of cube   18.8 %
    arithmetic floor = 219.49 TFLOP / (104.93 TFLOP/s x 0.188) = 11.134 s
    traffic floor                                              =  6.934 s

104.93 TFLOP/s is the budget's own qb2 p300c cube, so the floor is stated on the part the floor of
record lives on. The fractions come from a p150a; the two Blackhole parts differ by 1.23x on the
cube and 1.18x on core count, so the per-core rate is within 4 % and the fractions carry.

## Why this is not the traffic floor wearing a different hat

Every arm moves bytes, so a rate limited by DRAM would show up here as a low fraction of cube and
be double-counted. It does not. The same session's starved 8192^2 add measures the DRAM roof at
**415.8 GB/s** on the p150a (**236.1 GB/s** on the Wormhole part), and the arms that carry the
floor sit well under it:

| arm | GB/s | % of the DRAM roof |
|---|---|---|
| TriangleAttention SDPA | 75.1 | 18.1 % |
| TriangleMultiplication triangle product | 103.4 | 24.9 % |
| TriangleMultiplication in-proj | 178.6 | 43.0 % |
| TriangleAttention in-proj | 183.0 | 44.0 % |
| the two 128->128 out-projections | 251.7 | 60.5 % |

Only the thin 128->128 output projection, 4.4 % of the fold's FLOPs, is meaningfully
bandwidth-shaped. The rest is the MAC array failing to fill at a 4-tile K.

## How hard is it to break the verdict

    as measured                                          11.134 s
    SDPA's pure-matmul rate twice its fused arm           9.867 s
    SDPA at the full dense cube rate                      8.978 s
    every class 1.5x faster than measured                 7.423 s
    break-even: every class 1.606x faster                 6.934 s

The SDPA row is the one place the construction is not pure: the fold's QK^T and AV live inside one
fused kernel with an online softmax that cannot be taken out without changing what it computes, so
that row is a fused-kernel ceiling rather than a matmul-only one. Removing its softmax cost
entirely still leaves the arithmetic floor at 8.98 s, above the traffic floor. **Nothing short of a
1.61x uniform speedup on every matmul class in the fold puts bandwidth back in charge.**

## What reproduces and what to correct

`trans_mm3only` reproduces `roof-pair-transition`'s Blackhole reading: 22.9 % of cube here against
20.9 % there, both at the shipped h = 16. Their headline 28.0 % was `full_h48_mm3only`, a row
blocking the fold does not ship. Splitting the unit into its two shapes and dropping the chunk
(which allocates, and is not a matmul) gives 29.9 % and 23.4 %, a combined 27.4 % -- so 28 % is
right for the pair Transition and it is the third-best class in the fold, not a typical one.

Cross-session reproducibility, same part, a separate process 20 minutes later: cube within 1.1 %,
every re-run arm within 1.4 % on Blackhole and 0.5 % on Wormhole.

Every Phase A brief saying "the fold is bandwidth-bound" is wrong. It is arithmetic-bound at these
shapes, and the binding term is triangle attention and triangle multiplication, 61 % of the floor
between them.

## Reproducing

    perf/roof_shape/shape_roofs.py    # the arms; one process, one device, interleaved, min over blocks
    perf/roof_shape/weigh.py          # joins the rates to perf/roof_budget's shape census

    python3 shape_roofs.py --blocks 6 --out shape_roofs_pc_bh.json
    python3 weigh.py --roofs shape_roofs_pc_bh.json

The Blackhole session ran on pc's p150a, which is the card
`pc-card0-512aa-fold-nondeterminism` records as computing some matmuls wrong. This row reads
times, never values, so the fault does not reach it; the Wormhole arm is on a known-good part and
agrees to 0.6 points.
