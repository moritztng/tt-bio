# Every op of the 512 aa fold, and how many of the 110 cores it gets

`ws:util-grid-coverage`. Blackhole, 11x10 = 110 worker cores. The core counts and device
times are read out of the shipped fold: the profiler captures committed by
`b2z-kernel-cycle-census` on qb2 card 0 (tt-metal v0.68.0 source build, `ENABLE_TRACY=ON`).
The scaling curves are measured separately on qb2 card 2 with the same build.

## Headline

- **pairformer block: 103.8 of 110 cores**, weighted by device kernel time (94.4 %). 272 programs, 35.8397 ms of kernel in a 36.2456 ms span (1.1 % of the span is in no kernel at all).
- **diffusion step: 83.3 of 110 cores**, weighted by device kernel time (75.7 %). 1066 programs, 22.0152 ms of kernel in a 45.5048 ms span (51.6 % of the span is in no kernel at all).

## The census

### pairformer block — 264 calls/fold, 9.462 s/fold of kernel time

7.4178 ms of the 35.8397 ms runs on a partial grid (20.7 %).

| op | cores | progs | kernel ms | % | s/fold | out tiles BxMxNxK | why not 110 |
|---|---:|---:|---:|---:|---:|---|---|
| Matmul | 86 | 64 | 4.4598 | 12.44 | 1.177 | 16x16x16x4 | 1D split over M only: 3 tile-rows/core is the smallest block that fits the 86 cores it needs; a 111th core would have no row to take |
| Matmul | 64 | 2 | 1.5990 | 4.46 | 0.422 | 128x16x16x16 | 2D split 2x2 tiles/core; the output is 16x16 tiles and no finer rectangle fits an 11x10 grid |
| Matmul | 86 | 32 | 1.1421 | 3.19 | 0.302 | 16x16x4x16 | 1D split over M only: 3 tile-rows/core is the smallest block that fits the 86 cores it needs; a 111th core would have no row to take |
| Matmul | 32 | 1 | 0.0436 | 0.12 | 0.012 | 16x16x16x1 | 2D split 8x16 tiles/core; the output is 16x16 tiles and no finer rectangle fits an 11x10 grid |
| Matmul | 32 | 1 | 0.0359 | 0.10 | 0.009 | 16x16x1x16 | 2D split 8x1 tiles/core; the output is 16x1 tiles and no finer rectangle fits an 11x10 grid |
| LayerNorm | 16 | 2 | 0.0233 | 0.06 | 0.006 | 1x16x12x12 | row-parallel kernel: 16 tile rows exist, so 16 cores |
| NlpCreateHeads | 16 | 1 | 0.0221 | 0.06 | 0.006 | 16x16x1x48 | one core per head, 16 heads |
| Matmul | 80 | 3 | 0.0599 | 0.17 | 0.016 | 1x16x48x12 | 2D split 2x5 tiles/core; the output is 16x48 tiles and no finer rectangle fits an 11x10 grid |
| Matmul | 48 | 1 | 0.0170 | 0.05 | 0.004 | 1x16x12x48 | 2D split 2x2 tiles/core; the output is 16x12 tiles and no finer rectangle fits an 11x10 grid |
| Matmul | 48 | 2 | 0.0152 | 0.04 | 0.004 | 1x16x12x12 | 2D split 2x2 tiles/core; the output is 16x12 tiles and no finer rectangle fits an 11x10 grid |
| LayerNorm | 110 | 7 | 2.8815 | 8.04 | 0.761 | 512x16x4x4 | full grid |
| ReshapeView | 110 | 2 | 0.4993 | 1.39 | 0.132 | 512x16x1x16 | full grid |
| GenericOp | 110 | 2 | 2.3630 | 6.59 | 0.624 | 512x16x16x4 | full grid |
| GenericOp | 110 | 4 | 3.9612 | 11.05 | 1.046 | 128x16x16x16 | full grid |
| Transpose | 110 | 2 | 0.6941 | 1.94 | 0.183 | 128x16x16x16 | full grid |
| BinaryNg | 110 | 2 | 0.9005 | 2.51 | 0.238 | 128x16x16x16 | full grid |
| GenericOp | 110 | 2 | 0.9870 | 2.75 | 0.261 | 512x16x4x16 | full grid |
| Matmul | 110 | 4 | 1.1576 | 3.23 | 0.306 | 512x16x4x4 | full grid |
| BinaryNg | 110 | 7 | 2.9300 | 8.18 | 0.774 | 512x16x4x4 | full grid |
| Matmul | 110 | 3 | 0.6817 | 1.90 | 0.180 | 512x16x1x4 | full grid |
| GenericOp | 110 | 2 | 1.7177 | 4.79 | 0.453 | 2048x16x1x4 | full grid |
| GenericOp | 110 | 2 | 0.8197 | 2.29 | 0.216 | 2048x16x1x4 | full grid |
| GenericOp | 110 | 2 | 2.6213 | 7.31 | 0.692 | 2048x16x1x1 | full grid |
| BinaryNg | 110 | 2 | 1.0188 | 2.84 | 0.269 | 2048x16x1x1 | full grid |
| GenericOp | 110 | 2 | 0.8274 | 2.31 | 0.218 | 512x16x4x1 | full grid |
| Transpose | 110 | 2 | 1.4596 | 4.07 | 0.385 | 512x16x4x4 | full grid |
| Slice | 110 | 32 | 0.3853 | 1.08 | 0.102 | 16x16x4x4 | full grid |
| LayerNorm | 110 | 32 | 0.7464 | 2.08 | 0.197 | 16x16x4x4 | full grid |
| BinaryNg | 110 | 34 | 0.9221 | 2.57 | 0.243 | 16x16x16x16 | full grid |
| Concat | 110 | 1 | 0.3476 | 0.97 | 0.092 | 512x16x4x4 | full grid |

### diffusion step — 200 calls/fold, 4.403 s/fold of kernel time

11.6907 ms of the 22.0152 ms runs on a partial grid (53.1 %).

| op | cores | progs | kernel ms | % | s/fold | out tiles BxMxNxK | why not 110 |
|---|---:|---:|---:|---:|---:|---|---|
| Matmul | 64 | 193 | 3.8771 | 17.61 | 0.775 | 1x16x24x24 | 2D split 2x3 tiles/core; the output is 16x24 tiles and no finer rectangle fits an 11x10 grid |
| NlpCreateHeads | 16 | 24 | 1.0213 | 4.64 | 0.204 | 16x16x2x96 | one core per head, 16 heads |
| LayerNorm | 16 | 52 | 0.8874 | 4.03 | 0.177 | 1x16x24x24 | row-parallel kernel: 16 tile rows exist, so 16 cores |
| LayerNorm | 16 | 48 | 0.7057 | 3.21 | 0.141 | 1x16x24x24 | row-parallel kernel: 16 tile rows exist, so 16 cores |
| Matmul | 80 | 76 | 1.6875 | 7.67 | 0.338 | 1x16x48x24 | 2D split 2x5 tiles/core; the output is 16x48 tiles and no finer rectangle fits an 11x10 grid |
| Matmul | 64 | 26 | 0.5959 | 2.71 | 0.119 | 1x16x24x48 | 2D split 2x3 tiles/core; the output is 16x24 tiles and no finer rectangle fits an 11x10 grid |
| Matmul | 88 | 24 | 0.9505 | 4.32 | 0.190 | 1x16x96x24 | 2D split 2x9 tiles/core; the output is 16x96 tiles and no finer rectangle fits an 11x10 grid |
| Matmul | 70 | 18 | 0.4422 | 2.01 | 0.088 | 140x1x8x4 | 1D split over M only: 2 tile-rows/core is the smallest block that fits the 70 cores it needs; a 111th core would have no row to take |
| Matmul | 70 | 24 | 0.3839 | 1.74 | 0.077 | 140x1x4x4 | 1D split over M only: 2 tile-rows/core is the smallest block that fits the 70 cores it needs; a 111th core would have no row to take |
| Matmul | 94 | 6 | 0.4877 | 2.22 | 0.098 | 140x4x8x4 | 1D split over M only: 6 tile-rows/core is the smallest block that fits the 94 cores it needs; a 111th core would have no row to take |
| Matmul | 70 | 6 | 0.1908 | 0.87 | 0.038 | 140x1x4x8 | 1D split over M only: 2 tile-rows/core is the smallest block that fits the 70 cores it needs; a 111th core would have no row to take |
| Matmul | 64 | 1 | 0.1211 | 0.55 | 0.024 | 1x24x16x24 | 2D split 3x2 tiles/core; the output is 24x16 tiles and no finer rectangle fits an 11x10 grid |
| Matmul | 90 | 6 | 0.2214 | 1.01 | 0.044 | 16x4x35x9 | 2D split 7x4 tiles/core; the output is 4x35 tiles and no finer rectangle fits an 11x10 grid |
| Matmul | 70 | 1 | 0.0273 | 0.12 | 0.005 | 1x4x140x16 | 1D split over M only: 4 tile-rows/core is the smallest block that fits the 70 cores it needs; a 111th core would have no row to take |
| Matmul | 80 | 1 | 0.0357 | 0.16 | 0.007 | 1x140x24x4 | 2D split 14x3 tiles/core; the output is 140x24 tiles and no finer rectangle fits an 11x10 grid |
| LayerNorm | 1 | 1 | 0.0085 | 0.04 | 0.002 | 1x1x8x8 | row-parallel kernel: 1 tile rows exist, so 1 cores |
| Matmul | 32 | 1 | 0.0104 | 0.05 | 0.002 | 1x16x4x24 | 2D split 2x1 tiles/core; the output is 16x4 tiles and no finer rectangle fits an 11x10 grid |
| Matmul | 24 | 1 | 0.0076 | 0.03 | 0.002 | 1x1x24x8 | 1D split over M only: 1 tile-rows/core is the smallest block that fits the 24 cores it needs; a 111th core would have no row to take |
| Matmul | 70 | 1 | 0.0147 | 0.07 | 0.003 | 1x140x4x1 | 1D split over M only: 2 tile-rows/core is the smallest block that fits the 70 cores it needs; a 111th core would have no row to take |
| Matmul | 8 | 1 | 0.0049 | 0.02 | 0.001 | 1x1x8x1 | 1D split over M only: 1 tile-rows/core is the smallest block that fits the 8 cores it needs; a 111th core would have no row to take |
| Matmul | 70 | 1 | 0.0091 | 0.04 | 0.002 | 1x140x1x4 | 1D split over M only: 2 tile-rows/core is the smallest block that fits the 70 cores it needs; a 111th core would have no row to take |
| BinaryNg | 110 | 54 | 0.5140 | 2.33 | 0.103 | 140x1x4x4 | full grid |
| Permute | 110 | 6 | 0.7194 | 3.27 | 0.144 | 16x4x9x4 | full grid |
| Permute | 110 | 6 | 0.7504 | 3.41 | 0.150 | 1120x1x4x35 | full grid |
| ReshapeView | 110 | 6 | 0.6446 | 2.93 | 0.129 | 140x4x4x4 | full grid |
| NlpCreateHeads | 110 | 6 | 0.4459 | 2.03 | 0.089 | 560x4x1x4 | full grid |
| SDPAOperation | 110 | 6 | 0.3089 | 1.40 | 0.062 | 560x1x1x1 | full grid |
| BinaryNg | 110 | 50 | 0.6486 | 2.95 | 0.130 | 1x16x48x48 | full grid |
| BinaryNg | 110 | 219 | 2.1595 | 9.81 | 0.432 | 1x16x24x24 | full grid |
| SDPAOperation | 110 | 24 | 2.1956 | 9.97 | 0.439 | 16x16x2x2 | full grid |
| ReshapeView | 110 | 24 | 0.5215 | 2.37 | 0.104 | 1x24x16x16 | full grid |

## The measured scaling curves

Each class rebuilt at its shipped shape, dtype, fidelity, memory config and fused
activation, with the matmul program config written out explicitly so the engaged core
count is exact. Device kernel time, median of 20 back-to-back calls, qb2 card 2.

| class | in-situ cores | max reachable | best cores | in-situ us | best us | gain | s/fold | recoverable s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B1-transition-fc1 | 86 | 86 | 86 | 191.62 | 191.62 | 1.000x | 1.177 | 0.000 |
| B2-trimul-batched | 64 | 64 | 64 | 1639.58 | 1639.58 | 1.000x | 0.422 | 0.000 |
| S4-dit-ff-up | 80 | 80 | 80 | 27.55 | 27.55 | 1.000x | 0.338 | 0.000 |
| S1-dit-attn-proj | 64 | 64 | 64 | 23.85 | 23.85 | 1.000x | 0.330 | 0.000 |
| B3-transition-fc2 | 86 | 86 | 86 | 60.32 | 60.32 | 1.000x | 0.302 | 0.000 |
| S5-dit-ff-up-4x | 88 | 88 | 88 | 61.95 | 61.95 | 1.000x | 0.190 | 0.000 |
| S6-atom-proj | 70 | 70 | 70 | 40.19 | 40.19 | 1.000x | 0.088 | 0.000 |
| **total** | | | | | | | | **0.000** |

Full ladders, including the points a wider grid refuses, are in `curve_512_qb2c2.json`.


## What the curves say

Every partial-grid matmul in the fold is already on the most cores its own tile counts can reach.
The ladder in `grid_sweep.py` enumerates every legal split of each op, and in all seven classes the
core count production dispatches today is both the maximum and the fastest point. Handing the op
the whole 11x10 grid does not widen it: with the same per-core quantum the device still reports
the same core count and the same time. The 4x-expansion projection, given `(x=11;y=10)` instead of
`(x=11;y=8)` with `per_core_M=2, per_core_N=9` in both, engages **88 cores either way** and runs
61.98 us against 61.93 us, 0.08 % apart.

The reason is the per-core quantum, not the grid. A 1D matmul's critical path is `per_core_M` tile
rows; a 2D matmul's is `per_core_M x per_core_N` tiles. Transition's fc1 folds its batch into
M = 256 tiles, and 256 tiles over 110 cores needs 3 rows per core, which needs 86 cores. Give it
110 and the quantum is still 3, so 24 cores draw a share of zero. The idle cores are a *consequence*
of the quantum already being at its floor, not a cause of anything.

The one class that is genuinely under-parallelised is the diffusion step's LayerNorm: 16 cores
because `LayerNormDefaultProgramConfig` splits over tile rows only and a (1,1,512,768) normalise
has 16 of them. A block-sharded layout does split the width, and the kernel alone gets faster:
**19.62 us on 64 cores against 28.05 us on 16, 1.43x**. It loses the win to its own layout
conversions. Interleaved-to-sharded costs 4.24 us and sharded-to-interleaved 5.09 us, so the round
trip is 28.95 us against the shipped 28.05 us, **0.97x**, and the output is not bit-exact
(max |delta| 3.1e-02 against the default kernel). Worth revisiting only as part of a residency
change that keeps the tensor sharded across several ops, which is a different campaign.

Not swept, and the only shortfall left with real time in it: `NlpCreateHeads` at 16 of 110 cores,
24 programs and 1.0213 ms per diffusion step, **0.204 s/fold**. It splits one core per head and
exposes no program config, so there is no knob to sweep; moving it would mean a different kernel.
It shifts 6 MB per call in 42.6 us, which is 141 GB/s against the card's measured 435 GB/s DRAM
roof, so a full-grid version has about 3x in it, or **0.14 s of kernel time per fold**. That 0.14 s
sits inside the diffusion step, which spends 34-52 % of its span in no kernel at all, so most of
it would be absorbed before it reached the wall.

## The two caveats on these numbers

**The sweep ran at the idle clock.** A Blackhole p300c idles at 800 MHz and boosts to 1350 under
sustained load. A short isolated op burst never boosts, so every point in `curve_512_qb2c2.json`
reads 1.25-2.01x slow against the same program measured inside a real fold; a burn preamble did not
move it. Ratios within a case are unaffected, which is what prices the recoverable seconds, and the
in-situ point is carried in every row so the offset is visible rather than hidden. `1350 / 800 =
1.6875` and five of the seven classes read between 1.56 and 1.69, so the offset is the clock.

**Idle-core-milliseconds is not recoverable time.** The census's `idle_core_ms` column, kernel time
times the idle fraction of the grid, totals 4.13 ms of the two captures. None of it is available.
It is in the table because it is the right way to *rank* shortfalls, and the wrong way to price
them.
