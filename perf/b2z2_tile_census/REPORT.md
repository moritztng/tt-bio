# The Blackhole tile census, and what the input-tile wait actually is

`b2z2-bh-tile-census`, 2026-09-12. Every number here is BH. No device was opened: the inputs are
three committed artifacts from `b2z-kernel-cycle-census`, all taken on qb2 card 0, one Blackhole
processor of a p300c, 11x10 grid, tt-metal v0.68.0 source build.

Reproduce with `census_tiles.py <artdir> <outdir>`; the header says how to fetch the inputs.

## 1. The count

**236,118 CB tile arrivals per core-averaged PairformerLayer at 512 aa.** Derived from the operand
shapes, the matmul program configs (`per_core_M`, `per_core_N`, `in0_block_w`, read verbatim out of
the profiler's `ATTRIBUTES`, not modelled) and the grid split, over all 272 programs. Also
2,146,037 tile-pair MACs and 151,862 pack events per core.

Break-even for a datum-rate-bound wait was 880,720 arrivals. The count is 3.73x under it.

| | per core-averaged block |
|---|---|
| CB tile arrivals | **236,118** |
| of those, crossing the DRAM interface | 20,911 (8.9 %) |
| of those, read out of L1-interleaved | 7,025 (3.0 %) |
| multicast or CB replications of tiles the grid read once | 208,182 (88.2 %) |

The diffusion step: **84,607** arrivals per core, 10,390 of them (12.3 %) from DRAM.

## 2. The identity

Against the **18.3366 ms** of measured input-tile wait (MEASURED, `b2z-kernel-cycle-census`, stall
accounting, per core-averaged block; this pass's own window of the same capture reads 18.3696 ms):

    236,118 arrivals  x  20.82 ns/tile  =  4.9160 ms  =  26.8 % of the wait

The falsifier's "near 30 %" branch fired. **The wait is not datum-rate-bound.** 26.8 % is also an
upper bound on the datum rate's share, not an estimate of it, because those 236,118 arrivals have
to be delivered before they can be unpacked and the delivery is the slower stage.

## 3. What the remainder is

Over the 40 sites that run a compute kernel, fitted against the measured per-site wait:

| model | R2 |
|---|---|
| tile arrivals only | **-0.5559** |
| tile-pair MACs only | -0.5744 |
| DRAM bytes only | 0.7011 |
| DRAM + L1 bytes | **0.7872** |
| DRAM + L1 + a per-program constant | 0.7903 |
| DRAM + L1 + tile arrivals | 0.7879 |

A tile-count model fits **worse than predicting the mean**. A bytes model fits at 0.79. Adding the
tile term on top of bytes buys 0.0007 of R2 and lands a coefficient of 0.50 ns/tile, 40x under the
measured 20.82 — the fit puts essentially no weight on it. Spearman against the measured wait:
bytes delivered **+0.909**, tile arrivals +0.767, programs in the site +0.445.

The two-bandwidth fit: **DRAM-interleaved reads at 352.2 GB/s** (79 % of the measured 444.9 GB/s
roof) and **L1-interleaved reads at 651.9 GB/s**. Those two terms are 68.1 % and 13.2 % of the
wait, **81.3 % together**.

A per-program constant is worth **3.0 us/program**, about 3.8 % of the wait across the block's 232
compute programs. So the remainder is not dispatch, not launch latency and not dependency
serialisation. Dest capacity (`b2z2-tile-shape-and-format`) and CB topology
(`b2z2-cb-depth-prefetch`) were already closed by measurement. What is left is delivery bandwidth.

## 4. Why wave 1 read DRAM as idle, and why that is not a contradiction

CONTEXT §3 excludes DRAM at "41.1 % of a measured 444.9 GB/s". That number is exactly
`b2x-baseline-attrib`'s deduped **6,650.7 MB** over the block's 36.3438 ms span: 183.0 GB/s,
41.1 %. This row counts operand traffic **per program**, which is what the interface sees, and gets
8,049.3 MB of DRAM plus 3,097.7 MB of L1. The two counts differ because a global dedup by buffer
address charges a buffer once no matter how many programs read it, and a buffer read by six
programs is six DRAM reads.

Both are right about their own question, and neither contradicts the other:

* **Averaged over the whole block the interface is idle**: 8,049.3 MB in 36.3438 ms is 221.5 GB/s,
  49.8 % of roof. The block is not globally bandwidth-saturated.
* **While the reads are happening it is near its roof**: the fitted read bandwidth is 79 % of it.
  The block's average is diluted by the time it spends writing and computing, and the math thread
  does its waiting inside the reads, not inside the average.

That is the whole reason a block-average roof check missed this.

## 5. What it means for the wave

**The floor is unchanged. The lever class is not.** The movement-free bound is a stall-accounting
identity (`wait_in 18.3366 + wait_out 3.1066 + compute 10.7010 + non-resident 4.1995 = 36.3437`
against a 36.3438 ms span) and nothing here touches it. What changes is what you have to do to
reach it.

* **Deleting a tile PASS is worth 20.82 ns only if it also deletes the operand's round trip.** A
  fusion that keeps an intermediate in DST and stops it going to DRAM and back deletes bytes at
  352 GB/s, which on this block is 27x the tile-pass saving. A fusion that merely saves a pass on
  an operand that still has to be read is worth the 20.82 ns and no more.
* **Residency is the direct lever.** DERIVED, at the fitted bandwidths: if every DRAM-interleaved
  read in the block became L1-interleaved, the wait's DRAM term falls 12.5130 -> 6.7604 ms and the
  block goes 36.3438 -> 30.5912 ms, **1.188x on the Pairformer block**. That is a ratio against a
  block this pass did not re-measure, and the L1 bandwidth in it is fitted, not measured
  (`SILICON-EXPERIMENT.md` E2).
* **bfp8_b comes back, on the other axis.** `b2z2-datum-rate-floor` killed it as a movement lever
  because the BH packer is datum-limited. On the delivery axis it is worth its bytes: 0.531x the
  bytes of bf16 on an operand that has to cross the DRAM interface is 0.531x of that operand's
  contribution to the wait. It was never a tile lever and it is a real byte lever.
* **The sampler is a different problem.** 2,340.6 MB of DRAM reads per step against the trunk's
  4,710.8 MB per block, in a step now 26.40-26.60 ms against the block's 36.34 ms. Its input-tile
  wait has never been measured on any architecture (`SILICON-EXPERIMENT.md` E1) and should not be
  assumed to be 57 %.
