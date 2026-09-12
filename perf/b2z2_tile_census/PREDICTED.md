# PREDICTED — written before the first tile was counted

Row `b2z2-bh-tile-census`. Committed before `census_tiles.py` was written or run, so the
falsifier is on the record and not retrofitted.

## The quantity

Tile arrivals per **core-averaged** PairformerLayer at 512 aa on Blackhole, against the measured
**18.3366 ms** of TRISC1 input-tile wait in a 36.3438 ms block (BH, qb2 card 0, stall accounting,
`b2z-kernel-cycle-census`). The datum rate to divide by is **20.82 ns/tile** (BH, MEASURED by
`b2z2-datum-rate-floor`, binary shape, 1.35 GHz), not the 71.3 ns/tile Wormhole constant.

Break-even: 18.3366 ms / 20.82 ns = **880,720 tile arrivals per core-averaged block** would make
the wait exactly datum-rate-bound.

## PREDICTED

Two counts, because they are not the same thing and the campaign has been conflating them:

1. **CB arrivals** — operand tiles a reader must deliver into a circular buffer, each delivery
   counted once. This is what `cb_wait_front` literally blocks on.
   **PREDICTED: 150,000 - 350,000 per core-averaged block, centre 250,000.**
   Reasoning without counting: the pair tensor z at 512 aa is 512 x 512 x 128 bf16 = 32,768 tiles,
   298 tiles/core on a 110-core grid. A block makes on the order of 20-40 full passes over z
   (2 trimul, 2 triangle attention, transition, pair-from-single, plus the gating and norm
   passes each of those decomposes into). 30 passes x 32,768 = ~1.0 M tiles, /110 cores = ~9,000
   ... which is two orders of magnitude BELOW break-even, so the arrival count alone cannot be the
   term. I therefore expect the honest count to be dominated by matmul operand re-delivery and to
   land in the 150k-350k band only if the trunk's matmuls re-push their operands per k-block.

2. **Unpack events** — tile-pair MACs x 2 for matmul plus operands x out-tiles for eltwise. This
   counts a tile every time it enters SrcA/SrcB, re-reads included.
   **PREDICTED: 5 - 30 million per core-averaged block**, i.e. 6x to 34x ABOVE break-even.
   Reasoning: one trimul is 512 channels' worth of 512x512x512 matmul work; in tiles that is
   ~524,000 tile-MACs per trimul before the grid split.

## Therefore

**PREDICTED WAIT-FRACTION-EXPLAINED (CB arrivals x 20.82 ns): 17 - 40 %, centre 28 %.**

That is the falsifier's "near 30 %" branch. I am predicting the campaign has been pricing the
wrong stage: the math thread is not waiting for a datum pipe to drain, it is waiting for a
*delivery* that has not started or has not arrived, and the lever class that attacks delivery
latency is not the lever class that deletes tile passes.

**PREDICTED REMAINDER-CAUSE, in the order I expect to find it:**
1. Dependency serialisation — the math thread waiting on a producer that has not run, not on
   bytes in flight. Expect this to be the largest single term.
2. NOC / DRAM read latency under the trunk's access pattern (transactions in flight too few to
   cover the round trip).
3. Grid-split imbalance: a core-averaged wait includes cores that finished early and are waiting
   for a barrier, which is not movement at all.

## Falsifier, stated in advance

* If **CB arrivals x 20.82 ns lands within 85-115 % of 18.3366 ms**, the wait IS datum-rate-bound,
  wave 1's mechanism survives in corrected form, and pass-deleting levers stay the right target
  at 3.42x less value than advertised. My prediction is then WRONG.
* If it lands **below 50 %**, the campaign has been optimising the wrong stage and the remainder
  must be named.
* If **unpack events x 20.82 ns exceeds 18.3366 ms**, then "tiles x a per-tile constant" is not a
  floor at all in that currency and the pass model is over-determined — which would mean the
  currency, not the constant, is what was wrong.

No card is held and none is needed for the count: it is arithmetic over committed BH artifacts
(`perf/b2z_kernel_census/ops_perf_block_qb2c0.csv.gz` and `ops_perf_step_qb2c0.csv.gz`, taken on
qb2 card 0 on a tt-metal v0.68.0 source build).
