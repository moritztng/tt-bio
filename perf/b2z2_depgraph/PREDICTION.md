# b2z2-dependency-overlap — PREDICTED, written before the first number exists

Committed before any per-core data was read. The brief asks for the prediction first because the
falsifier is the deliverable: if the wait is diffuse, the fold is at its structural floor at this
grid size and reordering cannot buy 2x.

## What I am measuring

One settled 512 aa `PairformerLayer` on Wormhole (whglx), from the per-core device profiler log
`b2z2-whglx-profiler-build` already captured on the STOCK kernel tree (6,963,440 rows, common
device clock, 8x9 grid). Two things nobody has built yet:

1. **A per-core wait histogram.** Total math-thread input-tile wait per physical core over the
   whole block, all 272 ops — not per op code, not inside one kernel family.
2. **A gating map.** Per op, per core, kernel begin and end on the shared clock. The core that
   finishes last gates every other core on that op. "Straggler surplus" = sum over cores of
   (op_last_end - core_end), in core-us.

## PREDICTED

**Concentrated — but in a structural site, not in a hot core.** Three sub-predictions, each with
its own falsifier, because they fail independently:

* **P1. The per-core wait histogram is FLAT.** Top core within 1.5x of the median, Gini < 0.20.
  The 110-core (here 72-core) grid runs a homogeneous work split, so there is no single core that
  many cores wait on. *This kills "a few cores gating many" — a fan-in — as the shape.*
  Falsified if the top decile of cores carries >35 % of the wait.

* **P2. The concentration is in OP SITES, not cores.** Fewer than 40 of the 272 ops carry more
  than 60 % of the block's straggler surplus. `b2z2-tile-arrival-latency` already showed 16 of 272
  programs carry 35.2 % of the input wait; I expect the straggler surplus to be at least as
  skewed. Falsified if the top 40 ops carry under 40 %.

* **P3. The shape is a CHAIN, and it is position-dependent.** This is the decisive one. In
  `tt_bio/mm_generic.py` the operand is forwarded core-to-core along a grid axis, so a core's
  start delay should rise **monotonically with its index along that axis**. I predict a positive
  rank correlation between chain position and per-core wait on the matmul ops, Spearman >= +0.5.
  *If it holds, the wait is a chain and a multicast collapses it.* Falsified if |Spearman| < 0.3,
  which would mean the delay does not know where on the chain the core sits, and then the wait is
  a property of the dataflow rather than of the forwarding order.

## The falsifier for the ROW, stated as the brief demands

If P1 holds AND P3 fails, the 67.6 % is **diffuse**: every core waiting a little on every other,
no ordering to fix, and the honest campaign answer is that **2x is not reachable by reordering at
this grid size**. I will say that plainly rather than shop for a lever.

If P1 holds AND P3 holds, the shape is a chain — concentrated in a mechanism, not in a core — and
the lever is the multicast `b2z2-tile-arrival-latency` priced but did not build.

## What would surprise me

P1 failing. A grid where 7 of 72 cores carry a third of the wait would mean the work split itself
is lopsided, which is a placement bug and a much cheaper fix than anything in the brief.
