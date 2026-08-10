# P4 / p2-layout-kernel — can a tt-metal kernel move tile faces at compute-engine rates?

Phase 2, EXPERIMENT. protenix-v2, trunk only, 298 aa (N=320 padded, c_z=256). qb2 **card 2**
(board 005 chip 0), ttnn 0.68.0, board mate chip 3 held idle. **Ratios only** (charter §4.8).

**No production change. Nothing under `tt_bio/` was touched** — the deliverable is a prototype and
two probes under `perf/p2_layout_kernel/`.

## Predictions (before measuring)

Committed to git before the device was opened. The shapes are the two production channel moves at
298 aa, both on the L1 path (`_triangle_mul_memory_config(320)` returns `L1_MEMORY_CONFIG`, chunk
width C=32):

- **in-move** `[1,320,320,32] -> [1,32,320,320]`, `permute(0,3,1,2)`, 3200 tiles, 6.554 MB each way
- **out-move** `[1,32,320,320] -> [1,320,320,32]`, `permute(0,2,3,1)`, 3200 tiles, 6.554 MB each way

T2's baseline for both: a fixed **2.92 / 2.56 us per tile per core** on the dataflow RISCs, against
**0.33-0.34 us/tile/core** for the in-tile `transpose(-2,-1)` that runs on the compute engine. Those
two numbers bracket the whole question and I am predicting where a kernel lands between them.

**P0 — the cheap test, run first: staging through L1 cannot recover these two sites, because they
are already in L1.** T5's 37.6-44.9 GB/s figure is a DRAM->DRAM permute at other sites; at 298 aa
the trimul chunks are L1-resident. Predict: forcing the destination to DRAM is **>=1.5x slower**
than the production L1 destination. Wrong if L1->DRAM comes within 10 % of L1->L1 or beats it — in
which case DRAM bank-row locality, not transaction count, is the mechanism and no kernel is needed.

**P1 — the move is transaction-count-bound at sub-line granularity.** Mechanism, stated so it can
lose: with C=32 (Ct=1) one 32x32 source tile, after an in-tile transpose, contributes exactly **one
row to each of 32 different destination tiles**. A tile row is 32 bf16 = 64 B, split across two
16-wide faces, so it is **2 NOC writes of 32 B**: **64 sub-line NOC writes per source tile**.
Predict the implied per-transaction cost is **40-50 ns** (2.92 us / 64 = 45.6 ns). Wrong if it comes
out under 20 ns or over 100 ns, which would mean something other than transaction issue is paying.

**P2 — a full-tile-write kernel wins 2.0-3.2x and does NOT reach the in-tile floor.** The prototype
(`reblock_permute`, already written in the `tt-metal-fused` worktree, bit-exact, 221 GB/s to DRAM at
N=1024) turns the 64 scattered 32 B writes into a **local L1 gather plus ONE 2 KB destination write
per output tile**. Predict **0.9-1.5 us/tile/core** at the production shape with an L1 destination,
i.e. **300-480 GB/s** of L1 traffic. Wrong if under 1.6x or over 4.0x of the 2.92 baseline. Predict
explicitly that it does **not** reach 0.33-0.34: the 32 face-row copies per output tile still get
issued, they only move from "scattered to the destination buffer" to "local in L1", so the
transaction count falls by the write-side factor only, not to zero.

**P3 — the prize, repriced before measuring.** At 2.5x the two sites go 2213 -> ~885 ms/fold.
Predict the recoverable figure lands in **1100-1500 ms/fold**, not the brief's 1805, because 1805
assumes the in-tile floor and P2 says the floor is not reachable. Wrong if outside that band.

**P4 — the honest floor is a clone, not a transpose.** The move touches 13.107 MB of L1 (6.554 read
+ 6.554 write), so no kernel can beat a same-bytes L1->L1 clone of the same tensor. Predict that
clone measures **12-18 us** on this card for 3200 tiles = **0.41-0.62 us/tile/core** at 110 cores.
Wrong if it comes in under 10 us — which would put 0.33-0.34 back in reach and make P2/P3 too
pessimistic.

**P5 — parity.** `torch.equal` True against the `ttnn.permute` result for both moves, at every shape
tested. A permute is a pure index move, so anything else is a kernel bug, not a parity trade. Wrong
if any single element differs.
