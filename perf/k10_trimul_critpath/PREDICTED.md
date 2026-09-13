# PREDICTED — written before any per-program thread number was computed

Committed first so the ordering is checkable from git, not from my word.

What I already knew when I wrote this, and what is therefore NOT a prediction:

- The census's single `GenericOp` row is seven distinct programs, identified by their in/out
  signature in `perf/b2z2_byte_floor/art/ops_perf_blocksum_qb2c0.csv.gz` (Blackhole, qb2 card 0).
  The largest is `reblock_permute_gated` at 29.6 % of GenericOp device time, 994.5 us a call.
- The three kernels of that program, read in full:
  `tt_bio/kernels/reblock_permute_gated/{reader,compute,writer}_reblock_permute_gated.cpp`.
- `trimul_tail` (F1) is not one of the seven. `F1_BLOCK_KEYS = {(8, 8)}` and Boltz-2's c_z = 128
  gives (4, 4), so it declines 100 % of this fold's calls.

## The predictions

1. **Dominant stage: the writer's L1 -> L1 gather.** Per group of 32 output tiles the writer issues
   32 x 64 = 2048 `noc_async_read_one_packet_with_state` transactions plus 32 DRAM writes, against
   the reader's 64 DRAM page reads and the compute cluster's 224 tile moves. The writer's own
   header already claims instruction count on that RISC is the binding resource. I expect the
   measurement to agree.

2. **The compute cluster's input stall on this program is back-pressure, not starvation.** The
   whole-`GenericOp` figure is 58.0 % (`k10-instrument`, qb2 card 0). On this program specifically I
   predict compute waits mostly because `out_cb` (depth 64) is not being drained by the writer, and
   that the reader is comparatively idle. Falsifier: if NCRISC NoC-read time on this program is well
   above the 22.6 % GenericOp average, the reader's per-tile-pair `noc_async_read_barrier()` is the
   limiter instead and prediction 1 is wrong.

3. **Fraction of the 3.76 s on the critical path: 60-70 %, so 1.1-1.5 s hidden.** The three threads
   are pipelined per group but joined twice per group (the reader's 32 serialised barriers, the
   writer's `cb_wait_front(out_cb, 32)` which is a whole-group join, not a tile-level one), so I
   expect substantially less overlap than a clean three-stage pipeline.

4. **Tile-movement to tile-math inside the compute kernel: 2.33 : 1**, counting `transpose_wh` as
   math (7 moves = 4 unpacks + 3 packs, against sigmoid + multiply + transpose). Counting only real
   arithmetic it is 3.5 : 1. Four of the seven moves exist only to hold a rounding point for
   bit-exactness against the two-op ttnn sequence they replaced.

5. **Recoverable, named in advance.** (a) The reader's barrier granularity: one barrier per tile
   pair caps in-flight DRAM reads at 2 and the p/g CBs are only `2*GRAN = 4` tiles deep, so the
   reader cannot run ahead even if it wanted to. (b) The `sig_cb` round trip: sigmoid packs to a CB
   and is immediately unpacked again, 2 of the 7 tile moves, kept only for bit-exactness. I predict
   (a) is worth more than (b), and that both are small next to the writer.
