# c12-genericop-rate — pre-registration, written before the first device arm

Committed before `perf/c12_genop_rate/roofs.py` ran on qb2 card 3. C10 lost two rows' worth of
work to predictions written afterwards, so this file is the record.

## What is already measured, and is not re-measured here

`insitu_sites.py` reads the per-site rate out of `c12-profiled-fold`'s ops reports
(`runs/pfl_prof2`, `runs/msal_prof`), which grabbed a PairformerLayer and an MSALayer out of a live
fold at a during-sampled 1350 MHz. The 14 `generic_op` programs inside one layer are identified by
`COMPUTE KERNEL SOURCE`, not by a census key label. That gives 3.3743 s of fold time over 3,920
calls, which reproduces the profiled fold's own 3.3830 s `GenericOpDeviceOperation` total to 0.26 %.
No arm below re-measures a site: an isolated arm is a worse estimate of an in-fold rate than the
in-fold rate.

## The one thing missing: the denominator

The brief compares this op's 26.02 TFLOP/s to 115.685 TFLOP/s on cube4096 from a different session.
`roofline-roof-must-be-measured-not-asserted`: that is not a roof for this op. Every site moves
67-403 MB of DRAM per call at 0-254 FLOP/byte, so what can bind them is the streaming roof.

## Predictions, in order of how much they can embarrass this row

1. **bw_add8192 on qb2 card 3 lands in 410-450 GB/s.** pc measures 415.8 GB/s on the same arm
   (`perf/roof_shape/bw_pc_bh.json`); the census measured 442.877 GB/s on qb2 node 1.
2. **cube4096 HiFi4 lands in 105-120 TFLOP/s.** 108.543 in an earlier qb2-card-3 session
   (`perf/roof_triatt_rate/rate_ab_512_qb2c3.json`), 115.685 in the session the brief quotes.
3. **cube2048 lands in 0.150-0.160 ms**, the known-answer control (b2z 0.1537 ms,
   c12-profiled-fold 0.1520 ms on this card). Outside that band the session is not measuring this
   part and nothing else in it is quotable.
4. **THE CLAIM UNDER TEST: all six sites are bandwidth-bound, not arithmetic-bound.** With
   machine balance = cube4096_TFLOPs / bw_GBs (predicted 235-290 FLOP/byte), the six sites sit at
   0, 0, 103.6, 106.6, 128.0 and 254.1 FLOP/byte, so per-site traffic floor > arithmetic floor at
   all six. If any site's arithmetic floor comes out above its traffic floor, this prediction is
   wrong and the row prints it that way. The brief's premise -- `generic_op` as "the only
   arithmetic-bound op big enough to matter" with "the largest rate headroom" -- stands or falls
   here.
5. **Prize band, pre-registered as a formula, not a number:**
   `prize(f) = 3.3743 s - 0.9127 TB / (f x roof)`, evaluated at f = 1.00 (every site at the full
   streaming roof, physically unreachable) and f = 0.90. At a 442.877 GB/s roof that is 1.3135 s /
   1773 Mc and 1.0845 s / 1464 Mc. Priced in cycles at 1350 MHz throughout.

## Kill criterion

This row builds no lever unless it can name a mechanism that is BOTH

- not already measured closed, and
- worth >= **0.30 s of fold time / 405 Mc** at the measured roof.

0.30 s is the campaign's own cheapest pair of already-built levers re-based by `c12-profiled-fold`
(cond-hoist 0.1445 -> 0.105 s, silu 0.270 -> 0.197 s). A mechanism under that does not deserve a
build slot ahead of landing those two.

Mechanisms already measured closed, so they do not count (each with the measurement that closes it,
because `eligibility-firing-condition-is-not-a-code-fact` cuts both ways):

- **Core count / grid coverage.** All 14 programs report `CORE COUNT` 110 of 110
  `AVAILABLE WORKER CORE COUNT` in the executed graph, on both legs. The core-count sweep
  (`perf/ttx_splitwork/core_count_sweep.py`) found the whole grid the fastest point at all four
  shapes. This is also how this row differs from the two refuted grid levers: those measured
  103.8/110 coverage across the `ttnn` op ladder (prize 0.000 s) and `core_grid` on `ttnn` hot-path
  sites (0.156 s in fold, 98.9 % of its traffic in one AdaLN pair). Neither touched a
  `generic_op` program, and this row is not reopening either -- it reports that the pin the
  matmul-key row handed over, `_triangle_mul_program_config(16)` at 64 of 110 cores, is a
  `ttnn.matmul` program config (`tenstorrent.py:3967`) that no `generic_op` site uses.
- **DRAM bank/page walk.** `TT_BIO_REBLOCK_WALK` measured at 512 aa: gated stride 0.964x, rotate
  1.030x; back stride 1.019x, rotate 1.018x (`perf/ttx_splitwork/prod_shapes_ab.json`). No walk
  ships, and the bank model picks the device's winner at 6 of 12.
- **Folding the qkv projection into SDPA** (`TT_BIO_TRIATT_FUSE_QKV`). Already fused at this shape:
  `rate_ab_512_qb2c3.json` records `fuse_rejects {qkv_already_fused_with_gate: 48}` and
  `sdpa_route_counts {fused: 60, stock: 0}`, and its arm read 3.7753 ms against a 3.7870 ms A/A.
- **Trimul operand L1 residency.** Priced <= 0.1066 s / 143.8 Mc by `c12-matmul-key-attribution`,
  under the kill criterion on its own, and the 67.1 MB chunk does not fit L1 at 512 aa
  (`TRIANGLE_MULT_L1_MAX_SEQ = 352`).

## Accuracy

No arm in this pass changes a number the model computes: the roof arms are synthetic and the
per-site rates are read out of committed artifacts. If a lever does open, its accuracy is scored on
this row's own fixture against a seed floor measured here, never against the 1.84 A figure.
