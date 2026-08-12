# protenix-to-4x — the L1-blocking screen, spec only

Full reasoning, byte models, predicted landing and stop rule: `~/.coworker/state/protenix-to-4x.md`.
This file exists so the instrument and the plan sit in the same tree.

Instrument: extend `perf/trimul_abs/tape2.py` (real `PairformerLayer`, real protenix-v2 layer-0
weights, 512 aa pair shape, tapes `generic_op` calls, and already carries the L1 capacity probe
whose result decides this program).

The measurement that motivates the screen, already on disk in
`perf/trimul_abs/tape2_512_qb2c0.json` under `l1_residency`, qb2 card 0, 11x10 grid, 512 aa:

    C=32   3x[1,32,512,512] L1-resident   48.0 MB   L1 matmul 0.2118 ms  x8 = 1.694 ms
    C=64   3x[1,64,512,512] L1-resident   96.0 MB   L1 matmul 0.3872 ms  x4 = 1.549 ms
    C=128  OOM: 67108864 B across 110 banks, 610304 B per bank

against the production DRAM triangle matmul at 1.717 ms/call. So:
  1. L1 residency of the chunk triple IS feasible at 512 aa up to C=64 (the
     TRIANGLE_MULT_L1_MAX_SEQ=352 gate is keyed on token count, not on chunk x seq^2);
  2. the triangle matmul does NOT want it -- 402.7 MB of deleted DRAM traffic returns 0.02-0.17 ms,
     because it is compute-bound at 40.0 TFLOP/s, 39% of the measured 101.99 TFLOP/s roof.

Legs (interleaved one sample per round, median of 7, warm 3, sync both sides, benchlock):
  A         production TriangleMultiplication.__call__          + its own A/A over 3 processes
  T_today   A minus the tail                    -> the tail in situ (today 4.032 ms, 1207.8 MB)
  T_block   the tail row-blocked, R in {64,128,256}, intermediates L1        [N1]
  C_today   the chunk core, DRAM, C=256
  C_l1      the chunk core, L1 chunk triple, C in {32,64}                    [N2]
  N3        in-projection at N=1280 vs N=1024 + separate g_out               [N3]
  INT       A with N1+N2+N3 on, interleaved against A

Stop rule, registered:
  INT >= A - 0.6 ms/call            -> NO-GO on the program
  INT <= A - 1.8 ms/call            -> GO, build, then ONE fold-level A/B
  in between                        -> GO only if N1 alone clears -0.7 ms/call
  any leg failing torch.equal       -> dead on the spot, no envelope run
  C=64 not building                 -> expected; C=32 is the fallback rung

Predicted landing, written before the build: -2.3 to -3.3 s/fold, i.e. 50.2-51.2 s = 4.12-4.20x
against the H200 12.186 s. 4.00x needs 48.744 s. The program does not reach it and the doc names the
mechanism holding every remaining second.

N2 needs `reblock_permute.eligible_gated`s `shape[3] != 4*slice_c` guard relaxed so the fused gated
move can take a 64-wide value/gate pair out of the single 1024-wide projection; the kernel already
takes (value_offset, gate_offset, width) in tile units. Do NOT instead run the in-projection per
group: that re-reads x_norm_in 4x, +402.7 MB, +1.2 ms/call, and eats the whole win.
