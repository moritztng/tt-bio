# WH shipped sets vs TT_BIO_LEVERS=none (gradeexw, partial), unpaired seed-cluster bootstrap

Read 2026-11-02 20:2xZ on .111, `perf/spd/bg_unpaired.py --expect 8`. Base `wh-ee0..7` (8 runs, 8 designs each),
tree te11e990dc. This is the Wormhole half of the question the Blackhole grade answered: does the shipped lever set
cost design_iptm against exact?

| arm | runs | design_iptm | design_ptm | complex_plddt |
|---|---|---|---|---|
| f = shipped WH fast | 4 (ef0-3) | +0.0043 [-0.0151, +0.0219] | +0.0250 [+0.0073, +0.0433] | +0.0022 [-0.0031, +0.0074] |
| n = shipped WH normal | 1 usable (en0) | +0.0452 [+0.0266, +0.0612] | +0.0164 [-0.0004, +0.0344] | +0.0041 [-0.0003, +0.0084] |

The fast arm PASSES: iptm covers 0, ptm better, plddt flat. Wormhole does not show the ~0.03 design_iptm creep that
Blackhole's pre-drop set did, which is consistent with the WH sets never having carried the BH trunk group.

The normal arm is NOT graded yet: only seeds 0-2 have run and two of them are unusable (below). One run is not a
verdict. gradeexw stays active; it fills seeds 3-7 as WH chips lend.

## Why two runs are unusable: a truncated analyze CSV reads as a clean run

`wh-en1` and `wh-en2` both finished rc=0 with all 8 inverse-folded structures on disk, and both wrote an
`aggregate_metrics_analyze.csv` holding only part of the run: en1 design 0 alone, en2 designs 6 and 7 alone. Averaging
them would have been averaging whichever designs survived the last write, and it skewed the first read of this table
(+0.050 iptm over 11 designs instead of +0.045 over 8). Nothing in the run marks the loss.

Fixed in `perf/spd/bg_unpaired.py` (dbc78ea34): `--expect` (default 8) drops any run whose analyze CSV holds fewer
designs and says so. `bg_grade.py` compares the CSV against the run's own `ca_coords_sequences.pkl.gz` and warns.
Every grade input used before today was re-checked under the guard: the BH 7-seed D verdict is unaffected (all 15 runs
full), so is the WH fast arm.
