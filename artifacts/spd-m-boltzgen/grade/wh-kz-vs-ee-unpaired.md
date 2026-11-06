# WH BoltzGen: dit_kl1 + device z_init stack vs TT_BIO_LEVERS=none (2026-11-06 01Z)

perf/spd/bg_unpaired.py seed-cluster bootstrap, base wh-ee0-7 (none), bg_R3, 8 designs a run, all runs full (8 in CSV).
Tree a3e477e10 (kl1 034fd977d + device z_init on Wormhole), TT_BIO_LEVERS = mode set + dit_kl1. CSVs pc:/tmp/bgw1106/grade.

| test | runs | design_iptm | design_ptm | complex_plddt | verdict |
|---|---|---|---|---|---|
| wh-kzn0-7 (normal set + dit_kl1 + z_init) | 8 | +0.0092 [-0.0138, +0.0306] | +0.0255 [+0.0065, +0.0454] | +0.0074 [+0.0022, +0.0125] | PASS (normal) |
| wh-kzf0-4 (fast set + dit_kl1 + z_init) | 5 | +0.0126 [-0.0148, +0.0403] | +0.0234 [+0.0030, +0.0448] | +0.0077 [+0.0029, +0.0125] | PASS (fast) |
| wh-kk0-7 (normal set + dit_kl1, host z_init) | 8 | +0.0058 [-0.0158, +0.0275] | +0.0233 [+0.0057, +0.0417] | +0.0041 [-0.0020, +0.0098] | PASS |
| wh-en0-7 (shipped normal, main e11e990dc) | 8 | +0.0137 [-0.0124, +0.0383] | +0.0286 [+0.0112, +0.0469] | +0.0040 [-0.0017, +0.0100] | PASS |

Trunk fidelity A/Bs (.111, 1000 MHz on the run chip, 6 designs, medians designs 2,3,5,6, wall s):
tkf fast (chip 0): m1 32.84 == b md5; b1 32.80 b2 32.27; x1 31.73 x2 31.69; y1 31.38 y2 31.88 -> misses <= b-1.0 on both pairs: KILLED.
tkn normal (chip 27): b1 33.82 (chip mostly 810 MHz) b2 33.23; h1 33.08 h2 32.87 -> clean pair -0.36: KILLED.
