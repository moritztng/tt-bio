# WH BoltzGen kzm stack (c02150635): fast grade vs TT_BIO_LEVERS=none and speed vs main (2026-11-06 03Z)

Tree c02150635 = main 85441859d + dit_kl1 (both WH sets) + device z_init on Wormhole. Defaults only, no TT_BIO_LEVERS.

Grade, perf/spd/bg_unpaired.py seed-cluster bootstrap, base wh-ee0-7 (none), test wh-km0-4 (--fast, seeds 0-4), bg_R3,
8 designs a run, all runs full, all rc 0 (.108 chips 2/30, .111 chips 2/24/7). CSVs pc:/tmp/bgw1106/grade.

| metric | delta [95 %] | |
|---|---|---|
| design_iptm | +0.0020 [-0.0228, +0.0263] | fast bar (drop <= 0.03): PASS |
| design_ptm | +0.0329 [+0.0145, +0.0517] | better |
| complex_plddt | +0.0040 [-0.0013, +0.0092] | |

Normal needs no new grade: WH_LEVERS_NORMAL on c02150635 is the kzn set (wh-kz-vs-ee-unpaired.md, PASS).

Speed, kzab (.108, chip 12 normal / chip 1 fast, AICLK 1000 on the run chip for ~85-95 % of samples, rest 500 idle
between designs), order b k b k, 6 designs seed 0, warm medians designs 2,3,5,6, rc 0 x8, md5 deterministic per arm and
k != b:

| mode | b1 | k1 | b2 | k2 | ms_step b -> k | pre b -> k | trunk b -> k |
|---|---|---|---|---|---|---|---|
| normal | 33.42 | 32.29 | 34.07 | 31.82 | 36.0 -> 34.4 | 1.7 -> 0.9 | 14.4 -> 13.5 |
| fast | 32.22 | 31.10 | 31.98 | 30.90 | 33.1 -> 33.1 | 1.9 -> 0.9 | 14.2 -> 13.2 |

Pre-registered bar k <= b - 1.0 s wall on both pairs: met in both modes (normal -1.1/-2.3, fast -1.1/-1.1).
In fast, dit_kl1 replaces dit_kc at equal speed (no fp32-dest multi-tile exposure); the fast gain is device z_init.
