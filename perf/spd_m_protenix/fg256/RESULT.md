# fg256: trimul_t1_fgate at Protenix-v2's T1 key (8, 16, 8), p150a

qb1 logical card 1, 2026-10-29 17:27-17:30Z, AICLK 1350 min/max on every rep (sampled during the runs).
512 tokens, compute-only T1W8, 20 reps per run, base and fg alternated 3x per fidelity.

| fidelity | base w8_ms (3 runs)   | fg w8_ms (3 runs)     | mean change | a rel-rms vs f64 base -> fg |
|----------|-----------------------|-----------------------|-------------|-----------------------------|
| HiFi2    | 2.869 2.847 2.852     | 2.781 2.741 2.738     | -3.6 %      | 0.0147 -> 0.0126            |
| HiFi3    | 2.919 2.878 2.878     | 2.810 2.790 2.847     | -2.7 %      | 0.0089 -> 0.0086            |

g is unchanged (same digest path for g). End to end: T1W8 is 2.44 s of the 20.37 s BH fast kernel (bhcen3), so the
lever is worth about 0.09 s per fold, 0.4 %, below fold-timing noise and not worth a paired-fold grade.
