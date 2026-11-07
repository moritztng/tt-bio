# WH BoltzGen: gradewan2, gradewtg, trgab (read 2026-11-07 05Z)

Runs on .108 (plus wh-an5/7 on .111) between 11-06 20:34Z and 11-07 01:09Z, tree c02150635, all rc 0, every analyze
CSV carries 8 designs. Local copies pc:/tmp/bgw1107/{grade,trgab}.

## Grades vs TT_BIO_LEVERS=none (bg_unpaired, base wh-ee0-7)

| arm | runs | design_iptm | design_ptm | complex_plddt |
|---|---|---|---|---|
| an: WH normal + adaln_at | 8 | -0.0082 [-0.0302, +0.0135] | +0.0089 [-0.0171, +0.0345] | +0.0034 [-0.0030, +0.0099] |
| tg: WH fast + trimul_sig6, triatt_ln_b8, triatt_ln_b8g, trimul_t1w8 | 5 | +0.0225 [-0.0111, +0.0626] | +0.0264 [+0.0077, +0.0455] | +0.0012 [-0.0049, +0.0075] |

Both PASS (normal: interval covers 0; fast: no metric worse than 0.03).

## trgab (.108 chip 29, AICLK 1000 MHz on all ~385 samples per arm, --fast, medians designs 2,3,5,6)

| arm | loop s | trunk s | ms_step | wall s | md5 |
|---|---|---|---|---|---|
| b1 | 11.96 | 13.22 | 37.12 | 33.30 | 914d232eed |
| g1 | 11.49 | 12.82 | 37.31 | 33.28 | 3b3f128a35 |
| a1 | 11.47 | 13.07 | 37.78 | 33.70 | 3b3f128a35 |
| b2 | 11.97 | 13.25 | 35.91 | 32.34 | 914d232eed |
| g2 | 11.48 | 12.84 | 36.27 | 32.39 | 3b3f128a35 |
| a2 | 11.46 | 12.72 | 35.23 | 31.69 | 3b3f128a35 |

g = b + the four: loop -0.47/-0.49 s, past the -0.3 bar on both pairs. KEEP. md5 g == a: triatt_k1, opm_b8cfg,
trimul_t1_2g, triatt_q10, trimul_t2w are byte-neutral in BoltzGen. Wall is diffusion-noise dominated (box serving).

Handed as wk/spd-m-boltzgen-tga 6596b188a.
