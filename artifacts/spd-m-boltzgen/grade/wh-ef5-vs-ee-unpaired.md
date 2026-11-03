# WH shipped fast set vs TT_BIO_LEVERS=none, 5 seeds (11-03 05:3xZ)

perf/spd/bg_unpaired.py, seed-cluster bootstrap, 20000 draws. Base wh-ee0..7 (main e11e990dc, none, 8 runs x 8 designs,
.111/.112). Runs copied to pc /tmp/bgw1103/grade (analyze CSVs only). wh-ef5/ef6 excluded: rc 130 recall kills.

| test | runs | design_iptm | design_ptm | complex_plddt |
|---|---|---|---|---|
| wh-ef0..4 (shipped fast, e11e990dc) | 5 | -0.0007 [-0.0223, +0.0185] | +0.0261 [+0.0084, +0.0443] | +0.0033 [-0.0021, +0.0085] |
| wh-zs0..2 (048b7015f fast stack, adaln_at) | 3 | +0.0087 [-0.0344, +0.0467] | +0.0228 [+0.0044, +0.0420] | +0.0012 [-0.0048, +0.0070] |
| wh-en0 (shipped normal) | 1 (en1, en2 truncated CSV, dropped) | +0.0452 | +0.0164 | +0.0041 |

Shipped WH fast PASSES against the exact model (interval covers 0, ptm better). 048b7015f is pass-shaped but its lower
bound -0.034 is past the 0.03 fast bar at 3 seeds; the pre-registered 8 seeds stand. wh-zs3 finished rc=1 (recall abort
at 2:05) but the old command wrote `done` anyway, so the feed counted it; fixed in the grid (done on rc 0 only), the bad
marker moved aside on .111.
