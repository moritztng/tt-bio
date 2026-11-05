# WH BoltzGen fast: shipped set (ef, 8 seeds) and 048b7015f stack (zs, 7 seeds) vs TT_BIO_LEVERS=none (ee, 8 seeds)

2026-11-05 21Z, bg_R3, 8 designs a run. Runs: ee .108, ef .108/.111/.112, zs .107/.111. zs7 lost to a lend recall.

```
== design_iptm
base wh-ee?: 8 runs, per-run means [np.float64(0.163), np.float64(0.2), np.float64(0.185), np.float64(0.146), np.float64(0.161), np.float64(0.152), np.float64(0.156), np.float64(0.223)]
wh-ef?: 8 runs, 64 designs, design_iptm -0.0078 [-0.0287, +0.0115], per-run [np.float64(0.179), np.float64(0.166), np.float64(0.177), np.float64(0.188), np.float64(0.153), np.float64(0.157), np.float64(0.16), np.float64(0.143)]
wh-zs?: 7 runs, 56 designs, design_iptm +0.0032 [-0.0235, +0.0291], per-run [np.float64(0.137), np.float64(0.216), np.float64(0.192), np.float64(0.15), np.float64(0.204), np.float64(0.172), np.float64(0.163)]
== design_ptm
base wh-ee?: 8 runs, per-run means [np.float64(0.757), np.float64(0.783), np.float64(0.717), np.float64(0.745), np.float64(0.697), np.float64(0.759), np.float64(0.747), np.float64(0.727)]
wh-ef?: 8 runs, 64 designs, design_ptm +0.0251 [+0.0078, +0.0432], per-run [np.float64(0.759), np.float64(0.772), np.float64(0.768), np.float64(0.767), np.float64(0.772), np.float64(0.767), np.float64(0.757), np.float64(0.77)]
wh-zs?: 7 runs, 56 designs, design_ptm +0.0261 [+0.0069, +0.0460], per-run [np.float64(0.762), np.float64(0.773), np.float64(0.758), np.float64(0.746), np.float64(0.773), np.float64(0.782), np.float64(0.778)]
== complex_plddt
base wh-ee?: 8 runs, per-run means [np.float64(0.635), np.float64(0.637), np.float64(0.625), np.float64(0.638), np.float64(0.63), np.float64(0.641), np.float64(0.647), np.float64(0.631)]
wh-ef?: 8 runs, 64 designs, complex_plddt +0.0065 [+0.0007, +0.0121], per-run [np.float64(0.637), np.float64(0.642), np.float64(0.633), np.float64(0.638), np.float64(0.643), np.float64(0.644), np.float64(0.647), np.float64(0.651)]
wh-zs?: 7 runs, 56 designs, complex_plddt +0.0036 [-0.0019, +0.0091], per-run [np.float64(0.638), np.float64(0.64), np.float64(0.632), np.float64(0.638), np.float64(0.639), np.float64(0.648), np.float64(0.639)]
```

Verdict: both pass the fast bar (mean design_iptm drop <= 0.03). zs handed as wk/spd-m-boltzgen-zsf 85441859d.
