# B200 accuracy (acc_arms.py, CA Kabsch RMSD over 5 folds; box 55990925, sessions b1+b2)

Session b2 (opendde, pxdesign, rf3 and the fast arms of esmfold2 x2 / openfold3 / openbind follow from b3/b4, after the kit refused them on sm_100).


boltz2: pair | folds | CA RMSD A median (max) | per-copy CA RMSD A median (max) | max |dx| A
| exact vs off | 5 | 1.484 (21.234) | 1.444 (10.593) | 67.5438 |
| fast vs off | 5 | 1.555 (21.301) | 1.502 (10.586) | 67.3474 |
| off-det vs off (stock floor) | 5 | 1.554 (21.420) | 1.476 (10.566) | 71.4007 |
| off fold i vs fold 0 (floor) | 4 | 10.454 (21.234) | 4.723 (10.593) | 67.5438 |

protenix_v2: pair | folds | CA RMSD A median (max) | per-copy CA RMSD A median (max) | max |dx| A
| exact vs off | 5 | 4.151 (23.127) | 3.335 (15.504) | 66.2326 |
| fast vs off | 5 | 4.444 (23.712) | 2.397 (11.592) | 63.3531 |
| exact-det vs off-det | 5 | 0.000 (0.000) | 0.000 (0.000) | 0.0000 |
| off-det vs off (stock floor) | 5 | 3.322 (24.433) | 3.249 (13.130) | 67.6592 |
| off fold i vs fold 0 (floor) | 4 | 12.672 (23.446) | 5.591 (13.723) | 80.9804 |

openfold3: pair | folds | CA RMSD A median (max) | per-copy CA RMSD A median (max) | max |dx| A
| exact vs off | 5 | 0.288 (1.411) | 0.280 (1.816) | 7.6707 |
| exact-det vs off-det | 5 | 0.000 (0.000) | 0.000 (0.000) | 0.0000 |
| off-det vs off (stock floor) | 5 | 4.168 (8.665) | 4.101 (8.043) | 31.8575 |
| off fold i vs fold 0 (floor) | 4 | 6.813 (7.533) | 6.614 (7.115) | 49.9951 |

openfold3_ob0: pair | folds | CA RMSD A median (max) | per-copy CA RMSD A median (max) | max |dx| A
| exact vs off | 5 | 9.924 (25.120) | 9.941 (13.081) | 46.8709 |
| exact-det vs off-det | 5 | 11.740 (20.440) | 11.572 (19.151) | 46.9428 |
| off-det vs off (stock floor) | 5 | 0.121 (0.246) | 0.121 (0.247) | 1.0534 |
| off fold i vs fold 0 (floor) | 4 | 14.401 (19.646) | 13.601 (18.085) | 43.0318 |

esmfold2-full_nomsa: pair | folds | CA RMSD A median (max) | per-copy CA RMSD A median (max) | max |dx| A
| exact vs off | 5 | 0.011 (0.023) | 0.009 (0.020) | 0.3360 |
| exact-det vs off-det | 5 | 0.000 (0.000) | 0.000 (0.000) | 0.0000 |
| off-det vs off (stock floor) | 5 | 0.034 (0.043) | 0.028 (0.033) | 1.2960 |
| off fold i vs fold 0 (floor) | 4 | 0.013 (0.021) | 0.010 (0.014) | 0.2550 |

esmfold2-fast: pair | folds | CA RMSD A median (max) | per-copy CA RMSD A median (max) | max |dx| A
| exact vs off | 5 | 0.012 (0.014) | 0.011 (0.013) | 0.2400 |
| exact-det vs off-det | 5 | 0.000 (0.000) | 0.000 (0.000) | 0.0000 |
| off-det vs off (stock floor) | 5 | 0.098 (0.102) | 0.076 (0.078) | 4.0400 |
| off fold i vs fold 0 (floor) | 4 | 0.014 (0.018) | 0.011 (0.017) | 0.1470 |

rfdiffusion3: pair | folds | CA RMSD A median (max) | per-copy CA RMSD A median (max) | max |dx| A
| exact vs off | 5 | 4.248 (7.001) | 5.909 (9.835) | 29.7823 |
| fast vs off | 5 | 2.720 (5.685) | 3.826 (7.968) | 26.8355 |
| exact-det vs off-det | 5 | 0.000 (0.000) | 0.000 (0.000) | 0.0000 |
| off-det vs off (stock floor) | 5 | 2.637 (7.918) | 3.625 (11.079) | 28.2059 |

ESMC x3: pooled-embedding digest of exact == off on all three sizes (300m cd1a40ffbe2b8ccb, 600m ffd90b14692daa11, 6b f3f64477ba69cdfb).
openbind exact-det vs off-det is not bitwise (11.7 A, inside the 14.4 A off fold-to-fold floor): the kit's exact runs a bf16 base, off fp32 (A100 and H200 read the same).
boltz2 has no det recipe on the kit's exact route (exact-det rc 2, as on A100).

## Session b5 (box 56030188, 2026-10-15): the arms b1 could not run on sm_100

Kit levers withheld with the kit's own switch (MODEL_OPT_LEVERS_OFF) because they do not run on sm_100: rf3 prefetch,awrite,fpf_xln (exact and fast); openfold3 and openbind pair_transition (fast). opendde's kit modes run with --allow-partial (the cu12 cuEquivariance sm_100 triangle kernel serves some of fpf_trimul_exact's calls). pxdesign runs upstream's --use_fast_ln false on torch 2.7.1+cu128 (Protenix 0.5.0's fused LayerNorm has no sm_100 code).

opendde: pair | folds | CA RMSD A median (max) | per-copy CA RMSD A median (max) | max |dx| A
| exact vs off | 5 | 0.007 (0.011) | 0.006 (0.009) | 0.0236 |
| fast vs off | 5 | 0.167 (0.183) | 0.121 (0.132) | 1.4113 |
| exact-det vs off-det | 5 | 0.000 (0.000) | 0.000 (0.000) | 0.0000 |
| off-det vs off (stock floor) | 5 | 0.036 (0.037) | 0.030 (0.032) | 0.3393 |
| off fold i vs fold 0 (floor) | 4 | 0.008 (0.008) | 0.006 (0.007) | 0.0248 |

pxdesign: pair | folds | CA RMSD A median (max) | per-copy CA RMSD A median (max) | max |dx| A
| exact vs off | 5 | 0.001 (0.005) | 0.002 (0.007) | 0.0348 |
| fast vs off | 5 | 0.057 (0.600) | 0.079 (0.823) | 3.6742 |

rosettafold3-no-prefetch-awrite-fpf_xln: pair | folds | CA RMSD A median (max) | per-copy CA RMSD A median (max) | max |dx| A
| exact vs off | 10 | 0.186 (6.328) | 0.169 (6.306) | 15.2296 |
| fast vs off | 10 | 11.997 (23.245) | 11.108 (20.896) | 56.9412 |
| off fold i vs fold 0 (floor) | 9 | 22.288 (25.146) | 16.296 (18.582) | 60.7833 |

openfold3-no-pair_transition: pair | folds | CA RMSD A median (max) | per-copy CA RMSD A median (max) | max |dx| A
| fast vs off | 5 | 5.838 (8.260) | 5.312 (7.574) | 28.1139 |
| off fold i vs fold 0 (floor) | 4 | 6.801 (7.503) | 6.580 (7.109) | 49.8066 |

openfold3_ob0-no-pair_transition: pair | folds | CA RMSD A median (max) | per-copy CA RMSD A median (max) | max |dx| A
| fast vs off | 5 | 12.071 (19.732) | 11.861 (17.979) | 50.2584 |
| off fold i vs fold 0 (floor) | 4 | 14.408 (19.642) | 13.608 (18.081) | 42.8035 |

opendde exact-det vs off-det is bitwise. openfold3 and openbind exact were measured in b2 (above); their b5 fast arms compare against the b5 off arm on the same box. esmfold2 fast refused again in b5: t15 (same fused transition kernel family as t10) requests 528 of 512 tensor-memory columns; rerun as b6 with t10,t15,t15msa withheld.
