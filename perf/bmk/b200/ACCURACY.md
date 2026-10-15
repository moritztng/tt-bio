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
