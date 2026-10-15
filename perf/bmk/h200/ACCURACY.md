# H200 accuracy (acc_arms.py, CA Kabsch RMSD over 5 folds; boxes 55990865 h1, 56013210 h2)

boltz2: pair | folds | CA RMSD A median (max) | per-copy CA RMSD A median (max) | max |dx| A
| exact vs off | 5 | 1.481 (21.230) | 1.438 (10.590) | 67.6478 |
| fast vs off | 5 | 1.543 (21.289) | 1.496 (10.579) | 66.9492 |
| off-det vs off (stock floor) | 5 | 1.559 (5.192) | 1.481 (4.419) | 71.6014 |
| off fold i vs fold 0 (floor) | 4 | 10.414 (21.057) | 4.788 (10.837) | 82.9568 |

protenix_v2: pair | folds | CA RMSD A median (max) | per-copy CA RMSD A median (max) | max |dx| A
| exact vs off | 5 | 5.303 (20.321) | 4.934 (17.576) | 56.3540 |
| fast vs off | 5 | 3.371 (23.489) | 2.235 (9.323) | 65.1248 |
| exact-det vs off-det | 5 | 0.000 (0.000) | 0.000 (0.000) | 0.0000 |
| off-det vs off (stock floor) | 5 | 2.996 (13.849) | 2.080 (7.434) | 38.0452 |
| off fold i vs fold 0 (floor) | 4 | 4.543 (8.445) | 3.474 (6.598) | 81.8611 |

opendde: pair | folds | CA RMSD A median (max) | per-copy CA RMSD A median (max) | max |dx| A
| exact vs off | 5 | 0.006 (0.744) | 0.006 (0.576) | 1.6585 |
| fast vs off | 5 | 0.985 (0.990) | 0.737 (0.741) | 2.3865 |
| exact-det vs off-det | 5 | 0.000 (0.000) | 0.000 (0.000) | 0.0000 |
| off-det vs off (stock floor) | 5 | 0.200 (0.201) | 0.138 (0.138) | 0.8365 |
| off fold i vs fold 0 (floor) | 4 | 0.009 (0.009) | 0.008 (0.008) | 0.0226 |

openfold3: pair | folds | CA RMSD A median (max) | per-copy CA RMSD A median (max) | max |dx| A
| exact vs off | 5 | 1.263 (1.711) | 1.260 (1.724) | 7.8077 |
| fast vs off | 5 | 4.713 (7.594) | 4.529 (6.939) | 50.0757 |
| exact-det vs off-det | 5 | 0.000 (0.000) | 0.000 (0.000) | 0.0000 |
| off-det vs off (stock floor) | 5 | 1.275 (1.723) | 1.383 (1.716) | 7.8848 |
| off fold i vs fold 0 (floor) | 4 | 5.967 (7.251) | 5.629 (6.845) | 53.2355 |

openfold3_ob0: pair | folds | CA RMSD A median (max) | per-copy CA RMSD A median (max) | max |dx| A
| exact vs off | 5 | 9.884 (19.642) | 9.619 (17.828) | 45.5410 |
| fast vs off | 5 | 19.282 (20.397) | 18.291 (18.819) | 51.6601 |
| exact-det vs off-det | 5 | 9.870 (19.662) | 9.605 (17.843) | 45.5232 |
| off-det vs off (stock floor) | 5 | 0.063 (0.205) | 0.067 (0.205) | 0.7015 |
| off fold i vs fold 0 (floor) | 4 | 14.352 (19.565) | 13.567 (18.024) | 44.5938 |

rosettafold3-no-prefetch-awrite: pair | folds | CA RMSD A median (max) | per-copy CA RMSD A median (max) | max |dx| A
| exact vs off | 10 | 0.566 (7.232) | 0.541 (7.535) | 22.8306 |
| fast vs off | 10 | 10.794 (19.266) | 10.878 (19.216) | 46.7276 |
| off fold i vs fold 0 (floor) | 9 | 20.378 (24.517) | 16.799 (18.677) | 68.7728 |

esmfold2-full_nomsa: pair | folds | CA RMSD A median (max) | per-copy CA RMSD A median (max) | max |dx| A
| exact vs off | 5 | 0.016 (0.021) | 0.012 (0.017) | 0.1750 |
| fast vs off | 5 | 0.055 (0.057) | 0.048 (0.051) | 0.5490 |
| exact-det vs off-det | 5 | 0.000 (0.000) | 0.000 (0.000) | 0.0000 |
| off-det vs off (stock floor) | 5 | 0.024 (0.031) | 0.016 (0.022) | 0.2830 |
| off fold i vs fold 0 (floor) | 4 | 0.011 (0.019) | 0.010 (0.018) | 0.1850 |

esmfold2-fast: pair | folds | CA RMSD A median (max) | per-copy CA RMSD A median (max) | max |dx| A
| exact vs off | 5 | 0.012 (0.013) | 0.009 (0.009) | 0.1470 |
| fast vs off | 5 | 0.090 (0.096) | 0.098 (0.099) | 1.0830 |
| exact-det vs off-det | 5 | 0.000 (0.000) | 0.000 (0.000) | 0.0000 |
| off-det vs off (stock floor) | 5 | 0.111 (0.116) | 0.115 (0.121) | 1.0370 |
| off fold i vs fold 0 (floor) | 4 | 0.011 (0.013) | 0.009 (0.010) | 0.1710 |

rfdiffusion3: pair | folds | CA RMSD A median (max) | per-copy CA RMSD A median (max) | max |dx| A
| exact vs off | 5 | 3.646 (4.862) | 5.052 (6.663) | 28.6815 |
| fast vs off | 5 | 2.761 (4.327) | 3.854 (5.922) | 30.1982 |
| exact-det vs off-det | 5 | 0.000 (0.000) | 0.000 (0.000) | 0.0000 |
| off-det vs off (stock floor) | 5 | 1.739 (5.331) | 2.416 (7.322) | 39.4378 |

