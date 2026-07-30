# AbAg-XM ranker rank ranges (paired bootstrap)

Same paired bootstrap as abag-xm-ranker-cis.md: 12 cells (3 generators x N in (5, 50) x DockQ thresholds (0.23, 0.8)), B=10,000 target resamples, seed 20260729. Per resample the 10 ranker columns are ranked within each cell (average ranks for ties, rank 1 = best; oracle/random are reference lines, not ranked). Rank range = [2.5, 97.5] percentiles of the rank distribution. Overlapping rank ranges mean the ordering is not resolvable at this panel size.

## gap-recovered (primary), DockQ >= 0.23, N=5

| generator | ranker | median rank | rank range | metric [95% CI] |
|---|---|---|---|---|
| opendde-abag | iptm | 3 | [1, 6.5] | 0.252 [-0.054, 0.545] |
| opendde-abag | ptm | 5 | [3, 8] | 0.232 [-0.078, 0.528] |
| opendde-abag | ranking_score | 4 | [1.5, 7] | 0.251 [-0.054, 0.544] |
| opendde-abag | complex_plddt | 4 | [1, 8] | 0.275 [-0.052, 0.600] |
| opendde-abag | pdockq2 | 10 | [7, 10] | -0.017 [-0.293, 0.261] |
| opendde-abag | ipsae | 9 | [4, 10] | 0.019 [-0.259, 0.274] |
| opendde-abag | anticonf | 7 | [4, 9] | 0.174 [-0.120, 0.459] |
| opendde-abag | pss | 3 | [1, 10] | 0.298 [0.024, 0.557] |
| opendde-abag | deeprank_ab | 7 | [2, 10] | 0.171 [0.012, 0.337] |
| opendde-abag | abag_rank | 2 | [1, 8] | 0.312 [0.064, 0.525] |

| protenix-v2 | iptm | 2 | [1, 4] | 0.352 [0.176, 0.522] |
| protenix-v2 | ptm | 7 | [6, 8] | 0.207 [0.046, 0.369] |
| protenix-v2 | ranking_score | 3 | [2, 5] | 0.348 [0.171, 0.519] |
| protenix-v2 | complex_plddt | 10 | [10, 10] | -0.318 [-0.471, -0.153] |
| protenix-v2 | pdockq2 | 3 | [1, 5] | 0.345 [0.202, 0.483] |
| protenix-v2 | ipsae | 8 | [6, 9] | 0.160 [0.009, 0.314] |
| protenix-v2 | anticonf | 6 | [4, 6] | 0.271 [0.108, 0.432] |
| protenix-v2 | pss | 5 | [1, 8] | 0.283 [0.117, 0.447] |
| protenix-v2 | deeprank_ab | 1 | [1, 6] | 0.387 [0.248, 0.519] |
| protenix-v2 | abag_rank | 9 | [8, 9] | 0.028 [-0.121, 0.178] |

| boltz2 | iptm | 8 | [6, 9] | -0.006 [-0.107, 0.097] |
| boltz2 | ptm | 10 | [9, 10] | -0.057 [-0.152, 0.040] |
| boltz2 | ranking_score | 4 | [3, 7] | 0.144 [0.034, 0.259] |
| boltz2 | complex_plddt | 2 | [1, 3] | 0.326 [0.168, 0.478] |
| boltz2 | pdockq2 | 5 | [3, 6] | 0.083 [-0.052, 0.227] |
| boltz2 | ipsae | 9 | [7, 10] | -0.012 [-0.119, 0.102] |
| boltz2 | anticonf | 7 | [5, 8] | 0.043 [-0.083, 0.178] |
| boltz2 | pss | 2 | [1, 6] | 0.305 [0.107, 0.493] |
| boltz2 | deeprank_ab | 3 | [1, 4] | 0.263 [0.133, 0.405] |
| boltz2 | abag_rank | 6 | [4, 9] | 0.063 [-0.051, 0.188] |

## gap-recovered (primary), DockQ >= 0.23, N=50

| generator | ranker | median rank | rank range | metric [95% CI] |
|---|---|---|---|---|
| opendde-abag | iptm | 4 | [2.5, 6.5] | 0.090 [-0.115, 0.341] |
| opendde-abag | ptm | 4 | [2.5, 6.5] | 0.090 [-0.115, 0.341] |
| opendde-abag | ranking_score | 4 | [2.5, 6.5] | 0.090 [-0.115, 0.341] |
| opendde-abag | complex_plddt | 4 | [2.5, 6.5] | 0.090 [-0.115, 0.341] |
| opendde-abag | pdockq2 | 8 | [1.5, 9] | -0.076 [-0.433, 0.233] |
| opendde-abag | ipsae | 10 | [9, 10] | -0.479 [-1.071, -0.074] |
| opendde-abag | anticonf | 6.5 | [3.5, 9] | 0.013 [-0.163, 0.235] |
| opendde-abag | pss | 7.5 | [2, 9] | 0.013 [-0.131, 0.205] |
| opendde-abag | deeprank_ab | 6.5 | [1, 9.5] | 0.004 [-0.365, 0.318] |
| opendde-abag | abag_rank | 1 | [1, 7.5] | 0.167 [-0.047, 0.425] |

| protenix-v2 | iptm | 3.5 | [2.5, 5.5] | 0.151 [0.015, 0.302] |
| protenix-v2 | ptm | 6 | [4, 7] | 0.074 [-0.048, 0.206] |
| protenix-v2 | ranking_score | 3.5 | [2.5, 5.5] | 0.151 [0.015, 0.302] |
| protenix-v2 | complex_plddt | 10 | [9, 10] | -0.344 [-0.522, -0.190] |
| protenix-v2 | pdockq2 | 2 | [1, 6.5] | 0.176 [0.017, 0.335] |
| protenix-v2 | ipsae | 8 | [8, 9] | -0.155 [-0.304, -0.015] |
| protenix-v2 | anticonf | 5 | [3, 7] | 0.099 [-0.036, 0.238] |
| protenix-v2 | pss | 6 | [2, 7] | 0.085 [-0.037, 0.218] |
| protenix-v2 | deeprank_ab | 1 | [1, 2] | 0.306 [0.153, 0.469] |
| protenix-v2 | abag_rank | 9 | [8, 10] | -0.160 [-0.318, -0.011] |

| boltz2 | iptm | 8 | [6, 9] | -0.059 [-0.167, 0.056] |
| boltz2 | ptm | 10 | [9, 10] | -0.177 [-0.268, -0.093] |
| boltz2 | ranking_score | 6 | [3, 8] | 0.030 [-0.091, 0.161] |
| boltz2 | complex_plddt | 1 | [1, 3.5] | 0.264 [0.112, 0.435] |
| boltz2 | pdockq2 | 6 | [3, 7.5] | 0.031 [-0.104, 0.175] |
| boltz2 | ipsae | 9 | [8, 10] | -0.147 [-0.240, -0.057] |
| boltz2 | anticonf | 6 | [3, 8] | 0.031 [-0.100, 0.173] |
| boltz2 | pss | 3 | [1, 6] | 0.175 [0.046, 0.318] |
| boltz2 | deeprank_ab | 2 | [1, 5] | 0.208 [0.066, 0.367] |
| boltz2 | abag_rank | 4 | [2, 8] | 0.089 [-0.032, 0.227] |

## gap-recovered (primary), DockQ >= 0.8, N=5

| generator | ranker | median rank | rank range | metric [95% CI] |
|---|---|---|---|---|
| opendde-abag | iptm | 7 | [4, 10] | 0.028 [-0.137, 0.185] |
| opendde-abag | ptm | 4 | [1, 7] | 0.127 [-0.031, 0.273] |
| opendde-abag | ranking_score | 5 | [2, 7] | 0.113 [-0.049, 0.262] |
| opendde-abag | complex_plddt | 1 | [1, 6] | 0.207 [0.025, 0.389] |
| opendde-abag | pdockq2 | 3 | [1, 6] | 0.158 [0.042, 0.273] |
| opendde-abag | ipsae | 10 | [7, 10] | -0.130 [-0.304, 0.021] |
| opendde-abag | anticonf | 2 | [1, 5] | 0.184 [0.066, 0.303] |
| opendde-abag | pss | 5 | [1, 8] | 0.101 [-0.042, 0.248] |
| opendde-abag | deeprank_ab | 8 | [5, 10] | -0.031 [-0.134, 0.070] |
| opendde-abag | abag_rank | 9 | [6, 10] | -0.038 [-0.143, 0.072] |

| protenix-v2 | iptm | 8 | [4, 9] | -0.086 [-0.268, 0.109] |
| protenix-v2 | ptm | 8 | [3, 9] | -0.083 [-0.267, 0.109] |
| protenix-v2 | ranking_score | 7 | [3, 8] | -0.069 [-0.255, 0.127] |
| protenix-v2 | complex_plddt | 3 | [1, 9] | 0.080 [-0.140, 0.271] |
| protenix-v2 | pdockq2 | 4 | [2, 8] | 0.041 [-0.081, 0.159] |
| protenix-v2 | ipsae | 2 | [1, 7] | 0.130 [-0.010, 0.275] |
| protenix-v2 | anticonf | 6 | [3, 9] | -0.022 [-0.157, 0.126] |
| protenix-v2 | pss | 1 | [1, 4] | 0.250 [0.063, 0.438] |
| protenix-v2 | deeprank_ab | 4 | [2, 7] | 0.035 [-0.112, 0.177] |
| protenix-v2 | abag_rank | 10 | [9, 10] | -0.227 [-0.363, -0.099] |

| boltz2 | iptm | 7 | [5, 9] | 0.185 [-0.115, 0.438] |
| boltz2 | ptm | 8 | [6, 10] | 0.168 [-0.127, 0.421] |
| boltz2 | ranking_score | 4 | [1, 6] | 0.322 [0.048, 0.555] |
| boltz2 | complex_plddt | 2 | [1, 6] | 0.452 [0.266, 0.632] |
| boltz2 | pdockq2 | 5 | [1, 8] | 0.302 [0.042, 0.502] |
| boltz2 | ipsae | 10 | [7, 10] | 0.069 [-0.209, 0.335] |
| boltz2 | anticonf | 6 | [3, 7] | 0.254 [-0.023, 0.475] |
| boltz2 | pss | 3 | [1, 9] | 0.426 [0.165, 0.660] |
| boltz2 | deeprank_ab | 1 | [1, 6] | 0.508 [0.356, 0.651] |
| boltz2 | abag_rank | 9 | [4, 10] | 0.150 [-0.078, 0.405] |

## gap-recovered (primary), DockQ >= 0.8, N=50

| generator | ranker | median rank | rank range | metric [95% CI] |
|---|---|---|---|---|
| opendde-abag | iptm | 8 | [3, 10] | -0.095 [-0.304, 0.097] |
| opendde-abag | ptm | 1 | [1, 4.5] | 0.136 [-0.016, 0.307] |
| opendde-abag | ranking_score | 4 | [1, 8] | 0.021 [-0.156, 0.200] |
| opendde-abag | complex_plddt | 4 | [2, 7.5] | 0.023 [-0.103, 0.159] |
| opendde-abag | pdockq2 | 5.5 | [1, 9] | -0.020 [-0.233, 0.176] |
| opendde-abag | ipsae | 10 | [4, 10] | -0.200 [-0.475, 0.025] |
| opendde-abag | anticonf | 7 | [2.5, 10] | -0.058 [-0.271, 0.141] |
| opendde-abag | pss | 4 | [1, 8] | 0.023 [-0.090, 0.144] |
| opendde-abag | deeprank_ab | 4 | [1, 9] | 0.021 [-0.168, 0.204] |
| opendde-abag | abag_rank | 8.5 | [4, 10] | -0.132 [-0.285, 0.005] |

| protenix-v2 | iptm | 6 | [3, 8.5] | -0.054 [-0.276, 0.151] |
| protenix-v2 | ptm | 6 | [1, 9] | -0.054 [-0.289, 0.175] |
| protenix-v2 | ranking_score | 6 | [2, 9] | -0.053 [-0.290, 0.174] |
| protenix-v2 | complex_plddt | 6 | [1.5, 9] | -0.059 [-0.371, 0.211] |
| protenix-v2 | pdockq2 | 9 | [4, 10] | -0.218 [-0.556, 0.051] |
| protenix-v2 | ipsae | 3 | [1, 6] | 0.109 [-0.015, 0.257] |
| protenix-v2 | anticonf | 7 | [2.5, 9.5] | -0.110 [-0.405, 0.134] |
| protenix-v2 | pss | 1 | [1, 4] | 0.171 [0.046, 0.333] |
| protenix-v2 | deeprank_ab | 3 | [1, 7.5] | 0.106 [-0.082, 0.315] |
| protenix-v2 | abag_rank | 10 | [8, 10] | -0.377 [-0.627, -0.181] |

| boltz2 | iptm | 8.5 | [5.5, 9.5] | -0.382 [-1.206, 0.219] |
| boltz2 | ptm | 9.5 | [7, 10] | -0.542 [-1.467, 0.087] |
| boltz2 | ranking_score | 7 | [3, 9] | -0.231 [-1.079, 0.373] |
| boltz2 | complex_plddt | 3 | [1.5, 7] | 0.282 [-0.025, 0.710] |
| boltz2 | pdockq2 | 4 | [1.5, 7] | 0.117 [-0.292, 0.563] |
| boltz2 | ipsae | 8.5 | [3, 10] | -0.386 [-1.258, 0.262] |
| boltz2 | anticonf | 6 | [2.5, 8.5] | -0.047 [-0.567, 0.426] |
| boltz2 | pss | 5 | [3, 9] | 0.075 [-0.335, 0.507] |
| boltz2 | deeprank_ab | 1.5 | [1, 6.5] | 0.418 [-0.021, 0.842] |
| boltz2 | abag_rank | 2 | [1, 5] | 0.432 [0.097, 0.836] |

## ranked success (secondary), DockQ >= 0.23, N=5

| generator | ranker | median rank | rank range | metric [95% CI] |
|---|---|---|---|---|
| opendde-abag | iptm | 3 | [1, 6.5] | 0.666 [0.593, 0.737] |
| opendde-abag | ptm | 5 | [3, 8] | 0.665 [0.592, 0.737] |
| opendde-abag | ranking_score | 4 | [1.5, 7] | 0.666 [0.593, 0.737] |
| opendde-abag | complex_plddt | 4 | [1, 8] | 0.666 [0.593, 0.738] |
| opendde-abag | pdockq2 | 10 | [7, 10] | 0.657 [0.585, 0.728] |
| opendde-abag | ipsae | 9 | [4, 10] | 0.658 [0.586, 0.729] |
| opendde-abag | anticonf | 7 | [4, 9] | 0.663 [0.591, 0.735] |
| opendde-abag | pss | 3 | [1, 10] | 0.668 [0.595, 0.740] |
| opendde-abag | deeprank_ab | 7 | [2, 10] | 0.663 [0.591, 0.734] |
| opendde-abag | abag_rank | 2 | [1, 8] | 0.668 [0.595, 0.739] |

| protenix-v2 | iptm | 2 | [1, 4] | 0.447 [0.374, 0.520] |
| protenix-v2 | ptm | 7 | [6, 8] | 0.431 [0.359, 0.504] |
| protenix-v2 | ranking_score | 3 | [2, 5] | 0.446 [0.373, 0.519] |
| protenix-v2 | complex_plddt | 10 | [10, 10] | 0.373 [0.305, 0.443] |
| protenix-v2 | pdockq2 | 3 | [1, 5] | 0.446 [0.375, 0.519] |
| protenix-v2 | ipsae | 8 | [6, 9] | 0.426 [0.355, 0.497] |
| protenix-v2 | anticonf | 6 | [4, 6] | 0.438 [0.366, 0.511] |
| protenix-v2 | pss | 5 | [1, 8] | 0.439 [0.366, 0.513] |
| protenix-v2 | deeprank_ab | 1 | [1, 6] | 0.450 [0.381, 0.521] |
| protenix-v2 | abag_rank | 9 | [8, 9] | 0.411 [0.340, 0.481] |

| boltz2 | iptm | 8 | [6, 9] | 0.287 [0.224, 0.354] |
| boltz2 | ptm | 10 | [9, 10] | 0.283 [0.220, 0.349] |
| boltz2 | ranking_score | 4 | [3, 7] | 0.301 [0.236, 0.368] |
| boltz2 | complex_plddt | 2 | [1, 3] | 0.317 [0.251, 0.385] |
| boltz2 | pdockq2 | 5 | [3, 6] | 0.295 [0.230, 0.362] |
| boltz2 | ipsae | 9 | [7, 10] | 0.287 [0.222, 0.353] |
| boltz2 | anticonf | 7 | [5, 8] | 0.291 [0.227, 0.358] |
| boltz2 | pss | 2 | [1, 6] | 0.315 [0.248, 0.385] |
| boltz2 | deeprank_ab | 3 | [1, 4] | 0.311 [0.246, 0.378] |
| boltz2 | abag_rank | 6 | [4, 9] | 0.293 [0.229, 0.361] |

## ranked success (secondary), DockQ >= 0.23, N=50

| generator | ranker | median rank | rank range | metric [95% CI] |
|---|---|---|---|---|
| opendde-abag | iptm | 4 | [2.5, 6.5] | 0.666 [0.590, 0.739] |
| opendde-abag | ptm | 4 | [2.5, 6.5] | 0.666 [0.590, 0.739] |
| opendde-abag | ranking_score | 4 | [2.5, 6.5] | 0.666 [0.590, 0.739] |
| opendde-abag | complex_plddt | 4 | [2.5, 6.5] | 0.666 [0.590, 0.739] |
| opendde-abag | pdockq2 | 8 | [1.5, 9] | 0.653 [0.578, 0.727] |
| opendde-abag | ipsae | 10 | [9, 10] | 0.622 [0.547, 0.696] |
| opendde-abag | anticonf | 6.5 | [3.5, 9] | 0.659 [0.584, 0.733] |
| opendde-abag | pss | 7.5 | [2, 9] | 0.659 [0.584, 0.733] |
| opendde-abag | deeprank_ab | 6.5 | [1, 9.5] | 0.659 [0.584, 0.733] |
| opendde-abag | abag_rank | 1 | [1, 7.5] | 0.672 [0.596, 0.745] |

| protenix-v2 | iptm | 3.5 | [2.5, 5.5] | 0.442 [0.366, 0.522] |
| protenix-v2 | ptm | 6 | [4, 7] | 0.423 [0.348, 0.503] |
| protenix-v2 | ranking_score | 3.5 | [2.5, 5.5] | 0.442 [0.366, 0.522] |
| protenix-v2 | complex_plddt | 10 | [9, 10] | 0.323 [0.255, 0.398] |
| protenix-v2 | pdockq2 | 2 | [1, 6.5] | 0.448 [0.373, 0.528] |
| protenix-v2 | ipsae | 8 | [8, 9] | 0.369 [0.299, 0.443] |
| protenix-v2 | anticonf | 5 | [3, 7] | 0.429 [0.354, 0.509] |
| protenix-v2 | pss | 6 | [2, 7] | 0.426 [0.351, 0.501] |
| protenix-v2 | deeprank_ab | 1 | [1, 2] | 0.479 [0.404, 0.559] |
| protenix-v2 | abag_rank | 9 | [8, 10] | 0.367 [0.292, 0.441] |

| boltz2 | iptm | 8 | [6, 9] | 0.274 [0.205, 0.342] |
| boltz2 | ptm | 10 | [9, 10] | 0.249 [0.186, 0.317] |
| boltz2 | ranking_score | 6 | [3, 8] | 0.292 [0.224, 0.366] |
| boltz2 | complex_plddt | 1 | [1, 3.5] | 0.342 [0.267, 0.416] |
| boltz2 | pdockq2 | 6 | [3, 7.5] | 0.292 [0.224, 0.366] |
| boltz2 | ipsae | 9 | [8, 10] | 0.255 [0.186, 0.323] |
| boltz2 | anticonf | 6 | [3, 8] | 0.292 [0.224, 0.366] |
| boltz2 | pss | 3 | [1, 6] | 0.323 [0.254, 0.397] |
| boltz2 | deeprank_ab | 2 | [1, 5] | 0.330 [0.261, 0.404] |
| boltz2 | abag_rank | 4 | [2, 8] | 0.305 [0.236, 0.379] |

## ranked success (secondary), DockQ >= 0.8, N=5

| generator | ranker | median rank | rank range | metric [95% CI] |
|---|---|---|---|---|
| opendde-abag | iptm | 7 | [4, 10] | 0.267 [0.205, 0.331] |
| opendde-abag | ptm | 4 | [1, 7] | 0.272 [0.209, 0.338] |
| opendde-abag | ranking_score | 5 | [2, 7] | 0.272 [0.209, 0.336] |
| opendde-abag | complex_plddt | 1 | [1, 6] | 0.277 [0.214, 0.343] |
| opendde-abag | pdockq2 | 3 | [1, 6] | 0.274 [0.211, 0.340] |
| opendde-abag | ipsae | 10 | [7, 10] | 0.258 [0.196, 0.321] |
| opendde-abag | anticonf | 2 | [1, 5] | 0.276 [0.213, 0.341] |
| opendde-abag | pss | 5 | [1, 8] | 0.271 [0.206, 0.339] |
| opendde-abag | deeprank_ab | 8 | [5, 10] | 0.263 [0.202, 0.328] |
| opendde-abag | abag_rank | 9 | [6, 10] | 0.263 [0.200, 0.329] |

| protenix-v2 | iptm | 8 | [4, 9] | 0.128 [0.083, 0.177] |
| protenix-v2 | ptm | 8 | [3, 9] | 0.128 [0.084, 0.177] |
| protenix-v2 | ranking_score | 7 | [3, 8] | 0.128 [0.084, 0.177] |
| protenix-v2 | complex_plddt | 3 | [1, 9] | 0.136 [0.092, 0.186] |
| protenix-v2 | pdockq2 | 4 | [2, 8] | 0.134 [0.090, 0.183] |
| protenix-v2 | ipsae | 2 | [1, 7] | 0.138 [0.092, 0.190] |
| protenix-v2 | anticonf | 6 | [3, 9] | 0.131 [0.087, 0.179] |
| protenix-v2 | pss | 1 | [1, 4] | 0.144 [0.095, 0.198] |
| protenix-v2 | deeprank_ab | 4 | [2, 7] | 0.134 [0.089, 0.182] |
| protenix-v2 | abag_rank | 10 | [9, 10] | 0.121 [0.078, 0.169] |

| boltz2 | iptm | 7 | [5, 9] | 0.114 [0.071, 0.162] |
| boltz2 | ptm | 8 | [6, 10] | 0.114 [0.071, 0.162] |
| boltz2 | ranking_score | 4 | [1, 6] | 0.118 [0.074, 0.166] |
| boltz2 | complex_plddt | 2 | [1, 6] | 0.121 [0.076, 0.170] |
| boltz2 | pdockq2 | 5 | [1, 8] | 0.117 [0.073, 0.165] |
| boltz2 | ipsae | 10 | [7, 10] | 0.111 [0.069, 0.158] |
| boltz2 | anticonf | 6 | [3, 7] | 0.116 [0.072, 0.164] |
| boltz2 | pss | 3 | [1, 9] | 0.120 [0.075, 0.170] |
| boltz2 | deeprank_ab | 1 | [1, 6] | 0.122 [0.077, 0.172] |
| boltz2 | abag_rank | 9 | [4, 10] | 0.113 [0.070, 0.161] |

## ranked success (secondary), DockQ >= 0.8, N=50

| generator | ranker | median rank | rank range | metric [95% CI] |
|---|---|---|---|---|
| opendde-abag | iptm | 8 | [3, 10] | 0.249 [0.186, 0.317] |
| opendde-abag | ptm | 1 | [1, 4.5] | 0.287 [0.217, 0.354] |
| opendde-abag | ranking_score | 4 | [1, 8] | 0.268 [0.199, 0.335] |
| opendde-abag | complex_plddt | 4 | [2, 7.5] | 0.268 [0.199, 0.335] |
| opendde-abag | pdockq2 | 5.5 | [1, 9] | 0.261 [0.193, 0.329] |
| opendde-abag | ipsae | 10 | [4, 10] | 0.233 [0.170, 0.297] |
| opendde-abag | anticonf | 7 | [2.5, 10] | 0.255 [0.193, 0.323] |
| opendde-abag | pss | 4 | [1, 8] | 0.268 [0.199, 0.335] |
| opendde-abag | deeprank_ab | 4 | [1, 9] | 0.268 [0.199, 0.335] |
| opendde-abag | abag_rank | 8.5 | [4, 10] | 0.243 [0.180, 0.311] |

| protenix-v2 | iptm | 6 | [3, 8.5] | 0.124 [0.075, 0.180] |
| protenix-v2 | ptm | 6 | [1, 9] | 0.125 [0.075, 0.174] |
| protenix-v2 | ranking_score | 6 | [2, 9] | 0.125 [0.075, 0.180] |
| protenix-v2 | complex_plddt | 6 | [1.5, 9] | 0.124 [0.075, 0.180] |
| protenix-v2 | pdockq2 | 9 | [4, 10] | 0.106 [0.062, 0.155] |
| protenix-v2 | ipsae | 3 | [1, 6] | 0.143 [0.093, 0.199] |
| protenix-v2 | anticonf | 7 | [2.5, 9.5] | 0.118 [0.068, 0.168] |
| protenix-v2 | pss | 1 | [1, 4] | 0.150 [0.099, 0.206] |
| protenix-v2 | deeprank_ab | 3 | [1, 7.5] | 0.143 [0.093, 0.199] |
| protenix-v2 | abag_rank | 10 | [8, 10] | 0.087 [0.050, 0.130] |

| boltz2 | iptm | 8.5 | [5.5, 9.5] | 0.093 [0.050, 0.143] |
| boltz2 | ptm | 9.5 | [7, 10] | 0.087 [0.043, 0.130] |
| boltz2 | ranking_score | 7 | [3, 9] | 0.099 [0.056, 0.149] |
| boltz2 | complex_plddt | 3 | [1.5, 7] | 0.118 [0.068, 0.168] |
| boltz2 | pdockq2 | 4 | [1.5, 7] | 0.112 [0.068, 0.161] |
| boltz2 | ipsae | 8.5 | [3, 10] | 0.093 [0.050, 0.143] |
| boltz2 | anticonf | 6 | [2.5, 8.5] | 0.106 [0.062, 0.155] |
| boltz2 | pss | 5 | [3, 9] | 0.110 [0.067, 0.161] |
| boltz2 | deeprank_ab | 1.5 | [1, 6.5] | 0.124 [0.075, 0.174] |
| boltz2 | abag_rank | 2 | [1, 5] | 0.124 [0.075, 0.180] |

