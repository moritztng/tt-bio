# AbAg-XM ranker bootstrap CIs

161 scorable targets resampled with replacement, B=10,000, seed 20260729. Budget-N estimator: mean over 200 without-replacement subsamples per fold (seeded per fold). Gap-recovered = (ranked - random)/(oracle - random). 95% percentile CIs; a difference is significant iff its CI excludes 0.

## DockQ >= 0.23, N=5: gap-recovered by ranker

| generator | ranker | gap-recovered | 95% CI | vs ranking_score |
|---|---|---|---|---|
| opendde-abag | iptm | 0.252 [-0.054, 0.545] | +0.001 [-0.002, +0.005] |
| opendde-abag | ptm | 0.232 [-0.078, 0.528] | -0.019 [-0.038, -0.004] **sig** |
| opendde-abag | ranking_score | 0.251 [-0.054, 0.544] | - |
| opendde-abag | complex_plddt | 0.275 [-0.052, 0.600] | +0.024 [-0.264, +0.350] |
| opendde-abag | pdockq2 | -0.017 [-0.293, 0.261] | -0.268 [-0.515, -0.054] **sig** |
| opendde-abag | ipsae | 0.019 [-0.259, 0.274] | -0.232 [-0.558, +0.105] |
| opendde-abag | anticonf | 0.174 [-0.120, 0.459] | -0.077 [-0.157, -0.014] **sig** |
| opendde-abag | pss | 0.298 [0.024, 0.557] | +0.047 [-0.398, +0.518] |
| opendde-abag | deeprank_ab | 0.171 [0.012, 0.337] | -0.080 [-0.438, +0.284] |
| opendde-abag | abag_rank | 0.312 [0.064, 0.525] | +0.061 [-0.375, +0.489] |

| protenix-v2 | iptm | 0.352 [0.176, 0.522] | +0.005 [-0.001, +0.012] |
| protenix-v2 | ptm | 0.207 [0.046, 0.369] | -0.141 [-0.240, -0.059] **sig** |
| protenix-v2 | ranking_score | 0.348 [0.171, 0.519] | - |
| protenix-v2 | complex_plddt | -0.318 [-0.471, -0.153] | -0.665 [-0.906, -0.426] **sig** |
| protenix-v2 | pdockq2 | 0.345 [0.202, 0.483] | -0.002 [-0.109, +0.112] |
| protenix-v2 | ipsae | 0.160 [0.009, 0.314] | -0.187 [-0.300, -0.081] **sig** |
| protenix-v2 | anticonf | 0.271 [0.108, 0.432] | -0.077 [-0.144, -0.014] **sig** |
| protenix-v2 | pss | 0.283 [0.117, 0.447] | -0.064 [-0.218, +0.090] |
| protenix-v2 | deeprank_ab | 0.387 [0.248, 0.519] | +0.040 [-0.114, +0.202] |
| protenix-v2 | abag_rank | 0.028 [-0.121, 0.178] | -0.320 [-0.486, -0.162] **sig** |

| boltz2 | iptm | -0.006 [-0.107, 0.097] | -0.150 [-0.248, -0.050] **sig** |
| boltz2 | ptm | -0.057 [-0.152, 0.040] | -0.201 [-0.303, -0.098] **sig** |
| boltz2 | ranking_score | 0.144 [0.034, 0.259] | - |
| boltz2 | complex_plddt | 0.326 [0.168, 0.478] | +0.182 [+0.076, +0.287] **sig** |
| boltz2 | pdockq2 | 0.083 [-0.052, 0.227] | -0.061 [-0.193, +0.079] |
| boltz2 | ipsae | -0.012 [-0.119, 0.102] | -0.156 [-0.278, -0.027] **sig** |
| boltz2 | anticonf | 0.043 [-0.083, 0.178] | -0.101 [-0.222, +0.029] |
| boltz2 | pss | 0.305 [0.107, 0.493] | +0.161 [-0.020, +0.339] |
| boltz2 | deeprank_ab | 0.263 [0.133, 0.405] | +0.120 [-0.028, +0.272] |
| boltz2 | abag_rank | 0.063 [-0.051, 0.188] | -0.081 [-0.199, +0.043] |

## DockQ >= 0.23, N=50: gap-recovered by ranker

| generator | ranker | gap-recovered | 95% CI | vs ranking_score |
|---|---|---|---|---|
| opendde-abag | iptm | 0.090 [-0.115, 0.341] | +0.000 [+0.000, +0.000] |
| opendde-abag | ptm | 0.090 [-0.115, 0.341] | +0.000 [+0.000, +0.000] |
| opendde-abag | ranking_score | 0.090 [-0.115, 0.341] | - |
| opendde-abag | complex_plddt | 0.090 [-0.115, 0.341] | +0.000 [+0.000, +0.000] |
| opendde-abag | pdockq2 | -0.076 [-0.433, 0.233] | -0.165 [-0.547, +0.129] |
| opendde-abag | ipsae | -0.479 [-1.071, -0.074] | -0.569 [-1.214, -0.119] **sig** |
| opendde-abag | anticonf | 0.013 [-0.163, 0.235] | -0.077 [-0.258, +0.000] |
| opendde-abag | pss | 0.013 [-0.131, 0.205] | -0.077 [-0.432, +0.275] |
| opendde-abag | deeprank_ab | 0.004 [-0.365, 0.318] | -0.086 [-0.411, +0.174] |
| opendde-abag | abag_rank | 0.167 [-0.047, 0.425] | +0.078 [-0.271, +0.432] |

| protenix-v2 | iptm | 0.151 [0.015, 0.302] | +0.000 [+0.000, +0.000] |
| protenix-v2 | ptm | 0.074 [-0.048, 0.206] | -0.077 [-0.169, +0.000] |
| protenix-v2 | ranking_score | 0.151 [0.015, 0.302] | - |
| protenix-v2 | complex_plddt | -0.344 [-0.522, -0.190] | -0.494 [-0.751, -0.274] **sig** |
| protenix-v2 | pdockq2 | 0.176 [0.017, 0.335] | +0.026 [-0.129, +0.175] |
| protenix-v2 | ipsae | -0.155 [-0.304, -0.015] | -0.305 [-0.522, -0.116] **sig** |
| protenix-v2 | anticonf | 0.099 [-0.036, 0.238] | -0.052 [-0.160, +0.047] |
| protenix-v2 | pss | 0.085 [-0.037, 0.218] | -0.065 [-0.256, +0.127] |
| protenix-v2 | deeprank_ab | 0.306 [0.153, 0.469] | +0.156 [+0.000, +0.313] |
| protenix-v2 | abag_rank | -0.160 [-0.318, -0.011] | -0.310 [-0.524, -0.121] **sig** |

| boltz2 | iptm | -0.059 [-0.167, 0.056] | -0.089 [-0.225, +0.033] |
| boltz2 | ptm | -0.177 [-0.268, -0.093] | -0.207 [-0.365, -0.073] **sig** |
| boltz2 | ranking_score | 0.030 [-0.091, 0.161] | - |
| boltz2 | complex_plddt | 0.264 [0.112, 0.435] | +0.234 [+0.092, +0.400] **sig** |
| boltz2 | pdockq2 | 0.031 [-0.104, 0.175] | +0.001 [-0.165, +0.168] |
| boltz2 | ipsae | -0.147 [-0.240, -0.057] | -0.177 [-0.329, -0.056] **sig** |
| boltz2 | anticonf | 0.031 [-0.100, 0.173] | +0.001 [-0.145, +0.144] |
| boltz2 | pss | 0.175 [0.046, 0.318] | +0.145 [-0.028, +0.323] |
| boltz2 | deeprank_ab | 0.208 [0.066, 0.367] | +0.178 [+0.000, +0.380] |
| boltz2 | abag_rank | 0.089 [-0.032, 0.227] | +0.059 [-0.122, +0.247] |

## DockQ >= 0.8, N=5: gap-recovered by ranker

| generator | ranker | gap-recovered | 95% CI | vs ranking_score |
|---|---|---|---|---|
| opendde-abag | iptm | 0.028 [-0.137, 0.185] | -0.085 [-0.180, -0.016] **sig** |
| opendde-abag | ptm | 0.127 [-0.031, 0.273] | +0.015 [-0.081, +0.108] |
| opendde-abag | ranking_score | 0.113 [-0.049, 0.262] | - |
| opendde-abag | complex_plddt | 0.207 [0.025, 0.389] | +0.094 [-0.060, +0.257] |
| opendde-abag | pdockq2 | 0.158 [0.042, 0.273] | +0.045 [-0.110, +0.209] |
| opendde-abag | ipsae | -0.130 [-0.304, 0.021] | -0.243 [-0.488, -0.017] **sig** |
| opendde-abag | anticonf | 0.184 [0.066, 0.303] | +0.071 [-0.065, +0.215] |
| opendde-abag | pss | 0.101 [-0.042, 0.248] | -0.012 [-0.244, +0.247] |
| opendde-abag | deeprank_ab | -0.031 [-0.134, 0.070] | -0.144 [-0.328, +0.042] |
| opendde-abag | abag_rank | -0.038 [-0.143, 0.072] | -0.151 [-0.295, +0.018] |

| protenix-v2 | iptm | -0.086 [-0.268, 0.109] | -0.017 [-0.049, +0.009] |
| protenix-v2 | ptm | -0.083 [-0.267, 0.109] | -0.014 [-0.110, +0.074] |
| protenix-v2 | ranking_score | -0.069 [-0.255, 0.127] | - |
| protenix-v2 | complex_plddt | 0.080 [-0.140, 0.271] | +0.149 [-0.208, +0.470] |
| protenix-v2 | pdockq2 | 0.041 [-0.081, 0.159] | +0.110 [-0.129, +0.338] |
| protenix-v2 | ipsae | 0.130 [-0.010, 0.275] | +0.199 [-0.058, +0.446] |
| protenix-v2 | anticonf | -0.022 [-0.157, 0.126] | +0.047 [-0.134, +0.204] |
| protenix-v2 | pss | 0.250 [0.063, 0.438] | +0.319 [+0.083, +0.573] **sig** |
| protenix-v2 | deeprank_ab | 0.035 [-0.112, 0.177] | +0.104 [-0.040, +0.236] |
| protenix-v2 | abag_rank | -0.227 [-0.363, -0.099] | -0.158 [-0.285, -0.035] **sig** |

| boltz2 | iptm | 0.185 [-0.115, 0.438] | -0.137 [-0.215, -0.056] **sig** |
| boltz2 | ptm | 0.168 [-0.127, 0.421] | -0.153 [-0.226, -0.076] **sig** |
| boltz2 | ranking_score | 0.322 [0.048, 0.555] | - |
| boltz2 | complex_plddt | 0.452 [0.266, 0.632] | +0.131 [-0.097, +0.376] |
| boltz2 | pdockq2 | 0.302 [0.042, 0.502] | -0.020 [-0.159, +0.156] |
| boltz2 | ipsae | 0.069 [-0.209, 0.335] | -0.253 [-0.417, -0.104] **sig** |
| boltz2 | anticonf | 0.254 [-0.023, 0.475] | -0.068 [-0.160, +0.034] |
| boltz2 | pss | 0.426 [0.165, 0.660] | +0.105 [-0.225, +0.448] |
| boltz2 | deeprank_ab | 0.508 [0.356, 0.651] | +0.187 [-0.113, +0.539] |
| boltz2 | abag_rank | 0.150 [-0.078, 0.405] | -0.171 [-0.451, +0.126] |

## DockQ >= 0.8, N=50: gap-recovered by ranker

| generator | ranker | gap-recovered | 95% CI | vs ranking_score |
|---|---|---|---|---|
| opendde-abag | iptm | -0.095 [-0.304, 0.097] | -0.116 [-0.268, +0.000] |
| opendde-abag | ptm | 0.136 [-0.016, 0.307] | +0.115 [-0.074, +0.325] |
| opendde-abag | ranking_score | 0.021 [-0.156, 0.200] | - |
| opendde-abag | complex_plddt | 0.023 [-0.103, 0.159] | +0.001 [-0.179, +0.193] |
| opendde-abag | pdockq2 | -0.020 [-0.233, 0.176] | -0.041 [-0.322, +0.228] |
| opendde-abag | ipsae | -0.200 [-0.475, 0.025] | -0.221 [-0.560, +0.061] |
| opendde-abag | anticonf | -0.058 [-0.271, 0.141] | -0.080 [-0.353, +0.177] |
| opendde-abag | pss | 0.023 [-0.090, 0.144] | +0.002 [-0.176, +0.192] |
| opendde-abag | deeprank_ab | 0.021 [-0.168, 0.204] | -0.001 [-0.265, +0.259] |
| opendde-abag | abag_rank | -0.132 [-0.285, 0.005] | -0.153 [-0.396, +0.075] |

| protenix-v2 | iptm | -0.054 [-0.276, 0.151] | -0.001 [-0.152, +0.155] |
| protenix-v2 | ptm | -0.054 [-0.289, 0.175] | -0.000 [-0.216, +0.213] |
| protenix-v2 | ranking_score | -0.053 [-0.290, 0.174] | - |
| protenix-v2 | complex_plddt | -0.059 [-0.371, 0.211] | -0.005 [-0.359, +0.326] |
| protenix-v2 | pdockq2 | -0.218 [-0.556, 0.051] | -0.165 [-0.528, +0.146] |
| protenix-v2 | ipsae | 0.109 [-0.015, 0.257] | +0.162 [-0.106, +0.464] |
| protenix-v2 | anticonf | -0.110 [-0.405, 0.134] | -0.057 [-0.428, +0.285] |
| protenix-v2 | pss | 0.171 [0.046, 0.333] | +0.225 [-0.055, +0.558] |
| protenix-v2 | deeprank_ab | 0.106 [-0.082, 0.315] | +0.159 [-0.148, +0.490] |
| protenix-v2 | abag_rank | -0.377 [-0.627, -0.181] | -0.324 [-0.598, -0.101] **sig** |

| boltz2 | iptm | -0.382 [-1.206, 0.219] | -0.151 [-0.513, +0.000] |
| boltz2 | ptm | -0.542 [-1.467, 0.087] | -0.311 [-0.852, +0.000] |
| boltz2 | ranking_score | -0.231 [-1.079, 0.373] | - |
| boltz2 | complex_plddt | 0.282 [-0.025, 0.710] | +0.513 [-0.151, +1.540] |
| boltz2 | pdockq2 | 0.117 [-0.292, 0.563] | +0.348 [-0.242, +1.244] |
| boltz2 | ipsae | -0.386 [-1.258, 0.262] | -0.155 [-0.692, +0.341] |
| boltz2 | anticonf | -0.047 [-0.567, 0.426] | +0.184 [-0.327, +0.896] |
| boltz2 | pss | 0.075 [-0.335, 0.507] | +0.306 [-0.274, +1.176] |
| boltz2 | deeprank_ab | 0.418 [-0.021, 0.842] | +0.649 [+0.000, +1.651] |
| boltz2 | abag_rank | 0.432 [0.097, 0.836] | +0.663 [+0.000, +1.706] |

## Spearman (per-target mean and global pooled, bootstrap over targets)

| generator | ranker | per-target mean [CI] | global [CI] |
|---|---|---|---|
| opendde-abag | iptm | -0.008 [-0.065, 0.050] | n/a |
| opendde-abag | ptm | 0.031 [-0.026, 0.090] | n/a |
| opendde-abag | ranking_score | 0.017 [-0.042, 0.076] | 0.760 [0.683, 0.822] |
| opendde-abag | complex_plddt | 0.008 [-0.048, 0.064] | n/a |
| opendde-abag | pdockq2 | 0.026 [-0.024, 0.077] | n/a |
| opendde-abag | ipsae | -0.045 [-0.102, 0.010] | n/a |
| opendde-abag | anticonf | 0.037 [-0.016, 0.092] | n/a |
| opendde-abag | pss | 0.153 [0.097, 0.207] | n/a |
| opendde-abag | deeprank_ab | 0.035 [0.005, 0.064] | 0.573 [0.467, 0.668] |
| opendde-abag | abag_rank | -0.000 [-0.044, 0.043] | 0.493 [0.418, 0.560] |
| protenix-v2 | iptm | 0.131 [0.066, 0.196] | n/a |
| protenix-v2 | ptm | 0.105 [0.046, 0.164] | n/a |
| protenix-v2 | ranking_score | 0.133 [0.069, 0.198] | 0.719 [0.651, 0.777] |
| protenix-v2 | complex_plddt | -0.001 [-0.060, 0.059] | n/a |
| protenix-v2 | pdockq2 | 0.126 [0.065, 0.188] | n/a |
| protenix-v2 | ipsae | 0.187 [0.118, 0.256] | n/a |
| protenix-v2 | anticonf | 0.109 [0.050, 0.168] | n/a |
| protenix-v2 | pss | 0.217 [0.151, 0.284] | n/a |
| protenix-v2 | deeprank_ab | 0.106 [0.063, 0.149] | 0.614 [0.537, 0.680] |
| protenix-v2 | abag_rank | 0.051 [0.003, 0.099] | 0.428 [0.341, 0.506] |
| boltz2 | iptm | 0.009 [-0.030, 0.048] | n/a |
| boltz2 | ptm | 0.004 [-0.035, 0.043] | n/a |
| boltz2 | ranking_score | 0.021 [-0.022, 0.062] | 0.622 [0.534, 0.698] |
| boltz2 | complex_plddt | 0.036 [-0.017, 0.088] | n/a |
| boltz2 | pdockq2 | 0.031 [-0.013, 0.077] | n/a |
| boltz2 | ipsae | 0.008 [-0.032, 0.049] | n/a |
| boltz2 | anticonf | 0.026 [-0.016, 0.069] | n/a |
| boltz2 | pss | 0.145 [0.078, 0.210] | n/a |
| boltz2 | deeprank_ab | 0.079 [0.039, 0.118] | 0.493 [0.397, 0.584] |
| boltz2 | abag_rank | 0.025 [-0.012, 0.061] | 0.393 [0.306, 0.475] |

## Claim verdicts (the assertions this table exists to settle)

- "DeepRank-Ab recovers a fraction of the oracle gap": at DockQ>=0.23, N=50 the point estimates are opendde-abag 0.4% [-36.5%, 31.8%], protenix-v2 30.6% [15.3%, 46.9%], boltz2 20.8% [6.6%, 36.7%]. Significant (CI excludes 0) on protenix-v2 and boltz2 only; on opendde-abag it is consistent with noise.
- "DeepRank-Ab beats ranking by native ipTM": paired gap-recovered difference deeprank_ab - iptm at 0.23/N=50 is opendde-abag -0.086 [-0.411, +0.174] (includes 0), protenix-v2 +0.156 [+0.000, +0.313] (includes 0), boltz2 +0.267 [+0.075, +0.472] (significant).
- "ABAG-Rank does not transfer": abag_rank gap-recovered on protenix-v2 is N=5: 2.8% [-12.1%, 17.8%]; N=50: -16.0% [-31.8%, -1.1%] -- negative, CI excludes 0 at N=50. Verdict stands.
- abag_rank vs native ranking_score at 0.23/N=50: opendde-abag +0.078 [-0.271, +0.432] (includes 0), protenix-v2 -0.310 [-0.524, -0.121] (significant), boltz2 +0.059 [-0.122, +0.247] (includes 0).
- Largest gap-recovered any ranker achieves at 0.23/N=50: 30.6% [15.3%, 46.9%] (deeprank_ab on protenix-v2). The earlier session claim "no ranker exceeds ~22%" is REVISED upward by this table.
