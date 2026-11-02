/home/moritz/.coworker/wt/spd-m-boltzgen/perf/spd/bg_grade.py:43: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df["k"] = df["id"].map(idx)
/home/moritz/.coworker/wt/spd-m-boltzgen/perf/spd/bg_grade.py:43: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df["k"] = df["id"].map(idx)
/home/moritz/.coworker/wt/spd-m-boltzgen/perf/spd/bg_grade.py:43: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df["k"] = df["id"].map(idx)
/home/moritz/.coworker/wt/spd-m-boltzgen/perf/spd/bg_grade.py:43: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df["k"] = df["id"].map(idx)
/home/moritz/.coworker/wt/spd-m-boltzgen/perf/spd/bg_grade.py:43: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df["k"] = df["id"].map(idx)
/home/moritz/.coworker/wt/spd-m-boltzgen/perf/spd/bg_grade.py:43: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df["k"] = df["id"].map(idx)
/home/moritz/.coworker/wt/spd-m-boltzgen/perf/spd/bg_grade.py:43: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df["k"] = df["id"].map(idx)
/home/moritz/.coworker/wt/spd-m-boltzgen/perf/spd/bg_grade.py:43: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df["k"] = df["id"].map(idx)
/home/moritz/.coworker/wt/spd-m-boltzgen/perf/spd/bg_grade.py:43: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df["k"] = df["id"].map(idx)
# BoltzGen grade: bh-bT0 bh-bT1 bh-bT3 vs bh-ee0 bh-ee1 bh-ee3, normal mode

| metric | floor median abs | paired mean (test - base) | 95 % CI | better |
|---|---|---|---|---|
| num_prot_tokens | 0.0000 | +0.0000 | [+0.0000, +0.0000] |  |
| num_lig_atoms | 0.0000 | +0.0000 | [+0.0000, +0.0000] |  |
| num_resolved_tokens | 0.0000 | +0.0000 | [+0.0000, +0.0000] |  |
| num_tokens | 0.0000 | +0.0000 | [+0.0000, +0.0000] |  |
| num_design | 0.0000 | +0.0000 | [+0.0000, +0.0000] |  |
| UNK_fraction | 0.0000 | +0.0000 | [+0.0000, +0.0000] |  |
| GLY_fraction | 0.0100 | +0.0004 | [-0.0113, +0.0129] |  |
| ALA_fraction | 0.0100 | +0.0237 | [-0.0125, +0.0833] |  |
| CYS_fraction | 0.0000 | -0.0021 | [-0.0054, +0.0008] |  |
| SER_fraction | 0.0500 | -0.0042 | [-0.0229, +0.0146] |  |
| PRO_fraction | 0.0100 | -0.0017 | [-0.0125, +0.0088] |  |
| THR_fraction | 0.0700 | -0.0067 | [-0.0317, +0.0158] |  |
| VAL_fraction | 0.0300 | -0.0083 | [-0.0258, +0.0087] |  |
| ILE_fraction | 0.0100 | +0.0029 | [-0.0046, +0.0108] |  |
| ASN_fraction | 0.0200 | -0.0092 | [-0.0158, -0.0025] |  |
| ASP_fraction | 0.0100 | -0.0042 | [-0.0121, +0.0029] |  |
| LEU_fraction | 0.0300 | +0.0050 | [-0.0083, +0.0183] |  |
| MET_fraction | 0.0000 | +0.0000 | [-0.0037, +0.0033] |  |
| GLN_fraction | 0.0200 | -0.0029 | [-0.0137, +0.0096] |  |
| GLU_fraction | 0.0200 | +0.0004 | [-0.0133, +0.0138] |  |
| LYS_fraction | 0.0200 | +0.0079 | [-0.0079, +0.0237] |  |
| HIS_fraction | 0.0000 | +0.0021 | [-0.0013, +0.0050] |  |
| PHE_fraction | 0.0200 | +0.0017 | [-0.0058, +0.0092] |  |
| ARG_fraction | 0.0100 | -0.0021 | [-0.0104, +0.0062] |  |
| TYR_fraction | 0.0200 | -0.0029 | [-0.0129, +0.0063] |  |
| TRP_fraction | 0.0000 | -0.0000 | [-0.0021, +0.0025] |  |
| loop | 0.0450 | -0.0217 | [-0.0492, +0.0013] |  |
| helix | 0.0300 | +0.0229 | [-0.0163, +0.0925] |  |
| sheet | 0.0550 | -0.0013 | [-0.0567, +0.0362] |  |
| liability_score | 30.0000 | +4.3750 | [-12.9167, +21.2500] |  |
| liability_num_violations | 4.0000 | +0.4167 | [-1.3750, +2.1667] |  |
| liability_high_severity_violations | 3.0000 | +0.4583 | [-1.2500, +2.1250] |  |
| liability_medium_severity_violations | 1.0000 | -0.0417 | [-0.6667, +0.5000] |  |
| liability_low_severity_violations | 0.0000 | +0.0000 | [+0.0000, +0.0000] |  |
| liability_AspBridge_count | 0.0000 | +0.0000 | [+0.0000, +0.0000] |  |
| liability_AspCleave_count | 0.0000 | +0.0000 | [+0.0000, +0.0000] |  |
| liability_ProtTryp_count | 0.0000 | +0.0000 | [+0.0000, +0.0000] |  |
| liability_DPP4_count | 0.0000 | +0.0000 | [+0.0000, +0.0000] |  |
| liability_HydroPatch_count | 0.0000 | +0.0000 | [+0.0000, +0.0000] |  |
| liability_AspBridge_position | 20.0000 | -0.8333 | [-14.9167, +14.2083] |  |
| liability_AspBridge_length | 2.0000 | -0.1667 | [-0.6667, +0.3333] |  |
| liability_AspBridge_severity | 10.0000 | -0.8333 | [-3.3333, +1.6667] |  |
| liability_AspBridge_num_positions | 0.0000 | +0.0417 | [-0.5000, +0.5833] |  |
| liability_AspBridge_avg_severity | 0.0000 | +0.4167 | [-2.0833, +2.9167] |  |
| liability_AspCleave_position | 32.5000 | -5.0833 | [-24.0833, +15.2083] |  |
| liability_AspCleave_length | 1.0000 | -0.3333 | [-0.9167, +0.2500] |  |
| liability_AspCleave_severity | 5.0000 | -1.6667 | [-4.5833, +1.2500] |  |
| liability_AspCleave_num_positions | 1.5000 | +0.2917 | [-0.2917, +0.8750] |  |
| liability_AspCleave_avg_severity | 5.0000 | +1.2500 | [-1.2500, +3.7500] |  |
| liability_NTCycl_count | 0.0000 | +0.0000 | [+0.0000, +0.0000] |  |
| liability_NTCycl_position | 0.0000 | +0.0000 | [-0.2500, +0.2500] |  |
| liability_NTCycl_length | 0.0000 | +0.0000 | [-0.1250, +0.1250] |  |
| liability_NTCycl_severity | 0.0000 | +0.0000 | [-0.6250, +0.6250] |  |
| liability_NTCycl_num_positions | 0.0000 | +0.0000 | [+0.0000, +0.0000] |  |
| liability_NTCycl_avg_severity | 0.0000 | +0.0000 | [+0.0000, +0.0000] |  |
| liability_ProtTryp_position | 0.0000 | +0.0000 | [+0.0000, +0.0000] |  |
| liability_ProtTryp_length | 0.0000 | +0.0000 | [+0.0000, +0.0000] |  |
| liability_ProtTryp_severity | 0.0000 | +0.0000 | [+0.0000, +0.0000] |  |
| liability_ProtTryp_num_positions | 3.0000 | +0.5000 | [-1.0417, +2.0417] |  |
| liability_ProtTryp_avg_severity | 0.0000 | +0.0000 | [+0.0000, +0.0000] |  |
| liability_DPP4_position | 0.0000 | -0.2500 | [-0.8333, +0.3333] |  |
| liability_DPP4_length | 0.0000 | -0.1250 | [-0.4167, +0.1667] |  |
| liability_DPP4_severity | 0.0000 | -0.6250 | [-2.0833, +0.8333] |  |
| liability_DPP4_num_positions | 0.0000 | +0.0000 | [+0.0000, +0.0000] |  |
| liability_DPP4_avg_severity | 0.0000 | +0.0000 | [+0.0000, +0.0000] |  |
| liability_MetOx_count | 0.0000 | +0.0000 | [+0.0000, +0.0000] |  |
| liability_MetOx_position | 0.0000 | +0.7917 | [-3.1667, +6.3333] |  |
| liability_MetOx_length | 0.0000 | +0.0000 | [-0.2083, +0.2083] |  |
| liability_MetOx_severity | 0.0000 | +0.0000 | [-1.0417, +1.0417] |  |
| liability_MetOx_num_positions | 0.0000 | +0.0000 | [-0.3333, +0.2917] |  |
| liability_MetOx_avg_severity | 0.0000 | +0.0000 | [-0.6250, +0.6250] |  |
| liability_TrpOx_count | 0.0000 | +0.0000 | [+0.0000, +0.0000] |  |
| liability_TrpOx_position | 5.5000 | +1.7083 | [-4.3750, +7.8333] |  |
| liability_TrpOx_length | 0.0000 | +0.0417 | [-0.1667, +0.2500] |  |
| liability_TrpOx_severity | 0.0000 | +0.4167 | [-1.6667, +2.5000] |  |
| liability_TrpOx_num_positions | 0.0000 | -0.0417 | [-0.3333, +0.2500] |  |
| liability_TrpOx_avg_severity | 0.0000 | -0.4167 | [-1.6667, +0.8333] |  |
| liability_HydroPatch_position | 26.0000 | -0.2083 | [-15.7083, +16.2500] |  |
| liability_HydroPatch_length | 3.0000 | -0.6667 | [-1.5417, +0.2083] |  |
| liability_HydroPatch_severity | 5.0000 | -0.8333 | [-2.0833, +0.6250] |  |
| liability_HydroPatch_num_positions | 0.0000 | +0.2500 | [-0.2917, +0.7917] |  |
| liability_HydroPatch_avg_severity | 0.0000 | +0.4167 | [-0.8333, +1.4583] |  |
| native_rmsd | 0.0000 | +0.0000 | [+0.0000, +0.0000] |  |
| native_rmsd_bb | 0.0000 | +0.0000 | [+0.0000, +0.0000] |  |
| native_rmsd_refolded | 0.0000 | +0.0000 | [+0.0000, +0.0000] |  |
| native_rmsd_bb_refolded | 0.0000 | +0.0000 | [+0.0000, +0.0000] |  |
| designfolding-bb_rmsd | 0.3572 | +0.2166 | [-0.1140, +0.5889] |  |
| designfolding-bb_rmsd_design | 0.3572 | +0.2166 | [-0.1140, +0.5889] |  |
| designfolding-ligand_iptm | 0.0000 | +0.0000 | [+0.0000, +0.0000] |  |
| designfolding-interaction_pae | 0.0000 | +0.0000 | [+0.0000, +0.0000] |  |
| designfolding-min_interaction_pae | 0.0000 | +0.0000 | [+0.0000, +0.0000] |  |
| designfolding-min_design_to_target_pae | 0.0000 | +0.0000 | [+0.0000, +0.0000] |  |
| designfolding-iptm | 0.0000 | +0.0000 | [+0.0000, +0.0000] |  |
| designfolding-ptm | 0.0368 | +0.0182 | [-0.0183, +0.0547] |  |
| designfolding-protein_iptm | 0.0000 | +0.0000 | [+0.0000, +0.0000] |  |
| designfolding-design_iptm | 0.0000 | +0.0000 | [+0.0000, +0.0000] |  |
| designfolding-design_iiptm | 0.0000 | +0.0000 | [+0.0000, +0.0000] |  |
| designfolding-design_to_target_iptm | 0.0000 | +0.0000 | [+0.0000, +0.0000] |  |
| designfolding-design_residue_iptm | 0.0000 | +0.0000 | [+0.0000, +0.0000] |  |
| designfolding-target_ptm | 0.0000 | +0.0000 | [+0.0000, +0.0000] |  |
| designfolding-design_ptm | 0.0368 | +0.0182 | [-0.0183, +0.0547] |  |
| designfolding-design_ipsae_min | 0.0000 | +0.0000 | [+0.0000, +0.0000] |  |
| designfolding-design_to_target_ipsae | 0.0000 | +0.0000 | [+0.0000, +0.0000] |  |
| designfolding-target_to_design_ipsae | 0.0000 | +0.0000 | [+0.0000, +0.0000] |  |
| designfolding-complex_plddt | 0.0633 | +0.0216 | [-0.0002, +0.0474] |  |
| designfolding-complex_iplddt | 0.0633 | +0.0216 | [-0.0002, +0.0474] |  |
| designfolding-complex_pde | 0.1629 | -0.0365 | [-0.1797, +0.1097] |  |
| designfolding-complex_ipde | 0.0000 | +0.0000 | [+0.0000, +0.0000] |  |
| bb_rmsd | 8.2785 | +1.9705 | [-0.8675, +4.8229] | lower |
| bb_rmsd_design | 0.3257 | -0.3303 | [-1.8151, +0.8194] |  |
| bb_rmsd_target | 0.1396 | -0.0709 | [-0.1460, +0.0030] |  |
| bb_rmsd_design_target | 16.1853 | +3.5051 | [-2.0219, +9.0497] |  |
| bb_target_aligned_rmsd_design | 23.8350 | +5.8299 | [-1.7307, +13.5444] |  |
| ligand_iptm | 0.0000 | +0.0000 | [+0.0000, +0.0000] |  |
| interaction_pae | 2.7208 | +0.9780 | [-0.1757, +2.1835] |  |
| min_interaction_pae | 5.6661 | +1.8794 | [-0.2796, +4.1312] |  |
| min_design_to_target_pae | 5.6661 | +1.8794 | [-0.2796, +4.1312] |  |
| iptm | 0.0952 | -0.0432 | [-0.0921, +0.0029] | higher |
| ptm | 0.0192 | -0.0109 | [-0.0229, +0.0001] | higher |
| protein_iptm | 0.0952 | -0.0432 | [-0.0921, +0.0029] |  |
| design_iptm | 0.0952 | -0.0432 | [-0.0921, +0.0029] | higher |
| design_iiptm | 0.1028 | -0.0408 | [-0.0920, +0.0075] |  |
| design_to_target_iptm | 0.0952 | -0.0432 | [-0.0921, +0.0029] | higher |
| design_residue_iptm | 0.0952 | -0.0432 | [-0.0921, +0.0029] |  |
| target_ptm | 0.0075 | -0.0056 | [-0.0119, +0.0000] |  |
| design_ptm | 0.0326 | -0.0003 | [-0.0317, +0.0302] | higher |
| design_ipsae_min | 0.0000 | -0.0045 | [-0.0109, -0.0004] |  |
| design_to_target_ipsae | 0.0000 | -0.0093 | [-0.0212, -0.0002] |  |
| target_to_design_ipsae | 0.0000 | -0.0045 | [-0.0109, -0.0004] |  |
| complex_plddt | 0.0159 | +0.0035 | [-0.0043, +0.0111] | higher |
| complex_iplddt | 0.0393 | +0.0060 | [-0.0080, +0.0200] |  |
| complex_pde | 0.1189 | +0.0386 | [-0.0110, +0.0928] |  |
| complex_ipde | 2.0124 | +0.9938 | [-0.2120, +2.2630] |  |
| design_largest_hydrophobic_patch_refolded | 365.7956 | +173.8047 | [-46.9641, +391.0845] |  |
| design_chain_hydrophobicity | 3.8893 | +0.8465 | [-1.5504, +3.0796] |  |
| design_hydrophobicity | 3.8893 | +0.8465 | [-1.5504, +3.0796] |  |
| delta_sasa_refolded | 242.9648 | -27.2129 | [-139.7741, +90.6344] |  |
| design_sasa_unbound_refolded | 385.6201 | +72.6165 | [-119.4465, +261.5353] |  |
| design_sasa_bound_refolded | 423.0107 | +99.8293 | [-160.4946, +357.1959] |  |
| plip_hbonds_refolded | 2.0000 | +0.3750 | [-1.0000, +1.7917] |  |
| plip_saltbridge_refolded | 2.0000 | -0.9167 | [-1.6667, -0.1667] |  |

Design CA RMSD, same seed test vs base: median 4.197 A, max 14.615 A (n=24); floor (base seed vs seed) median 10.455 A; kill bar 0.6 A.
Sequence identity test vs base: mean 0.285; floor 0.114.
Filter pass: base 24, test 24, floor run 24.
Per design: 0: 1.77 A / id 0.40, 1: 14.61 A / id 0.05, 2: 2.65 A / id 0.40, 3: 0.94 A / id 0.61, 4: 8.23 A / id 0.11, 5: 5.07 A / id 0.20, 6: 2.46 A / id 0.33, 7: 3.46 A / id 0.35, 10000: 4.18 A / id 0.19, 10001: 0.33 A / id 0.81, 10002: 3.56 A / id 0.47, 10003: 1.60 A / id 0.71, 10004: 3.95 A / id 0.23, 10005: 12.76 A / id 0.08, 10006: 6.29 A / id 0.14, 10007: 13.80 A / id 0.06, 20000: 12.57 A / id 0.07, 20001: 6.91 A / id 0.05, 20002: 2.42 A / id 0.40, 20003: 5.36 A / id 0.24, 20004: 2.44 A / id 0.46, 20005: 4.80 A / id 0.14, 20006: 4.82 A / id 0.27, 20007: 4.22 A / id 0.07

VERDICT normal: PASS

