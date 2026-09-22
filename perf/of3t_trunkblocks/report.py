#!/usr/bin/env python3
"""Print BLOCKCENSUS_N384.json as the tables the state doc quotes. No measurement here."""
import json, sys

d = json.load(open(sys.argv[1] if len(sys.argv) > 1
                   else "perf/of3t_trunkblocks/BLOCKCENSUS_N384.json"))
ci, cf, fl, cp = (d["CENSUS_IN_FRAME"], d["CENSUS_vs_FLOAT64"],
                  d["CENSUS_THE_FLOOR"], d["CENSUS_PINNED_FRAME"])
ex, cx = d["EXCESS_per_block"], d["COUNTERFACTUAL"]

print("of3t-trunkblocks -- per-block census of the pairformer trunk at crop 384")
print("host %s   device_involved %s" % (d["host"], d["device_involved"]))
for nm, v in d["digests"].items():
    print("  %-22s %s  %d B" % (nm, v["sha256"], v["bytes"]))
print()
print("CONTROLS")
c = d["controls"]
print("  A/A in-frame reference against itself      %.17g  exactly zero %s"
      % (c["AA_the_in_frame_reference_against_itself"]["mass_weighted_rel_l2"],
         c["AA_the_in_frame_reference_against_itself"]["exactly_zero"]))
print("  A/A our device arm against itself          %.17g  exactly zero %s"
      % (c["AA_our_device_arm_against_itself"]["mass_weighted_rel_l2"],
         c["AA_our_device_arm_against_itself"]["exactly_zero"]))
dv = c.get("AA_device_two_runs_of3t_modelboundary", {})
if dv:
    print("  A/A device, two runs (of3t-modelboundary)  %d of %d bit-identical, worst move %.17g"
          % (dv["n_a"] - dv["n_moved"], dv["n_a"], dv["worst_rel_move"]))
print("  A16 zero-gradient baseline                 %.17g  exactly one %s"
      % (c["A16_zero_gradient_baseline"]["mass_weighted_rel_l2"],
         c["A16_zero_gradient_baseline"]["exactly_one"]))
for k, v in c["REPRODUCTION"].items():
    if isinstance(v, dict):
        print("  repro %-28s recomputed %.16f  published %.16f  rel diff %.3e"
              % (k, v["recomputed"], v["published"], v["relative_difference"]))
print()
print("IN FRAME  ours vs upstream 0.4.3's own bf16 autocast step, same capture")
print("  E = %.16f   R = %.16f   pooled = %.16f"
      % (ci["E_absolute_error_mass"], ci["R_reference_squared_norm"], ci["pooled_rel_l2"]))
print("  sum check  sum_b e_b - E = %.3e   relative %.3e"
      % (ci["sum_check_sum_of_blocks_minus_E"], ci["sum_check_relative"]))
print("  denominator floor on the reference norm %.1e: %d of %d tensors below it, "
      "%.4f %% of the mass, %.4f %% of the error; smallest reference norm %.6e"
      % (ci["denominator_floor_on_reference_norm"], ci["n_below_the_floor"], ci["tensors"],
         100 * ci["mass_share_below_the_floor"], 100 * ci["error_share_below_the_floor"],
         ci["smallest_reference_norm"]))
print("  worst per tensor by relative L2  %.6f  %s"
      % (ci["worst_rel_l2_over_tensors"], ci["worst_rel_l2_tensor"]))
print("  worst per tensor by absolute err %.6e  %s"
      % (ci["worst_absolute_error"], ci["worst_absolute_error_tensor"]))
print()
hdr = ("%4s %12s %9s %9s %10s %11s %8s %9s %-52s"
       % ("blk", "e_b", "share E", "mass", "rel in blk", "floor e_b", "ours/fl", "worst rel",
          "worst tensor (in frame)"))
print(hdr); print("-" * len(hdr))
for i in map(str, range(48)):
    p = ci["per_block"][i]
    print("%4s %12.6e %8.4f%% %8.4f%% %10.5f %11.4e %8.3f %9.4f %-52s"
          % (i, p["absolute_error_mass"], 100 * p["share_of_E"],
             100 * p["mass_share_of_the_stack"], p["pooled_rel_l2_in_block"],
             fl["per_block"][i]["absolute_error_mass"],
             ex[i]["ratio_ours_over_floor_in_absolute_mass"], p["worst_rel_l2"],
             p["worst_rel_tensor"].split("blocks.", 1)[1]))
print()
print("RANKED in frame, cumulative share of E")
for r in ci["cumulative"]:
    print("  blk %-3d %8.4f %%   cumulative %8.4f %%"
          % (r["block"], 100 * r["share_of_E"], 100 * r["cumulative"]))
print()
print("THE SAME CENSUS IN THE PINNED FRAME (the graded artifact's own denominator)")
print("  pooled %.16f reproduces the section's published %.16f at rel diff %.3e"
      % (cp["pooled_rel_l2"], cp["reproduces_the_sections_published_reading"],
         cp["relative_difference"]))
print("  in-frame top three %s   pinned top three %s   same set %s"
      % (cp["in_frame_top3"], cp["pinned_top3"], cp["top3_set_matches_in_frame"]))
for i in cp["ranked"][:10]:
    p = cp["per_block"][str(i)]
    print("  blk %-3d e=%.6e share %7.4f %%  rel in blk %8.4f  worst rel %8.4f  %s"
          % (i, p["absolute_error_mass"], 100 * p["share_of_E"],
             p["pooled_rel_l2_in_block"], p["worst_rel_l2"], p["worst_rel_tensor"]))
print()
print("FLOAT64 FRAME, for comparison: ranked %s" % cf["ranked"][:6])
print()
print("COUNTERFACTUAL  bar %.16f   trunk target %.10f"
      % (cx["A26_reachable_bar_vs_their_bf16"], cx["clause_target_for_the_trunk"]))
print("  E %.10f   E allowed %.10f   fraction of E that must go %.6f %%"
      % (cx["E_absolute_error_mass"], cx["E_allowed_by_the_clause"],
         100 * cx["fraction_of_E_that_must_be_removed"]))
for k, v in cx["controls"].items():
    print("  control %-44s %.16f%s"
          % (k, v["recomputed"],
             ("  published %.10f" % v["published"]) if "published" in v else ""))
print()
h2 = ("%3s %-34s %9s %10s %10s %7s %6s %10s %10s %7s %6s"
      % ("k", "blocks", "cum E", "trunk EX", "model EX", "x bar", "pass",
         "trunk A26", "model A26", "x bar", "pass"))
print(h2); print("-" * len(h2))
for k, v in cx["top_k"].items():
    e = v["EXACT_our_error_on_those_blocks_set_to_zero"]
    a = v["A26_our_error_there_reduced_to_upstreams_own_level"]
    if int(k) > 12 and int(k) not in (24, 48):
        continue
    print("%3s %-34s %8.4f%% %10.6f %10.6f %7.3f %6s %10.6f %10.6f %7.3f %6s"
          % (k, str(v["blocks"])[:34], 100 * v["cumulative_share_of_E"],
             e["trunk_in_frame"], e["model_scope"], e["multiple_of_the_bar"],
             e["clause_would_pass"], a["trunk_in_frame"], a["model_scope"],
             a["multiple_of_the_bar"], a["clause_would_pass"]))
print()
print("SINGLE BLOCK, top eight by share of E")
for i, v in list(cx["single_block"].items())[:8]:
    e = v["EXACT_our_error_on_those_blocks_set_to_zero"]
    a = v["A26_our_error_there_reduced_to_upstreams_own_level"]
    print("  blk %-3s share %7.4f %%  EXACT model %.6f (%.3fx, pass %s)  "
          "A26 model %.6f (%.3fx, pass %s)"
          % (i, 100 * v["share_of_E"], e["model_scope"], e["multiple_of_the_bar"],
             e["clause_would_pass"], a["model_scope"], a["multiple_of_the_bar"],
             a["clause_would_pass"]))
