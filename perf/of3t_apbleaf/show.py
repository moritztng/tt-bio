#!/usr/bin/env python3
"""of3t-apbleaf: read CENSUS96_N384.json back as the tables the state doc quotes."""
import json, sys
R = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "perf/of3t_apbleaf/CENSUS96_N384.json"))
print("resid mass-weighted", json.dumps(R["residue_mass_weighted"]))
print("worst", json.dumps(R["worst_case"], indent=1))
print("worst residue factors:")
for d in R["residue_worst"]:
    print("  ", d["param"], round(d["factor"], 4))
print("best residue factors:")
for d in R["residue_best"]:
    print("  ", d["param"], round(d["factor"], 4))
print()
print("leafset per frame:")
for fr, s in R["leafset"].items():
    print("  %-18s abs=%.6e refn=%.6e rel=%.6f ratio=%.4f cos=%+.4f shareErr=%.6f%% shareRef=%.6f%% enr=%.4f"
          % (fr, s["abs_err"], s["ref_norm"], s["mass_weighted_rel_l2"],
             s["mass_weighted_norm_ratio"], s["mass_weighted_cos"],
             100 * s["share_of_error_mass"], 100 * s["share_of_reference_mass"], s["enrichment"]))
print()
print("trunk per frame:")
for fr, s in R["trunk"].items():
    print("  %-18s rel=%r abs=%.6e refn=%.6e" % (fr, s["mass_weighted_rel_l2"], s["abs_err"], s["ref_norm"]))
print()
print("top blocks by share of trunk error mass (ours_vs_f64):")
bb = R["by_block"]["ours_vs_f64"]
for b, s in sorted(bb.items(), key=lambda kv: -kv[1]["share_of_error_mass"])[:10]:
    u = R["by_block"]["bf16auto_vs_f64"][b]
    print("  block %2s abs=%.6e refn=%.6e share=%.6f%% rel=%.4f up_rel=%.4f resid=%.4fx cos=%+.4f"
          % (b, s["abs_err"], s["ref_norm"], 100 * s["share_of_error_mass"],
             s["mass_weighted_rel_l2"], u["mass_weighted_rel_l2"],
             s["mass_weighted_rel_l2"] / u["mass_weighted_rel_l2"], s["mass_weighted_cos"]))
print()
print("the three named blocks, per tensor (ours_vs_f64 | upstream bf16 floor):")
for b in (44, 4, 0):
    for leaf in ("weight", "bias"):
        k = "pairformer_stack.blocks.%d.attn_pair_bias.layer_norm_a.%s" % (b, leaf)
        o = R["by_tensor"]["ours_vs_f64"][k]
        u = R["by_tensor"]["bf16auto_vs_f64"][k]
        v = R["residue_over_upstream_floor"][k]
        print("  b%-2d %-6s refn=%.6e absErr=%.6e rel=%.4f cos=%+.4f | up_rel=%.4f up_cos=%+.4f resid=%.4fx shareErr=%.4f%%"
              % (b, leaf, o["ref_norm"], o["abs_err"], o["mass_weighted_rel_l2"],
                 o["mass_weighted_cos"], u["mass_weighted_rel_l2"], u["mass_weighted_cos"],
                 v["residue_factor"], 100 * o["share_of_error_mass"]))
