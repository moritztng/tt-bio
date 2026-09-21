import json, sys
d = json.load(open(sys.argv[1]))
print("union", d["union_n_tensors"], "coverage", d["coverage_total"])
print("\n-- bars --")
for k, v in d["bars"].items():
    if k != "note":
        print("  %-52s %r" % (k, v))
print("\n-- coverage by section --")
for s, c in d["coverage_by_section"].items():
    print("  %-44s %9.4f  compared %9.4f  %d/%d"
          % (s, c["pct_of_model"], c["pct_of_model_compared"], c["n_compared"], c["n_total"]))
for arm, r in d["reconciliation"].items():
    print("\n-- reconciliation %s --" % arm)
    print("  assembled            %r" % r["assembled_over_union"])
    print("  composed (published float64 shares) %r  reldiff %r"
          % (r["composed_from_published_float64_mass_shares"], r["rel_difference_published"]))
    print("  composed (arm's own reference mass) %r  reldiff %r"
          % (r["composed_from_the_arms_own_reference_mass"], r["rel_difference_exact"]))
    w = r.get("with_other_boundary_terms")
    if w:
        print("  + other-boundary terms -> %.4f %% at %r"
              % (w["pct_of_model_mass"], w["mass_weighted_rel_l2"]))
print("\n-- per-section, arm vs UPSTREAM_BF16, against each section's own floor/r --")
fl = d["per_section"]["UPSTREAM_BF16_vs_FLOAT64"]
for key in sorted(k for k in d["per_section"] if k.endswith("_vs_UPSTREAM_BF16") and not k.startswith("ZERO")):
    print(" [%s]" % key)
    for s, v in sorted(d["per_section"][key].items(), key=lambda kv: -kv[1]["pct_of_model_mass"]):
        f = fl[s]; rr = f["mass_weighted_norm_ratio"]
        th = f["mass_weighted_rel_l2"] / rr if rr else float("nan")
        print("  %-44s %8.4f %%  ours %.6e  floor/r %.6e  x%.4f  cos %.4f  r %.4f  over-bar %d/%d"
              % (s, v["pct_of_model_mass"], v["mass_weighted_rel_l2"], th,
                 v["mass_weighted_rel_l2"] / th, v["mass_weighted_cos"] or float("nan"),
                 v["mass_weighted_norm_ratio"], v["n_over_per_tensor_bar"], v["n"]))
    print("   worst: %s  %r" % (d["stats"][key]["worst_tensor"], d["stats"][key]["worst_rel_l2"]))
print("\n-- headline stats --")
for label, s in d["stats"].items():
    print("  %-40s rel %r median %r r %r cos %r" % (label, s["mass_weighted_rel_l2"],
          s["median_rel_l2_over_tensors"], s["mass_weighted_norm_ratio"], s["mass_weighted_cos"]))
    print("  %-40s over per-tensor bar %d of %d measurable, worst %s %r"
          % ("", s["n_over_per_tensor_bar"], s["n_rel_measurable"], s["worst_tensor"], s["worst_rel_l2"]))
