import json, sys
d = json.load(open(sys.argv[1]))
labs = sys.argv[2:] or ["ours_vs_bf16auto", "ours_vs_f64", "bf16auto_vs_f64", "bf16full_vs_f64"]
for lab in labs:
    if lab not in d["by_class"]:
        continue
    print("=" * 96)
    o = d["overall"][lab]
    print("%s   OVERALL rel=%.10f abs_err=%.6e ref_norm=%.6e cos=%.4f nr=%.4f" % (
        lab, o["mass_weighted_rel_l2"], o["abs_err"], o["ref_norm"],
        o["mass_weighted_cos"], o["mass_weighted_norm_ratio"]))
    hdr = ("class", "n", "abs_err", "err%", "ref%", "enrich", "relL2", "cos", "normratio")
    print("%-10s %5s %14s %8s %8s %7s %10s %8s %9s" % hdr)
    for k, v in d["by_class"][lab].items():
        print("%-10s %5d %14.6e %7.3f%% %7.3f%% %7.3f %10.4f %8.4f %9.4f" % (
            k, v["n_tensors"], v["abs_err"], 100 * v["share_of_error_mass"],
            100 * v["share_of_reference_mass"], v["enrichment"],
            v["mass_weighted_rel_l2"], v["mass_weighted_cos"],
            v["mass_weighted_norm_ratio"]))
    print("  -- families --")
    for k, v in d["by_family"][lab].items():
        print("  %-20s %5d %14.6e %7.3f%% %7.3f%% %7.3f %10.4f" % (
            k, v["n_tensors"], v["abs_err"], 100 * v["share_of_error_mass"],
            100 * v["share_of_reference_mass"], v["enrichment"],
            v["mass_weighted_rel_l2"]))
