#!/usr/bin/env python3
import json, sys
d = json.load(open(sys.argv[1]))


def f(x, n=6):
    if x is None:
        return "None"
    return ("%.*e" % (n, x)) if isinstance(x, float) else str(x)


print("== stats over the whole union ==")
for k, v in d["stats"].items():
    print("%-32s n=%5d over=%5d rel=%s median=%s r=%s cos=%s" % (
        k, v["n"], v["n_over_per_tensor_bar"], f(v["mass_weighted_rel_l2"]),
        f(v["median_rel_l2_over_tensors"]), f(v["mass_weighted_norm_ratio"], 5),
        f(v["mass_weighted_cos"], 5)))
    print("%-32s worst %s %s" % ("", v["worst_tensor"], f(v["worst_rel_l2"])))

print()
print("== per_section: pairformer_stack (this row's 2,736) ==")
for lab, secs in d["per_section"].items():
    s = secs.get("pairformer_stack")
    if not s:
        continue
    print("%-32s n=%d over=%d pct=%.5f rel=%s r=%s cos=%s" % (
        lab, s["n"], s["n_over_per_tensor_bar"], s["pct_of_model_mass"],
        f(s["mass_weighted_rel_l2"]), f(s["mass_weighted_norm_ratio"], 5),
        f(s["mass_weighted_cos"], 5)))
    print("%-32s worst %s %s" % ("", s["worst_tensor"], f(s["worst_rel_l2"])))

print()
print("== per_section: every section, renorm_vs_FLOAT64 ==")
for sec, s in sorted(d["per_section"]["renorm_vs_FLOAT64"].items(),
                     key=lambda kv: -kv[1]["pct_of_model_mass"]):
    print("%-42s pct=%9.5f n=%4d over=%4d rel=%s r=%s" % (
        sec, s["pct_of_model_mass"], s["n"], s["n_over_per_tensor_bar"],
        f(s["mass_weighted_rel_l2"]), f(s["mass_weighted_norm_ratio"], 5)))

print()
print("== coverage_by_section ==")
for k, v in d["coverage_by_section"].items():
    print("%-42s pct=%9.5f compared=%9.5f  %d/%d" % (
        k, v["pct_of_model"], v["pct_of_model_compared"], v["n_compared"], v["n_total"]))
print("TOTAL", json.dumps(d["coverage_total"]))
