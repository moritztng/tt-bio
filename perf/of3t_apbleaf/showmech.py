#!/usr/bin/env python3
"""of3t-apbleaf: read MECH_N384.json back as the tables the state doc quotes."""
import json, sys, math
R = json.load(open(sys.argv[1]))
print(json.dumps(R["mass_weighted"], indent=1))
print(json.dumps(R["clause"], indent=1))
print("\nvalidation (closed form vs torch float64 autograd vs central finite differences):")
for b, v in R["validation"].items():
    print("  block %-3s closed_vs_autograd dW=%.3e db=%.3e   fd_rel=%.3e (bar %.0e)"
          % (b, v["closed_vs_torch_autograd_dW"], v["closed_vs_torch_autograd_db"],
             v["fd_rel"], v["bar"]))
bb = R["by_block"]
print("\nper block, sorted by E_tot:")
print("  blk  dtype_x/g       refN       E_tot      E_loc      E_inh    L      H      A_x        A_g       cancel_med cancel_max  g_rows")
for b, r in sorted(bb.items(), key=lambda kv: -kv[1].get("E_tot", 0))[:14]:
    print("  %3s  %-13s %.3e  %.3e  %.3e  %.3e  %.4f %.4f  %.3e  %.3e  %8.1f  %10.1f  %d/%d"
          % (b, str(r["dtype_g"]).replace("DataType.", ""), r["ref_norm_dW"], r["E_tot"],
             r["E_loc"], r["E_inh"], r["L_block"], r["H_block"], r["A_x"], r["A_g"],
             r["cancellation_median"], r["cancellation_max"],
             r["cotangent_rows_nonzero"], r["cotangent_rows_total"]))
print("\nthe named blocks:")
for b in ("44", "4", "0", "42", "15"):
    r = bb[b]
    print("  blk %-3s L=%.6f H=%.6f  E_tot=%.4e E_loc=%.4e E_inh=%.4e  A_x=%.4e A_g=%.4e"
          % (b, r["L_block"], r["H_block"], r["E_tot"], r["E_loc"], r["E_inh"], r["A_x"], r["A_g"]))
    print("       refops_reproduce_reference=%.3e  eps_dev=%g  rel_dev=%.4f rel_f64dev=%.4f"
          % (r["refops_reproduce_reference"], r["eps_device"], r["rel_dev_vs_ref"],
             r["rel_f64dev_vs_ref"]))
    print("       cancellation median=%.1f max=%.1f  cotangent rows nonzero %d of %d  dtypes x=%s g=%s"
          % (r["cancellation_median"], r["cancellation_max"], r["cotangent_rows_nonzero"],
             r["cotangent_rows_total"], r["dtype_x"], r["dtype_g"]))
# dW-only mass sums, so A_x and A_g share the denominator they are divided by
t2 = sum(r["E_tot"] ** 2 for r in bb.values())
l2 = sum(r["E_loc"] ** 2 for r in bb.values())
i2 = sum(r["E_inh"] ** 2 for r in bb.values())
x2 = sum(r["A_x"] ** 2 for r in bb.values())
g2 = sum(r["A_g"] ** 2 for r in bb.values())
print("\ndW ONLY (48 weight tensors), same denominator for every ratio:")
print("  E_tot=%.6e L=%.6f H=%.6f A_x=%.6f A_g=%.6f  A_g/A_x=%.1fx"
      % (math.sqrt(t2), math.sqrt(l2 / t2), math.sqrt(i2 / t2),
         math.sqrt(x2 / t2), math.sqrt(g2 / t2), math.sqrt(g2 / x2)))
d2 = sum(r["db_E_tot"] ** 2 for r in bb.values())
dl = sum(r["db_E_loc"] ** 2 for r in bb.values())
di = sum(r["db_E_inh"] ** 2 for r in bb.values())
print("db ONLY (48 bias tensors): E_tot=%.6e L=%.6f H=%.6f"
      % (math.sqrt(d2), math.sqrt(dl / d2), math.sqrt(di / d2)))
