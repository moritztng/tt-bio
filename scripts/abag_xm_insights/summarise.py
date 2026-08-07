"""Print the headline table straight from insights.json -- the numbers the docs quote."""

import json
import sys

d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "site/data/insights.json"))


def f(c, p=4):
    if c is None or c.get("mean") is None:
        return "n/a"
    return "%+.*f [%+.*f, %+.*f]" % (p, c["mean"], p, c["lo"], p, c["hi"])


def at(c, i):
    return {k: (v[i] if isinstance(v, list) else v) for k, v in c.items()}


print("=" * 78, "\nQ1 selection")
for m, a in d["q1_selection"]["per_model"].items():
    kg = a["k_grid"]
    i, i16 = kg.index(256), kg.index(16)
    print("\n%s  (%d targets)" % (m, a["n_targets"]))
    print("  random          %s" % f(a["random_baseline"]))
    print("  oracle @16/@256 %.4f / %s" % (a["oracle"]["mean"][i16], f(at(a["oracle"], i))))
    print("  user   @16/@256 %.4f / %s" % (a["user"]["mean"][i16], f(at(a["user"], i))))
    print("  SE @256         %s" % f(at(a["selection_efficiency"], -1), 3))
    print("  N_eff           %s" % f(a["effective_n"], 2))
    print("  gap @256        %s" % f(a["gap_256"]))
    print("  user gain 16->256   %s" % f(a["user_gain_16_to_256"]))
    print("  oracle gain 16->256 %s" % f(a["oracle_gain_16_to_256"]))
    for t, tv in a["thresholds"].items():
        print("  thr %-11s oracle %.3f  user %.3f  N_eff %s"
              % (t, tv["oracle"]["mean"][i], tv["user"]["mean"][i], f(tv["effective_n"], 2)))

print("\n" + "=" * 78, "\nQ2 confidence")
for m, a in d["q2_confidence"].items():
    print("\n%s  (%d targets)" % (m, a["n_targets"]))
    for fl in a["flavor_names"]:
        v = a["flavors"][fl]
        w = v["within_target"]
        print("  %-17s within med %+.3f  mean %s  IQR [%+.2f,%+.2f]  across %.3f"
              % (fl, w["median"], f(w["mean"], 3), w["q25"], w["q75"],
                 v["across_target_mean_dockq"]))

print("\n" + "=" * 78, "\nQ3 epitope   EJ* = %.3f" % d["q3_epitope"]["ej_star"])
for m, a in d["q3_epitope"]["per_model"].items():
    s, f2 = a["states"], a["f2"]
    print("  %-14s complete=%-5s depth=%-4d states %s" % (m, a["ej_labels_complete"],
                                                          a["curve_depth"], s))
    print("      %d unsolved; %.0f%% of failures never find the site; "
          "max-EJ median %.3f unsolved vs %.3f solved"
          % (f2["n_unsolved"], 100 * f2["frac_failures_that_never_find_site"],
             f2["max_ej_median_unsolved"], f2["max_ej_median_solved"]))
    print("      P(site)  %s" % [round(x, 3) for x in a["p_finds_site"]["mean"]])
    print("      P(pose)  %s" % [round(x, 3) for x in a["p_acceptable_pose"]["mean"]])
c = d["q3_epitope"]["cross_model"]
print("  cross-model (%d common): solved by >=1 %d, failed by all four %d, per-model %s"
      % (c["n_common_targets"], c["solved_by_at_least_one"], c["failed_by_all_four"],
         c["per_model_solved"]))

print("\n" + "=" * 78, "\nQ4 pareto  (%d targets)" % d["q4_pareto"]["n_targets"])
q4 = d["q4_pareto"]
keys = ["boltz2", "opendde", "protenix", "esmfold2", "boltz2+opendde+protenix+esmfold2"]
for metric in ("oracle", "delivered"):
    print("\n  %s  budgets %s" % (metric, q4["budgets_card_h"]))
    for k in keys:
        print("   %-33s %s" % (k, [round(x["mean"], 3) for x in q4["strategies"][k][metric]]))
print("\n  P(>=0.23)")
for k in keys:
    print("   %-33s %s"
          % (k, [round(x["mean"], 3) for x in q4["strategies"][k]["solved"]["acceptable"]]))
b = q4["budgets_card_h"]
i4, isng = b.index(0.08), b.index(2.5)
u = q4["strategies"]["boltz2+opendde+protenix+esmfold2"]
for k in ("opendde", "protenix"):
    s = q4["strategies"][k]
    print("\n  4-way @%.2f vs %s @%.2f card-h  (%.0fx less compute)"
          % (b[i4], k, b[isng], b[isng] / b[i4]))
    print("    oracle    %s   vs %s" % (f(u["oracle"][i4], 3), f(s["oracle"][isng], 3)))
    print("    P(>=0.23) %s   vs %s" % (f(u["solved"]["acceptable"][i4], 3),
                                        f(s["solved"]["acceptable"][isng], 3)))

print("\n" + "=" * 78, "\nQ6 forecast")
for m, a in d["q6_forecast"]["per_model"].items():
    fit = a["fits"]["oracle_dockq"]
    print("  %-14s power a=%.3f alpha=%.3f degenerate=%s rmse=%.4f | log rmse=%.4f"
          % (m, fit["power"]["a"], fit["power"]["alpha"], fit["power"]["degenerate"],
             fit["power"]["rmse"], fit["log"]["rmse"]))
    for t, s in a["n_for_80pct"].items():
        print("      80%% @%s: measured@256 %.3f  log-linear N=%.3g (%.3g card-h/target)"
              % (t, s["measured_at_256"], s["log_n"], s["log_card_h_per_target"]))

print("\n" + "=" * 78, "\nQ7 antibody")
for m, a in d["q7_antibody"].items():
    w = a["within_target_rho_dockq_vs_h3"]
    p = a["h3_penalty_of_dockq_pick"]
    print("  %-14s depth=%-4d n=%-4d rho(DockQ,-H3) med %+.3f  frac>0.5 %.3f  "
          "H3 penalty med %.2f A mean %s"
          % (m, a["depth"], a["n_targets"], w["median"], w["frac_above_0_5"],
             p["median_angstrom"], f(p["mean"], 2)))
    print("      H3 oracle RMSD %s" % [round(x, 2) for x in a["h3_oracle_rmsd"]["mean"]])
