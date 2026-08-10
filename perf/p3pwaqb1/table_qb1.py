import json, statistics as st
p = json.load(open("perf/p3pwaqb1/probe_qb1.json"))
print("probe ttnn label:", p.get("ttnn"), "grid:", p.get("compute_grid_main_after_device_open"))
for arm in ("pwa", "template"):
    print("==", arm, json.dumps(p[arm + "_meta"]))
    for lbl, row in p[arm].items():
        print("  %-14s region=%-9s norm=%-8s proj=%-8s eq=%-5s maxabs=%-6s l1free_live=%-9s %s" % (
            lbl, row.get("region_us"), row.get("norm_us"), row.get("proj_us"),
            row.get("torch_equal_vs_prod_cg"), row.get("max_abs_vs_prod_cg"),
            row.get("l1_free_all_consumers_live"), str(row.get("err", ""))[:60]))
print("== pair legs")
for lbl, row in p["pair"].items():
    if isinstance(row, dict):
        print("  %-22s proj=%-9s eq=%-5s %s" % (lbl, row.get("proj_us"),
              row.get("torch_equal_vs_prod_cg"), str(row.get("err", ""))[:100]))
print()
arms = {}
for a in ("base", "base2", "base3", "pwa", "tpl", "all"):
    d = json.load(open("perf/p3pwaqb1/ops_%s.json" % a)); w = d["wall"]
    g = lambda k: w[k]["wall_ms"] if k in w else None
    md = lambda k: w[k]["median_us"] if k in w else None
    arms[a] = dict(pwa_op=g("_narrow_proj_linear|in0=[298, 320, 256]|w=[256, 1]"),
                   pwa_op_med=md("_narrow_proj_linear|in0=[298, 320, 256]|w=[256, 1]"),
                   tpl_op=g("_narrow_proj_linear|in0=[1, 298, 320, 256]|w=[256, 64]"),
                   tpl_op_med=md("_narrow_proj_linear|in0=[1, 298, 320, 256]|w=[256, 64]"),
                   pwa_region=g("body:PairWeightedAveraging"), block=g("block:PairformerLayer"),
                   ln484=g("shared_layer_norm|in=[1, 298, 320, 256]"),
                   ln30=g("shared_layer_norm|in=[298, 320, 256]"),
                   plddt=d["instrumented_plddt"], cold_plddt=d["plddt"], grid=d["grid"],
                   refused=d["l1_out_refused"], fold=d["instrumented_fold_s"], cold=d["cold_s"])
    print(a, json.dumps(arms[a]))
base = [arms[a] for a in ("base", "base2", "base3")]
print()
for k in ("pwa_op", "pwa_op_med", "tpl_op", "tpl_op_med", "pwa_region", "block"):
    v = [b[k] for b in base]; m = st.mean(v)
    print("CONTROL %-11s mean=%9.3f  spread=%7.3f = %.2f %% of the mean  %s" % (
        k, m, max(v) - min(v), 100 * (max(v) - min(v)) / m, v))
    for a in ("pwa", "tpl", "all"):
        if arms[a][k] is not None:
            print("    %-4s %9.3f  delta %+8.3f  (%.1f %% of control)" % (
                a, arms[a][k], arms[a][k] - m, 100 * (arms[a][k] - m) / m))
