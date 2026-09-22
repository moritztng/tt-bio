#!/usr/bin/env python3
"""Re-price D129 and D30 on the repaired 0.4.3 denominator.

Reads only the per-tensor sidecars the arms already wrote, so nothing here needs a card. Every
figure it prints is a division of two published numbers, and both sides are named.
"""
import json, os, statistics as st

os.chdir("/home/ttuser/.coworker/wt/of3t-ditref")
OUT = "perf/of3t_ditref"

FLOOR_043 = 1.5931532097e-01      # of3t-cond043, D129 UPDATE 2: this leaf's own 0.4.3 bf16 floor
FLOOR_050 = 1.5815634233e-01      # the same leaf's 0.5.0 floor
BAR_043 = 0.2253059               # A26's bar rebuilt on the 0.4.3 floor
BAR_050 = 0.223667
D129_FILED = 0.693974             # what D129 was filed at, at the 0.5.0 capture
LEAF = "conditioned_transition.layer_norm.layer_norm_s.weight"


def load(tag):
    s = json.load(open(f"{OUT}/device_gradient_{tag}_per_tensor.json"))
    return {r["tensor"]: r for r in s["per_tensor"]}, s["provenance"]


def leaf_rows(pt):
    return {k: v for k, v in pt.items() if k.endswith(LEAF)}


def bucket(name):
    if name.startswith("diffusion_transformer.blocks."):
        return "dit"
    if name.startswith("atom_attn_enc"):
        return "enc"
    if name.startswith("atom_attn_dec"):
        return "dec"
    return "other"


rep = {}
arms = {}
for tag in ("r043_ours", "r043_ours2", "r043_permcot", "r050_ours"):
    pt, prov = load(tag)
    arms[tag] = pt
    rep[tag] = {"n": len(pt), "median_rel_l2": st.median(r["rel_l2"] for r in pt.values()),
                "cap": prov["cap"], "ref_tree": prov["ref_tree"]}

a43, a50 = arms["r043_ours"], arms["r050_ours"]

# ---- DENOM -----------------------------------------------------------------------------------
m43 = rep["r043_ours"]["median_rel_l2"]
m50 = rep["r050_ours"]["median_rel_l2"]
only43 = sorted(set(a43) - set(a50))
common = sorted(set(a43) & set(a50))
rep["DENOM"] = {
    "scope_median_043": m43, "scope_median_050": m50,
    "repair_factor_x": m50 / m43,
    "ditcot_claimed_x": 6.625214353225869,
    "compared_043": len(a43), "compared_050": len(a50),
    "newly_comparable": len(only43),
    "newly_comparable_all_perblock_lnz": all("attention_pair_bias.layer_norm_z" in k
                                             for k in only43),
    "newly_comparable_sample": only43[:3],
    # the +24 are new tensors, so the repair must also be shown on the tensors BOTH arms had
    "common_n": len(common),
    "common_median_043": st.median(a43[k]["rel_l2"] for k in common),
    "common_median_050": st.median(a50[k]["rel_l2"] for k in common),
    "newly_comparable_median_043": st.median(a43[k]["rel_l2"] for k in only43),
}
rep["DENOM"]["common_repair_factor_x"] = (rep["DENOM"]["common_median_050"]
                                          / rep["DENOM"]["common_median_043"])

# ---- D129 ------------------------------------------------------------------------------------
l43, l50 = leaf_rows(a43), leaf_rows(a50)
by43 = {k: v["rel_l2"] for k, v in l43.items()}
by50 = {k: v["rel_l2"] for k, v in l50.items()}
dit43 = {k: v for k, v in by43.items() if bucket(k) == "dit"}
dit50 = {k: v for k, v in by50.items() if bucket(k) == "dit"}
med_all_43, med_all_50 = st.median(by43.values()), st.median(by50.values())
med_dit_43, med_dit_50 = st.median(dit43.values()), st.median(dit50.values())
improved = sum(1 for k in by43 if k in by50 and by43[k] < by50[k])
rep["D129"] = {
    "leaf": LEAF, "n_instances_043": len(by43), "n_instances_050": len(by50),
    "median_all_043": med_all_43, "median_all_050": med_all_50,
    "median_dit24_043": med_dit_43, "median_dit24_050": med_dit_50,
    "filed_reading_050": D129_FILED,
    "floor_043": FLOOR_043, "floor_050": FLOOR_050, "bar_043": BAR_043, "bar_050": BAR_050,
    "ratio_to_own_floor_043": med_all_43 / FLOOR_043,
    "ratio_to_own_floor_050_as_filed": D129_FILED / FLOOR_050,
    "ratio_to_bar_043": med_all_43 / BAR_043,
    "instances_over_bar_043": sum(1 for v in by43.values() if v > BAR_043),
    "instances_over_floor_043": sum(1 for v in by43.values() if v > FLOOR_043),
    "instances_improved_043_vs_050": improved,
    "per_instance_043": {k: by43[k] for k in sorted(by43)},
    "per_instance_050": {k: by50[k] for k in sorted(by50)},
    "by_bucket_043": {b: st.median([v for k, v in by43.items() if bucket(k) == b])
                      for b in ("dit", "enc", "dec")},
}

# ---- D30 -------------------------------------------------------------------------------------
fwd = json.load(open(f"{OUT}/device_gradient_r043_ours.json"))["forward_rel_median"]
fwd50 = json.load(open(f"{OUT}/device_gradient_r050_ours.json"))["forward_rel_median"]
rep["D30"] = {
    "forward_median_043": fwd, "gradient_median_043": m43, "ratio_043_x": m43 / fwd,
    "forward_median_050": fwd50, "gradient_median_050": m50, "ratio_050_x": m50 / fwd50,
    "filed": {"forward": 8.474800850934073e-03, "gradient": 0.16588485538135056,
              "ratio_x": 19.6, "source": "perf/of3t_rebase/device_gradient_043all.json"},
    "forward_bit_identical_to_filed": fwd == 8.474800850934073e-03,
    "gradient_move_vs_filed_x": 0.16588485538135056 / m43,
}

# ---- CONTROLS --------------------------------------------------------------------------------
aa = [(k, a43[k]["rel_l2"], arms["r043_ours2"][k]["rel_l2"]) for k in a43]
aa_bad = [(k, x, y) for k, x, y in aa if x != y]
pc = arms["r043_permcot"]
rep["CONTROL"] = {
    "AA_n": len(aa), "AA_bit_identical": len(aa) - len(aa_bad), "AA_differing": len(aa_bad),
    "AA_median_a": m43, "AA_median_b": rep["r043_ours2"]["median_rel_l2"],
    "permute_cot_median": rep["r043_permcot"]["median_rel_l2"],
    "permute_cot_move_x": rep["r043_permcot"]["median_rel_l2"] / m43,
    "permute_cot_tensors_moved": sum(1 for k in a43 if pc[k]["rel_l2"] != a43[k]["rel_l2"]),
    "permute_cot_leaf_median": st.median(pc[k]["rel_l2"] for k in l43),
}

json.dump(rep, open(f"{OUT}/REPRICE.json", "w"), indent=1, sort_keys=True, default=str)

def p(*x): print(*x)
p("=== DENOM ===")
for k in ("scope_median_050", "scope_median_043", "repair_factor_x", "ditcot_claimed_x",
          "compared_050", "compared_043", "newly_comparable",
          "newly_comparable_all_perblock_lnz", "common_n", "common_median_050",
          "common_median_043", "common_repair_factor_x", "newly_comparable_median_043"):
    p(f"  {k:38s} {rep['DENOM'][k]}")
p("=== D129 ===")
for k in ("n_instances_043", "median_all_050", "median_all_043", "median_dit24_050",
          "median_dit24_043", "filed_reading_050", "ratio_to_own_floor_050_as_filed",
          "ratio_to_own_floor_043", "ratio_to_bar_043", "instances_over_bar_043",
          "instances_over_floor_043", "instances_improved_043_vs_050", "by_bucket_043"):
    p(f"  {k:38s} {rep['D129'][k]}")
p("=== D30 ===")
for k, v in rep["D30"].items():
    p(f"  {k:38s} {v}")
p("=== CONTROL ===")
for k, v in rep["CONTROL"].items():
    p(f"  {k:38s} {v}")
p(f"\nwrote {OUT}/REPRICE.json")
