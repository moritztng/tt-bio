#!/usr/bin/env python3
"""Our repaired arm against upstream 0.4.3's OWN bf16 gradient, at the same capture.

A26's bar is upstream's own bf16 floor, and `of3t-cond043` measured it on exactly the capture
this row's arms are scored against -- `perf/of3t_cond043/BARS043.json`, arm `bf16auto`, 761
tensors, `cap = /home/ttuser/of3t_softgrad/diffcap043`, `upstream_pkg = of3pkg043`. So the
comparison is like for like: two bf16-class gradients, one theirs and one ours, both divided by
the same float64 reference at the same boundary.

Grouped the way the floor is grouped -- by LEAF, the parameter path with its block index
stripped -- because that is the unit A26's bar is stated in.
"""
import json, os, statistics as st, math

os.chdir("/home/ttuser/.coworker/wt/of3t-ditref")
OUT = "perf/of3t_ditref"
bars = json.load(open("perf/of3t_cond043/BARS043.json"))
floor = bars["arms"]["bf16auto"]["by_leaf"]
LEAF129 = "conditioned_transition.layer_norm.layer_norm_s.weight"

ours = {r["tensor"]: r for r in
        json.load(open(f"{OUT}/device_gradient_r043_ours_per_tensor.json"))["per_tensor"]}


def leafkey(name):
    """the floor's own grouping: everything after the last `blocks.<N>.`, else the full path.

    `BARS043.json` keys its 84 leaves that way -- 30 instances of
    `conditioned_transition.layer_norm.layer_norm_s.weight` across the DiT and the two atom
    transformers collapse to one leaf, which is the unit A26's bar is stated in."""
    parts = name.split(".")
    for i in range(len(parts) - 1, 0, -1):
        if parts[i].isdigit() and parts[i - 1] == "blocks":
            return ".".join(parts[i + 1:])
    return name


grp = {}
for n, r in ours.items():
    grp.setdefault(leafkey(n), []).append(r)

rows, matched, unmatched = [], 0, []
for k, rs in sorted(grp.items()):
    f = floor.get(k)
    if f is None:
        unmatched.append(k)
        continue
    matched += 1
    om = st.median(r["rel_l2"] for r in rs)
    err = math.sqrt(sum((r["rel_l2"] * r["ref_norm"]) ** 2 for r in rs))
    ref = math.sqrt(sum(r["ref_norm"] ** 2 for r in rs))
    rows.append({"leaf": k, "n_ours": len(rs), "n_floor": f["n"],
                 "ours_median": om, "floor_median": f["median_rel"],
                 "ours_mass_weighted": err / ref if ref else None,
                 "floor_mass_weighted": f["mass_weighted_rel"],
                 "ratio_median": om / f["median_rel"] if f["median_rel"] else None,
                 "floor_mass_share_of_scope": f["mass_share_of_scope"]})

# scope, mass-weighted, the same statistic the floor publishes at scope
err = math.sqrt(sum((r["rel_l2"] * r["ref_norm"]) ** 2 for r in ours.values()))
ref = math.sqrt(sum(r["ref_norm"] ** 2 for r in ours.values()))
scope_mw = err / ref
FLOOR_SCOPE_MW = json.load(open("perf/of3t_cond043/FLOOR043_bf16auto.json"))[
    "scope_mass_weighted_vs_f64"]

# LIKE FOR LIKE, and the coverage stated rather than glossed. Our port produces a device
# gradient for 547 of the reference's 761 tensors, so a scope figure over our 547 against the
# floor's 761 is two different sets. `exact` keeps only the leaves where our instance count
# equals the floor's, which is the subset on which the two statistics are the same statistic.
exact = [r for r in rows if r["n_ours"] == r["n_floor"]]
cov = sum(r["floor_mass_share_of_scope"] for r in rows)
cov_exact = sum(r["floor_mass_share_of_scope"] for r in exact)

over = [r for r in rows if r["ratio_median"] and r["ratio_median"] > 1.0]
d129 = next(r for r in rows if r["leaf"] == LEAF129)

# A14: the denominator floor on OUR compared set, read rather than assumed
minref = min(r["ref_norm"] for r in ours.values())
rep = {
    "SCOPE": {"ours_mass_weighted": scope_mw, "floor_mass_weighted": FLOOR_SCOPE_MW,
              "ours_over_floor_x": scope_mw / FLOOR_SCOPE_MW,
              "ours_median": st.median(r["rel_l2"] for r in ours.values()),
              "n_ours": len(ours), "n_floor": 761,
              "cap": "/home/ttuser/of3t_softgrad/diffcap043"},
    "D129_LEAF": d129,
    "EXACT_SUBSET": {
        "n_leaves": len(exact), "of_leaves": len(rows),
        "n_tensors_ours": sum(r["n_ours"] for r in exact),
        "floor_mass_share_covered": cov_exact,
        "all_leaves_mass_share_covered": cov,
        "ours_mass_weighted": (math.sqrt(sum((r["ours_mass_weighted"] or 0) ** 2 *
                                             r["floor_mass_share_of_scope"] for r in exact)
                                         / (cov_exact or 1))),
        "floor_mass_weighted": (math.sqrt(sum(r["floor_mass_weighted"] ** 2 *
                                              r["floor_mass_share_of_scope"] for r in exact)
                                          / (cov_exact or 1))),
        "over_floor": sum(1 for r in exact if r["ratio_median"] and r["ratio_median"] > 1.0),
        "at_or_under_floor": sum(1 for r in exact
                                 if r["ratio_median"] and r["ratio_median"] <= 1.0),
        "what": "leaves where our instance count equals the floor's, so the two mass-weighted "
                "figures are over the same tensors. Each leaf's mass-weighted rel is combined "
                "with the floor's own published mass share as the weight."},
    "LEAVES": {"matched": matched, "unmatched": unmatched,
               "over_floor": len(over), "at_or_under_floor": len(rows) - len(over),
               "worst10": sorted([r for r in rows if r["ratio_median"]],
                                 key=lambda r: -r["ratio_median"])[:10],
               "all": rows},
    "A14": {"floor_on_reference_norm": 1e-12,
            "scope_min_ref_norm": minref,
            "scope_min_tensor": min(ours.values(), key=lambda r: r["ref_norm"])["tensor"],
            "tensors_below_the_floor": sum(1 for r in ours.values() if r["ref_norm"] < 1e-12)},
    "A16": {"zero_model_median_rel": 1.0,
            "why": "rel_l2 of a zero gradient against any non-zero reference is exactly 1 by "
                   "construction, which BARS043.json measures at 1.0 over all 761 tensors",
            "ours_median": st.median(r["rel_l2"] for r in ours.values()),
            "break_control_median": 1.5675916191030375,
            "break_control_is_above_the_zero_model": True},
}
json.dump(rep, open(f"{OUT}/VS_FLOOR.json", "w"), indent=1, sort_keys=True, default=str)

print("=== SCOPE, mass-weighted, both sides at diffcap043 ===")
for k, v in rep["SCOPE"].items():
    print(f"  {k:26s} {v}")
print("=== D129's leaf ===")
for k, v in d129.items():
    print(f"  {k:26s} {v}")
print(f"=== LEAVES: {matched} matched, {len(over)} over their own floor, "
      f"{len(rows)-len(over)} at or under; unmatched {unmatched}")
print("  worst 10 by median ratio to their own bf16 floor:")
for r in rep["LEAVES"]["worst10"]:
    print(f"    {r['ratio_median']:8.3f}x  ours {r['ours_median']:.6f} floor "
          f"{r['floor_median']:.6f}  n={r['n_ours']:3d}  mass_share "
          f"{r['floor_mass_share_of_scope']:.5f}  {r['leaf']}")
e = rep["EXACT_SUBSET"]
print(f"=== EXACT SUBSET ({e['n_leaves']} of {e['of_leaves']} leaves, "
      f"{e['n_tensors_ours']} tensors, {e['floor_mass_share_covered']:.4f} of the floor's "
      f"scope mass) ===")
for k in ("ours_mass_weighted", "floor_mass_weighted", "over_floor", "at_or_under_floor",
          "all_leaves_mass_share_covered"):
    print(f"  {k:26s} {e[k]}")
print(f"  {'ours_over_floor_x':26s} {e['ours_mass_weighted']/e['floor_mass_weighted']}")
rep["EXACT_SUBSET"]["ours_over_floor_x"] = e["ours_mass_weighted"] / e["floor_mass_weighted"]
json.dump(rep, open(f"{OUT}/VS_FLOOR.json", "w"), indent=1, sort_keys=True, default=str)
print("=== A14 ===")
for k, v in rep["A14"].items(): print(f"  {k:26s} {v}")
print("=== A16 ===")
for k, v in rep["A16"].items(): print(f"  {k:26s} {v}")
print(f"\nwrote {OUT}/VS_FLOOR.json")
