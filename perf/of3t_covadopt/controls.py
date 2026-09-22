#!/usr/bin/env python3
"""of3t-covadopt: C1-C4 against the predictions committed at b2cb0e6d5.

C1 is the one that matters: restricted to the original 3,643, the new artifact must reproduce
MODEL_withtrunk_n384.json -- and it will not, because --device-refatom moves cl0/plm0 onto the
card and they feed the whole DiffusionModule. The disagreement was predicted from refcov's C2
BEFORE this ran; what is checked here is predicted against measured, per tensor and pooled.

    controls.py <new-artifact.json> <new-sidecar-dir> <A/A artifact.json> <A/A sidecar-dir>
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SIDE = REPO / "perf/of3t_modelboundary/sidecar_modelboundary"
PUB = REPO / "perf/of3t_modelboundary/MODEL_withtrunk_n384.json"
PRED = REPO / "perf/of3t_covadopt/PREDICTION.json"
OUT = Path(__file__).with_name("CONTROLS.json")
PER_TENSOR_BAR = 5.0e-02


def load(p):
    return json.loads(Path(p).read_text())


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 23), b""):
            h.update(c)
    return h.hexdigest()


def pooled(rows):
    sq = sum(r["ref_norm"] ** 2 for r in rows)
    d2 = sum((r["diff_norm"] or 0.0) ** 2 for r in rows)
    a2 = sum((r["arm_norm"] or 0.0) ** 2 for r in rows)
    dot = sum(r.get("dot") or 0.0 for r in rows)
    rels = [r["rel_l2"] for r in rows if r["rel_l2"] is not None]
    return {"n": len(rows), "mass_weighted_rel_l2": math.sqrt(d2 / sq) if sq else None,
            "mass_weighted_norm_ratio": math.sqrt(a2 / sq) if sq else None,
            "mass_weighted_cos": dot / math.sqrt(a2 * sq) if a2 > 0 and sq > 0 else None,
            "n_rel_measurable": len(rels),
            "n_over_per_tensor_bar": sum(1 for x in rels if x > PER_TENSOR_BAR)}


def drift(new, old):
    """Worst and mass-weighted disagreement of new per-tensor rel_l2 against old ones."""
    worst, worst_n, num, den, identical = 0.0, None, 0.0, 0.0, 0
    for k, r in new.items():
        p, q = old[k]["rel_l2"], r["rel_l2"]
        if r == old[k]:
            identical += 1
        if p in (None, 0.0) or q is None:
            continue
        d = abs(q - p) / p
        num += old[k]["mass_sq"] * d
        den += old[k]["mass_sq"]
        if d > worst:
            worst, worst_n = d, k
    return {"n": len(new), "n_rows_bit_identical": identical, "worst_rel_diff": worst,
            "worst_tensor": worst_n,
            "mass_weighted_rel_diff": (num / den) if den else None}


def main() -> int:
    art_a, side_a, art_b, side_b = (Path(x) for x in sys.argv[1:5])
    new = load(art_a)
    pub = load(PUB)
    pred = load(PRED)

    res = {"artifact": str(art_a), "artifact_sha256": sha256(art_a)}

    # --- C4, the A/A -------------------------------------------------------------------------
    aa = {"artifact_identical": sha256(art_a) == sha256(art_b), "sidecars": {}}
    for f in sorted(side_a.glob("*.json")):
        aa["sidecars"][f.name] = sha256(f) == sha256(side_b / f.name)
    aa["pass"] = aa["artifact_identical"] and all(aa["sidecars"].values())
    res["C4_AA"] = aa

    # --- C3, the reference digests and the denominator ---------------------------------------
    res["C3_reference_and_denominator"] = {
        "digests_read_back_in_process": {k: v["sha256"] for k, v in new["inputs"].items()},
        "all_match_their_pin": all(v.get("matches_pin") for v in new["inputs"].values()),
        "float64_pin": "1d4ea9225f854afe06a8adedcb5f8aa1c53656fbf8cefdaa3a451d0327895cc4",
        "denominator_measured": new["model_squared_gradient_norm_measured"],
        "denominator_published": 10.279642678524981,
        "relative_drift": abs(new["model_squared_gradient_norm_measured"] - 10.279642678524981)
                          / 10.279642678524981,
        "pass": all(v.get("matches_pin") for v in new["inputs"].values())
                and abs(new["model_squared_gradient_norm_measured"] - 10.279642678524981)
                / 10.279642678524981 < 1e-12,
    }

    # --- C2, coverage -------------------------------------------------------------------------
    cov = new["coverage_total"]["pct_of_model_compared"]
    res["C2_coverage"] = {
        "measured": cov, "refcov": 99.50523155277378, "difference": cov - 99.50523155277378,
        "within_1e_9": abs(cov - 99.50523155277378) < 1e-9,
        "bar": 99.2594, "over_the_bar_by": cov - 99.2594,
        "n_compared": new["coverage_total"]["n_compared"],
        "n_reference_tensors": new["coverage_total"]["n_reference_tensors"],
        "union_n_tensors": new["union_n_tensors"],
        "absent_from_reference": new["n_union_tensors_absent_from_float64_reference"],
        "pass": abs(cov - 99.50523155277378) < 1e-9 and new["union_n_tensors"] == 3660,
    }

    # --- C1, restricted to the original 3,643 -------------------------------------------------
    c1 = {}
    for leg in ("renorm_vs_FLOAT64", "renorm_vs_UPSTREAM_BF16",
                "UPSTREAM_BF16_vs_FLOAT64", "UPSTREAM_F32_vs_FLOAT64"):
        old_rows = {r["param"]: r for r in load(SIDE / f"{leg}.json")}
        new_rows = {r["param"]: r for r in load(side_a / f"{leg}.json")}
        assert set(old_rows) <= set(new_rows)
        restricted = [new_rows[k] for k in old_rows]
        entered = [new_rows[k] for k in new_rows if k not in old_rows]
        moved = {k: new_rows[k] for k in old_rows if new_rows[k] != old_rows[k]}
        held = {k: new_rows[k] for k in old_rows if new_rows[k] == old_rows[k]}
        c1[leg] = {
            "restricted_to_3643": pooled(restricted),
            "published": {k: pub["stats"][leg][k] for k in
                          ("mass_weighted_rel_l2", "mass_weighted_norm_ratio",
                           "mass_weighted_cos", "n_rel_measurable", "n_over_per_tensor_bar")},
            "n_rows_bit_identical_to_published": len(held),
            "n_rows_that_moved": len(moved),
            "drift_over_the_rows_that_moved": drift(moved, old_rows) if moved else None,
            "entering": pooled(entered) if entered else None,
            "full_3660": {k: new["stats"][leg][k] for k in
                          ("mass_weighted_rel_l2", "mass_weighted_norm_ratio",
                           "mass_weighted_cos", "n_rel_measurable", "n_over_per_tensor_bar",
                           "worst_rel_l2", "worst_tensor")},
        }
        c1[leg]["restricted_minus_published"] = (
            c1[leg]["restricted_to_3643"]["mass_weighted_rel_l2"]
            - c1[leg]["published"]["mass_weighted_rel_l2"])
    p1 = pred["C1_restricted_to_the_original_3643"]
    c1["predicted_vs_measured"] = {
        "vs_FLOAT64_restricted": {
            "predicted": p1["vs_FLOAT64_predicted"],
            "measured": c1["renorm_vs_FLOAT64"]["restricted_to_3643"]["mass_weighted_rel_l2"],
        },
        "vs_UPSTREAM_BF16_restricted": {
            "predicted_point_estimate": p1["vs_UPSTREAM_BF16_point_estimate"],
            "measured": c1["renorm_vs_UPSTREAM_BF16"]["restricted_to_3643"]
                          ["mass_weighted_rel_l2"],
        },
        "n_bit_identical": {"predicted": p1["n_tensors_that_must_be_BIT_IDENTICAL"],
                            "measured": c1["renorm_vs_FLOAT64"]
                                          ["n_rows_bit_identical_to_published"]},
        "n_moved": {"predicted": p1["n_tensors_that_must_move"],
                    "measured": c1["renorm_vs_FLOAT64"]["n_rows_that_moved"]},
        "worst_per_tensor_rel_diff": {
            "predicted": p1["per_tensor_worst_rel_diff"],
            "measured": c1["renorm_vs_FLOAT64"]["drift_over_the_rows_that_moved"]
                          ["worst_rel_diff"],
            "predicted_tensor": p1["per_tensor_worst_tensor"],
            "measured_tensor": c1["renorm_vs_FLOAT64"]["drift_over_the_rows_that_moved"]
                                 ["worst_tensor"]},
        "mass_weighted_per_tensor_rel_diff": {
            "predicted": p1["per_tensor_mass_weighted_rel_diff"],
            "measured": c1["renorm_vs_FLOAT64"]["drift_over_the_rows_that_moved"]
                          ["mass_weighted_rel_diff"]},
    }
    for k, v in c1["predicted_vs_measured"].items():
        pv = v.get("predicted", v.get("predicted_point_estimate"))
        mv = v["measured"]
        v["relative_difference"] = (abs(mv - pv) / abs(pv)) if pv else (0.0 if mv == pv else None)
    res["C1_restricted_to_the_original_3643"] = c1

    # --- the headline legs against their predictions -----------------------------------------
    hp = {
        "coverage": {"predicted": pred["coverage_total_pct_of_model_compared"], "measured": cov},
        "renorm_vs_FLOAT64": {
            "predicted": pred["renorm_vs_FLOAT64"]["mass_weighted_rel_l2"],
            "measured": new["stats"]["renorm_vs_FLOAT64"]["mass_weighted_rel_l2"]},
        "renorm_vs_FLOAT64_n_over_per_tensor_bar": {
            "predicted": pred["renorm_vs_FLOAT64"]["n_over_per_tensor_bar"],
            "measured": new["stats"]["renorm_vs_FLOAT64"]["n_over_per_tensor_bar"]},
        "UPSTREAM_BF16_vs_FLOAT64": {
            "predicted": pred["UPSTREAM_BF16_vs_FLOAT64"]["mass_weighted_rel_l2"],
            "measured": new["stats"]["UPSTREAM_BF16_vs_FLOAT64"]["mass_weighted_rel_l2"]},
        "UPSTREAM_BF16_vs_FLOAT64_n_over_per_tensor_bar": {
            "predicted": pred["UPSTREAM_BF16_vs_FLOAT64"]["n_over_per_tensor_bar"],
            "measured": new["stats"]["UPSTREAM_BF16_vs_FLOAT64"]["n_over_per_tensor_bar"]},
        "UPSTREAM_F32_vs_FLOAT64": {
            "predicted": pred["UPSTREAM_F32_vs_FLOAT64"]["mass_weighted_rel_l2"],
            "measured": new["stats"]["UPSTREAM_F32_vs_FLOAT64"]["mass_weighted_rel_l2"]},
        "A26_reachable_bar_vs_their_bf16": {
            "predicted": pred["bars"]["A26_reachable_bar_vs_their_bf16"],
            "measured": new["bars"]["A26_reachable_bar_vs_their_bf16"]},
        "renorm_vs_UPSTREAM_BF16": {
            "predicted_point_estimate":
                pred["renorm_vs_UPSTREAM_BF16"]["point_estimate_holding_the_published_geometry"],
            "rigorous_interval": pred["renorm_vs_UPSTREAM_BF16"]["rigorous_interval"],
            "measured": new["stats"]["renorm_vs_UPSTREAM_BF16"]["mass_weighted_rel_l2"]},
    }
    for k, v in hp.items():
        pv = v.get("predicted", v.get("predicted_point_estimate"))
        mv = v["measured"]
        v["relative_difference"] = (abs(mv - pv) / abs(pv)) if pv else (0.0 if mv == pv else None)
    lo, hi = pred["renorm_vs_UPSTREAM_BF16"]["rigorous_interval"]
    m = hp["renorm_vs_UPSTREAM_BF16"]["measured"]
    hp["renorm_vs_UPSTREAM_BF16"]["inside_the_rigorous_interval"] = lo <= m <= hi
    hp["renorm_vs_UPSTREAM_BF16"]["within_5pct_of_the_point_estimate"] = (
        hp["renorm_vs_UPSTREAM_BF16"]["relative_difference"] < 0.05)
    res["predicted_vs_measured"] = hp

    # --- the two clauses, at ONE scope --------------------------------------------------------
    a26 = new["bars"]["A26_reachable_bar_vs_their_bf16"]
    rel = new["stats"]["renorm_vs_UPSTREAM_BF16"]["mass_weighted_rel_l2"]
    res["the_two_clauses_at_one_scope"] = {
        "scope": "the composed 3,660, both clauses read from this one artifact",
        "coverage_clause": {"value": cov, "bar": 99.2594, "passes": cov >= 99.2594},
        "A26_clause": {"value": rel, "bar": a26, "x_bar": rel / a26, "passes": rel <= a26},
        "per_tensor_clause_D176": {
            "ours": new["stats"]["renorm_vs_FLOAT64"]["n_over_per_tensor_bar"],
            "upstream_bf16": new["stats"]["UPSTREAM_BF16_vs_FLOAT64"]["n_over_per_tensor_bar"],
            "passes": (new["stats"]["renorm_vs_FLOAT64"]["n_over_per_tensor_bar"]
                       <= new["stats"]["UPSTREAM_BF16_vs_FLOAT64"]["n_over_per_tensor_bar"])},
    }
    OUT.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))
    print("->", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
