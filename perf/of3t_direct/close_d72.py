#!/usr/bin/env python3
"""Close D72's five AT-OR-BETTER rows against upstream's own gradient, one table.

D72 re-scored every section against upstream's own bf16 error instead of a float64 ideal
upstream never runs, and read 42.2794 % of OpenFold3's squared gradient norm as AT OR BETTER
than the recipe's own deviation. Both of D72's columns are distances from the SAME float64
reference, and two distances from one reference do not order each other, so that reading was
limited by construction.

`of3t-trajectory` removed the shared subtrahend for the two of those rows inside its 547-tensor
device arm; this row measures the other three. Every AT-OR-BETTER row in D72's table now has a
direct column, so the headline can be recomputed rather than argued.

Every entry carries where its numbers come from, per entry and not as a comment on the file.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRAJ = HERE.parent / "of3t_trajectory" / "D72_ROW_BY_ROW_WITHOUT_THE_SHARED_SUBTRAHEND.json"

# D72's table, as published, for the five rows it read AT OR BETTER. pct is D72's own figure so
# the recomputed headline divides by the same denominator D72 used.
AT_OR_BETTER = {
    "diffusion_module.diffusion_conditioning": 36.9462,
    "aux_heads": 2.8431,
    "msa_module": 1.2400,
    "diffusion_module.layer_norm_s": 0.9835,
    "diffusion_module.layer_norm_a": 0.2666,
}
# this row's own measurements, scope -> the DIRECT_*.json it produced
MINE = {
    "diffusion_module.diffusion_conditioning": "DIRECT_diffusion_conditioning.json",
    "aux_heads": "DIRECT_aux_heads.json",
    "msa_module": "DIRECT_msa_module.json",
}


def from_mine(path):
    d = json.loads((HERE / path).read_text())
    pre = d["PRE_REGISTERED_READING"]
    head = d["pairs"]["DEVICE_vs_UPSTREAM_BF16"]["sets"][0]
    ctl = {k: d["pairs"][k]["sets"][0]["mass_weighted_rel_l2"]
           for k in ("ZERO_vs_UPSTREAM_BF16", "UPSTREAM_BF16_vs_UPSTREAM_F32",
                     "UPSTREAM_PERMUTED_DRAWS_vs_UPSTREAM_BF16")
           if k in d["pairs"]}
    if "DEVICE_PERMUTED_COTANGENT_vs_UPSTREAM_BF16" in d["pairs"]:
        ctl["our_cotangent_break"] = (d["pairs"]["DEVICE_PERMUTED_COTANGENT_vs_UPSTREAM_BF16"]
                                       ["sets"][0]["mass_weighted_rel_l2"])
    return {
        "pct_of_model_mass_reached": d["scope"]["pct_of_model_mass"],
        "n_tensors": d["scope"]["n_tensors"],
        "device_vs_float64": pre["our_distance_from_float64_on_this_scope"],
        "device_vs_bf16_direct": pre["headline_mass_weighted_rel_l2"],
        "median_over_tensors": head["median_rel_l2_over_tensors"],
        "norm_ratio": head["mass_weighted_norm_ratio"],
        "cos": head["mass_weighted_cos"],
        "n_over_per_tensor_bar": head["n_over_per_tensor_bar"],
        "n_rel_measurable": head["n_rel_measurable"],
        "worst_tensor": head["worst_tensor"],
        "worst_rel_l2": head["worst_rel_l2"],
        "scope_floor_bf16_vs_float64": pre["scope_floor_rel_bf16_vs_float64"],
        "scope_norm_ratio_r": pre["scope_norm_ratio_r_bf16_over_float64"],
        "threshold_floor_over_r": pre["threshold_a_perfect_port_would_read__floor_over_r"],
        "multiples_of_threshold": pre["multiples_of_that_threshold"],
        "survives": pre["headline_mass_weighted_rel_l2"] <= pre[
            "threshold_a_perfect_port_would_read__floor_over_r"],
        "error_geometry": d["ERROR_GEOMETRY"],
        "controls": ctl,
        "capture_identity": d["diffcap_is_the_bundles_float64"],
        "reference_digests": {k: v["sha256"] for k, v in d["inputs"].items()},
        "source": f"perf/of3t_direct/{path}, this row, set 'the device arm's scope'",
    }


def from_traj(section, tbl):
    row = next(r for r in tbl if r["section"] == section)
    return {
        "pct_of_model_mass_reached": row.get("pct_of_model_mass_here"),
        "device_vs_float64": row["device_vs_float64_D72"],
        "device_vs_bf16_direct": row["device_vs_bf16_direct"],
        "median_over_tensors": row.get("median_over_tensors"),
        "norm_ratio": row.get("norm_ratio"),
        "cos": row.get("cos"),
        "worst_tensor": row.get("worst_tensor"),
        "worst_rel_l2": row.get("worst_rel_l2"),
        "scope_floor_bf16_vs_float64": row.get("bf16_vs_float64_here"),
        "multiples_of_this_sections_own_bf16_floor": row.get(
            "multiples_of_this_sections_own_bf16_floor"),
        "controls": {"our_cotangent_break": row.get("break_control_permuted_cotangent")},
        "source": ("perf/of3t_trajectory/D72_ROW_BY_ROW_WITHOUT_THE_SHARED_SUBTRAHEND.json, "
                   f"row {section!r}, measured by of3t-trajectory at pass 175"),
        "note": ("floor/r is not recomputed for this row: trajectory scored it against the "
                 "547-tensor scope constant. Its own floor and multiple are carried as "
                 "published."),
    }


def main():
    tbl = json.loads(TRAJ.read_text())["table"]
    rows = {}
    for sec in AT_OR_BETTER:
        rows[sec] = (from_mine(MINE[sec]) if sec in MINE else from_traj(sec, tbl))
        rows[sec]["pct_of_model_mass_D72"] = AT_OR_BETTER[sec]
        rows[sec]["D72_band"] = "AT OR BETTER"

    # trajectory's two rows report the multiple against `floor`, not against `floor / r`. Both
    # are far enough from 1.0 that the verdict does not turn on the 1-2 % the division moves,
    # and the verdict it published is carried rather than recomputed.
    rows["diffusion_module.layer_norm_s"]["survives"] = False
    rows["diffusion_module.layer_norm_a"]["survives"] = True

    surv = sorted(s for s, r in rows.items() if r["survives"])
    fell = sorted(s for s, r in rows.items() if not r["survives"])
    pct_s = sum(AT_OR_BETTER[s] for s in surv)
    pct_f = sum(AT_OR_BETTER[s] for s in fell)

    out = {
        "what": __doc__.strip().splitlines()[0],
        "why": " ".join(__doc__.strip().split("\n\n")[1].split()),
        "D72_headline_being_tested": {
            "claim": "42.2794 % of the model deviates from float64 by no more than upstream's "
                     "own bf16 training recipe does",
            "pct_of_model": round(sum(AT_OR_BETTER.values()), 4),
            "limit": "both columns are distances from the same float64 reference, and two "
                     "distances from one reference do not order each other",
        },
        "recomputed_without_the_shared_subtrahend": {
            "survives_the_direct_test": {"sections": surv, "pct_of_model": round(pct_s, 4)},
            "does_not_survive": {"sections": fell, "pct_of_model": round(pct_f, 4)},
            "reading": (f"D72's {sum(AT_OR_BETTER.values()):.4f} % becomes {pct_s:.4f} %. "
                        f"{pct_f:.4f} % of the model was read as at or better than upstream's "
                        f"own recipe and does not agree with what that recipe actually "
                        f"computes."),
        },
        "threshold": "floor / r, where floor = rel(bf16, float64) over the scope and "
                     "r = ||bf16|| / ||float64|| over the same scope. rel(device, bf16) divides "
                     "by ||bf16|| and floor divides by ||float64||, so a port that reproduced "
                     "float64 exactly reads floor / r and not floor (D76).",
        "rows": rows,
    }
    p = HERE / "D72_AT_OR_BETTER_ROWS_DIRECTLY_TESTED.json"
    p.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out["recomputed_without_the_shared_subtrahend"], indent=1))
    for s, r in sorted(rows.items(), key=lambda kv: -kv[1]["pct_of_model_mass_D72"]):
        print(f"{r['pct_of_model_mass_D72']:8.4f} %  {s:45s} f64 {r['device_vs_float64']:.6e}  "
              f"direct {r['device_vs_bf16_direct']:.6e}  "
              f"{'SURVIVES' if r['survives'] else 'DOES NOT SURVIVE'}")
    print("wrote", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
