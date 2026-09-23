#!/usr/bin/env python3
"""of3t-covadopt: what the model-boundary instrument MUST read over the composed 3,660.

Pre-registered before the arm. Everything here comes from artifacts that were committed before
this row started -- `perf/of3t_modelboundary/sidecar_modelboundary/*.json` (the published
per-tensor rows behind MODEL_withtrunk_n384.json) and `perf/of3t_refcov/*` (the composition) --
plus the float64/bf16/f32 REFERENCES, which carry no arm data. No device run feeds this file.

The statistic is `agreement.stat`'s, and it is a POOLED ratio, not a mass-weighted mean:

    mass_weighted_rel_l2 = sqrt( sum_i ||a_i - b_i||^2 / sum_i ||b_i||^2 )

so it composes by sums of squares, which is what makes an exact prediction possible. (refcov's
`delta` block reports the OTHER statistic, a mass-weighted mean of per-tensor rel_l2; the two
are not interchangeable and mixing them is worth 1.2 % here. Only `headline_vs_float64` in that
file is this statistic.)

Three things enter the composed set that are not in the published 3,643:
  * the 547 diffusion-scope tensors are REPLACED by their --device-refatom values,
  * 8 ref-atom tensors enter diffusion_module,
  * 9 tensors enter input_embedder.
The reference is unchanged, so every ref_norm outside the 17 is the published one.

The float64 leg is therefore predictable to the last bit. The bf16 leg is NOT: no arm was ever
scored against upstream's bf16 step with the lever on. It gets a RIGOROUS INTERVAL from the
triangle inequality over the concatenated vectors, plus a point estimate that holds the
published geometry fixed. Both are written here before the run.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SIDE = REPO / "perf/of3t_modelboundary/sidecar_modelboundary"
RC = REPO / "perf/of3t_refcov"
OUT = Path(__file__).with_name("PREDICTION.json")

PIN_F64 = "1d4ea9225f854afe06a8adedcb5f8aa1c53656fbf8cefdaa3a451d0327895cc4"
PIN_BF16 = "ff78d7bc0bf7a4355014a470fbe931592bd4896fe06a8b400c161c08b5607ccb"
PIN_F32 = "09f1217c8ea254d04f1bfdae73585f058cb2aec51557699bfae3a8090dadd548"
REF_DIR = Path("/home/ttuser/of3t_refprec")
F64 = REF_DIR / "bundle_ref/grads_f64_043.pt"
BF16 = REF_DIR / "pinned_p175/arm4_bf16_autocast/grads_f64.pt"
F32 = REF_DIR / "pinned_p175/arm2_f32_upstream/grads_f64.pt"
PER_TENSOR_BAR = 5.0e-02
MODEL_TOTAL_SQ = 10.279642678524985
REF_NORM_FLOOR = 1e-12


def load(p):
    return json.loads(Path(p).read_text())


def pooled(diff_sq, ref_sq):
    return math.sqrt(diff_sq / ref_sq) if ref_sq else None


def main() -> int:
    pub_f64 = load(SIDE / "renorm_vs_FLOAT64.json")
    pub_bf16 = load(SIDE / "renorm_vs_UPSTREAM_BF16.json")
    pub_bf16f64 = load(SIDE / "UPSTREAM_BF16_vs_FLOAT64.json")
    pub_f32f64 = load(SIDE / "UPSTREAM_F32_vs_FLOAT64.json")
    comp = load(RC / "COVERAGE_COMPOSED.json")
    lever = load(RC / "device_gradient_rc_refatom_on_per_tensor.json")["per_tensor"]

    union = [r["param"] for r in pub_f64]
    uset = set(union)
    entering = comp["accuracy"]["by_arm"]["renorm"]["entering"]["per_tensor"]
    s17 = {r["param"] for r in entering}
    assert len(union) == 3643 and len(s17) == 17, (len(union), len(s17))

    # --- the lever's per-tensor rows, from refcov's own dump-per-tensor artifact -------------
    # keys there are relative to diffusion_module; rel_l2 is against the same float64 reference,
    # so diff_norm = rel_l2 * ref_norm. VALIDATED below against refcov's published C2 drift.
    lev = {"diffusion_module." + r["tensor"]: r for r in lever}
    scope = sorted(k for k in lev if k in uset)
    new_only = sorted(k for k in lev if k not in uset)
    assert len(scope) == 547 and len(new_only) == 8, (len(scope), len(new_only))

    pf64 = {r["param"]: r for r in pub_f64}
    worst_d, worst_n, num, den = 0.0, None, 0.0, 0.0
    for k in scope:
        p, q = pf64[k]["rel_l2"], lev[k]["rel_l2"]
        if p in (None, 0.0) or q is None:
            continue
        d = abs(q - p) / p
        num += pf64[k]["mass_sq"] * d
        den += pf64[k]["mass_sq"]
        if d > worst_d:
            worst_d, worst_n = d, k
    c2 = comp["accuracy"]["by_arm"]["renorm"]["C2_the_levers_reach_into_the_547_it_was_not_sold_on"]
    validation = {
        "what": "refcov's C2 drift recomputed from the dump-per-tensor artifact. It must "
                "reproduce the published C2, or the per-tensor file is not the scoring the "
                "composition used and nothing below stands.",
        "worst_rel_diff": worst_d, "refcov_C2_worst_rel_diff": c2["worst_rel_diff"],
        "worst_tensor": worst_n, "refcov_C2_worst_tensor": c2["worst_tensor"],
        "mass_weighted_rel_diff": num / den,
        "refcov_C2_mass_weighted_rel_diff": c2["mass_weighted_rel_diff"],
        "worst_agrees": abs(worst_d - c2["worst_rel_diff"]) / c2["worst_rel_diff"] < 1e-12,
        "tensor_agrees": worst_n == c2["worst_tensor"],
    }
    if not (validation["worst_agrees"] and validation["tensor_agrees"]):
        print("STOP: the per-tensor artifact does not reproduce refcov's C2", validation)
        return 1

    # --- the float64 leg, exactly -------------------------------------------------------------
    sum_ref_sq = sum(r["ref_norm"] ** 2 for r in pub_f64)
    sum_diff_sq = sum((r["diff_norm"] or 0.0) ** 2 for r in pub_f64)
    sum_arm_sq = sum((r["arm_norm"] or 0.0) ** 2 for r in pub_f64)
    sum_dot = sum(r.get("dot") or 0.0 for r in pub_f64)
    old_diff_sq = sum((pf64[k]["diff_norm"] or 0.0) ** 2 for k in scope)
    old_arm_sq = sum((pf64[k]["arm_norm"] or 0.0) ** 2 for k in scope)
    old_dot = sum(pf64[k].get("dot") or 0.0 for k in scope)
    # the lever's own rows on those same 547
    new_diff_sq = sum((lev[k]["rel_l2"] * lev[k]["ref_norm"]) ** 2 for k in scope)
    new_arm_sq = sum(lev[k]["device_norm"] ** 2 for k in scope)
    new_dot = sum(lev[k]["cos"] * lev[k]["device_norm"] * lev[k]["ref_norm"] for k in scope)
    ent_ref_sq = sum(r["ref_norm"] ** 2 for r in entering)
    ent_diff_sq = sum(r["diff_norm"] ** 2 for r in entering)
    ent_arm_sq = sum(r["arm_norm"] ** 2 for r in entering)
    ent_dot = sum(r["dot"] for r in entering)

    d2_3643 = sum_diff_sq - old_diff_sq + new_diff_sq
    a2_3643 = sum_arm_sq - old_arm_sq + new_arm_sq
    dot_3643 = sum_dot - old_dot + new_dot
    d2_3660 = d2_3643 + ent_diff_sq
    a2_3660 = a2_3643 + ent_arm_sq
    r2_3660 = sum_ref_sq + ent_ref_sq
    dot_3660 = dot_3643 + ent_dot

    rel_f64 = {r["param"]: r["rel_l2"] for r in pub_f64}
    rel_new = dict(rel_f64)
    for k in scope:
        rel_new[k] = lev[k]["rel_l2"]
    for r in entering:
        rel_new[r["param"]] = r["rel_l2"]
    over_f64 = sum(1 for v in rel_new.values() if v is not None and v > PER_TENSOR_BAR)
    meas_f64 = sum(1 for v in rel_new.values() if v is not None)
    wn, wv = max(((k, v) for k, v in rel_new.items() if v is not None), key=lambda kv: kv[1])

    f64_leg = {
        "mass_weighted_rel_l2": pooled(d2_3660, r2_3660),
        "mass_weighted_norm_ratio": pooled(a2_3660, r2_3660),
        "mass_weighted_cos": dot_3660 / math.sqrt(a2_3660 * r2_3660),
        "n": 3660, "n_rel_measurable": meas_f64, "n_over_per_tensor_bar": over_f64,
        "worst_tensor": wn, "worst_rel_l2": wv,
        "pct_of_model_mass": 100.0 * (sum(r["mass_sq"] for r in pub_f64)
                                      + sum(r["mass_sq"] for r in entering)) / MODEL_TOTAL_SQ,
        "cross_check_refcov_headline_after": comp["accuracy"]["by_arm"]["renorm"]
                                                 ["headline_vs_float64"]["after"]
                                                 ["mass_weighted_rel_l2"],
    }
    f64_leg["cross_check_rel_difference"] = abs(
        f64_leg["mass_weighted_rel_l2"] - f64_leg["cross_check_refcov_headline_after"]
    ) / f64_leg["cross_check_refcov_headline_after"]

    # --- C1: the same artifact restricted to the ORIGINAL 3,643 -------------------------------
    pub_stat_f64 = 0.5327948845677762
    pub_stat_bf16 = 0.5201243840984896
    c1 = {
        "what": "the new artifact restricted to the published 3,643. It CANNOT reproduce "
                "MODEL_withtrunk_n384.json, because --device-refatom moves cl0/plm0 onto the "
                "card and they are an input to the whole DiffusionModule (refcov C2).",
        "vs_FLOAT64_predicted": pooled(d2_3643, sum_ref_sq),
        "vs_FLOAT64_published": pub_stat_f64,
        "predicted_difference": pooled(d2_3643, sum_ref_sq) - pub_stat_f64,
        "n_tensors_that_must_be_BIT_IDENTICAL": len(union) - len(scope),
        "n_tensors_that_must_move": len(scope),
        "per_tensor_worst_rel_diff": c2["worst_rel_diff"],
        "per_tensor_worst_tensor": c2["worst_tensor"],
        "per_tensor_mass_weighted_rel_diff": c2["mass_weighted_rel_diff"],
    }

    pred = {
        "what": "of3t-covadopt, pre-registered. What model_scope.py must read over the "
                "composed 3,660, from committed artifacts only.",
        "statistic": "mass_weighted_rel_l2 = sqrt(sum ||a-b||^2 / sum ||b||^2), agreement.stat",
        "validation_of_the_lever_per_tensor_artifact": validation,
        "counts": {"union_n_tensors": 3660, "n_compared": 3660, "n_reference_tensors": 4170,
                   "n_union_tensors_absent_from_float64_reference": 0},
        "coverage_total_pct_of_model_compared": comp["coverage"]["after"],
        "coverage_bar": 99.2594,
        "model_squared_gradient_norm_measured": 10.279642678524981,
        "renorm_vs_FLOAT64": f64_leg,
        "C1_restricted_to_the_original_3643": c1,
    }

    # --- stage 2: the 17 entering tensors against the references ------------------------------
    if "--no-refs" not in sys.argv:
        import torch
        import hashlib

        def sha256(p):
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for c in iter(lambda: f.read(1 << 23), b""):
                    h.update(c)
            return h.hexdigest()

        names = sorted(s17)
        got = {"float64": sha256(F64), "upstream_bf16": sha256(BF16), "upstream_f32": sha256(F32)}
        pins = {"float64": PIN_F64, "upstream_bf16": PIN_BF16, "upstream_f32": PIN_F32}
        pred["reference_digests"] = {k: {"sha256": v, "matches_pin": v == pins[k]}
                                     for k, v in got.items()}
        if any(got[k] != pins[k] for k in pins):
            print("STOP: a reference digest does not match its pin", got)
            return 1
        sub = {}
        for label, p in (("f64", F64), ("bf16", BF16), ("f32", F32)):
            d = torch.load(p, map_location="cpu", weights_only=False)
            sub[label] = {n: (d[n].to(torch.float64).reshape(-1) if d.get(n) is not None
                              else None) for n in names}
            del d
        for leg, key in (("UPSTREAM_BF16_vs_FLOAT64", "bf16"),
                         ("UPSTREAM_F32_vs_FLOAT64", "f32")):
            base = pub_bf16f64 if key == "bf16" else pub_f32f64
            b_ref_sq = sum(r["ref_norm"] ** 2 for r in base)
            b_diff_sq = sum((r["diff_norm"] or 0.0) ** 2 for r in base)
            b_arm_sq = sum((r["arm_norm"] or 0.0) ** 2 for r in base)
            b_dot = sum(r.get("dot") or 0.0 for r in base)
            n_over = sum(1 for r in base
                         if r["rel_l2"] is not None and r["rel_l2"] > PER_TENSOR_BAR)
            e_ref = e_diff = e_arm = e_dot = 0.0
            e_over = 0
            for n in names:
                b, a = sub["f64"][n], sub[key][n]
                rb = float(torch.linalg.vector_norm(b))
                ra = float(torch.linalg.vector_norm(a))
                dd = float(torch.linalg.vector_norm(a - b))
                e_ref += rb ** 2
                e_arm += ra ** 2
                e_diff += dd ** 2
                e_dot += float(torch.dot(a, b))
                if rb >= REF_NORM_FLOOR and dd / rb > PER_TENSOR_BAR:
                    e_over += 1
            pred[leg] = {
                "mass_weighted_rel_l2": pooled(b_diff_sq + e_diff, b_ref_sq + e_ref),
                "mass_weighted_norm_ratio": pooled(b_arm_sq + e_arm, b_ref_sq + e_ref),
                "mass_weighted_cos": (b_dot + e_dot) / math.sqrt((b_arm_sq + e_arm)
                                                                 * (b_ref_sq + e_ref)),
                "n_over_per_tensor_bar": n_over + e_over,
                "published_over_3643": {"mass_weighted_rel_l2": pooled(b_diff_sq, b_ref_sq),
                                        "n_over_per_tensor_bar": n_over},
                "_sums": {"ref_sq": b_ref_sq + e_ref, "diff_sq": b_diff_sq + e_diff,
                          "arm_sq": b_arm_sq + e_arm},
            }
        fl = pred["UPSTREAM_BF16_vs_FLOAT64"]
        floor = fl["mass_weighted_rel_l2"]
        rr = fl["mass_weighted_norm_ratio"]
        pred["bars"] = {
            "upstream_own_bf16_floor_vs_float64": floor,
            "norm_ratio_bf16_over_float64": rr,
            "perfect_port_reads_floor_over_r": floor / rr,
            "A26_reachable_bar_vs_their_bf16": math.sqrt(2.0) * floor / rr,
            "published_A26_over_3643": 0.14735268326440318,
        }

        # --- the bf16 leg: an interval that cannot be wrong, and a point estimate -------------
        # over the concatenated set, ||A-B16|| is within ||B16-F64|| of ||A-F64|| (Minkowski),
        # and the denominator is sum||b16||^2, which stage 2 measured above.
        D = math.sqrt(d2_3660)                      # ||A - F64||
        E = math.sqrt(fl["_sums"]["diff_sq"])       # ||B16 - F64||
        den_b = math.sqrt(fl["_sums"]["arm_sq"])    # ||B16||
        # published geometry: the angle between (A-F64) and (B16-F64) over the 3,643
        Dp = math.sqrt(sum_diff_sq)
        Ep = math.sqrt(sum((r["diff_norm"] or 0.0) ** 2 for r in pub_bf16f64))
        Bp = math.sqrt(sum((r["diff_norm"] or 0.0) ** 2 for r in pub_bf16))
        cos_pub = (Dp ** 2 + Ep ** 2 - Bp ** 2) / (2 * Dp * Ep)
        point = math.sqrt(max(D ** 2 + E ** 2 - 2 * D * E * cos_pub, 0.0)) / den_b
        bar = pred["bars"]["A26_reachable_bar_vs_their_bf16"]
        pred["renorm_vs_UPSTREAM_BF16"] = {
            "not_predictable_exactly": "no arm was ever scored against upstream's bf16 step "
                                       "with --device-refatom on, so this leg is the one new "
                                       "measurement. It gets a bound and an estimate.",
            "rigorous_interval": [abs(D - E) / den_b, (D + E) / den_b],
            "point_estimate_holding_the_published_geometry": point,
            "published_geometry_cos": cos_pub,
            "published_over_3643": pub_stat_bf16,
            "x_A26_bar_point_estimate": point / bar,
            "clause_verdict_predicted": "FAIL -- the A26 clause is expected to read about "
                                        f"{point / bar:.2f}x its bar. This row does not flip it.",
            "falsifier": "a reading outside the rigorous interval means the composition or the "
                         "references are not what this predicts, and stops the row. A reading "
                         "inside the interval but more than 5 % from the point estimate means "
                         "the entering tensors and the lever change the geometry more than the "
                         "published set's average -- reportable, not a stop.",
        }
        c1["vs_UPSTREAM_BF16_point_estimate"] = math.sqrt(
            max(d2_3643 + sum((r["diff_norm"] or 0.0) ** 2 for r in pub_bf16f64)
                - 2 * math.sqrt(d2_3643) * Ep * cos_pub, 0.0)
        ) / math.sqrt(sum((r["arm_norm"] or 0.0) ** 2 for r in pub_bf16f64))
        c1["vs_UPSTREAM_BF16_published"] = pub_stat_bf16

    OUT.write_text(json.dumps(pred, indent=1) + "\n")
    print(json.dumps({k: v for k, v in pred.items() if not k.startswith("_")}, indent=1))
    print("->", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
