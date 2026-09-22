#!/usr/bin/env python3
"""Per-block ABSOLUTE error-mass census over all 48 pairformer blocks, in the clause's frame.

Three rows (of3t-blk4544, of3t-trunkact, of3t-vjpln) searched blocks 45 and 44 on one shared
assumption: that the carrier of the trunk's gradient error is in the last two blocks. The
evidence for it is a step in the per-block COTANGENT curve at 45->44. of3t-tapeamp showed that
curve is ramp / step / saturation, which is a cancellation signature, and a cancellation moves a
RELATIVE reading sharply while the absolute error barely moves. So the block where a relative
curve steps need not be the block that holds the mass.

This census is over the absolute, differenced quantity the clause is built from:

    e_b = SUM over block b's leaf tensors of || ours - upstream_bf16 ||^2
    E   = SUM_b e_b,   R = SUM || upstream_bf16 ||^2,   v = sqrt(E/R)

v is the trunk reading the GRADIENTS clause consumes. Blocks partition the 2736 leaf tensors, so
the split is additive by construction and the sum check can only catch a coding error; what is
not additive is ATTRIBUTION, and PREDICTION.md says so in advance.

Two frames, because a share moves when its denominator collapses
(`a-difference-of-absolute-errors-locates-the-carrier`):

  IN-FRAME    ours against upstream 0.4.3's own bf16 autocast step, both driven from the same
              capture, with the local float64 as the reference that validates the floor. This is
              where the live 1.0293953378 was measured (of3t-frame384).
  PINNED      the graded artifact's own denominator, read off its committed per-tensor sidecar
              (pinned float64 sha256 1d4ea922..., pinned upstream bf16 ff78d7bc...). Free, and
              it is the denominator the charter clause is scored in.

Scoring math is of3t_trunkg043/score.py, imported rather than reimplemented, so this row shares
one definition of the per-tensor triple with every row above it.

CPU only. No Tenstorrent device is opened at any point, so no figure here carries a clock and
none is a perf claim.
"""
import argparse, hashlib, json, os, sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "of3t_trunkg043"))
from score import score                                                      # noqa: E402

PRE = "pairformer_stack.blocks."
SECTION = "pairformer_stack"
# A14: a relative reading needs a denominator floor. A reference norm of 1.4e-19 once produced a
# 1.142e+13 headline that meant nothing. Tensors under this floor keep their ABSOLUTE error mass
# in the census -- the census is absolute -- and are excluded from per-tensor RELATIVE readings
# and from the worst-relative search only. The count and the mass they hold are reported.
REL_FLOOR = 1.0e-12


def sha256_file(p, chunk=1 << 24):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def trunk(p):
    d = torch.load(p, map_location="cpu", weights_only=False)
    g = d["grads"] if isinstance(d, dict) and "grads" in d else d
    return ({k: v for k, v in g.items() if k.startswith(PRE) and v is not None},
            d if isinstance(d, dict) else {})


def blk(name):
    return int(name[len(PRE):].split(".", 1)[0])


def rows_of(a, b, keys):
    """Per tensor: the absolute squared error, the reference's squared norm, and the triple."""
    out = []
    for k in keys:
        x = a[k].reshape(-1).to(torch.float64)
        y = b[k].reshape(-1).to(torch.float64)
        nref = float(torch.linalg.vector_norm(y))
        nours = float(torch.linalg.vector_norm(x))
        d = float(torch.linalg.vector_norm(x - y))
        out.append({"tensor": k, "block": blk(k), "err_sq": d * d, "err": d,
                    "ref_norm": nref, "ref_sq": nref * nref, "our_norm": nours,
                    "rel_l2": (d / nref) if nref > REL_FLOOR else None,
                    "norm_ratio": (nours / nref) if nref > REL_FLOOR else None,
                    "cos": (float(x @ y) / (nours * nref)) if nours and nref > REL_FLOOR else None})
    return out


def census(rows, nb, tag):
    E = sum(r["err_sq"] for r in rows)
    R = sum(r["ref_sq"] for r in rows)
    per = {}
    for i in range(nb):
        sel = [r for r in rows if r["block"] == i]
        if not sel:
            continue
        e = sum(r["err_sq"] for r in sel)
        m = sum(r["ref_sq"] for r in sel)
        meas = [r for r in sel if r["rel_l2"] is not None]
        wrel = max(meas, key=lambda r: r["rel_l2"]) if meas else None
        wabs = max(sel, key=lambda r: r["err_sq"])
        per[i] = {"n": len(sel), "n_rel_measurable": len(meas),
                  "absolute_error_mass": e, "share_of_E": e / E if E else 0.0,
                  "reference_squared_norm": m, "mass_share_of_the_stack": m / R if R else 0.0,
                  "pooled_rel_l2_in_block": float(np.sqrt(e / m)) if m else None,
                  "worst_rel_l2": wrel["rel_l2"] if wrel else None,
                  "worst_rel_tensor": wrel["tensor"] if wrel else None,
                  "worst_by_absolute_error": wabs["err"],
                  "worst_by_absolute_error_tensor": wabs["tensor"]}
    ranked = sorted(per, key=lambda i: -per[i]["absolute_error_mass"])
    cum, acc = [], 0.0
    for i in ranked:
        acc += per[i]["share_of_E"]
        cum.append({"block": i, "share_of_E": per[i]["share_of_E"], "cumulative": acc})
    meas = [r for r in rows if r["rel_l2"] is not None]
    wrel = max(meas, key=lambda r: r["rel_l2"])
    wabs = max(rows, key=lambda r: r["err_sq"])
    below = [r for r in rows if r["rel_l2"] is None]
    return {"frame": tag, "tensors": len(rows), "blocks": len(per),
            "E_absolute_error_mass": E, "R_reference_squared_norm": R,
            "pooled_rel_l2": float(np.sqrt(E / R)) if R else None,
            "sum_check_sum_of_blocks_minus_E":
                sum(v["absolute_error_mass"] for v in per.values()) - E,
            "sum_check_relative":
                abs(sum(v["absolute_error_mass"] for v in per.values()) - E) / E if E else 0.0,
            "denominator_floor_on_reference_norm": REL_FLOOR,
            "n_below_the_floor": len(below),
            "mass_share_below_the_floor": sum(r["ref_sq"] for r in below) / R if R else 0.0,
            "error_share_below_the_floor": sum(r["err_sq"] for r in below) / E if E else 0.0,
            "smallest_reference_norm": min(r["ref_norm"] for r in rows),
            "worst_rel_l2_over_tensors": wrel["rel_l2"],
            "worst_rel_l2_tensor": wrel["tensor"],
            "worst_absolute_error": wabs["err"],
            "worst_absolute_error_tensor": wabs["tensor"],
            "per_block": per, "ranked": ranked, "cumulative": cum}


def repool(mb, v, section=SECTION, stat="renorm_vs_UPSTREAM_BF16"):
    """Model scope with the trunk section reading replaced by v.

    The pooling is the artifact's own: model_rel = sqrt(SUM_sec rel_sec^2 * ref_sq_sec /
    SUM_sec ref_sq_sec), over each section's OWN ref_sq. of3t-frame384's projection composed
    from the published float64 mass shares instead and came out 11 % high; the orchestrator's
    pass-359 correction re-pooled from ref_sq and reproduces the published headline exactly,
    which is the control below. What it still assumes is what that projection assumed: that the
    other nine sections' arms are already frame-matched. None of them was measured here, so
    every model figure this function returns is a LOWER BOUND on an in-frame model reading.
    """
    ps = mb["per_section"][stat]
    num = den = 0.0
    for sec, s in ps.items():
        r = v if sec == section else s["mass_weighted_rel_l2"]
        num += r * r * s["ref_sq"]
        den += s["ref_sq"]
    return float(np.sqrt(num / den))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-f64", required=True)
    ap.add_argument("--ref-bf16", required=True)
    ap.add_argument("--ours", required=True)
    ap.add_argument("--model-artifact", required=True)
    ap.add_argument("--sidecar", required=True)
    ap.add_argument("--device-aa", default="", help="of3t-modelboundary's own device A/A artifact")
    ap.add_argument("--blocks", type=int, default=48)
    ap.add_argument("--clause-target", type=float, default=0.4361680548)
    ap.add_argument("--published-inframe", type=float, default=1.0293953377723410)
    ap.add_argument("--published-ours-vs-f64", type=float, default=0.8354121633458239)
    ap.add_argument("--published-floor", type=float, default=0.37393839211303687)
    ap.add_argument("--published-model", type=float, default=0.5201243840984896)
    ap.add_argument("--published-model-inframe", type=float, default=0.2785749654)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    out = {"what": __doc__.strip().splitlines()[0],
           "host": os.uname().nodename, "device_involved": False,
           "note": "CPU only. The device arm was banked by of3t-modelboundary on qb2 and is "
                   "scored here, not re-run, so no figure in this file carries a clock.",
           "crop": 384, "digests": {}, "controls": {}}

    for nm, p in (("REF_LOCAL_f64_n384", a.ref_f64), ("REF_LOCAL_bf16_n384", a.ref_bf16),
                  ("OURS_n384", a.ours)):
        out["digests"][nm] = {"path": p, "bytes": os.path.getsize(p), "sha256": sha256_file(p)}
        print("SHA", nm, out["digests"][nm]["sha256"], flush=True)

    f64, _ = trunk(a.ref_f64)
    bf16, _ = trunk(a.ref_bf16)
    ours, _ = trunk(a.ours)
    keys = sorted(bf16)
    out["scope"] = {"reference_tensors": len(keys),
                    "ours_absent": len([k for k in keys if k not in ours]),
                    "f64_absent": len([k for k in keys if k not in f64]),
                    "blocks": a.blocks}
    print("SCOPE", out["scope"], flush=True)

    # ---- FLOORS AND CONTROLS, all before any census number is read -------------------------
    aa = score(bf16, bf16, keys); aa.pop("_rows")
    aao = score(ours, ours, keys); aao.pop("_rows")
    z = score({k: torch.zeros_like(bf16[k]) for k in keys}, bf16, keys); z.pop("_rows")
    out["controls"]["AA_the_in_frame_reference_against_itself"] = {
        "mass_weighted_rel_l2": aa["mass_weighted_rel_l2"],
        "exactly_zero": aa["mass_weighted_rel_l2"] == 0.0}
    out["controls"]["AA_our_device_arm_against_itself"] = {
        "mass_weighted_rel_l2": aao["mass_weighted_rel_l2"],
        "exactly_zero": aao["mass_weighted_rel_l2"] == 0.0,
        "what": "an in-process identity, so it bounds the SCORER's determinism and not the "
                "device's. The device A/A is of3t-modelboundary's own and is quoted below."}
    out["controls"]["A16_zero_gradient_baseline"] = {
        "mass_weighted_rel_l2": z["mass_weighted_rel_l2"],
        "exactly_one": z["mass_weighted_rel_l2"] == 1.0,
        "what": "measured in this row's own scorer, not inherited: a reading of 0.9 against "
                "this ceiling is not a pass"}
    if a.device_aa:
        with open(a.device_aa) as fh:
            out["controls"]["AA_device_two_runs_of3t_modelboundary"] = {
                "from": a.device_aa, **json.load(fh)}
    print("AA", aa["mass_weighted_rel_l2"], aao["mass_weighted_rel_l2"],
          "A16", z["mass_weighted_rel_l2"], flush=True)

    # ---- the two frames --------------------------------------------------------------------
    r_in = rows_of(ours, bf16, keys)
    c_in = census(r_in, a.blocks, "ours_vs_upstream_own_bf16__IN_FRAME")
    r_f64 = rows_of(ours, f64, keys)
    c_f64 = census(r_f64, a.blocks, "ours_vs_local_float64__IN_FRAME")
    r_fl = rows_of(bf16, f64, keys)
    c_fl = census(r_fl, a.blocks, "upstream_own_bf16_vs_local_float64__THE_FLOOR")

    out["controls"]["REPRODUCTION"] = {
        "in_frame_clause_reading": {"recomputed": c_in["pooled_rel_l2"],
                                    "published": a.published_inframe,
                                    "relative_difference": abs(c_in["pooled_rel_l2"]
                                                               - a.published_inframe)
                                    / a.published_inframe},
        "ours_vs_local_float64": {"recomputed": c_f64["pooled_rel_l2"],
                                  "published": a.published_ours_vs_f64,
                                  "relative_difference": abs(c_f64["pooled_rel_l2"]
                                                             - a.published_ours_vs_f64)
                                  / a.published_ours_vs_f64},
        "the_floor": {"recomputed": c_fl["pooled_rel_l2"], "published": a.published_floor,
                      "relative_difference": abs(c_fl["pooled_rel_l2"] - a.published_floor)
                      / a.published_floor},
        "what": "the three published in-frame readings, recomputed through this row's own "
                "absolute-mass path. If they do not come back the arm or the scorer is not the "
                "one the clause figure was measured with."}
    print("REPRO", json.dumps({k: v["relative_difference"]
                               for k, v in out["controls"]["REPRODUCTION"].items()
                               if isinstance(v, dict)}), flush=True)

    out["CENSUS_IN_FRAME"] = c_in
    out["CENSUS_vs_FLOAT64"] = c_f64
    out["CENSUS_THE_FLOOR"] = c_fl

    # excess over upstream's own bf16, per block: the only separation of generated from
    # inherited this instrument offers.
    out["EXCESS_per_block"] = {
        i: {"ours_absolute_error_mass_vs_f64": c_f64["per_block"][i]["absolute_error_mass"],
            "floor_absolute_error_mass": c_fl["per_block"][i]["absolute_error_mass"],
            "ratio_ours_over_floor_in_absolute_mass":
                c_f64["per_block"][i]["absolute_error_mass"]
                / c_fl["per_block"][i]["absolute_error_mass"],
            "ratio_ours_over_floor_in_rel":
                c_f64["per_block"][i]["pooled_rel_l2_in_block"]
                / c_fl["per_block"][i]["pooled_rel_l2_in_block"]}
        for i in c_f64["per_block"]}

    # ---- COUNTERFACTUAL --------------------------------------------------------------------
    with open(a.model_artifact) as fh:
        mb = json.load(fh)
    E, R = c_in["E_absolute_error_mass"], c_in["R_reference_squared_norm"]
    bar = mb["bars"]["A26_reachable_bar_vs_their_bf16"]
    E_allowed = a.clause_target ** 2 * R
    cf = {"what": "model scope with named blocks at upstream's own level, in the graded "
                  "artifact's own pooling. A projection, and a LOWER BOUND: see repool().",
          "A26_reachable_bar_vs_their_bf16": bar,
          "clause_target_for_the_trunk": a.clause_target,
          "E_absolute_error_mass": E, "R_reference_squared_norm": R,
          "E_allowed_by_the_clause": E_allowed,
          "fraction_of_E_that_must_be_removed": 1.0 - E_allowed / E,
          "controls": {
              "model_as_published": {"recomputed": repool(mb, mb["per_section"]
                                                          ["renorm_vs_UPSTREAM_BF16"][SECTION]
                                                          ["mass_weighted_rel_l2"]),
                                     "published": a.published_model},
              "model_with_the_trunk_in_frame": {
                  "recomputed": repool(mb, c_in["pooled_rel_l2"]),
                  "published": a.published_model_inframe},
              "model_with_the_trunk_exact": {"recomputed": repool(mb, 0.0)},
              "model_with_the_trunk_at_the_clause_target": {
                  "recomputed": repool(mb, a.clause_target)}},
          "single_block": {}, "top_k": {}}

    def arm(remove_exact, remove_a26, label):
        vE = max(E - remove_exact, 0.0)
        vA = max(E - remove_exact + remove_a26, 0.0)
        ve, va = float(np.sqrt(vE / R)), float(np.sqrt(vA / R))
        return {"label": label,
                "EXACT_our_error_on_those_blocks_set_to_zero": {
                    "trunk_in_frame": ve, "model_scope": repool(mb, ve),
                    "multiple_of_the_bar": repool(mb, ve) / bar,
                    "clause_would_pass": repool(mb, ve) <= bar,
                    "trunk_meets_its_target": ve <= a.clause_target},
                "A26_our_error_there_reduced_to_upstreams_own_level": {
                    "arithmetic": "e_b -> 2 * (the floor's own absolute error mass in block b). "
                                  "Two independent errors of equal magnitude compose as "
                                  "sqrt(2), which is the same factor the A26 bar carries.",
                    "trunk_in_frame": va, "model_scope": repool(mb, va),
                    "multiple_of_the_bar": repool(mb, va) / bar,
                    "clause_would_pass": repool(mb, va) <= bar,
                    "trunk_meets_its_target": va <= a.clause_target}}

    for i in c_in["ranked"]:
        cf["single_block"][i] = {
            "share_of_E": c_in["per_block"][i]["share_of_E"],
            **arm(c_in["per_block"][i]["absolute_error_mass"],
                  2.0 * c_fl["per_block"][i]["absolute_error_mass"], f"block {i}")}
    acc_e = acc_f = 0.0
    for k, i in enumerate(c_in["ranked"], 1):
        acc_e += c_in["per_block"][i]["absolute_error_mass"]
        acc_f += 2.0 * c_fl["per_block"][i]["absolute_error_mass"]
        cf["top_k"][k] = {"blocks": c_in["ranked"][:k], "cumulative_share_of_E": acc_e / E,
                          **arm(acc_e, acc_f, f"top {k}")}
    out["COUNTERFACTUAL"] = cf

    # ---- P4: the same census in the PINNED frame, off the graded artifact's own sidecar -----
    with open(a.sidecar) as fh:
        sc = json.load(fh)
    sel = [r for r in sc if r["param"].startswith(PRE)]
    per = {}
    for r in sel:
        i = blk(r["param"])
        p = per.setdefault(i, {"n": 0, "absolute_error_mass": 0.0,
                               "reference_squared_norm": 0.0, "worst_rel_l2": 0.0,
                               "worst_rel_tensor": None, "worst_by_absolute_error": 0.0,
                               "worst_by_absolute_error_tensor": None})
        p["n"] += 1
        p["absolute_error_mass"] += r["diff_norm"] ** 2
        p["reference_squared_norm"] += r["ref_norm"] ** 2
        rl = r.get("rel_l2")
        if r["ref_norm"] > REL_FLOOR and rl is not None and rl > p["worst_rel_l2"]:
            p["worst_rel_l2"], p["worst_rel_tensor"] = rl, r["param"]
        if r["diff_norm"] > p["worst_by_absolute_error"]:
            p["worst_by_absolute_error"] = r["diff_norm"]
            p["worst_by_absolute_error_tensor"] = r["param"]
    Ep = sum(p["absolute_error_mass"] for p in per.values())
    Rp = sum(p["reference_squared_norm"] for p in per.values())
    for p in per.values():
        p["share_of_E"] = p["absolute_error_mass"] / Ep
        p["pooled_rel_l2_in_block"] = float(np.sqrt(p["absolute_error_mass"]
                                                    / p["reference_squared_norm"]))
    rankedp = sorted(per, key=lambda i: -per[i]["absolute_error_mass"])
    sec = mb["per_section"]["renorm_vs_UPSTREAM_BF16"][SECTION]
    out["CENSUS_PINNED_FRAME"] = {
        "frame": "ours_vs_pinned_upstream_bf16__THE_GRADED_ARTIFACTS_OWN_DENOMINATOR",
        "from": a.sidecar, "tensors": len(sel),
        "E_absolute_error_mass": Ep, "R_reference_squared_norm": Rp,
        "pooled_rel_l2": float(np.sqrt(Ep / Rp)),
        "reproduces_the_sections_published_reading": sec["mass_weighted_rel_l2"],
        "relative_difference": abs(float(np.sqrt(Ep / Rp)) - sec["mass_weighted_rel_l2"])
        / sec["mass_weighted_rel_l2"],
        "per_block": per, "ranked": rankedp,
        "top3_set_matches_in_frame": sorted(rankedp[:3]) == sorted(c_in["ranked"][:3]),
        "in_frame_top3": c_in["ranked"][:3], "pinned_top3": rankedp[:3]}

    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2)

    print(json.dumps({
        "pooled_in_frame": c_in["pooled_rel_l2"],
        "E": E, "R": R,
        "sum_check_relative": c_in["sum_check_relative"],
        "must_remove": cf["fraction_of_E_that_must_be_removed"],
        "in_frame_top6": [(i, round(c_in["per_block"][i]["share_of_E"], 6))
                          for i in c_in["ranked"][:6]],
        "pinned_top6": [(i, round(per[i]["share_of_E"], 6)) for i in rankedp[:6]],
        "block45_share": c_in["per_block"][45]["share_of_E"],
        "block44_share": c_in["per_block"][44]["share_of_E"],
        "top3_set_matches": out["CENSUS_PINNED_FRAME"]["top3_set_matches_in_frame"],
        "model_controls": {k: v for k, v in cf["controls"].items()},
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
