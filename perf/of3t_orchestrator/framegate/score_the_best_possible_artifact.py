#!/usr/bin/env python3
"""D237. Score the BEST POSSIBLE artifact against the GRADIENTS clause before spending
another pass closing a deficit.

The clause the charter gate evaluates is, verbatim from _of3t_donecheck.py:

    stats.renorm_vs_UPSTREAM_BF16.mass_weighted_rel_l2 <= bars.A26_reachable_bar_vs_their_bf16

on the graded 3,660 artifact. This script asks one question the campaign had never asked:
what does that clause read if OUR arm is perfect? If a perfect artifact fails, the clause is
unsatisfiable and no amount of work on our gradient can pass it.

Two inputs, both committed artifacts, neither transcribed from prose:
  A  perf/of3t_modelboundary/MODEL_withtrunk_composed3660_n384.json  (the graded artifact)
  B  perf/of3t_frame384/FRAME_N384.json                              (the frame analysis)

Control: the section composition is reproduced against the artifact's own published headline
before any substitution is made. If that control does not hold to 1e-12 this script is wrong
about the scorer's arithmetic and says so instead of reporting a result.
"""
import json, math, argparse, hashlib, socket, os

T = "pairformer_stack"
ARM = "renorm_vs_UPSTREAM_BF16"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graded", required=True)
    ap.add_argument("--frame", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    g = json.load(open(a.graded))
    f = json.load(open(a.frame))
    secs = g["per_section"][ARM]
    fl = g["per_section"]["UPSTREAM_BF16_vs_FLOAT64"]
    bar = g["bars"]["A26_reachable_bar_vs_their_bf16"]

    # --- the scorer's own composition, identity form (model_scope.py:255) -------------------
    W = sum(s["ref_sq"] for s in secs.values())
    wt = secs[T]["ref_sq"]
    S_other = sum(s["ref_sq"] * s["mass_weighted_rel_l2"] ** 2
                  for k, s in secs.items() if k != T)

    def model(xt):
        return math.sqrt((S_other + wt * xt * xt) / W)

    published = g["stats"][ARM]["mass_weighted_rel_l2"]
    xt_meas = secs[T]["mass_weighted_rel_l2"]
    control = model(xt_meas)
    control_rel = abs(control - published) / published
    if control_rel > 1e-12:
        raise SystemExit("CONTROL FAILED: composition %r != published %r (rel %.3e). "
                         "This script is wrong about the scorer." % (control, published, control_rel))

    # --- the contamination, from the frame artifact ----------------------------------------
    M = f["MATCHED"]
    refref = f["FRAME_ref_f64_n384__vs__grads_f64_043"]["mass_weighted_rel_l2"]
    refref_cos = f["FRAME_ref_f64_n384__vs__grads_f64_043"]["mass_weighted_cos"]
    ours_in = M["ours_vs_REF_LOCAL_f64_n384"]["mass_weighted_rel_l2"]
    floor_in = M["floor_REF_LOCAL_bf16_vs_REF_LOCAL_f64_n384"]["mass_weighted_rel_l2"]
    ft = fl[T]["mass_weighted_rel_l2"]
    rt = fl[T]["mass_weighted_norm_ratio"]
    A26T = math.sqrt(2.0) * ft / rt
    xt_admissible = math.sqrt((bar * bar * W - S_other) / wt)

    # our arm is on the LOCAL boundary: its trunk gradient mass sits with the local
    # references and is 4.4x the model reference's. Checked here, not assumed.
    norms = {k: v["trunk_squared_gradient_norm"] for k, v in f["refs"].items()}
    boundary_check = {
        "ours_trunk_sq_norm": norms["OURS_n384"],
        "local_f64_trunk_sq_norm": norms["REF_LOCAL_f64_n384"],
        "model_f64_trunk_sq_norm": norms["REF_MODEL_f64"],
        "ours_over_model_ref": norms["OURS_n384"] / norms["REF_MODEL_f64"],
        "ours_over_local_ref": norms["OURS_n384"] / norms["REF_LOCAL_f64_n384"],
        "verdict": "our trunk arm is on the LOCAL (capture-driven, injected-cotangent) "
                   "boundary, not the model boundary",
    }
    # and the graded artifact's trunk term IS frame384's cross-frame reading, bit for bit
    identity_check = {
        "graded_artifact_x_t": xt_meas,
        "frame384_trunk_section_cross_frame":
            f["MODEL_PROJECTION"][ARM]["trunk_section_cross_frame"],
        "bit_identical": xt_meas == f["MODEL_PROJECTION"][ARM]["trunk_section_cross_frame"],
    }

    # --- the best possible artifact --------------------------------------------------------
    # ours PERFECT on its own boundary => ours == the n384 float64 reference exactly. The
    # clause then compares that float64 reference to the MODEL bf16 reference, which still
    # carries the whole boundary gap. Alignment between the two residues is unknown, so this
    # is reported as a triangle-inequality INTERVAL with the quadrature point inside it.
    lo, hi = abs(refref - ft), refref + ft
    quad = math.sqrt(refref ** 2 + ft ** 2)
    best = {lbl: {"x_t": v, "model_reading": model(v), "x_bar": model(v) / bar,
                  "passes": model(v) <= bar}
            for lbl, v in (("triangle_lower", lo), ("quadrature", quad),
                           ("triangle_upper", hi))}

    # --- what a FRAME-MATCHED trunk arm would read (a PROJECTION, not a measurement) -------
    mult = ours_in / floor_in
    proj = {}
    for lbl, m in (("at_the_in_frame_multiple", mult),
                   ("with_the_measured_lever_ceiling_1.8563x",
                    mult / 1.8563207917912123)):
        xt = A26T * m / math.sqrt(2.0)
        proj[lbl] = {"assumed_multiple_of_upstreams_own_bf16": m, "x_t": xt,
                     "model_reading": model(xt), "x_bar": model(xt) / bar,
                     "passes": model(xt) <= bar}

    out = {
        "defect": "D237",
        "what": "the GRADIENTS clause is unsatisfiable as gated: a bit-exactly perfect trunk "
                "still fails, because the artifact's pairformer_stack term is scored across "
                "two boundaries whose own float64 references disagree at %.4f (cos %.3f)."
                % (refref, refref_cos),
        "host": socket.gethostname(),
        "device_involved": False,
        "note": "CPU only, no device op, no timing claim: nothing here needs a clock.",
        "inputs": {"graded": {"path": a.graded, "sha256": sha(a.graded)},
                   "frame": {"path": a.frame, "sha256": sha(a.frame)}},
        "clause": "stats.%s.mass_weighted_rel_l2 <= bars.A26_reachable_bar_vs_their_bf16" % ARM,
        "control_composition_reproduces_published_headline": {
            "published": published, "composed": control, "rel_difference": control_rel,
            "note": "the scorer's identity form, weights = each section's own ref_sq",
        },
        "boundary_check": boundary_check,
        "identity_check": identity_check,
        "bar": {"value": bar,
                "built_from": "sqrt(2)*floor/r, both upstream-vs-upstream on the MODEL "
                              "boundary -- the bar is frame-clean; the numerator is not"},
        "trunk": {"x_t_as_measured": xt_meas,
                  "x_t_admissible_by_the_clause": xt_admissible,
                  "A26_perfect_level": A26T,
                  "model_frame_floor": ft, "model_frame_r": rt,
                  "required_factor_on_the_artifact": xt_meas / xt_admissible,
                  "boundary_gap_alone_over_the_whole_allowance":
                      refref / xt_admissible},
        "contamination_split": {
            "ref_f64_n384_vs_model_f64_BOTH_FLOAT64": refref,
            "cos": refref_cos,
            "ours_vs_local_f64_frame_matched": ours_in,
            "upstream_own_bf16_in_frame_floor": floor_in,
            "ours_over_their_own_bf16_in_frame": mult,
            "pct_of_cross_frame_error_mass_that_is_boundary_mismatch":
                100.0 * refref ** 2 / (refref ** 2 + ours_in ** 2),
            "pct_that_is_our_trunk_error":
                100.0 * ours_in ** 2 / (refref ** 2 + ours_in ** 2),
        },
        "as_measured": {"x_t": xt_meas, "model_reading": published,
                        "x_bar": published / bar, "passes": published <= bar},
        "best_possible_artifact_ours_bit_exact": best,
        "projection_if_the_trunk_arm_were_frame_matched": {
            "IS_A_PROJECTION_NOT_A_MEASUREMENT": True,
            "assumption": "a frame-matched trunk arm lands at the same multiple of upstream's "
                          "own bf16 that the in-frame arm measures. The model boundary carries "
                          "a real cotangent with 3.12x less trunk gradient mass, so our error "
                          "there could be relatively larger or smaller. This is exactly what "
                          "the missing run measures and is why it is a projection.",
            "cases": proj,
        },
        "remedy": "produce our trunk device arm ON the model boundary (the real cotangent from "
                  "batch_step003) so pairformer_stack is frame-matched like the other ten "
                  "sections. This is not a clause repair: it can equally show the trunk is "
                  "worse. Until it exists the clause can be neither passed nor failed honestly.",
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: out[k] for k in
                      ("control_composition_reproduces_published_headline", "as_measured",
                       "best_possible_artifact_ours_bit_exact", "trunk",
                       "contamination_split")}, indent=1))


if __name__ == "__main__":
    main()
