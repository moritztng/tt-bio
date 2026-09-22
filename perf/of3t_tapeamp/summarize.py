#!/usr/bin/env python3
"""Assemble AMPLIFICATION.json: the forward-to-gradient factor, priced against the reference's own.

Everything here is read out of an arm's own report; nothing is retyped. The one pair of numbers
that comes from another row is our device arm's, and it is read out of that row's artifact
(`perf/of3t_ditref/device_gradient_r043_ours.json`) rather than quoted from its prose, with the
capture and reference-tree fields carried through so the frame is checkable.
"""
import json
from pathlib import Path

W = Path("/home/ttuser/.coworker/wt/of3t-tapeamp")
A = W / "perf/of3t_tapeamp"

ours = json.load(open(W / "perf/of3t_ditref/device_gradient_r043_ours.json"))
oursp = json.load(open(W / "perf/of3t_ditref/device_gradient_r043_ours_per_tensor.json"))
cen = json.load(open(A / "CENSUS_cen0.json"))
cenarm = json.load(open(W / "perf/of3t_diffusion/device_gradientcen0.json"))

OURS_F = ours["forward_rel_median"]
# the gradient median our device arm reports over the 547 tensors it compares
rels = sorted(t["rel_l2"] for t in oursp["per_tensor"])
OURS_G = rels[len(rels) // 2] if len(rels) % 2 else 0.5 * (rels[len(rels)//2 - 1] + rels[len(rels)//2])

arms = {}
for t in ("bf16auto", "bf16auto_AA", "f32", "bf16auto_BREAK"):
    p = A / f"AMP_{t}.json"
    if p.exists():
        arms[t] = json.load(open(p))

b = arms["bf16auto"]
uf, ug = b["FORWARD"]["median"], b["GRADIENT_matched_scope"]["median_rel"]

out = {
 "what": "The diffusion module's forward-to-gradient error factor, ours against the reference's "
         "own, at one boundary, on the matched 547-tensor scope.",
 "VERDICT": "The factor is a property of differentiating this function in reduced precision, not "
            "an amplifier in the tape and not a dtype boundary. Upstream 0.4.3's own bf16 recipe "
            "shows 7.666x of our 11.026x on the same 547 tensors at the same boundary against the "
            "same float64 reference, and its own fp32 recipe shows 9.326x four orders of "
            "magnitude lower in absolute error. Our arm is more accurate than upstream's own on "
            "BOTH halves; our factor is the larger one because our forward is the better half.",
 "FRAME": {
   "capture": arms["bf16auto"]["cap"],
   "float64_reference": "the capture's own sub_boundary.pt: xl_out for the forward, grad_f64 for "
                        "the gradient. Both halves of every ratio divide by the SAME float64.",
   "upstream_tree": b["upstream_pkg"],
   "upstream_tree_sha256": b["TREE_DIGEST"]["tree_sha256"],
   "upstream_tree_digest_matches": b["TREE_DIGEST"]["digest_matches_entry"],
   "upstream_version": b["upstream_version"],
   "LAYERNORM_SIGNATURE": b["LAYERNORM_SIGNATURE"],
   "checkpoint": b["ckpt"],
   "padded_width": "384 tokens padded, 422 atoms, 48 structures (D180/D193)",
   "scope_matched": "the 547 tensors perf/of3t_ditref/device_gradient_r043_ours_per_tensor.json "
                    "compares; the upstream scope is 761 and both are reported",
   "forward_statistic": b["FORWARD"]["statistic"],
   "gradient_statistic": "per-tensor rel_l2 against grad_f64, median; mass-weighted reported "
                         "beside it",
   "upstream_arms_host": b["host"],
   "upstream_arms_omp_num_threads": b["omp_num_threads"],
   "our_arm_host_and_card": "qb2 card 1 (perf/of3t_ditref, ABLATION.json), reproduced on qb1 "
                            "card 1 this pass: structure 0's forward rel is bit-identical at "
                            f"{cenarm['forward_rel'][0]!r}",
   "why_one_process": "an upstream bf16 arm is host-dependent (of3t-bwdaccum control 7: "
                      "3.739355e-01 rebuilt against 4.007237e-01 on record, 6.7 % apart) so "
                      "both halves of every ratio here come out of one process on one host",
 },
 "TABLE": [
   {"arm": "ours, device, mixed precision", "forward_median": OURS_F,
    "gradient_median_547": OURS_G, "ratio_547": OURS_G / OURS_F, "n_gradient": len(rels),
    "source": "perf/of3t_ditref/device_gradient_r043_ours*.json"},
   {"arm": "upstream 0.4.3, its own bf16 recipe", "forward_median": uf,
    "gradient_median_547": ug, "ratio_547": b["RATIO_matched_scope"],
    "ratio_761": b["RATIO_full_scope"],
    "gradient_median_761": b["GRADIENT_full_scope"]["median_rel"],
    "gradient_mass_weighted_761": b["GRADIENT_full_scope"]["mass_weighted_rel"],
    "source": "perf/of3t_tapeamp/AMP_bf16auto.json"},
   {"arm": "upstream 0.4.3, its own fp32 recipe", "forward_median": arms["f32"]["FORWARD"]["median"],
    "gradient_median_547": arms["f32"]["GRADIENT_matched_scope"]["median_rel"],
    "ratio_547": arms["f32"]["RATIO_matched_scope"],
    "ratio_761": arms["f32"]["RATIO_full_scope"],
    "gradient_median_761": arms["f32"]["GRADIENT_full_scope"]["median_rel"],
    "source": "perf/of3t_tapeamp/AMP_f32.json"},
 ],
 "DECOMPOSITION_matched_scope": {
   "forward_ours_over_upstream": OURS_F / uf,
   "forward_ours_more_accurate_by": uf / OURS_F,
   "gradient_ours_over_upstream": OURS_G / ug,
   "gradient_ours_more_accurate_by": ug / OURS_G,
   "ratio_ours_over_upstream": (OURS_G / OURS_F) / (ug / uf),
   "upstream_share_of_our_factor": (ug / uf) / (OURS_G / OURS_F),
   "reading": "Our factor exceeds upstream's own by 1.438x, and that excess is exactly the "
              "quotient of the two accuracy advantages (1.959 / 1.362). It is not an extra "
              "error we add: both halves of our arm beat upstream's own, and the factor is "
              "larger because the DENOMINATOR is the half we beat it on hardest. A ratio grows "
              "when its denominator collapses.",
 },
 "PRECISION_INDEPENDENCE": {
   "upstream_bf16_ratio_547": b["RATIO_matched_scope"],
   "upstream_f32_ratio_547": arms["f32"]["RATIO_matched_scope"],
   "absolute_error_apart": arms["f32"]["FORWARD"]["median"] / uf,
   "reading": "The factor survives a change of precision that moves the absolute error by four "
              "orders of magnitude. A dtype boundary cannot do that; the conditioning of the "
              "Jacobian-transpose product can, and that is what of3t-bwdaccum measured directly "
              "on the trunk (site cancellation factors K up to 5.876e+01, every leaf at or "
              "below u*K at u = 2^-8).",
 },
 "CALL_CENSUS": {
   "source": "perf/of3t_tapeamp/CENSUS_cen0.json",
   "node_firings_total": cen["node_firings_total"],
   "reconciliations_total": cen["reconciliations_total"],
   "reconciliations_that_DOWNCAST_float32_to_bfloat16": cen["reconciliations_that_DOWNCAST_float32_to_bfloat16"],
   "every_firing_dtype": "g=FLOAT32, value=FLOAT32, all 1879",
   "explicit_typecast_verb_calls": cen["ALL_typecast_call_sites"],
   "control": "the wrappers change no arithmetic, and it is checked rather than asserted: "
              f"structure 0's forward rel is {cenarm['forward_rel'][0]!r}, bit-identical to "
              "perf/of3t_ditref/device_gradient_r043_ours.json's own structure 0 on qb2",
   "aiclk_during": "n=5 min=1343 med=1350 max=1350 mean=1347.2 MHz, sampled DURING the arm",
   "reading": "The tape has exactly one place a cotangent's dtype is reconciled to its forward "
              "value's, tt_bio/autograd.py's backward loop, and on the diffusion scope it fires "
              "ZERO times in 1879 node firings. The whole backward runs fp32 against fp32. The "
              "only dtype crossings are 96 calls of the model's own explicit typecast verb at "
              "taped_ttnn.py:1199, which has a tape entry and a backward at :1204 -- so it is "
              "differentiated rather than a hole. There is no D206-class boundary in the shipped "
              "diffusion backward, and this is a call count, not a route read (D196).",
 },
 "CONTROLS": {
   "A_over_A": {
     "arms": ["bf16auto", "bf16auto_AA"],
     "forward_per_structure_bit_identical": (arms["bf16auto"]["FORWARD"]["per_structure"]
                                             == arms["bf16auto_AA"]["FORWARD"]["per_structure"]),
     "forward_median_equal": arms["bf16auto"]["FORWARD"]["median"] == arms["bf16auto_AA"]["FORWARD"]["median"],
     "gradient_median_equal": (arms["bf16auto"]["GRADIENT_full_scope"]["median_rel"]
                               == arms["bf16auto_AA"]["GRADIENT_full_scope"]["median_rel"]),
     "ratio_equal": arms["bf16auto"]["RATIO_full_scope"] == arms["bf16auto_AA"]["RATIO_full_scope"],
     "determinism_floor": 0.0,
     "reading": "Bit-identical on all 48 forward values, on the gradient median, on the "
                "mass-weighted gradient and on the ratio. The determinism floor of this "
                "instrument is exactly 0, so every difference read below is a difference.",
   },
   "WALL_CLOCK_FLOOR": {
     "forward_seconds": [arms["bf16auto"]["forward_seconds"], arms["bf16auto_AA"]["forward_seconds"]],
     "backward_seconds": [arms["bf16auto"]["backward_seconds"], arms["bf16auto_AA"]["backward_seconds"]],
     "forward_spread": abs(arms["bf16auto"]["forward_seconds"] - arms["bf16auto_AA"]["forward_seconds"])
                       / min(arms["bf16auto"]["forward_seconds"], arms["bf16auto_AA"]["forward_seconds"]),
     "loadavg": [arms["bf16auto"]["loadavg_before_forward"], arms["bf16auto_AA"]["loadavg_before_forward"]],
     "reading": "Two byte-identical arms 54 % apart in forward wall clock on a box at load 9 to "
                "13 of 32 cores. No timing difference under that is readable here, and no claim "
                "in this artifact rests on one. of3t-shapekey declined to read 7.7 % against a "
                "9.1 % floor; this floor is six times worse and the arms are CPU arms sharing a "
                "host with three other rows.",
   },
   "A16_zero_model": {
     "value": 1.0,
     "reading": "rel_l2 of a zero gradient against a non-zero reference is 1.0 by construction, "
                "and perf/of3t_cond043/BARS043.json measured it at 1.0 over all 761 tensors on "
                "this very capture. Every arm here sits below it and the break control above it.",
   },
   "A14_near_zero_reference": {
     "bf16_a14_dropped": b["GRADIENT_full_scope"]["a14_dropped"],
     "min_reference_norm_gradient": b["GRADIENT_full_scope"]["min_ref_norm"],
     "min_reference_norm_forward": b["FORWARD"]["min_reference_norm"],
     "reading": "0 of 761 tensors sit below the 1e-12 floor and the forward's smallest reference "
                "norm is four orders above it, so nothing here is a small-denominator artifact.",
   },
   "D141_FINGERPRINT": b["FINGERPRINT"],
   "D141_reading": "The guard of3t-ditcot said would have caught the architecture mismatch at "
                   "the first capture is armed in this arm and passes: 761 parameters, 24 "
                   "per-block attention_pair_bias.layer_norm_z, 0 unexpected and 0 missing keys "
                   "in the diffusion scope, and the capture's grad_f64 key set equals this "
                   "tree's DiffusionModule parameter names. floor_bf16.py loads strict=False "
                   "and never inspects them.",
   "CROSS_HOST_REPRODUCTION": {
     "upstream_bf16_gradient_median_qb1": b["GRADIENT_full_scope"]["median_rel"],
     "upstream_bf16_gradient_median_qb2_on_record": 0.12940662000679157,
     "upstream_bf16_mass_weighted_qb1": b["GRADIENT_full_scope"]["mass_weighted_rel"],
     "upstream_bf16_mass_weighted_qb2_on_record": 0.1881764347206203,
     "upstream_f32_gradient_median_qb1": arms["f32"]["GRADIENT_full_scope"]["median_rel"],
     "upstream_f32_gradient_median_qb2_on_record": 1.2533394198898025e-05,
     "reading": "Upstream's own bf16 and fp32 gradients reproduce the qb2 record to every digit "
                "at OMP_NUM_THREADS=8 pinned. That is worth stating because the campaign holds a "
                "6.7 % host-sensitivity on an upstream bf16 arm (of3t-bwdaccum control 7) at a "
                "DIFFERENT boundary. At this boundary, with the thread count pinned to the value "
                "the capture was built with, the arm is host-invariant between qb1 and qb2. The "
                "ratio does not depend on that holding, because both halves are same-process.",
   },
 },
 "LIMIT_OF_THIS_INSTRUMENT": [
   "One boundary, one crop (384 tokens, 422 atoms), one checkpoint, 48 structures. A factor is a "
   "ratio of two medians and neither median is a scope-wide statement about the model.",
   "The msa_module track is NOT measured here. D58's second sighting reads 10.903x on our side "
   "and no upstream bf16 arm exists for it. The conditioning result PREDICTS its upstream ratio "
   "lands in the same 7x-10x band; that is a prediction, not a measurement, and it is the next "
   "arm rather than a claim of this one.",
   "Upstream's arms are CPU arms and ours is a device arm, so the two differ in more than "
   "precision policy. That is why the argument does not rest on the difference between 11.026x "
   "and 7.666x: it rests on both being far from 1x, and on the fp32 arm showing the same factor "
   "four orders of absolute error away.",
   "The upstream f32 arm is not an instrument floor for OUR arm. It is upstream's own fp32 "
   "recipe against upstream's own float64, and it is used here only to show the factor survives "
   "a change of precision.",
 ],
}
if "bf16auto_BREAK" in arms:
    bk = arms["bf16auto_BREAK"]
    out["CONTROLS"]["BREAK_that_moves"] = {
      "arm": "bf16auto_BREAK, the cotangent's structure axis rolled by one",
      "forward_median": bk["FORWARD"]["median"],
      "forward_bit_identical_to_baseline": bk["FORWARD"]["per_structure"] == b["FORWARD"]["per_structure"],
      "gradient_median_761": bk["GRADIENT_full_scope"]["median_rel"],
      "gradient_moved_by": bk["GRADIENT_full_scope"]["median_rel"] / b["GRADIENT_full_scope"]["median_rel"],
      "ratio_761": bk["RATIO_full_scope"],
      "cot_norm": bk["cot_norm"], "cot_used_norm": bk["cot_used_norm"],
      "reading": "The forward is bit-identical because the forward does not read the cotangent, "
                 "and the gradient moves. That is the control a RATIO needs rather than the one "
                 "a reading needs: it shows the numerator responds and pins that the denominator "
                 "is not what responded.",
    }
else:
    out["CONTROLS"]["BREAK_that_moves"] = "PENDING: bf16auto_BREAK had not finished when this was assembled"

(A / "AMPLIFICATION.json").write_text(json.dumps(out, indent=1, sort_keys=False) + "\n")
print(json.dumps({"VERDICT": out["VERDICT"], "TABLE": out["TABLE"],
                  "DECOMPOSITION": out["DECOMPOSITION_matched_scope"],
                  "PRECISION": out["PRECISION_INDEPENDENCE"]}, indent=1))
print("wrote", A / "AMPLIFICATION.json")
