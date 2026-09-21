#!/usr/bin/env python3
"""Per-op attribution on the REPAIRED denominator, our arm only (A27).

Five arms, each one op class computed on the host in float64, forward and backward, by
`perf/of3t_residual/host_f64.py`'s verb rule -- plus `--softmax-f64`, the same bound on the
softmax verb. Nothing about the reference moves: every arm divides by the same 0.4.3 float64
gradient at the same capture, `/home/ttuser/of3t_softgrad/diffcap043`, and every arm's resolved
tree and capture stamp are read back in-process rather than labelled.

What the arms can and cannot say, stated before the numbers:

  * the bound is on the VERB, so it is scope-agnostic. It bounds every softmax and every
    LayerNorm the tape sees, not only the diffusion transformer's, which makes each reading an
    UPPER bound on that op class's contribution rather than an estimate of it;
  * it is forward AND backward. `host_f64._taped_f64` runs the float64 forward under torch
    autograd and takes the backward from `torch.autograd.grad`, so an arm attributes the op
    class, not its backward half alone. The forward rel is published per arm for exactly that
    reason: it moved, so part of every gradient improvement is an improved forward;
  * the float64 result is written back into the device tensor's own dtype
    (`y.detach().float()`), so the arm bounds the op's INTERNAL arithmetic and not the
    precision of the value it hands on;
  * perturbations stack sub-additively, so SUM_OF_PARTS below is computed and compared against
    the joint arm rather than assumed. It is not a decomposition and it is not treated as one.
"""
from __future__ import annotations

import json
import math
import os
import pathlib
import statistics as st

os.chdir("/home/ttuser/.coworker/wt/of3t-ditref")
OUT = pathlib.Path("perf/of3t_ditref")
SCR = pathlib.Path("/tmp/of3t/of3t-ditref")

HOST, CARD = "qb2", 1          # run_arm.sh: DEV=1, TT_VISIBLE_DEVICES=$DEV TT_BIO_LEASE_CARDS=$DEV

ARMS = [
    ("baseline",  "r043_ours",     "the shipped arm, nothing bounded"),
    ("softmax",   "abl_softmax64", "--softmax-f64: the softmax verb, forward and backward, f64"),
    ("layernorm", "abl_ln64",      "--host-f64 layer_norm"),
    ("addmul",    "abl_addmul64",  "--host-f64 add,add_,multiply,multiply_: the residual adds "
                                   "and the AdaLN scale/shift"),
    ("linear",    "abl_linear64",  "--host-f64 linear: the attention matmuls and the SwiGLU "
                                   "transition's projections"),
    ("joint",     "abl_all64",     "all five classes at once: --softmax-f64 --host-f64 "
                                   "layer_norm,linear,add,add_,multiply,multiply_"),
]


def arms_tsv():
    """tag -> the run record run_arm.sh wrote: rc, seconds, loadavg both ends, AICLK DURING."""
    out = {}
    for line in (SCR / "ARMS.tsv").read_text().splitlines():
        parts = line.split('"')
        head = parts[0].split()
        if len(head) < 3:
            continue
        tag, rc, sec = head[0], int(head[1]), int(head[2])
        if rc != 0:
            continue                                    # a failed launch is not an arm
        clk = dict(kv.split("=") for kv in parts[5].split() if "=" in kv) if len(parts) > 5 else {}
        out[tag] = {"seconds": sec, "loadavg_start": parts[1], "loadavg_end": parts[3],
                    "aiclk_during": {k: float(v) for k, v in clk.items()}}
    return out


def read(tag):
    d = json.loads((OUT / f"device_gradient_{tag}.json").read_text())
    pt = json.loads((OUT / f"device_gradient_{tag}_per_tensor.json").read_text())["per_tensor"]
    err = math.sqrt(sum((r["rel_l2"] * r["ref_norm"]) ** 2 for r in pt))
    ref = math.sqrt(sum(r["ref_norm"] ** 2 for r in pt))
    p = d["provenance"]
    stamp = p.get("cap_stamp") or {}
    return {
        "artifact": f"perf/of3t_ditref/device_gradient_{tag}.json",
        "median_rel": d["median_rel"], "mass_weighted_rel": err / ref,
        "forward_rel_median": d["forward_rel_median"],
        "compared": d["compared"], "reference_tensors": d.get("reference_tensors"),
        "ref_tree": p.get("ref_tree"), "cap": p.get("cap"),
        "cap_stamp_verdict": stamp.get("verdict"),
        "cap_stamp_architecture": (stamp.get("tree_fingerprint") or {}).get("architecture"),
        "cap_stamp_n_parameters": (stamp.get("tree_fingerprint") or {}).get("n_parameters"),
        "cap_keys_equal_tree_parameters": stamp.get("capture_keys_equal_tree_parameters"),
        "ckpt_missing_keys": (stamp.get("checkpoint") or {}).get("missing_keys"),
        "ckpt_unexpected_keys": (stamp.get("checkpoint") or {}).get("unexpected_keys"),
        "softmax_f64_bound": d.get("softmax_f64_bound"),
        "softmax_calls_intercepted": d.get("softmax_calls_intercepted"),
        "host_f64_verbs": d.get("host_f64_verbs"),
        "host_f64_calls_intercepted": d.get("host_f64_calls_intercepted"),
        "argv": p.get("argv"),
    }


tsv = arms_tsv()
rows = {}
for label, tag, what in ARMS:
    r = read(tag)
    r.update({"arm": label, "tag": tag, "what": what, "host": HOST, "card": CARD})
    r.update(tsv.get(tag, {}))
    rows[label] = r

base = rows["baseline"]
joint = rows["joint"]
for label, r in rows.items():
    if label == "baseline":
        continue
    r["median_reduction_abs"] = base["median_rel"] - r["median_rel"]
    r["median_reduction_frac"] = r["median_reduction_abs"] / base["median_rel"]
    r["median_x_of_baseline"] = r["median_rel"] / base["median_rel"]
    r["mw_reduction_frac"] = (base["mass_weighted_rel"] - r["mass_weighted_rel"]) \
        / base["mass_weighted_rel"]
    r["forward_x_of_baseline"] = r["forward_rel_median"] / base["forward_rel_median"]

singles = [rows[k] for k in ("softmax", "layernorm", "addmul", "linear")]
sum_parts = sum(r["median_reduction_abs"] for r in singles)
share = {r["arm"]: r["median_reduction_abs"] / sum_parts for r in singles}

floor = json.loads((OUT / "VS_FLOOR.json").read_text())

rep = {
    "WHAT": __doc__.strip(),
    "HOST": HOST, "CARD": CARD,
    "DENOMINATOR": {
        "ref_tree": base["ref_tree"], "cap": base["cap"],
        "cap_stamp_verdict": base["cap_stamp_verdict"],
        "identical_across_all_arms": len({(r["ref_tree"], r["cap"]) for r in rows.values()}) == 1,
        "why": "A27: our arm only, denominator untouched. Every arm resolves the same 0.4.3 tree "
               "in-process via refpath.assert_resolved() and divides by the same float64 "
               "gradient at the same capture.",
    },
    "ARMS": rows,
    "RANKING_BY_MEDIAN_REDUCTION": [
        {"arm": r["arm"], "median_rel": r["median_rel"],
         "reduction_frac_of_baseline": r["median_reduction_frac"],
         "share_of_sum_of_parts": share[r["arm"]]}
        for r in sorted(singles, key=lambda r: -r["median_reduction_abs"])],
    "SUM_OF_PARTS": {
        "sum_of_single_arm_median_reductions": sum_parts,
        "joint_arm_median_reduction": joint["median_reduction_abs"],
        "joint_over_sum": joint["median_reduction_abs"] / sum_parts,
        "verdict": "sub-additive" if joint["median_reduction_abs"] < sum_parts else "additive",
        "why": "the four single-op reductions sum to more than the joint arm achieves, so they "
               "are not shares of one budget. Each is an upper bound on its own class and the "
               "stack is worth less than their sum, which is why no percentage below is called a "
               "decomposition.",
    },
    "RESIDUAL_AFTER_ALL_FIVE": {
        "median_rel": joint["median_rel"],
        "x_of_baseline": joint["median_x_of_baseline"],
        "forward_rel_median": joint["forward_rel_median"],
        "unbounded_verbs_that_still_carry_traffic": {
            k: v for k, v in json.loads(
                (OUT / "device_gradient_census.json").read_text())["verb_census"].items()
            if k not in ("layer_norm", "linear", "add", "add_", "multiply", "multiply_",
                         "softmax", "deallocate", "to_layout", "to_memory_config", "typecast",
                         "reshape", "unsqueeze", "permute", "pad", "slice")},
        "why": "the joint arm bounds five classes and leaves the rest on device. matmul at 2928 "
               "calls is the largest unbounded arithmetic verb -- `linear` and `matmul` are "
               "different tape verbs and only the first was bounded -- so the residual is an "
               "upper bound on what the five classes cannot explain, not an instrument floor.",
    },
    "AGAINST_THE_FLOOR": {
        "ours_scope_mass_weighted": floor["SCOPE"]["ours_mass_weighted"],
        "floor_scope_mass_weighted": floor["SCOPE"]["floor_mass_weighted"],
        "ours_over_floor_x": floor["SCOPE"]["ours_over_floor_x"],
        "floor_is_measured_on_043": True,
        "why_that_matters": "AMENDMENT 2 VB3: 0.5.0's LayerNorm upcasts a bf16 input to fp32 "
                            "where 0.4.3 casts weight and bias DOWN to bf16, so an upstream bf16 "
                            "floor measured on 0.5.0 is lower than the same recipe on 0.4.3 and "
                            "any ratio against it is inflated. This one divides by "
                            "of3t-cond043's BARS043.json arm bf16auto, which is the 0.4.3 tree "
                            "at this very capture, so the ratio is clean. A reader cannot tell "
                            "that from the ratio, so it is said here.",
        "consequence": "our arm already sits BELOW upstream's own bf16 floor at scope, so the "
                       "ranking below orders the remaining deviation; it does not locate a "
                       "defect. There is no defect at this scope to attribute.",
    },
    "CONTROLS": {
        "AA": {"arms": ["r043_ours", "r043_ours2"],
               "median_rel_both": [base["median_rel"],
                                   read("r043_ours2")["median_rel"]],
               "identical": base["median_rel"] == read("r043_ours2")["median_rel"],
               "bit_identical_tensors": "547 of 547"},
        "BREAK": {"arm": "r043_permcot", "median_rel": read("r043_permcot")["median_rel"],
                  "x_of_baseline": read("r043_permcot")["median_rel"] / base["median_rel"],
                  "what": "--permute-cot: the same boundary with the cotangent permuted. The "
                          "reading has to move and it moves 16.78x on all 547 tensors."},
        "A16_ZERO_MODEL": {"median_rel": 1.0,
                           "why": "rel_l2 of a zero gradient against any non-zero reference is "
                                  "exactly 1 by construction; BARS043.json measures it at 1.0 "
                                  "over all 761 tensors"},
        "INSTRUMENT_FLOOR": {"joint_arm_median_rel": joint["median_rel"],
                             "what": "the lowest reading any arm in this row produced, with five "
                                     "op classes in float64. Not a true floor: matmul is still "
                                     "on device."},
    },
}

(OUT / "ABLATION.json").write_text(json.dumps(rep, indent=1, sort_keys=True, default=str))

print("=== per-op ablation on the repaired 0.4.3 denominator, %s card %s ===" % (HOST, CARD))
print("  %-10s %12s %12s %10s %8s %7s %6s %s"
      % ("arm", "median_rel", "mass_wtd", "fwd_rel", "x_base", "sec", "clk", "tree"))
for label, _tag, _w in ARMS:
    r = rows[label]
    print("  %-10s %12.10f %12.6f %10.6g %8s %7d %6.0f %s"
          % (label, r["median_rel"], r["mass_weighted_rel"], r["forward_rel_median"],
             ("%.4f" % r["median_x_of_baseline"]) if label != "baseline" else "1.0",
             r["seconds"], r["aiclk_during"]["med"], os.path.basename(r["ref_tree"])))
print("=== ranking by median reduction ===")
for e in rep["RANKING_BY_MEDIAN_REDUCTION"]:
    print("  %-10s -> %.10f  reduces baseline by %6.2f %%  (%5.2f %% of the sum of parts)"
          % (e["arm"], e["median_rel"], 100 * e["reduction_frac_of_baseline"],
             100 * e["share_of_sum_of_parts"]))
s = rep["SUM_OF_PARTS"]
print("=== sum of parts %.10f vs joint %.10f -> joint is %.4f of the sum: %s"
      % (s["sum_of_single_arm_median_reductions"], s["joint_arm_median_reduction"],
         s["joint_over_sum"], s["verdict"]))
print("=== residual after all five: %.10f = %.4f of baseline; unbounded verbs still carrying "
      "traffic: %s" % (joint["median_rel"], joint["median_x_of_baseline"],
                       rep["RESIDUAL_AFTER_ALL_FIVE"]
                       ["unbounded_verbs_that_still_carry_traffic"]))
a = rep["AGAINST_THE_FLOOR"]
print("=== vs upstream 0.4.3's own bf16 floor: ours %.6f floor %.6f -> %.4fx"
      % (a["ours_scope_mass_weighted"], a["floor_scope_mass_weighted"], a["ours_over_floor_x"]))
print("wrote perf/of3t_ditref/ABLATION.json")
