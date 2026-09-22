#!/usr/bin/env python3
"""of3t-recut job 2 at crop 384, on the model's own frame: the repaired injection against the
reference's own full-model backward, and the legacy flag against the banked arm.

  CORRECTED   `ref_f64_model_n384_corrected.pt` against `grads_f64_043.pt`, block 47 and then
              all 2,736 trunk tensors. Block 47 must reach the 1e-12 bar fixed in 2520681ed;
              of3t-frameself got 3.0392623414001263e-15 there through its own break control and
              this is the same claim through the shipped instrument.
  FALSIFIER   the norm of dL/dz_in, which of3t-frameself banked BEFORE any of this ran:
              0.000848887340907281 if the injection is exact, 0.0014907294032500784 if it
              overcounts.
  LEGACY      `ref_f64_model_n384_legacy.pt` against of3t-twoside's `ctrl_f64.pt`. Same host,
              same tree, same 14 threads, same producer modulo this repair, so bit for bit.

usage: n384_check.py [--legacy]
"""
from __future__ import annotations

import argparse
import json
import socket
from pathlib import Path

import torch

O = Path("/home/ttuser/of3t_recut")
REF = Path("/home/ttuser/of3t-campaign-refs/bundle_min_043/grads_f64_043.pt")
BANKED = Path("/home/ttuser/of3t_twoside/ctrl_f64.pt")
BANKED_REP = Path("/home/ttuser/of3t_twoside/ctrl_f64.json")
PRE = "pairformer_stack.blocks."
BAR = 1e-12
FRAMESELF_BLOCK47 = 3.0392623414001263e-15
FALSIFIER = {"exact_means": 0.000848887340907281,
             "overcounts_means": 0.0014907294032500784,
             "source": "perf/of3t_frameself/BREAK_DOUBLECOUNT.json prereg_falsifier"}


def fit(arm, ref, keys):
    """of3t-frameself's fit, unchanged: pooled rel_l2 over a key set."""
    dot = a2 = r2 = e2 = 0.0
    worst, worst_name, n_bit, n_absent = -1.0, None, 0, 0
    for n in keys:
        r = ref[n].to(torch.float64).reshape(-1)
        rn2 = float(torch.dot(r, r))
        r2 += rn2
        m = arm.get(n)
        if m is None:
            n_absent += 1
            e2 += rn2
            continue
        m = m.to(torch.float64).reshape(-1)
        dot += float(torch.dot(m, r))
        a2 += float(torch.dot(m, m))
        d = float(torch.dot(m - r, m - r))
        e2 += d
        if torch.equal(m, r):
            n_bit += 1
        rel = (d / rn2) ** 0.5 if rn2 > 0 else (0.0 if d == 0 else float("inf"))
        if rel > worst:
            worst, worst_name = rel, n
    return {"n_tensors": len(keys), "n_absent_from_arm": n_absent, "n_bit_identical": n_bit,
            "rel_l2_as_is": (e2 / r2) ** 0.5 if r2 else None,
            "norm_ratio_arm_over_ref": (a2 / r2) ** 0.5 if r2 else None,
            "cos": dot / (a2 * r2) ** 0.5 if a2 > 0 and r2 > 0 else None,
            "best_scalar_a_star": dot / a2 if a2 > 0 else None,
            "residual_after_best_scalar_frac_of_ref":
                (max(r2 - dot * dot / a2, 0.0) / r2) ** 0.5 if a2 > 0 and r2 else None,
            "ref_squared_norm": r2, "arm_squared_norm": a2,
            "worst_rel_l2": worst, "worst_tensor": worst_name}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--legacy", action="store_true",
                    help="also score the --legacy-total-cotangent arm against the banked one")
    a = ap.parse_args()

    out = {"what": __doc__.strip().splitlines()[0], "host": socket.gethostname(),
           "row": "of3t-recut", "defect": "D242", "device_involved": False,
           "why_no_aiclk": "CPU only, no Tenstorrent device is opened",
           "bar": BAR}

    ref = torch.load(REF, map_location="cpu", weights_only=False)
    ref = ref.get("grads", ref) if isinstance(ref, dict) else ref
    arm = torch.load(O / "ref_f64_model_n384_corrected.pt", map_location="cpu",
                     weights_only=False)
    rep = json.loads(Path("perf/of3t_recut/REF_F64_MODEL_N384_CORRECTED.json").read_text())
    g = arm["grads"]
    b47 = sorted(k for k in ref if k.startswith(f"{PRE}47."))
    allk = sorted(k for k in ref if k.startswith(PRE))

    out["CORRECTED"] = {
        "stamp": arm["injection_convention"],
        "injection": rep["injection"],
        "block47": fit(g, ref, b47),
        "all_trunk_tensors": fit(g, ref, allk),
        "of3t_frameself_got_at_block47": FRAMESELF_BLOCK47,
    }
    out["CORRECTED"]["block47_clears_the_bar"] = (
        out["CORRECTED"]["block47"]["rel_l2_as_is"] <= BAR)
    out["CORRECTED"]["all_trunk_clears_the_bar"] = (
        out["CORRECTED"]["all_trunk_tensors"]["rel_l2_as_is"] <= BAR)

    out["FALSIFIER"] = dict(FALSIFIER, measured_dz_in_norm=rep["dz_in_norm"])
    e, o_ = FALSIFIER["exact_means"], FALSIFIER["overcounts_means"]
    m = rep["dz_in_norm"]
    out["FALSIFIER"]["rel_to_exact"] = abs(m - e) / e
    out["FALSIFIER"]["rel_to_overcounting"] = abs(m - o_) / o_
    out["FALSIFIER"]["verdict"] = ("the injection is the exact one"
                                   if abs(m - e) / e < abs(m - o_) / o_
                                   else "the injection still overcounts")

    if a.legacy:
        lg = torch.load(O / "ref_f64_model_n384_legacy.pt", map_location="cpu",
                        weights_only=False)
        bk = torch.load(BANKED, map_location="cpu", weights_only=False)
        lrep = json.loads(Path("perf/of3t_recut/REF_F64_MODEL_N384_LEGACY.json").read_text())
        brep = json.loads(BANKED_REP.read_text())
        out["LEGACY"] = {
            "stamp": lg["injection_convention"],
            "banked": {"pt": str(BANKED), "loss": brep["loss"],
                       "squared_gradient_norm": brep["gradient"]["squared_norm_total"],
                       "dz_in_norm": brep["dz_in_norm"], "host": brep["provenance"]["host"],
                       "threads": brep["threads"]},
            "legacy": {"loss": lrep["loss"],
                       "squared_gradient_norm": lrep["gradient"]["squared_norm_total"],
                       "dz_in_norm": lrep["dz_in_norm"]},
            "loss_bit_identical": lrep["loss"] == brep["loss"],
            "squared_gradient_norm_bit_identical":
                lrep["gradient"]["squared_norm_total"]
                == brep["gradient"]["squared_norm_total"],
            "tensors": fit(lg["grads"], bk["grads"], allk),
        }
        t = out["LEGACY"]["tensors"]
        out["LEGACY"]["verdict"] = (
            "the flag IS the banked behaviour, bit for bit"
            if t["n_bit_identical"] == t["n_tensors"] and out["LEGACY"]["loss_bit_identical"]
            else "the flag does NOT reproduce the banked arm bit for bit")

    p = Path("perf/of3t_recut/N384_CONTROLS.json")
    p.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
