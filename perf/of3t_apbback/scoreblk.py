#!/usr/bin/env python3
"""Score the single-block device arms against the capture's own float64 gradient.

The reference is `block47_boundary.pt`'s own `grad`: upstream's float64 gradient for this block,
from the same run that produced the input and the cotangent the arms are fed. Upstream's block
re-run in float64 on those operands reproduces it BIT-IDENTICALLY on all 16 tensors of
`attn_pair_bias` and `single_transition` (BLK47_VALIDATION.json), so a substitution that moves an
arm toward it is recovering real error and not a frame difference.

RECOVERED is the campaign's error mass: sum_t mass_t * rel_t^2 over the scope, where mass_t is
the tensor's share of the reference's squared gradient norm. recovered = 1 - E_after / E_before.
"""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.getcwd(), "perf/of3t_trunkg043"))
from score import score

CAP = "/home/ttuser/of3t_gradients/cap/block47_boundary.pt"
SCOPE = lambda k: k.startswith("attn_pair_bias.") or k.startswith("single_transition.")
MODEL = "/home/ttuser/of3t_refprec/bundle_ref/grads_f64_043.pt"


def emass(rows):
    return sum(x["mass"] * x["rel_l2"] ** 2 for x in rows)


def main():
    arms = sys.argv[2:]
    cap = torch.load(CAP, map_location="cpu", weights_only=False)
    REF = {k: v.to(torch.float64) for k, v in cap["grad"].items() if v is not None}
    M = torch.load(MODEL, map_location="cpu", weights_only=False)
    M = M["grads"] if isinstance(M, dict) and "grads" in M else M
    P = "pairformer_stack.blocks.47."
    M47 = {k[len(P):]: v for k, v in M.items() if k.startswith(P)}
    out = {"what": __doc__.strip().splitlines()[0],
           "reference": {"path": CAP, "kind": "the capture's own float64 gradient for block 47",
                         "validated": "upstream 0.4.3/0.5.0 float64 reproduces it bit-identically "
                                      "on 16 of 16 scope tensors"},
           "scope": "attn_pair_bias.* and single_transition.*", "arms": {}}
    base = None
    for path in arms:
        nm = os.path.basename(path).replace("dev_blk47_", "").replace(".pt", "")
        d = torch.load(path, map_location="cpu", weights_only=False)
        G = d["grads"]
        keys = sorted(k for k in set(G) & set(REF) if SCOPE(k))
        s = score({k: G[k] for k in keys}, {k: REF[k] for k in keys}, keys)
        rows = s.pop("_rows")
        E = emass(rows)
        per = {x["tensor"]: {"rel_l2": x["rel_l2"], "r": x["norm_ratio"], "cos": x["cos"],
                             "mass": x["mass"], "error_mass": x["mass"] * x["rel_l2"] ** 2}
               for x in rows}
        kk = sorted(k for k in set(G) & set(M47) if SCOPE(k))
        s2 = score({k: G[k] for k in kk}, {k: M47[k] for k in kk}, kk); s2.pop("_rows")
        out["arms"][nm] = {"compared": len(keys), "mass_weighted_rel_l2": s["mass_weighted_rel_l2"],
                           "norm_ratio": s["mass_weighted_norm_ratio"],
                           "cos": s["mass_weighted_cos"], "error_mass": E,
                           "worst_by_error_mass": s["worst_by_error_mass"]["tensor"],
                           "worst_rel": s["worst_by_error_mass"]["rel_l2"],
                           "vs_MODEL_f64_mw": s2["mass_weighted_rel_l2"],
                           "per_tensor": per}
        if nm == "none":
            base = out["arms"][nm]
    if base:
        for nm, v in out["arms"].items():
            v["recovered_share_of_error_mass"] = 1.0 - v["error_mass"] / base["error_mass"]
            v["rel_x_baseline"] = v["mass_weighted_rel_l2"] / base["mass_weighted_rel_l2"]
    json.dump(out, open(sys.argv[1], "w"), indent=2)
    print("%-10s %14s %10s %8s %8s %12s" % ("arm", "mw rel L2", "err mass", "r", "cos", "recovered"))
    for nm, v in sorted(out["arms"].items(), key=lambda x: x[1]["error_mass"]):
        print("%-10s %14.6e %10.3e %8.4f %8.4f %11.2f %%"
              % (nm, v["mass_weighted_rel_l2"], v["error_mass"], v["norm_ratio"], v["cos"],
                 100.0 * v.get("recovered_share_of_error_mass", float("nan"))))


main()
