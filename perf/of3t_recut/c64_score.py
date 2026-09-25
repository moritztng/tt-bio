#!/usr/bin/env python3
"""of3t-recut job 2, crop 64: what the three arms of `c64_controls.sh` prove.

Three claims, and each is a comparison against something that existed before this row ran:

  LEGACY_REPRODUCES_THE_BANKED_ARM  --legacy-total-cotangent against of3t-trunkg043's
      `ref_f64_c64.pt` and the loss and squared gradient norm in its `REF_F64_c64.json`. Bit for
      bit over all 2,736 tensors, or the flag is not the old behaviour.
  DEFAULT_MOVES                     the default against that same banked arm. It must NOT
      reproduce it -- a bug fix that changes nothing fixed nothing -- and the size of the move is
      the size of the double count at this crop.
  CHECKPOINT_IS_INERT               default plain against default --checkpoint. Bit-identical,
      which is what lets the crop-384 arm use the flag.
"""
from __future__ import annotations

import json
import socket
import sys
from pathlib import Path

import torch

O = Path("/home/ttuser/of3t_recut")
BANKED_PT = Path("/home/ttuser/of3t_trunkg043/ref_f64_c64.pt")
BANKED_JSON = Path("perf/of3t_trunkg043/REF_F64_c64.json")
BANKED_LOSS = -0.3073181442478619
BANKED_SQ = 1.8714981803225081


def grads(p):
    d = torch.load(p, map_location="cpu", weights_only=False)
    return d, d["grads"]


def cmp_bits(a, b):
    """Tensor-wise bit comparison over the union of the two key sets."""
    keys = sorted(set(a) | set(b))
    n_bit, worst, worst_k, absent = 0, 0.0, None, []
    num = den = 0.0
    for k in keys:
        x, y = a.get(k), b.get(k)
        if x is None or y is None:
            absent.append(k)
            continue
        x = x.to(torch.float64)
        y = y.to(torch.float64)
        if torch.equal(x, y):
            n_bit += 1
        d = float(torch.linalg.vector_norm(x - y))
        r = float(torch.linalg.vector_norm(y))
        num += d ** 2
        den += r ** 2
        rel = d / r if r else (0.0 if d == 0 else float("inf"))
        if rel > worst:
            worst, worst_k = rel, k
    return {"compared": len(keys) - len(absent), "bit_identical": n_bit,
            "all_bit_identical": n_bit == len(keys) - len(absent) and not absent,
            "absent": absent[:4],
            "mass_weighted_rel_l2": (num / den) ** 0.5 if den else None,
            "worst_rel_l2": worst, "worst_tensor": worst_k}


def main() -> int:
    banked_rep = json.loads(BANKED_JSON.read_text())
    _, banked = grads(BANKED_PT)
    out = {"what": __doc__.strip().splitlines()[0], "host": socket.gethostname(),
           "device_involved": False,
           "why_no_aiclk": "CPU only, no Tenstorrent device is opened",
           "banked": {"pt": str(BANKED_PT), "report": str(BANKED_JSON),
                      "loss": banked_rep["loss"],
                      "squared_gradient_norm": banked_rep["gradient"]["squared_norm_total"],
                      "injection_convention": banked_rep.get("injection", {}).get(
                          "convention", "predates the stamp: legacy-total-cotangent")}}

    arms = {}
    for nm in ("c64_legacy", "c64_corrected", "c64_corrected_ckpt"):
        d, g = grads(O / f"{nm}.pt")
        rep = json.loads(Path(f"perf/of3t_recut/{nm.upper()}.json").read_text())
        arms[nm] = {"d": d, "g": g, "rep": rep}

    lg, cr, ck = arms["c64_legacy"], arms["c64_corrected"], arms["c64_corrected_ckpt"]

    out["LEGACY_REPRODUCES_THE_BANKED_ARM"] = {
        "stamp": lg["d"]["injection_convention"],
        "loss": {"banked": BANKED_LOSS, "legacy": lg["rep"]["loss"],
                 "bit_identical": lg["rep"]["loss"] == BANKED_LOSS},
        "squared_gradient_norm": {
            "banked": BANKED_SQ, "legacy": lg["rep"]["gradient"]["squared_norm_total"],
            "bit_identical": lg["rep"]["gradient"]["squared_norm_total"] == BANKED_SQ},
        "tensors": cmp_bits(lg["g"], banked),
    }
    out["LEGACY_REPRODUCES_THE_BANKED_ARM"]["verdict"] = (
        "the flag IS the old behaviour" if
        out["LEGACY_REPRODUCES_THE_BANKED_ARM"]["tensors"]["all_bit_identical"] and
        out["LEGACY_REPRODUCES_THE_BANKED_ARM"]["loss"]["bit_identical"] else
        "the flag does NOT reproduce the banked arm")

    inj = cr["rep"]["injection"]
    out["DEFAULT_MOVES"] = {
        "stamp": cr["d"]["injection_convention"],
        "graph_cut": inj["graph_cut"],
        "cot_z_norm_hooked": inj["cot_z_norm_hooked"],
        "cot_z_norm_injected": inj["cot_z_norm_injected"],
        "correction_norm": inj["correction_norm"],
        "duplicate_share_of_the_hooked_cot_z": inj["duplicate_share_of_the_hooked_cot_z"],
        "true_external_cotangent_is_smaller_by":
            inj["cot_z_norm_hooked"] / inj["cot_z_norm_injected"],
        "loss": {"banked": BANKED_LOSS, "corrected": cr["rep"]["loss"]},
        "squared_gradient_norm": {
            "banked": BANKED_SQ,
            "corrected": cr["rep"]["gradient"]["squared_norm_total"]},
        "tensors_vs_the_banked_arm": cmp_bits(cr["g"], banked),
    }
    out["DEFAULT_MOVES"]["verdict"] = (
        "the default is a different injection from the banked one, and z_out is an ancestor of "
        "s_out at this boundary too"
        if not out["DEFAULT_MOVES"]["tensors_vs_the_banked_arm"]["all_bit_identical"]
        and inj["graph_cut"]["ancestor_descendant_pairs"] == [["z_out", "s_out"]]
        else "the default did not move, or the boundary is a cut")

    out["CHECKPOINT_IS_INERT"] = {
        "plain_loss": cr["rep"]["loss"], "checkpointed_loss": ck["rep"]["loss"],
        "loss_bit_identical": cr["rep"]["loss"] == ck["rep"]["loss"],
        "correction_norm_plain": inj["correction_norm"],
        "correction_norm_checkpointed": ck["rep"]["injection"]["correction_norm"],
        "tensors": cmp_bits(cr["g"], ck["g"]),
        "what": "--checkpoint now leaves the last block eager, so this is the control that says "
                "the flag is still bit-neutral and the crop-384 arm may use it",
    }
    out["CHECKPOINT_IS_INERT"]["verdict"] = (
        "inert" if out["CHECKPOINT_IS_INERT"]["tensors"]["all_bit_identical"]
        and out["CHECKPOINT_IS_INERT"]["loss_bit_identical"] else "NOT inert")

    p = Path("perf/of3t_recut/C64_CONTROLS.json")
    p.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    ok = (out["LEGACY_REPRODUCES_THE_BANKED_ARM"]["verdict"].startswith("the flag IS")
          and out["DEFAULT_MOVES"]["verdict"].startswith("the default is")
          and out["CHECKPOINT_IS_INERT"]["verdict"] == "inert")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
