#!/usr/bin/env python3
"""of3t-refatom: what the discovery check's assertion-4 miss actually is. HOST ONLY, no card.

`CHECK.json` assertion 4 passes its cosine clause (1 - 1.5e-7 over 917,504 elements) and its
rel-L2 clause (1.24e-3) and misses its norm-ratio clause: the device `plm` is 0.998884 of the
host norm against a pre-registered 1e-3 tolerance. The bar stays where it was fixed. This
file asks what the miss IS, and it asks it without a card, because the answer is arithmetic.

THE HYPOTHESIS, AND WHAT IT FORBIDS. `trajwide` builds every device module with
`compute_kernel_config=None` -- `OF3DiffusionModule(dmsd, None)` on the of3t-trajwide leg and
`RefAtomFeatureEmbedder(rafe_sd, None)` on this one -- so the matmuls run at ttnn's DEFAULT
fidelity, which feeds the FPU bf16-truncated operand mantissas rather than the fp32 the
tensors are stored in. Truncation toward zero is a BIASED rounding, so it predicts three
things at once and a wrong-function explanation predicts none of them:

  1. the norm comes out LOW, not merely different -- a sign, not a magnitude;
  2. it comes out low by about the bf16 mantissa's own truncation bias, ~1e-3, on BOTH legs;
  3. the same host arithmetic with operands truncated to bf16 reproduces the device reading,
     while round-to-NEAREST bf16 (torch's own `.bfloat16()`) does NOT -- it is unbiased and
     lands near 1.0.

Point 3 is the one that can fail. If nearest-rounding matched the device as well as
truncation does, the reading would not be evidence of anything.

WHAT THIS IS NOT. Not a precision gate, not a proposal to change the arm's fidelity: raising
this one module to HiFi4 would make the atom featurization more accurate than every
neighbour it feeds, which is a different trajectory and not a repair. And not a claim that
the trajectory's forward is accurate to 1e-3 -- the arm's own forward rel at level 0 is
2.08e-02, two orders looser than this leg, so this leg is not what limits it.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

OUT = "perf/of3t_refatom/FIDELITY_CONTROL.json"


def load(name, path):
    d = os.path.dirname(os.path.abspath(path))
    if d not in sys.path:
        sys.path.insert(0, d)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def trunc_bf16(x):
    """bf16 by TRUNCATION of the fp32 mantissa, in fp32. Biased toward zero, unlike torch's
    `.bfloat16()`, which rounds to nearest even."""
    import torch
    u = x.float().view(torch.int32) & torch.tensor(-65536, dtype=torch.int32)
    return u.view(torch.float32)


def near_bf16(x):
    import torch
    return x.float().to(torch.bfloat16).float()


def stats(a, b):
    import torch
    a = a.double().reshape(-1)
    b = b.double().reshape(-1)
    na, nb = torch.linalg.vector_norm(a), torch.linalg.vector_norm(b)
    return {"rel_l2": float(torch.linalg.vector_norm(a - b) / (nb + 1e-300)),
            "cos": float((a @ b) / (na * nb + 1e-300)),
            "norm_ratio": float(na / (nb + 1e-300))}


def main() -> int:
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    tw = load("trajwide", "perf/of3t_trajwide/trajwide.py")
    tw.refpath.install()

    import torch
    from tt_bio import openfold3_host_prep as HP
    from tt_bio.openfold3_weights import _sub

    chk = json.load(open("perf/of3t_refatom/CHECK.json"))
    dev_read = chk["function"]

    D = torch.load(tw.DIFFCAP, map_location="cpu", weights_only=False)
    kw = D["kwargs"]
    batch = kw["batch"]
    sq = lambda x: (x.reshape(x.shape[2:]) if x.dim() > 2 and x.shape[0] == 1
                    and x.shape[1] == 1 else x.squeeze(0))
    feats = {k: sq(batch[k]).float() for k in
             ("ref_pos", "ref_charge", "ref_mask", "ref_element", "ref_atom_name_chars",
              "ref_space_uid")}
    feats["atom_mask"] = sq(batch["atom_mask"]).float()

    sd = torch.load(tw.CKPT, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    sd = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}
    rafe_sd = _sub(_sub(_sub(sd, "diffusion_module"), "atom_attn_enc"),
                   "ref_atom_feature_embedder")

    ref64 = HP.ref_atom_embed({k: v.double() for k, v in rafe_sd.items()},
                              {k: v.double() for k, v in feats.items()})
    arms = {}
    for label, r in (("fp32", lambda x: x.float()),
                     ("bf16_truncated_operands", trunc_bf16),
                     ("bf16_nearest_operands", near_bf16)):
        # Only the OPERANDS are rounded. The accumulation stays fp32, which is what the FPU
        # does: it truncates the mantissas it multiplies and accumulates the products wide.
        cl, plm = HP.ref_atom_embed({k: r(v) for k, v in rafe_sd.items()},
                                    {k: r(v) for k, v in feats.items()})
        arms[label] = {"cl_vs_host_f64": stats(cl, ref64[0]),
                       "plm_vs_host_f64": stats(plm, ref64[1])}

    dev = {"cl_vs_host_f64": dev_read["cl_device_vs_host_f64"],
           "plm_vs_host_f64": dev_read["plm_device_vs_host_f64"]}

    def agrees(arm, leg):
        """The arm reproduces the device on this leg: same sign of the norm bias, and the
        magnitude within a factor of two of it."""
        d, a = dev[leg]["norm_ratio"] - 1.0, arms[arm][leg]["norm_ratio"] - 1.0
        return bool(d * a > 0 and 0.5 <= abs(a / d) <= 2.0)

    verdict = {
        "truncation_reproduces_the_device": {
            leg: agrees("bf16_truncated_operands", leg) for leg in dev},
        "nearest_does_not": {
            leg: not agrees("bf16_nearest_operands", leg) for leg in dev},
    }
    res = {
        "what": "what CHECK.json assertion 4's norm-ratio miss is. Host arithmetic only.",
        "device_reading_being_explained": dev,
        "host_arms": arms,
        "norm_bias_vs_float64": {
            "device": {leg: dev[leg]["norm_ratio"] - 1.0 for leg in dev},
            **{a: {leg: arms[a][leg]["norm_ratio"] - 1.0 for leg in dev} for a in arms}},
        "verdict": verdict,
        "reads": ("TRUNCATION reproduces the device's negative norm bias on both legs and "
                  "NEAREST does not"
                  if all(verdict["truncation_reproduces_the_device"].values())
                  and all(verdict["nearest_does_not"].values())
                  else "the truncated-operand arm does NOT account for the device reading; "
                       "the miss is unexplained and belongs to the orchestrator"),
        "compute_kernel_config": "None, for every module on this arm: "
                                 "OF3DiffusionConditioning(_sub(dmsd,'diffusion_conditioning'),"
                                 " None), OF3DiffusionModule(dmsd, None), and "
                                 "RefAtomFeatureEmbedder(rafe_sd, None). Not changed here.",
        "bar_not_moved": "CHECK.json's function_norm_ratio_tol stays 1.0e-3 and assertion 4 "
                         "stays FAIL. This file explains the miss; it does not re-score it.",
        "host": os.uname().nodename,
        "card": "none -- host arithmetic",
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1, default=str)
    print(json.dumps(res, indent=1, default=str))
    print(f"wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
