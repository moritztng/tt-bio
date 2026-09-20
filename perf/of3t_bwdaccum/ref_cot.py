#!/usr/bin/env python3
"""The reference's COTANGENT at every block boundary, not just its parameter gradients.

`of3t-trunkg043` showed the trunk's weight gradient is at upstream's own bf16 floor where the
backward starts and tens of times it where the backward ends. Two mechanisms produce that
profile and they are distinguishable by ONE quantity: the cotangent entering each block.

  (A) an error injected at each block's backward and carried down the chain -- the cotangent
      arriving at block k is already wrong, and the leaf gradient there is a symptom;
  (B) a correct cotangent with a wrong leaf backward -- the cotangent matches the reference at
      every rung and only the parameter reduction diverges.

This writes the reference side of that scan. `build()`, the dtype policy and the boundary
handling are `of3t-trunkg043/ref_grad.py`'s, imported rather than copied, so this differentiates
the same function that row published and no arm is rebuilt.

RUNG k is the cotangent ENTERING block k, i.e. dL/d(s_in of block k). Rung 0 is the stack's own
input gradient, rung 48 is the captured output cotangent itself and is identical by construction
-- it is written out anyway, because a scan whose first rung is not exactly the seed has a
harness bug rather than a finding.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "of3t_trunkg043"))

import ref_grad                                  # noqa: E402  (of3t-trunkg043's, unchanged)

CKPT = ref_grad.CKPT
PRE = ref_grad.PRE


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--boundary", required=True)
    ap.add_argument("--cap-last", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--policy", default="f64", choices=("f64", "f32", "bf16auto"))
    ap.add_argument("--blocks", type=int, default=48)
    ap.add_argument("--crop", type=int, default=64)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    t0 = time.perf_counter()

    sys.path.insert(0, a.tree)
    import openfold3
    if not openfold3.__file__.startswith(a.tree):
        raise SystemExit(f"wrong tree on sys.path: {openfold3.__file__} is not under {a.tree}")

    torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    dt = torch.float32 if a.policy in ("f32", "bf16auto") else torch.float64

    sd = torch.load(CKPT, map_location="cpu", weights_only=False, mmap=True)
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    mods, dims, load = ref_grad.build(sd, a.blocks, dt)
    if load["missing"] or load["unexpected"]:
        raise SystemExit(f"strict load did not hold: {load['missing'][:4]} "
                         f"{load['unexpected'][:4]}")

    b = torch.load(a.boundary, map_location="cpu", weights_only=False)
    s0 = b["s_in"].to(dt).contiguous()
    z0 = b["z_in"].to(dt).contiguous()
    sm = b["single_mask"].to(dt)
    pm = b["pair_mask"].to(dt)

    cap = torch.load(a.cap_last, map_location="cpu", weights_only=False)
    cot_s, cot_z = cap["cot"][0], cap["cot"][1]
    if cot_s is None or cot_z is None:
        raise SystemExit("captured cotangent missing -- the boundary is unusable")
    c = a.crop
    cot_s = cot_s.to(torch.float64)[:, :c].contiguous() if c else cot_s.to(torch.float64)
    cot_z = (cot_z.to(torch.float64)[:, :c, :c].contiguous() if c else cot_z.to(torch.float64))
    cs, cz = cot_s.to(dt), cot_z.to(dt)

    ctx = (torch.autocast("cpu", dtype=torch.bfloat16) if a.policy == "bf16auto"
           else torch.autocast("cpu", enabled=False))

    s_in = s0.detach().requires_grad_(True)
    z_in = z0.detach().requires_grad_(True)
    inter, act = [], []
    with ctx:
        s, z = s_in, z_in
        for i, m in enumerate(mods):
            act.append({"rung": i, "s_in_norm": float(s.detach().to(torch.float64).norm()),
                        "z_in_norm": float(z.detach().to(torch.float64).norm())})
            s, z = m(s, z, sm, pm)
            s.retain_grad()
            z.retain_grad()
            inter.append((s, z))
        act.append({"rung": a.blocks, "s_in_norm": float(s.detach().to(torch.float64).norm()),
                    "z_in_norm": float(z.detach().to(torch.float64).norm())})
        loss = (s.to(torch.float64) * cot_s).sum() + (z.to(torch.float64) * cot_z).sum()
    loss.backward()

    # rung k = the cotangent ENTERING block k. Rung `blocks` is the seed itself.
    cot = {}
    for k in range(a.blocks + 1):
        if k == 0:
            ds, dz = s_in.grad, z_in.grad
        else:
            ds, dz = inter[k - 1][0].grad, inter[k - 1][1].grad
        cot[k] = {"ds": None if ds is None else ds.detach().to(torch.float64).clone(),
                  "dz": None if dz is None else dz.detach().to(torch.float64).clone()}

    torch.save({"cot": cot, "policy": a.policy, "tree": a.tree, "blocks": a.blocks,
                "loss": float(loss), "activations": act,
                "seed_cot_s": cot_s, "seed_cot_z": cot_z}, a.out)

    rep = {"what": __doc__.strip().splitlines()[0], "tree": a.tree,
           "openfold3_file": openfold3.__file__, "policy": a.policy, "blocks": a.blocks,
           "crop": a.crop, "dims": dims, "load": load,
           "dtype_policy": {
               "f64": "every parameter and every activation float64; checkpoint upcast once at "
                      "load; no cast on the path",
               "f32": "parameters and activations float32, no autocast",
               "bf16auto": "float32 parameters under torch.autocast('cpu', bfloat16), which is "
                           "upstream's own training recipe",
           }[a.policy],
           "boundary": a.boundary, "boundary_sha256": ref_grad.sha256_file(a.boundary),
           "cotangent_from": a.cap_last,
           "cotangent_sha256": ref_grad.sha256_file(a.cap_last),
           "loss": float(loss),
           "rung_definition": "rung k is dL/d(input of block k); rung 0 is the stack input, "
                              "rung N is the captured seed itself",
           "activations": act,
           "cotangent_norms": {k: {"ds": (None if cot[k]["ds"] is None
                                          else float(cot[k]["ds"].norm())),
                                   "dz": (None if cot[k]["dz"] is None
                                          else float(cot[k]["dz"].norm()))}
                               for k in cot},
           "seconds": time.perf_counter() - t0, "out": a.out}
    with open(a.report, "w") as fh:
        json.dump(rep, fh, indent=2)
    print(json.dumps({"policy": a.policy, "loss": rep["loss"],
                      "ds_rung0": rep["cotangent_norms"][0]["ds"],
                      "ds_rung47": rep["cotangent_norms"][47]["ds"],
                      "seconds": round(rep["seconds"], 1)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
