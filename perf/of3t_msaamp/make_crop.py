#!/usr/bin/env python3
"""Crop the msa_module boundary to its first N tokens (Addendum 1's discriminator).

stage 1: slice inputs, outputs and cotangent; keep the 384 capture's param_grads, so the f64 arm
         run on the crop scores itself against the 384 reference. That score IS test C0.
stage 2: replace outputs and param_grads with the f64 crop arm's own, so every other crop arm is
         scored in-frame.
"""
import argparse
from pathlib import Path

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--src", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--n", type=int, default=64)
ap.add_argument("--f64-dump", type=Path, default=None, help="stage 2: the f64 crop arm's --dump")
ap.add_argument("--f64-z", type=Path, default=None, help="stage 2: the f64 crop arm's z")
a = ap.parse_args()

B = torch.load(a.src, map_location="cpu", weights_only=False)
if a.f64_dump is None:
    n = a.n
    m, z = B["inputs"]["args"]
    kw = dict(B["inputs"]["kwargs"])
    kw["msa_mask"] = kw["msa_mask"][:, :, :n].clone()
    kw["pair_mask"] = kw["pair_mask"][:, :n, :n].clone()
    tot = float(B["cotangents"]["out"].norm())
    cot = B["cotangents"]["out"][:, :n, :n].clone()
    assert float(cot.norm()) == tot, "the crop would drop cotangent mass"
    C = {"inputs": {"args": (m[:, :, :n].clone(), z[:, :n, :n].clone()), "kwargs": kw},
         "outputs": B["outputs"][:, :n, :n].clone(), "cotangents": {"out": cot},
         "param_grads": B["param_grads"], "crop": {"n": n, "src": str(a.src), "stage": 1}}
else:
    D = torch.load(a.f64_dump, map_location="cpu", weights_only=False)
    assert D["policy"] == "f64"
    B["param_grads"] = D["grads"]
    B["outputs"] = torch.load(a.f64_z, map_location="cpu", weights_only=False)
    B["crop"]["stage"] = 2
    C = B
torch.save(C, a.out)
print("wrote", a.out, "stage", C["crop"]["stage"])
