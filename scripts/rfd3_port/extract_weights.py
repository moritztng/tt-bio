"""Extract real weights for the RFD3 submodules we port, from an RFD3 checkpoint.

    python extract_weights.py /root/work/ckpt/rfd3_latest.ckpt /root/work/capture

Saves, under out_dir:
  token_initializer.real_weights.pt   (model.token_initializer.*  , prefix stripped)
  diffusion_module.real_weights.pt    (model.diffusion_module.*   , prefix stripped)
  *.real_weights.meta.json            (key->shape/dtype + key list)
"""
import json
import os
import sys

import torch

ckpt_path, out_dir = sys.argv[1], sys.argv[2]
os.makedirs(out_dir, exist_ok=True)

ck = torch.load(ckpt_path, map_location="cpu", mmap=True, weights_only=False)
model_sd = ck["model"]


def extract(submodule):
    for prefix in (f"{submodule}.", f"model.{submodule}."):
        sub = {k[len(prefix):]: v for k, v in model_sd.items() if k.startswith(prefix)}
        if sub:
            return sub, prefix
    return {}, None


for sub in ("token_initializer", "diffusion_module"):
    weights, prefix = extract(sub)
    out_path = os.path.join(out_dir, f"{sub}.real_weights.pt")
    torch.save({k: v.detach().cpu() for k, v in weights.items()}, out_path)
    meta = {"n_keys": len(weights), "prefix": prefix,
            "keys": sorted(weights.keys()),
            "shapes": {k: [list(v.shape), str(v.dtype)] for k, v in weights.items()}}
    with open(os.path.join(out_dir, f"{sub}.real_weights.meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[extract] {sub}: {len(weights)} tensors (prefix '{prefix}') -> {out_path}", flush=True)
