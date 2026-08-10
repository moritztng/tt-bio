#!/usr/bin/env python3
"""Weight bytes the fold cannot avoid reading, counted off the real checkpoint."""
import collections
import json
import re
import sys

import torch

sd = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
for k in ("model", "state_dict", "ema", "module"):
    if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
        sd = sd[k]
groups = collections.defaultdict(int)
pf_block = collections.defaultdict(int)
total = 0
for k, v in sd.items():
    if not torch.is_tensor(v):
        continue
    n = v.numel()
    total += n
    m = re.search(r"pairformer_stack\.(?:blocks\.)?(\d+)\.", k)
    if m:
        groups["pairformer_stack"] += n
        if m.group(1) == "0":
            pf_block[k.split(".", 3)[-1].rsplit(".", 1)[0]] += n
    elif "msa_module" in k:
        groups["msa_module"] += n
    elif "diffusion" in k or "denoise" in k:
        groups["diffusion"] += n
    elif "confidence" in k:
        groups["confidence"] += n
    else:
        groups["other"] += n
print(f"total params {total/1e6:.2f} M  -> {total*2/1e6:.1f} MB bf16")
for g, n in sorted(groups.items(), key=lambda x: -x[1]):
    print(f"  {g:20s} {n/1e6:8.2f} M  {n*2/1e6:8.2f} MB")
nblk = len({re.search(r"pairformer_stack\.(?:blocks\.)?(\d+)\.", k).group(1) for k in sd if torch.is_tensor(sd[k]) and re.search(r"pairformer_stack\.(?:blocks\.)?(\d+)\.", k)})
print(f"pairformer blocks found: {nblk}")
if nblk:
    print(f"  per block: {groups['pairformer_stack']/nblk/1e6:.3f} M = {groups['pairformer_stack']/nblk*2/1e6:.3f} MB bf16")
print("block-0 submodules:")
for k, n in sorted(pf_block.items(), key=lambda x: -x[1])[:14]:
    print(f"    {k:44s} {n/1e6:7.3f} M")
json.dump({"total_params": total, "groups": dict(groups), "n_blocks": nblk}, open(sys.argv[2], "w"), indent=2)
