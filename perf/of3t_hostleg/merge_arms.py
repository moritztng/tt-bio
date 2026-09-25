#!/usr/bin/env python3
"""One arm out of the two boundaries the seventeen are read at.

`diffusion_module.atom_attn_enc`'s eight come from `perf/of3t_diffusion/device_gradient.py` at
the diffusion boundary, `input_embedder.atom_attn_enc`'s nine from `ie_arm.py` at the input
embedder's own two. They are different boundaries on the SAME batch, which is the shape
`of3t-wholemodel/model_scope.py` already accepts for its `cond`/`aux`/`msa` scopes, so they
union. An overlapping key would mean two scopes claiming one parameter and is a STOP.

    merge_arms.py OUT.pt IN1.pt IN2.pt ...
"""
import sys
import torch

out, ins = sys.argv[1], sys.argv[2:]
merged, owner = {}, {}
for p in ins:
    d = torch.load(p, map_location="cpu", weights_only=False)
    clash = sorted(set(d) & set(merged))
    if clash:
        raise SystemExit(f"STOP: {p} and {owner[clash[0]]} both carry {clash[0]!r} "
                         f"({len(clash)} overlap). Two scopes cannot own one parameter.")
    for k, v in d.items():
        if v is None:
            continue
        merged[k] = v
        owner[k] = p
    print(f"  +{len(d)} from {p} -> {len(merged)}", flush=True)
    del d
torch.save(merged, out)
print(f"wrote {len(merged)} tensors to {out}")
