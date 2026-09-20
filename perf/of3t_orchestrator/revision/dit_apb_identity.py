"""Function-identity test for the DiT attention-pair-bias block across upstream revisions.

Builds upstream's own diffusion AttentionPairBias from the SAME of3-p2-155k block-0 weights
under 0.4.3's construction and under 0.5.0's, in float64, and measures the separation.
0.4.3 applies a per-block layer_norm_z to the pair bias; 0.5.0's DiffusionAttentionPairBias
has no layer_norm_z at all.
"""
import os, sys, json, torch

TREE = sys.argv[1]          # extracted sdist root containing openfold3/
TAG  = sys.argv[2]          # "0.4.3" | "0.5.0"
N    = int(sys.argv[3])
OUT  = sys.argv[4]
sys.path.insert(0, TREE)

import openfold3
assert openfold3.__file__.startswith(TREE), openfold3.__file__
from openfold3.core.model.layers import attention_pair_bias as apb

torch.manual_seed(0)
DIMS = dict(c_q=768, c_k=768, c_v=768, c_s=384, c_z=128, c_hidden=48, no_heads=16)

if hasattr(apb, "DiffusionAttentionPairBias"):
    mod = apb.DiffusionAttentionPairBias(**DIMS); ctor = "DiffusionAttentionPairBias"
else:
    mod = apb.AttentionPairBias(use_ada_layer_norm=True, **DIMS); ctor = "AttentionPairBias(use_ada_layer_norm=True)"

sd = torch.load("/home/moritz/.boltz/of3-p2-155k.pt", map_location="cpu", weights_only=False)
sd = sd.get("state_dict", sd)
pre = "diffusion_module.diffusion_transformer.blocks.0.attention_pair_bias."
blk = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
missing, unexpected = mod.load_state_dict(blk, strict=False)

mod = mod.double().eval()
g = torch.Generator().manual_seed(1234)
a = torch.randn(1, N, 768, generator=g, dtype=torch.float64)
z = torch.randn(1, N, N, 128, generator=g, dtype=torch.float64)
s = torch.randn(1, N, 384, generator=g, dtype=torch.float64)
mask = torch.ones(1, N, dtype=torch.float64)

with torch.no_grad():
    out = mod(a=a, z=z, s=s, mask=mask)

torch.save(out, OUT)
json.dump({"tag": TAG, "ctor": ctor, "N": N,
           "missing_keys": sorted(missing), "unexpected_keys": sorted(unexpected),
           "out_absmax": float(out.abs().max()), "out_l2": float(out.norm())},
          open(OUT + ".json", "w"), indent=1)
print(f"[{TAG}] {ctor}  missing={sorted(missing)}  unexpected={sorted(unexpected)}  "
      f"|out|={float(out.norm()):.6f}")
