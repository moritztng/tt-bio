"""Negative control: is layer_norm_z the ENTIRE difference between the two constructions?

Runs 0.5.0's DiffusionAttentionPairBias on a z that has been pre-normalised with the p2
checkpoint's own per-block layer_norm_z weight, using 0.5.0's own LayerNorm primitive.
If layer_norm_z is the whole difference, this reproduces 0.4.3's output exactly.
"""
import sys, json, torch
TREE = "/tmp/of3t/of3t-orchestrator/src/x0.5.0/openfold3-0.5.0"
sys.path.insert(0, TREE)
import openfold3
assert openfold3.__file__.startswith(TREE)
from openfold3.core.model.layers import attention_pair_bias as apb
from openfold3.core.model.primitives.normalization import LayerNorm

N = int(sys.argv[1])
torch.manual_seed(0)
DIMS = dict(c_q=768, c_k=768, c_v=768, c_s=384, c_z=128, c_hidden=48, no_heads=16)
mod = apb.DiffusionAttentionPairBias(**DIMS)

sd = torch.load("/home/moritz/.boltz/of3-p2-155k.pt", map_location="cpu", weights_only=False)
sd = sd.get("state_dict", sd)
pre = "diffusion_module.diffusion_transformer.blocks.0.attention_pair_bias."
blk = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
mod.load_state_dict(blk, strict=False)
mod = mod.double().eval()

ln = LayerNorm(c_in=128, create_offset=False)
ln.load_state_dict({"weight": blk["layer_norm_z.weight"]})
ln = ln.double().eval()
print("layer_norm_z eps =", ln.eps, " bias =", ln.bias)

g = torch.Generator().manual_seed(1234)
a = torch.randn(1, N, 768, generator=g, dtype=torch.float64)
z = torch.randn(1, N, N, 128, generator=g, dtype=torch.float64)
s = torch.randn(1, N, 384, generator=g, dtype=torch.float64)
mask = torch.ones(1, N, dtype=torch.float64)

with torch.no_grad():
    out = mod(a=a, z=ln(z), s=s, mask=mask)

ref = torch.load(f"/tmp/of3t/of3t-orchestrator/run/out_043_n{N}.pt")
raw = torch.load(f"/tmp/of3t/of3t-orchestrator/run/out_050_n{N}.pt")
ctl = float((out - ref).norm() / ref.norm())
unc = float((raw - ref).norm() / ref.norm())
print(f"N={N}  UNCONTROLLED 0.5.0 vs 0.4.3 : {unc:.6e}")
print(f"N={N}  CONTROLLED   0.5.0(prenormed z) vs 0.4.3 : {ctl:.6e}")
json.dump({"N": N, "uncontrolled": unc, "controlled": ctl},
          open(f"/tmp/of3t/of3t-orchestrator/run/apb_control_n{N}.json", "w"), indent=1)
