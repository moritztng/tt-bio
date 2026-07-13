"""Host bisect: verify my z-embed + head math (fp32, no device) against the golden
pairformer input/outputs. Isolates z-embed/head-layout bugs from device pairformer
precision."""
import os, pickle, sys
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/ttuser/.coworker/wt/tt-bio-openfold3-p10-confidence-heads")
CKPT = os.path.expanduser("~/of3-weights/of3-p2-155k.pt")
GOLD = os.path.expanduser("~/of3_ref_out.pkl")

sd = torch.load(CKPT, map_location="cpu", weights_only=False)
if hasattr(sd, "state_dict"):
    sd = sd.state_dict()
aux = {k[len("aux_heads."):]: v for k, v in sd.items() if k.startswith("aux_heads.")}
g = pickle.load(open(GOLD, "rb"))["intermediates"]["confidence_heads_real"]

_MIN_BIN, _MAX_BIN, _NO_BIN, _INF = 3.25, 50.75, 39, 1e8
bins = torch.linspace(_MIN_BIN, _MAX_BIN, _NO_BIN, dtype=torch.float32)
sqb = bins ** 2
upper = torch.cat([sqb[1:], sqb.new_tensor([_INF])])


def pcc(a, b):
    a = a.flatten().double(); b = b.flatten().double()
    return float(((a - a.mean()) * (b - b.mean())).sum()
                 / ((a - a.mean()).norm() * (b - b.mean()).norm()))


gk = lambda k: aux[k].float()
si_input = g["si_input"].float()
si_trunk = g["si_trunk"].float()
zij_trunk = g["zij_trunk"].float()
repr_x = g["repr_x_pred"].float()
mask23 = g["max_atom_per_token_mask"].float()

# z-embed (my math)
z = (zij_trunk
     + F.linear(si_input, gk("pairformer_embedding.linear_i.weight")).unsqueeze(-2)
     + F.linear(si_input, gk("pairformer_embedding.linear_j.weight")).unsqueeze(-3))
dij = torch.sum((repr_x[..., None, :] - repr_x[..., None, :, :]) ** 2, dim=-1, keepdim=True)
oh = ((dij > sqb) & (dij < upper)).to(z.dtype)
z = z + F.linear(oh, gk("pairformer_embedding.linear_distance.weight"))

print("z-embed  vs zij_pf_in :", pcc(z, g["zij_pf_in"].float()))
print("si_trunk vs si_pf_in  :", pcc(si_trunk, g["si_pf_in"].float()))
print("z maxdiff vs pf_in    :", float((z - g["zij_pf_in"].float()).abs().max()))

# Heads using the GOLDEN si_conf/zij_conf (device-independent) -> must match golden
# head logits ~1.0 to prove head layout is correct.
s_single = g["si_conf"].float()
zf = g["zij_conf"].float()
dlog = F.linear(zij_trunk, gk("distogram.linear.weight"))
print("distogram (golden zf) :", pcc(dlog + dlog.transpose(-2, -3), g["distogram_logits"].float()))
pae = F.linear(F.layer_norm(zf, (128,)) * gk("pae.layer_norm.weight") + aux["pae.layer_norm.bias"].float(),
               gk("pae.linear.weight"))
print("pae (golden zf)       :", pcc(pae, g["pae_logits"].float()))
plog = F.linear(F.layer_norm(zf, (128,)) * gk("pde.layer_norm.weight") + aux["pde.layer_norm.bias"].float(),
                gk("pde.linear.weight"))
print("pde (golden zf)       :", pcc(plog + plog.transpose(-2, -3), g["pde_logits"].float()))


def atom_head(s, name, c_out):
    ln = F.layer_norm(s, (384,)) * gk(f"{name}.layer_norm.weight") + aux[f"{name}.layer_norm.bias"].float()
    lg = F.linear(ln, gk(f"{name}.linear.weight")).reshape(s.shape[0] * 23, c_out)
    return lg[mask23.bool()]


print("plddt (golden si_conf):", pcc(atom_head(s_single, "plddt", 50), g["plddt_logits"].float()))
print("exp_res (golden si_conf):", pcc(atom_head(s_single, "experimentally_resolved", 2),
                                       g["experimentally_resolved_logits"].float()))
