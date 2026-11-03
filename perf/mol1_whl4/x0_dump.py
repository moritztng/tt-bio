"""Block 43's single-Transition input (s + APB(s), float64) and its LayerNorm weights from upstream's last-cycle block
inputs, for ln_split_probe.py.   python -I x0_dump.py CKPT UP_blkin.pt OUT.pt"""
import sys, torch
from mol.model.modules.pairformer import PairformerBlock
ckpt, blk, out = sys.argv[1:4]
sd = torch.load(ckpt, map_location="cpu", weights_only=False); sd = sd.get("model", sd)
name, si, zo, so = next(r for r in torch.load(blk, weights_only=False) if r[0].split(".")[-1] == "43")
pre = next(p for p in ("module." + name + ".", name + ".") if any(k.startswith(p) for k in sd))
w = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
b = PairformerBlock(n_heads=16, c_z=w["attention_pair_bias.layernorm_z.weight"].shape[0], c_s=384, hidden_scale_up=True)
b.load_state_dict(w, strict=True); b.eval().double()
with torch.no_grad():
    x0 = si.double() + b.attention_pair_bias(a=si.double(), s=None, z=zo.double())
ln = b.single_transition.layernorm1
torch.save({"x": x0.reshape(-1, 384), "w": ln.weight.detach().clone(), "b": ln.bias.detach().clone()}, out)
