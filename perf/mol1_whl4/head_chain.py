"""Which bf16 store the confidence Pairformer's chain amplifies: float64 4-block chain from the TT head's block-0
input, with ONE kind of bf16 rounding switched on in every block, graded by the pLDDT it gives (upstream plddt head).

    python -I head_chain.py CKPT DUMP_DIR [site ...]
Sites: s_in (s at each block entry), z (z at each block entry), apb_ln (APB pre-norm output), apb_out (APB update),
tr_ln (single Transition pre-norm output), tr_out (single Transition update), tt (the TT stack's own output).
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from mol.model.modules.pairformer import PairformerBlock

torch.set_num_threads(8)
ckpt, d = sys.argv[1], Path(sys.argv[2])
sites = sys.argv[3:] or ["none", "tt", "s_in", "z", "apb_ln", "apb_out", "tr_ln", "tr_out"]
sd = torch.load(ckpt, map_location="cpu", weights_only=False); sd = sd.get("model", sd)
pre = next(p for p in ("module.confidence_head.", "confidence_head.") if any(k.startswith(p) for k in sd))
recs = torch.load(d / "headblk.pt", weights_only=False)
inp = torch.load(d / "inputs.pt", weights_only=False)
N = inp["s_trunk"].shape[0]
W = lambda k: sd[pre + k].double()
a2t, a2ta = inp["feats"]["atom_to_token_idx"].long(), inp["feats"]["atom_to_tokatom_idx"].long()
bf = lambda x: x.to(torch.bfloat16).to(x.dtype)


def plddt(s):
    aln = F.layer_norm(s[a2t], (384,)) * W("plddt_ln.weight") + (W("plddt_ln.bias") if pre + "plddt_ln.bias" in sd else 0)
    lg = torch.einsum("nc,ncb->nb", aln, W("plddt_weight")[a2ta])
    nb = lg.shape[-1]
    return float((torch.softmax(lg, -1) @ ((torch.arange(nb, dtype=torch.float64) + 0.5) / nb)).mean())


blocks = []
for i in range(len(recs)):
    bp = f"{pre}pairformer_stack.blocks.{i}."
    w = {k[len(bp):]: v for k, v in sd.items() if k.startswith(bp)}
    b = PairformerBlock(n_heads=16, c_z=w["attention_pair_bias.layernorm_z.weight"].shape[0], c_s=384, hidden_scale_up=True)
    b.load_state_dict(w, strict=True)
    blocks.append(b.eval().double())
s0 = recs[0][0].reshape(-1, recs[0][0].shape[-1])[:N].double()[None]
z0 = recs[0][1].reshape(-1, *recs[0][1].shape[-3:])[:, :N, :N].double()
tt = recs[-1][2].reshape(-1, recs[-1][2].shape[-1])[:N].double()


def chain(site):
    hooks = []
    for b in blocks:
        mods = {"apb_ln": b.attention_pair_bias.layernorm_a, "apb_out": b.attention_pair_bias,
                "tr_ln": b.single_transition.layernorm1, "tr_out": b.single_transition}
        if site in mods:
            hooks.append(mods[site].register_forward_hook(lambda m, i, o: bf(o)))
    s, z = s0, z0
    with torch.no_grad():
        for b in blocks:
            if site == "s_in":
                s = bf(s)
            if site == "z":
                z = bf(z)
            s, z = b(s, z, None)
    for h in hooks:
        h.remove()
    return s.reshape(-1, 384)


ref = chain("none")
for site in sites:
    s = tt if site == "tt" else (ref if site == "none" else chain(site))
    print(f"{site:8s} plddt {plddt(s):.4f}  s rel err vs float64 chain {float((s - ref).norm() / ref.norm()):.4g}", flush=True)
