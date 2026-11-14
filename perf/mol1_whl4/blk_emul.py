"""Which bf16 store in a trunk block's single track loses the per-token part, against float64 (upstream modules).

    python -I blk_emul.py CKPT UP_blkin.pt
UP_blkin.pt is blkin_dump.py's last-cycle (name, s_in, z_out, s_out). For every block: rel err of the output's
8-small-channel token part vs the float64 block, with ONE LayerNorm output rounded to bf16 at a time
(apb = AttentionPairBias pre-norm, tr = single Transition pre-norm), plain or centred on token 0's row
(round(c) + round(x - c): what `_centred_linear` hands each matmul).
"""
import sys, torch
from mol.model.modules.pairformer import PairformerBlock
torch.set_num_threads(6)
ckpt, blk = sys.argv[1:3]
sd = torch.load(ckpt, map_location="cpu", weights_only=False); sd = sd.get("model", sd)
recs = torch.load(blk, weights_only=False)
def rnd(x, mode):
    if mode == "plain":
        return x.to(torch.bfloat16).double()
    c = x[..., :1, :]
    return c.to(torch.bfloat16).double() + (x - c).to(torch.bfloat16).double()
def tok(x): return x - x.mean(0)
small = recs[-1][3].reshape(-1, recs[-1][3].shape[-1]).abs().amax(0) < 512
print("small channels", small.nonzero().flatten().tolist())
for i, (name, si, zo, so) in enumerate(recs):
    pre = next(p for p in ("module." + name + ".", name + ".") if any(k.startswith(p) for k in sd))
    w = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
    c_z = w["attention_pair_bias.layernorm_z.weight"].shape[0]
    b = PairformerBlock(n_heads=16, c_z=c_z, c_s=384, hidden_scale_up=True)
    b.load_state_dict(w, strict=True); b.eval().double()
    s, z = si.double(), zo.double()
    def run(site=None, mode=None):
        hooks = []
        if site:
            ln = b.attention_pair_bias.layernorm_a if site == "apb" else b.single_transition.layernorm1
            hooks.append(ln.register_forward_hook(lambda m, i, o: rnd(o, mode)))
        with torch.no_grad():
            x = s + b.attention_pair_bias(a=s, s=None, z=z)
            x = x + b.single_transition(x)
        for h in hooks: h.remove()
        return x.reshape(-1, x.shape[-1])
    ref = run()
    up = so.reshape(-1, so.shape[-1]).double()
    def e(x): return float((tok(x[:, small]) - tok(ref[:, small])).norm() / tok(ref[:, small]).norm())
    line = f"{i:2d} {name.split('.')[-1]:>3s} fp32-vs-64 {e(up):.2g}"
    for site in ("apb", "tr"):
        for mode in ("plain", "centred"):
            line += f" | {site}-{mode} {e(run(site, mode)):.3g}"
    print(line, flush=True)
