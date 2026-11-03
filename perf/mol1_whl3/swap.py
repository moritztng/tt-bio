"""Compare a TT dump's trunk outputs to upstream fp32's, and swap them one at a time into upstream's fp32 head.

    python -I swap.py CKPT TT_DUMP_DIR UP_TRUNK.pt
"""
import sys, torch
torch.set_num_threads(10)
from mol.model.modules.confidence import ConfidenceHead
ckpt, dump, up = sys.argv[1:4]
tt = torch.load(dump + "/inputs.pt", weights_only=False)
u = torch.load(up, weights_only=False)
u = {"s_inputs": u["s_inputs"].squeeze(0).float(), "s_trunk": u["s_trunk"].squeeze(0).float(),
     "z_trunk": u["z_trunk"].squeeze(0).float(), "coords": u["x_pred_coords"].reshape(-1, *tt["coords"].shape[-2:]).float()}
def rel(a, b): return float((a - b).norm() / b.norm())
for k in ("s_inputs", "s_trunk", "z_trunk"):
    a, b = tt[k], u[k]
    print(f"{k}: tt absmax {a.abs().max():.4g} up absmax {b.abs().max():.4g}  rel err {rel(a, b):.4g}")
s, us = tt["s_trunk"], u["s_trunk"]
small = us.abs() < 512
print(f"s_trunk: frac |up|<512 {small.float().mean():.4f}, frac |tt|<512 {(s.abs()<512).float().mean():.4f}; "
      f"clamped rel err {rel(s.clamp(-512,512), us.clamp(-512,512)):.4g}; sign agree {float((s.sign()==us.sign()).float().mean()):.4f}")
for nm, x in (("tt", s), ("up", us)):
    m = x.mean(0)
    print(f"{nm} s_trunk: shared-vector energy {float((m**2).sum()*x.shape[0]/(x**2).sum()):.5f}, "
          f"per-ch token std median {float(x.std(0).median()):.4g}, top ch {x.abs().mean(0).topk(5).indices.tolist()}")
cs = torch.nn.functional.cosine_similarity(s, us, dim=-1)
print("per-token cos(tt,up) min/median", float(cs.min()), float(cs.median()))
sd = torch.load(ckpt, map_location="cpu", weights_only=False); sd = sd.get("model", sd)
pre = next(p for p in ("module.confidence_head.", "confidence_head.") if any(k.startswith(p) for k in sd))
w = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
n_ta, c_s, _ = w["plddt_weight"].shape
head = ConfidenceHead(n_blocks=4, c_s=c_s, c_z=w["linear_no_bias_s1.weight"].shape[0],
                      c_s_inputs=w["linear_no_bias_s1.weight"].shape[1], max_atoms_per_token=n_ta, hidden_scale_up=True)
head.load_state_dict(w, strict=True); head.eval()
def run(src):
    with torch.no_grad():
        p, *_ = head(input_feature_dict=dict(tt["feats"]), s_inputs=src[0]["s_inputs"], s_trunk=src[1]["s_trunk"],
                     z_trunk=src[2]["z_trunk"], pair_mask=None, x_pred_coords=src[3]["coords"][:1])
    nb = p.shape[-1]
    return float((torch.softmax(p.double(), -1) @ ((torch.arange(nb, dtype=torch.float64) + 0.5) / nb)).mean())
for name, src in [("all TT", (tt, tt, tt, tt)), ("all upstream", (u, u, u, u)), ("TT, upstream s_trunk", (tt, u, tt, tt)),
                  ("TT, upstream z_trunk", (tt, tt, u, tt)), ("TT, upstream coords", (tt, tt, tt, u)),
                  ("TT, upstream s_inputs", (u, tt, tt, tt)), ("upstream, TT s_trunk", (u, tt, u, u))]:
    print(f"{name:24s} pLDDT {run(src):.4f}", flush=True)
