"""Upstream Mol-1 fp32 ConfidenceHead on one TT dump's z_trunk/s_inputs/coords, with s_trunk taken from each source.

    python -I swap_s.py CKPT BASE_DUMP_DIR UP_TRUNK.pt NAME=DUMP_DIR ...
Prints mean pLDDT per sample (first 2 samples) for s_trunk = upstream and = every named TT dump.
"""
import sys, torch
torch.set_num_threads(6)
from mol.model.modules.confidence import ConfidenceHead
ckpt, base, up = sys.argv[1:4]
tt = torch.load(base + "/inputs.pt", weights_only=False)
u = torch.load(up, weights_only=False)
srcs = [("upstream", u["s_trunk"].squeeze(0).float())]
srcs += [(a.split("=")[0], torch.load(a.split("=")[1] + "/inputs.pt", weights_only=False)["s_trunk"].float()) for a in sys.argv[4:]]
sd = torch.load(ckpt, map_location="cpu", weights_only=False); sd = sd.get("model", sd)
pre = next(p for p in ("module.confidence_head.", "confidence_head.") if any(k.startswith(p) for k in sd))
w = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
n_ta, c_s, _ = w["plddt_weight"].shape
head = ConfidenceHead(n_blocks=4, c_s=c_s, c_z=w["linear_no_bias_s1.weight"].shape[0],
                      c_s_inputs=w["linear_no_bias_s1.weight"].shape[1], max_atoms_per_token=n_ta, hidden_scale_up=True)
head.load_state_dict(w, strict=True); head.eval()
us = srcs[0][1]; small = us.abs().amax(0) < 512
def tok(x): return x - x.mean(0)
for name, s in srcs:
    ps = []
    for k in range(2):
        with torch.no_grad():
            p, *_ = head(input_feature_dict=dict(tt["feats"]), s_inputs=tt["s_inputs"], s_trunk=s, z_trunk=tt["z_trunk"],
                         pair_mask=None, x_pred_coords=tt["coords"][k:k + 1])
        nb = p.shape[-1]
        ps.append(float((torch.softmax(p.double(), -1) @ ((torch.arange(nb, dtype=torch.float64) + 0.5) / nb)).mean()))
    d = tok(s[:, small].double()) - tok(us[:, small].double())
    print(f"s_trunk={name:10s} pLDDT {ps[0]:.4f} {ps[1]:.4f}  small-ch token std {tok(s[:, small]).std():.4g} "
          f"(upstream {tok(us[:, small]).std():.4g}) token-part rel err {float(d.norm() / tok(us[:, small].double()).norm()):.4g}",
          flush=True)
