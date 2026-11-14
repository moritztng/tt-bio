"""swap2.py plus upstream s_trunk rounded to bf16 (erasure test), z=upstream only. Upstream Mol-1 fp32 ConfidenceHead on one TT dump's coords/feats, s_trunk x z_trunk from every source pair.

    python -I swap2.py CKPT BASE_DUMP_DIR UP_TRUNK.pt NAME=DUMP_DIR ...
Prints how far each TT source's s_inputs / s_trunk / z_trunk sit from upstream, then mean pLDDT (samples 0, 1) per (s, z).
"""
import sys, torch
torch.set_num_threads(6)
from mol.model.modules.confidence import ConfidenceHead
ckpt, base, up = sys.argv[1:4]
tt = torch.load(base + "/inputs.pt", weights_only=False)
u = torch.load(up, weights_only=False)
src = {"upstream": {k: u[k].squeeze(0).float() for k in ("s_inputs", "s_trunk", "z_trunk")}}
src["up_bf16"] = {k: (v.bfloat16().float() if k == "s_trunk" else v) for k, v in src["upstream"].items()}
for a in sys.argv[4:]:
    d = torch.load(a.split("=")[1] + "/inputs.pt", weights_only=False)
    src[a.split("=")[0]] = {k: d[k].float() for k in ("s_inputs", "s_trunk", "z_trunk")}
def rel(a, b): return float((a.double() - b.double()).norm() / b.double().norm())
small = src["upstream"]["s_trunk"].abs().amax(0) < 512
def tok(x): return x - x.mean(0)
for n, d in src.items():
    print(f"{n:10s} vs upstream: s_inputs {rel(d['s_inputs'], src['upstream']['s_inputs']):.3g} "
          f"s_trunk {rel(d['s_trunk'], src['upstream']['s_trunk']):.3g} "
          f"s small-ch token {rel(tok(d['s_trunk'][:, small]), tok(src['upstream']['s_trunk'][:, small])):.3g} "
          f"z_trunk {rel(d['z_trunk'], src['upstream']['z_trunk']):.3g}", flush=True)
sd = torch.load(ckpt, map_location="cpu", weights_only=False); sd = sd.get("model", sd)
pre = next(p for p in ("module.confidence_head.", "confidence_head.") if any(k.startswith(p) for k in sd))
w = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
n_ta, c_s, _ = w["plddt_weight"].shape
head = ConfidenceHead(n_blocks=4, c_s=c_s, c_z=w["linear_no_bias_s1.weight"].shape[0],
                      c_s_inputs=w["linear_no_bias_s1.weight"].shape[1], max_atoms_per_token=n_ta, hidden_scale_up=True)
head.load_state_dict(w, strict=True); head.eval()
s_in = src["upstream"]["s_inputs"]
for zn in ["upstream"]:
    for sn in src:
        ps = []
        for k in range(2):
            with torch.no_grad():
                p, *_ = head(input_feature_dict=dict(tt["feats"]), s_inputs=s_in, s_trunk=src[sn]["s_trunk"],
                             z_trunk=src[zn]["z_trunk"], pair_mask=None, x_pred_coords=tt["coords"][k:k + 1])
            nb = p.shape[-1]
            ps.append(float((torch.softmax(p.double(), -1) @ ((torch.arange(nb, dtype=torch.float64) + 0.5) / nb)).mean()))
        print(f"z={zn:10s} s={sn:10s} pLDDT {ps[0]:.4f} {ps[1]:.4f}", flush=True)
