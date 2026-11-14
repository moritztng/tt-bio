import sys, torch
torch.set_num_threads(10)
from mol.model.modules.confidence import ConfidenceHead
ckpt, dump, up = sys.argv[1:4]
tt = torch.load(dump + "/inputs.pt", weights_only=False)
u = torch.load(up, weights_only=False)
us = u["s_trunk"].squeeze(0).float()
sd = torch.load(ckpt, map_location="cpu", weights_only=False); sd = sd.get("model", sd)
pre = next(p for p in ("module.confidence_head.", "confidence_head.") if any(k.startswith(p) for k in sd))
w = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
n_ta, c_s, _ = w["plddt_weight"].shape
head = ConfidenceHead(n_blocks=4, c_s=c_s, c_z=w["linear_no_bias_s1.weight"].shape[0],
                      c_s_inputs=w["linear_no_bias_s1.weight"].shape[1], max_atoms_per_token=n_ta, hidden_scale_up=True)
head.load_state_dict(w, strict=True); head.eval()
def run(s):
    with torch.no_grad():
        p, *_ = head(input_feature_dict=dict(tt["feats"]), s_inputs=tt["s_inputs"], s_trunk=s,
                     z_trunk=tt["z_trunk"], pair_mask=None, x_pred_coords=tt["coords"][:1])
    nb = p.shape[-1]
    return float((torch.softmax(p.double(), -1) @ ((torch.arange(nb, dtype=torch.float64) + 0.5) / nb)).mean())
print("up s fp32", run(us)); print("up s bf16-rounded", run(us.bfloat16().float()))
print("up s clamped first then bf16", run(us.clamp(-512,512).bfloat16().float()))
d = tt["s_trunk"].float() - us
print("up + WH error on small ch only", run(torch.where(us.abs().amax(0) < 512, tt["s_trunk"].float(), us)))
print("up + WH error on big ch only", run(torch.where(us.abs().amax(0) >= 512, tt["s_trunk"].float(), us)))
