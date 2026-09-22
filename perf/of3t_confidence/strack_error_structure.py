"""What SHAPE is the device s error? bf16 rounding of the same track costs pLDDT 4.8e-03
and the device costs 2.42e-01, so the device error is not bf16 noise and the difference is
either its size or its direction. This separates the two, per channel and per block."""
import os, pickle, sys, torch, torch.nn.functional as F, ttnn
sys.path.insert(0, os.path.expanduser("~/.coworker/wt/of3t-confidence/perf/of3t_confidence"))
from forward_vs_float64 import ca_walk, rel_l2
from tt_bio.tenstorrent import get_device
from tt_bio.openfold3_confidence import OF3ConfidenceHead

CK = os.path.expanduser("~/of3-weights/of3-p2-155k.pt")
sd = torch.load(CK, map_location="cpu", weights_only=False)
if hasattr(sd, "state_dict"): sd = sd.state_dict()
aux = {k[len("aux_heads."):]: v for k, v in sd.items() if k.startswith("aux_heads.")}
g = pickle.load(open(os.path.expanduser("~/of3_ref_out.pkl"), "rb"))["intermediates"]
si_input = g["input_embedder_real"]["out"][0].float()
si_trunk, zij_trunk = (t.float() for t in g["pairformer_stack_real"]["out"])
N = si_trunk.shape[0]
dev = get_device()
ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                             fp32_dest_acc_en=True, packer_l1_acc=True)
head = OF3ConfidenceHead(aux, dev, ckc)
kw = dict(si_input=si_input, si_trunk=si_trunk, zij_trunk=zij_trunk, repr_x_pred=ca_walk(N),
          max_atom_per_token_mask=torch.ones(N * 23), use_zij_trunk_embedding=True)
h = head.forward(**kw, s_path="host"); d = head.forward(**kw, s_path="device")
sh, sd_ = h["si_conf"].float(), d["si_conf"].float()
lh, ld = F.layer_norm(sh, (384,)), F.layer_norm(sd_, (384,))
e = ld - lh
print(f"LN(s) err relL2 {rel_l2(ld, lh):.3e}")
pc = e.norm(dim=0) / (lh.norm(dim=0) + 1e-30)
top = torch.argsort(pc, descending=True)[:8]
print("worst channels (per-channel relL2 of LN(s)):")
print("  " + "  ".join(f"c{int(i)}={float(pc[i]):.2e}" for i in top))
print(f"  median channel {float(pc.median()):.2e}   max {float(pc.max()):.2e}")
# how much of the pLDDT logit is the LN-bias term, i.e. is the logit a small difference?
w = aux["plddt.layer_norm.weight"].float()
b = aux.get("plddt.layer_norm.bias"); b = b.float() if b is not None else torch.zeros(384)
L = aux["plddt.linear.weight"].float()
var_term = F.linear(lh * w, L); bias_term = F.linear(b, L)
tot = var_term + bias_term
print(f"\npLDDT logit decomposition: ||LN-part|| {float(var_term.norm()):.4g}  "
      f"||bias-part|| {float(bias_term.norm())*N**0.5:.4g}  ||total|| {float(tot.norm()):.4g}")
print(f"  cancellation factor ||LN-part||/||total|| = {float(var_term.norm()/tot.norm()):.2f}")
# direction test: same-size RANDOM error vs the actual device error
rnd = torch.randn_like(e); rnd = rnd / rnd.norm() * e.norm()
pl = lambda x: F.linear(x * w + b, L)
print(f"\npLDDT relL2 from the ACTUAL device LN error : {rel_l2(pl(lh + e), pl(lh)):.3e}")
print(f"pLDDT relL2 from a RANDOM error of equal norm: {rel_l2(pl(lh + rnd), pl(lh)):.3e}")
# is the error a per-row scale (i.e. a bad rsqrt)?
alpha = (e * lh).sum(-1) / (lh * lh).sum(-1)
resid = e - alpha[:, None] * lh
print(f"\nper-row-scale component: alpha mean {float(alpha.mean()):+.3e} std {float(alpha.std()):.3e}; "
      f"residual after removing it {float(resid.norm()/e.norm()*100):.1f}% of the error")
