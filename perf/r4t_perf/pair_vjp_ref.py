"""P2 step 0: the pair builder's analytic split-form VJP in float64 vs upstream autograd (host only, no card)."""
import sys, os, torch
sys.path.insert(0, os.path.join(os.environ["WT"], "perf", "r4t"))
from loss_parity_reference import load_batch
from tt_bio.rfd4 import network as N, engine as E
torch.manual_seed(0)
state, config = N.load_checkpoint(E._resolve_weights(E.DEFAULT_WEIGHTS))
glue = N.FlowGlue(state, config).double()
pb = glue.pair_repr_builder
for _m in (pb.init_repr_factory.linear_out, pb.cond_factory.linear_out):
    _m.register_forward_pre_hook(lambda m, inp: tuple(i.double() for i in inp))
fac = pb.init_repr_factory
print("pair builder params:", [(n, tuple(p.shape)) for n, p in pb.named_parameters()])
bt, meta, *_ = load_batch(sys.argv[1])
def take(d):
    if isinstance(d, dict): return {k: take(v) for k, v in d.items()}
    if torch.is_tensor(d):
        d = d[0:1] if d.dim() and d.shape[0] == bt["padding_mask"].shape[0] else d
        return d.double() if d.is_floating_point() else d
    return d
b = take(bt)
pb.requires_grad_(True)
z = pb(b)
G = torch.randn_like(z) * b["padding_mask"][0][:, None, None] * b["padding_mask"][0][None, :, None]
(z * G).sum().backward()
ref = {n: p.grad.clone() for n, p in pb.named_parameters() if p.grad is not None}
# analytic: features F, x = F W^T, n1 = x r1, a = n1 w, b = a r2, z = (b gamma + beta) * mask
with torch.no_grad():
    F = torch.cat([c(b).double() if c(b).ndim == 4 else c(b).double().reshape(1,1,1,-1).expand(1,z.shape[1],z.shape[1],-1) for c in fac.feat_creators], -1)
    W = fac.linear_out.weight; w = fac.ln_out.weight; e1 = fac.ln_out.eps; ada = pb.adaln; e2 = ada.norm.eps
    x = F @ W.t(); C = x.shape[-1]
    r1 = (x.pow(2).mean(-1, keepdim=True) + e1).rsqrt(); n1 = x * r1; a = n1 * w
    r2 = (a.pow(2).mean(-1, keepdim=True) + e2).rsqrt(); bb = a * r2
    nc = ada.norm_cond(pb.cond_factory(b)); gamma = ada.to_gamma(nc); beta = ada.to_beta(nc)
    m = (b["padding_mask"][0][:, None] * b["padding_mask"][0][None, :])[None, ..., None]
    Gm = G * m
    zz = (bb * gamma[:, None, None] + beta[:, None, None]) * m
    print("fwd max|z-zz|", (z - zz).abs().max().item())
    dbeta = Gm.sum((0, 1, 2)); dgamma = (Gm * bb).sum((0, 1, 2))
    db = Gm * gamma[:, None, None]
    da = r2 * (db - bb * (db * bb).mean(-1, keepdim=True))
    dw = (da * n1).sum((0, 1, 2))
    dn1 = da * w
    dx = r1 * (dn1 - n1 * (dn1 * n1).mean(-1, keepdim=True)) if not os.environ.get("BREAK") else r1 * dn1   # broken arm: no RMSNorm centring term
    dW = dx.reshape(-1, C).t() @ F.reshape(-1, F.shape[-1])
# dgamma/dbeta through the host cond graph
gamma_g = ada.to_gamma(ada.norm_cond(pb.cond_factory(b))); beta_g = ada.to_beta(ada.norm_cond(pb.cond_factory(b)))
for p in pb.parameters(): p.grad = None
torch.autograd.backward([gamma_g, beta_g], [dgamma[None], dbeta[None]])
w_grads = {"init_repr_factory.linear_out.weight": dW, "init_repr_factory.ln_out.weight": dw}
worst = 0
for n, g in ref.items():
    mine = w_grads.get(n)
    if mine is None:
        p = dict(pb.named_parameters())[n]; mine = p.grad
    rel = ((mine - g).norm() / g.norm().clamp_min(1e-300)).item() if mine is not None else float("inf")
    worst = max(worst, rel); print(f"{n:55s} rel {rel:.2e}")
print("WORST", worst, "PASS" if worst < 1e-9 else "FAIL")
