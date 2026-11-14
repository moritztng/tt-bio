"""Device micro-test 2: the block-43 LayerNorm composed from fp32 ttnn primitives, to name the step that loses the
per-token scale on Wormhole. TT_VISIBLE_DEVICES=N python ln_probe2.py TAP.pt LNW.pt OUT.json"""
import json, sys
import torch
from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import ttnn
from tt_bio.tenstorrent import get_device
tap = torch.load(sys.argv[1], weights_only=False); lnw = torch.load(sys.argv[2], weights_only=False)
x = tap["in"][0].float().reshape(1, 1, -1, 384)
lw, lb = lnw["w"].double(), lnw["b"].double()
xd = x.double().reshape(-1, 384); n = xd.shape[0]
mu = xd.mean(-1, keepdim=True); var = xd.var(-1, unbiased=False, keepdim=True)
xh = (xd - mu) / (var + 1e-5).sqrt(); ref = xh * lw + lb
dev = get_device()
Cfg = ttnn.types.WormholeComputeKernelConfig if dev.arch() == ttnn.Arch.WORMHOLE_B0 else ttnn.types.BlackholeComputeKernelConfig
ckc = Cfg(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False)
up = lambda t: ttnn.from_torch(t.float(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.float32)
back = lambda t: ttnn.to_torch(t).double().reshape(-1, t.shape[-1])[:n]
def grade(o):
    y = (o - lb) / lw
    A = torch.stack([xh, torch.ones_like(xh)], -1)
    a = torch.linalg.lstsq(A, y.unsqueeze(-1)).solution.squeeze(-1)[:, 0]
    d = o - ref
    return {"scale_std": float(a.std()), "tok_err_rms": float((d - d.mean(0)).pow(2).mean().sqrt())}
res = {"floor_bf16_ref": grade(ref.to(torch.bfloat16).double())}
X = up(x)
m = ttnn.mean(X, dim=-1, keepdim=True, compute_kernel_config=ckc)
res["mean_relerr"] = float(((back(m) - mu).abs() / var.sqrt()).max())
xc = ttnn.subtract(X, m)
v = ttnn.mean(ttnn.multiply(xc, xc), dim=-1, keepdim=True, compute_kernel_config=ckc)
res["var_relerr_max"] = float(((back(v) - var).abs() / var).max())
ve = ttnn.add(v, 1e-5)
invs = {"rsqrt": ttnn.rsqrt(ve), "rsqrt_fast": ttnn.rsqrt(ve, fast_and_approximate_mode=True) if True else None,
        "recip_sqrt": ttnn.reciprocal(ttnn.sqrt(ve)), "host_inv": up(1 / (back(v) + 1e-5).sqrt())}
for k, inv in invs.items():
    res[f"inv_{k}_relerr_std"] = float((back(inv) * (var + 1e-5).sqrt() - 1).std())
    o = ttnn.add(ttnn.multiply(ttnn.multiply(xc, inv), up(lw.reshape(1, 1, 1, -1))), up(lb.reshape(1, 1, 1, -1)))
    res[k] = grade(ttnn.to_torch(ttnn.typecast(o, ttnn.bfloat16)).double().reshape(-1, 384)[:n])
for k, v_ in res.items():
    print(k, v_, flush=True)
json.dump(res, open(sys.argv[3], "w"), indent=1)
