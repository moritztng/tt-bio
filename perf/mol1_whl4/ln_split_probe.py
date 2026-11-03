"""Device probe: Wormhole fp32 LayerNorm statistics, ttnn.mean as is vs a two-term split (bf16 head + fp32 tail),
on trunk block 43's single-Transition input (x0_dump.py). Graded vs float64: mean err / sigma, var rel err, and the
rel err of the LN output's token part (x - token mean). TT_VISIBLE_DEVICES=N python ln_split_probe.py X0.pt OUT.json"""
import json, sys
import torch
from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import ttnn
from tt_bio.tenstorrent import get_device, fp32_layer_norm
d = torch.load(sys.argv[1], weights_only=False)
xd, lw, lb = d["x"].double(), d["w"].double(), d["b"].double(); n = xd.shape[0]
mu = xd.mean(-1, keepdim=True); var = xd.var(-1, unbiased=False, keepdim=True)
ref = (xd - mu) / (var + 1e-5).sqrt() * lw + lb
dev = get_device()
ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                       fp32_dest_acc_en=True, packer_l1_acc=False)
up = lambda t, dt=ttnn.float32: ttnn.from_torch(t.float().reshape(1, -1, t.shape[-1]), layout=ttnn.TILE_LAYOUT,
                                                device=dev, dtype=dt)
back = lambda t: ttnn.to_torch(t).double().reshape(-1, t.shape[-1])[:n]
tok = lambda x: x - x.mean(0)
def g(o): return float((tok(o) - tok(ref)).norm() / tok(ref).norm())
def mean_plain(x): return ttnn.mean(x, dim=-1, keepdim=True, compute_kernel_config=ckc)
def mean_split(x):
    hi = ttnn.typecast(ttnn.typecast(x, ttnn.bfloat16), ttnn.float32)
    lo = ttnn.subtract(x, hi)
    return ttnn.add(mean_plain(hi), mean_plain(lo))
X = up(xd)
W, B = up(lw.reshape(1, -1)), up(lb.reshape(1, -1))
res = {}
for k, mean in (("plain", mean_plain), ("split", mean_split)):
    m = mean(X); xc = ttnn.subtract(X, m); v = mean(ttnn.multiply(xc, xc))
    o = ttnn.add(ttnn.multiply(ttnn.multiply(xc, ttnn.rsqrt(ttnn.add(v, 1e-5))), W), B)
    res[k] = {"mean_err_over_sigma": float(((back(m) - mu).abs() / var.sqrt()).max()),
              "var_relerr_max": float(((back(v) - var).abs() / var).max()), "ln_tok_relerr": g(back(o))}
res["fp32_layer_norm_today"] = {"ln_tok_relerr": g(back(fp32_layer_norm(
    X, ttnn.typecast(W, ttnn.bfloat16), ttnn.typecast(B, ttnn.bfloat16), 1e-5, ckc, ttnn.float32)))}
res["floor_fp32_ref"] = {"ln_tok_relerr": g(ref.float().double())}
for k, v_ in res.items():
    print(k, v_, flush=True)
json.dump(res, open(sys.argv[2], "w"), indent=1)
