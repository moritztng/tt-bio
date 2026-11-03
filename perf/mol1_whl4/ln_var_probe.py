"""Device probe: where the Wormhole fp32 LayerNorm variance loses precision, on trunk block 43's Transition input.
TT_VISIBLE_DEVICES=N python ln_var_probe.py X0.pt OUT.json  (X0.pt from x0_dump.py)
Mean by fp32 column-tile adds (exact in probe 4). Reports: the elementwise square's rel err vs float64 of the same
device xc, then var / LN token rel err for squares taken plainly or split (h = bf16(xc): h*h + 2*h*l + l*l)."""
import json, sys
import torch
from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import ttnn
from tt_bio.tenstorrent import get_device
d = torch.load(sys.argv[1], weights_only=False)
xd, lw, lb = d["x"].double(), d["w"].double(), d["b"].double(); n = xd.shape[0]
mu = xd.mean(-1, keepdim=True); var = xd.var(-1, unbiased=False, keepdim=True)
ref = (xd - mu) / (var + 1e-5).sqrt() * lw + lb
dev = get_device()
ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                       fp32_dest_acc_en=True, packer_l1_acc=False)
up = lambda t: ttnn.from_torch(t.float().reshape(1, -1, t.shape[-1]), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.float32)
back = lambda t: ttnn.to_torch(t).double().reshape(-1, t.shape[-1])[:n]
tok = lambda v: v - v.mean(0)
bf32 = lambda t: ttnn.typecast(ttnn.typecast(t, ttnn.bfloat16), ttnn.float32)
def tiles_sum(x):
    t = ttnn.slice(x, [0, 0, 0], [1, x.shape[1], 32])
    for c in range(32, x.shape[-1], 32):
        t = ttnn.add(t, ttnn.slice(x, [0, 0, c], [1, x.shape[1], c + 32]))
    hi = bf32(t); lo = ttnn.subtract(t, hi)
    s = ttnn.add(ttnn.sum(hi, dim=-1, keepdim=True, compute_kernel_config=ckc),
                 ttnn.sum(lo, dim=-1, keepdim=True, compute_kernel_config=ckc))
    return s
mean = lambda x: ttnn.multiply(tiles_sum(x), 1.0 / x.shape[-1])
W, B = up(lw.reshape(1, -1)), up(lb.reshape(1, -1))
X = up(xd)
m = mean(X); xc = ttnn.subtract(X, m)
xch = back(xc)
res = {"mean_err_over_sigma": float(((back(m) - mu).abs() / var.sqrt()).max()),
       "xc_relerr_vs_f64": float(((xch - (xd - mu)).norm() / (xd - mu).norm()))}
def sq_plain(v): return ttnn.multiply(v, v)
def sq_split(v):
    h = bf32(v); l = ttnn.subtract(v, h)
    return ttnn.add(ttnn.add(ttnn.multiply(h, h), ttnn.multiply(ttnn.multiply(h, l), 2.0)), ttnn.multiply(l, l))
def sq_split_bf(v):
    # same split, the products taken on bf16 operands (exact in any source width), accumulated fp32
    h = ttnn.typecast(v, ttnn.bfloat16); l = ttnn.subtract(v, ttnn.typecast(h, ttnn.float32))
    lh = ttnn.typecast(l, ttnn.bfloat16); ll = ttnn.subtract(l, ttnn.typecast(lh, ttnn.float32))
    f = lambda a, b: ttnn.multiply(a, b, dtype=ttnn.float32)
    out = ttnn.add(f(h, h), ttnn.multiply(f(h, lh), 2.0))
    out = ttnn.add(out, ttnn.multiply(ttnn.multiply(ttnn.typecast(h, ttnn.float32), ll), 2.0))
    return out
for k, sq in (("plain", sq_plain), ("split", sq_split), ("split_bf", sq_split_bf)):
    s2 = sq(xc)
    res[f"sq_{k}_relerr"] = float(((back(s2) - xch ** 2).abs() / (xch ** 2 + 1e-30)).max())
    v = mean(s2)
    res[f"var_{k}_relerr_max"] = float(((back(v) - var).abs() / var).max())
    o = ttnn.add(ttnn.multiply(ttnn.multiply(xc, ttnn.rsqrt(ttnn.add(v, 1e-5))), W), B)
    res[f"ln_{k}_tok_relerr"] = float((tok(back(o)) - tok(ref)).norm() / tok(ref).norm())
    hv = (xch ** 2).mean(-1, keepdim=True)
    o2 = ttnn.add(ttnn.multiply(ttnn.multiply(xc, up(1 / (hv + 1e-5).sqrt())), W), B)
res["ln_hostvar_tok_relerr"] = float((tok(back(o2)) - tok(ref)).norm() / tok(ref).norm())
res["rsqrt_relerr_max"] = float(((back(ttnn.rsqrt(up(var))) * var.sqrt() - 1).abs()).max())
for k, v_ in res.items():
    print(k, v_, flush=True)
json.dump(res, open(sys.argv[2], "w"), indent=1)
