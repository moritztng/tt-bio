"""Device probe: where the Wormhole fp32 LayerNorm variance loses precision, on trunk block 43's Transition input.
TT_VISIBLE_DEVICES=N python ln_var_probe.py X0.pt OUT.json  (X0.pt from x0_dump.py)
Mean by fp32 column-tile adds (exact in probe 4). Reports: the elementwise square's rel err vs float64 of the same
device xc, then var / LN token rel err for four ways to reduce the 32-column partial of xc^2 (probe 2: sum, cols, mm, halves)."""
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
def red_sum(t):      # current: bf16 head + fp32 tail through ttnn.sum
    hi = bf32(t); lo = ttnn.subtract(t, hi)
    return ttnn.add(ttnn.sum(hi, dim=-1, keepdim=True, compute_kernel_config=ckc),
                    ttnn.sum(lo, dim=-1, keepdim=True, compute_kernel_config=ckc))
def red_cols(t):     # 32 single-column slices added elementwise in fp32
    o = ttnn.slice(t, [0, 0, 0], [1, t.shape[1], 1])
    for c in range(1, t.shape[-1]):
        o = ttnn.add(o, ttnn.slice(t, [0, 0, c], [1, t.shape[1], c + 1]))
    return o
ONES = ttnn.from_torch(torch.ones(32, 32), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
def red_mm(t):       # bf16x3 split through a matmul with ones, fp32 out
    o = None; r = t
    for _ in range(3):
        h = ttnn.typecast(r, ttnn.bfloat16); r = ttnn.subtract(r, ttnn.typecast(h, ttnn.float32))
        p = ttnn.slice(ttnn.matmul(h, ONES, compute_kernel_config=ckc, dtype=ttnn.float32), [0, 0, 0], [1, t.shape[1], 1])
        o = p if o is None else ttnn.add(o, p)
    return o
def red_halves(t):   # pairwise halving by slices: 16, 8, 4, 2, 1 columns, each an fp32 add
    w = t.shape[-1]
    while w > 1:
        w //= 2
        t = ttnn.add(ttnn.slice(t, [0, 0, 0], [1, t.shape[1], w]), ttnn.slice(t, [0, 0, w], [1, t.shape[1], 2 * w]))
    return t
xc2 = ttnn.multiply(xc, xc)
P = ttnn.slice(xc2, [0, 0, 0], [1, xc2.shape[1], 32])
for c in range(32, xc2.shape[-1], 32):
    P = ttnn.add(P, ttnn.slice(xc2, [0, 0, c], [1, xc2.shape[1], c + 32]))
ph = back(P)
for k, red in (("sum", red_sum), ("cols", red_cols), ("mm", red_mm), ("halves", red_halves)):
    try:
        v = ttnn.multiply(red(P), 1.0 / 384)
        vh = back(v)[:, :1]
        res[f"red_{k}_vs_host_partial"] = float(((vh * 384 - ph.sum(-1, keepdim=True)).abs() / ph.sum(-1, keepdim=True)).max())
        res[f"var_{k}_relerr_max"] = float(((vh - var).abs() / var).max())
        o = ttnn.add(ttnn.multiply(ttnn.multiply(xc, ttnn.rsqrt(ttnn.add(v, 1e-5))), W), B)
        res[f"ln_{k}_tok_relerr"] = float((tok(back(o)) - tok(ref)).norm() / tok(ref).norm())
    except Exception as e:
        res[f"red_{k}_error"] = str(e)[:200]
res["partial_tiles_vs_f64"] = float(((ph.sum(-1, keepdim=True) / 384 - var).abs() / var).max())
for k, v_ in res.items():
    print(k, v_, flush=True)
json.dump(res, open(sys.argv[2], "w"), indent=1)
