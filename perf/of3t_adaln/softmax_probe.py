#!/usr/bin/env python3
"""Two things the sandwich arm cannot conclude without: does the ablation harness bite, and
where does the error actually enter?"""
import os, sys, json, time
sys.path.insert(0, os.getcwd())
import torch, ttnn
from tt_bio import autograd as ag
from tt_bio.autograd import precise_config
from tt_bio.tenstorrent import get_device
from tt_bio import taped_ttnn as TT
from tt_bio.taped_ttnn import taped_ttnn

tt = taped_ttnn(); dev = get_device()
H, N = 16, 384
def rel(x, y):
    x, y = x.reshape(-1).double(), y.reshape(-1).double()
    return float((x - y).norm() / (y.norm() + 1e-300))

gen = torch.Generator().manual_seed(20260920)
sc = torch.randn(1, H, N, N, generator=gen).bfloat16().float()
y64 = torch.softmax(sc.double(), dim=-1)

print("== FORWARD: ttnn.softmax against torch float64 on identical values ==")
rows = []
for dt, name in ((ttnn.bfloat16, "bf16"), (ttnn.float32, "fp32")):
    d = ttnn.from_torch(sc, layout=ttnn.TILE_LAYOUT, device=dev, dtype=dt)
    for ns in (True, False):
        for ckc, cname in ((None, "default"), (precise_config(), "precise")):
            kw = dict(dim=-1, numeric_stable=ns)
            if ckc is not None:
                kw["compute_kernel_config"] = ckc
            y = ttnn.softmax(d, **kw)
            r = rel(ttnn.to_torch(y), y64)
            rows.append({"in_dtype": name, "numeric_stable": ns, "kernel_config": cname,
                         "forward_rel": r})
            print(f"  in {name}  numeric_stable={ns!s:5s}  cfg {cname:8s}  rel {r:.6e}")
print("  torch float32 host floor:", rel(torch.softmax(sc, dim=-1), y64))

print()
print("== the ablation harness: does replacing _VERBS['softmax'] bite? ==")
shipped = TT._VERBS["softmax"]

def broken(shipped_fn, args, kwargs):
    """Negative control: drop the `inner` term entirely. dx = y*g instead of y*(g - inner).
    If the harness installs, this MUST move the number; if it does not move, the harness is
    what the sandwich arm measured, not the rule."""
    x = TT._wrap(args[0]); dim = kwargs.get("dim", -1)
    ra, rk = TT._raw(args, kwargs)
    y0 = shipped_fn(*ra, **rk); box = [y0]
    def make():
        def bw(g):
            x.add_grad(ttnn.multiply(box[0], g))
        return bw
    out = TT._tape(y0, [x], make)
    if out.node is not None: out.box = box
    return out

def run(rule, cancel=0.0137):
    g_gen = torch.Generator().manual_seed(7)
    row = torch.randn(1, H, N, 1, generator=g_gen)
    g_in = (row.expand(1, H, N, N) + cancel * torch.randn(1, H, N, N, generator=g_gen)
            ).bfloat16().float()
    x64 = sc.double().requires_grad_(True)
    yy = torch.softmax(x64, dim=-1); yy.backward(g_in.double())
    d64 = x64.grad.detach()
    TT._VERBS["softmax"] = rule
    try:
        ag.forget_parameters()
        sd = ttnn.from_torch(sc, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        gd = ttnn.from_torch(g_in, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        with ag.tape():
            X = ag.Tensor(sd, requires_grad=True)
            s32 = tt.typecast(X, ttnn.float32)
            y = tt.softmax(s32, dim=-1, numeric_stable=True)
            attn = tt.typecast(y, ttnn.bfloat16)
            ag.backward([attn], [gd])
        dd = ttnn.to_torch(X.grad).double()
    finally:
        TT._VERBS["softmax"] = shipped
    return rel(dd, d64), float(dd.norm()) / float(d64.norm())

r_ship, ratio_ship = run(shipped)
r_broken, ratio_broken = run(broken)
print(f"  shipped rule: rel {r_ship:.6e}  r {ratio_ship:.5f}")
print(f"  broken  rule: rel {r_broken:.6e}  r {ratio_broken:.5f}")
print(f"  harness bites: {r_broken != r_ship}")

print()
print("== where the backward error enters: inner computed on device vs float64 ==")
g_gen = torch.Generator().manual_seed(7)
row = torch.randn(1, H, N, 1, generator=g_gen)
g_in = (row.expand(1, H, N, N) + 0.0137 * torch.randn(1, H, N, N, generator=g_gen)).bfloat16().float()
sd = ttnn.from_torch(sc, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.float32)
y_dev = ttnn.softmax(sd, dim=-1, numeric_stable=True)
gd = ttnn.from_torch(g_in, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.float32)
inner_dev = ttnn.to_torch(ttnn.sum(ttnn.multiply(gd, y_dev), dim=-1, keepdim=True)).double()
inner_pre = ttnn.to_torch(ttnn.sum(ttnn.multiply(gd, y_dev), dim=-1, keepdim=True,
                                   compute_kernel_config=precise_config())).double()
y_dev_t = ttnn.to_torch(y_dev).double()
inner64 = (g_in.double() * y64).sum(-1, keepdim=True)
inner_hybrid = (g_in.double() * y_dev_t).sum(-1, keepdim=True)
print(f"  y (device fp32 softmax) vs float64          : {rel(y_dev_t, y64):.6e}")
print(f"  inner, device sum on device y               : {rel(inner_dev, inner64):.6e}")
print(f"  inner, precise sum on device y              : {rel(inner_pre, inner64):.6e}")
print(f"  inner, float64 sum on DEVICE y              : {rel(inner_hybrid, inner64):.6e}")
print("  -> if the last line ~ the first two, the reduction is innocent and y is the source")
json.dump({"forward": rows}, open("perf/of3t_adaln/softmax_probe.json", "w"), indent=1)
