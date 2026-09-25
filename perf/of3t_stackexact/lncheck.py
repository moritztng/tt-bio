#!/usr/bin/env python3
"""of3t-stackexact: the exact layer norm against torch float64 autograd, on the card, before any arm.

A lever built for this row is checked against a float64 reference, never against another
approximation. Shapes are the trunk's: a pair row block (bf16 and fp32) and the single track,
with and without affine. Reports rel L2 of y, dx, dgamma, dbeta vs float64, and the shipped
verb's own error beside it so the exact one's is readable. Also checks the raw path, the
scope's teardown, and that dev_cot-style verbs are restored.
"""
import json, os, socket, sys
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, ttnn
from tt_bio import autograd as ag
from tt_bio import taped_ttnn as tt
import exactln


def rel(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm())


def one(dev, shape, dtype, affine, lever):
    torch.manual_seed(0)
    x = torch.randn(*shape, dtype=torch.float64) * 3 + 1.5
    d = shape[-1]
    g0 = torch.randn(d, dtype=torch.float64) * 0.5 + 1
    b0 = torch.randn(d, dtype=torch.float64) * 0.1
    cot = torch.randn(*shape, dtype=torch.float64)
    tdt = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    xq = x.to(tdt).double()  # the reference sees the same rounded input the card does
    gq, bq, cq = g0.to(tdt).double(), b0.to(tdt).double(), cot.to(tdt).double()
    xr = xq.clone().requires_grad_(); gr = gq.clone().requires_grad_(); br = bq.clone().requires_grad_()
    yr = torch.nn.functional.layer_norm(xr, (d,), gr if affine else None, br if affine else None, 1e-5)
    yr.backward(cq)
    put = lambda t: ttnn.from_torch(t.to(tdt), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=dev)
    X = ag.Tensor(put(xq), requires_grad=True)
    G = ag.Tensor(put(gq.reshape(1, d)), requires_grad=True) if affine else None
    B = ag.Tensor(put(bq.reshape(1, d)), requires_grad=True) if affine else None
    ctx = exactln.exact_layer_norm() if lever else __import__("contextlib").nullcontext()
    with ctx, ag.tape():
        kw = {"weight": G, "bias": B} if affine else {}
        y = ag._TAPED["layer_norm"](ttnn.layer_norm, (X,), dict(kw, epsilon=1e-5))
        y.backward(put(cq))
    out = {"y": rel(ttnn.to_torch(y.value), yr), "dx": rel(ttnn.to_torch(X.grad), xr.grad)}
    if affine:
        out["dgamma"] = rel(ttnn.to_torch(G.grad).reshape(-1), gr.grad)
        out["dbeta"] = rel(ttnn.to_torch(B.grad).reshape(-1), br.grad)
    return out


def main():
    import tt_bio.tenstorrent as T
    dev = T.get_device()
    try:
        res = {}
        for shape in ((1, 32, 384, 128), (1, 384, 384)):
            for dtype in (ttnn.bfloat16, ttnn.float32):
                for affine in (True, False):
                    k = f"{shape}|{dtype}|affine={affine}"
                    res[k] = {"shipped": one(dev, shape, dtype, affine, False),
                              "exact": one(dev, shape, dtype, affine, True)}
                    print(k, json.dumps(res[k]), flush=True)
        # raw path + teardown
        before = (tt._VERBS["layer_norm"], ag._TAPED["layer_norm"], ttnn.layer_norm)
        xr = torch.randn(1, 64, 128, dtype=torch.float64)
        with exactln.exact_layer_norm():
            yv = ttnn.layer_norm(ttnn.from_torch(xr.float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev), epsilon=1e-5)
        raw = rel(ttnn.to_torch(yv), torch.nn.functional.layer_norm(xr.float().double(), (128,), eps=1e-5))
        after = (tt._VERBS["layer_norm"], ag._TAPED["layer_norm"], ttnn.layer_norm)
        worst_exact = max(v for r in res.values() for v in r["exact"].values())
        rep = {"host": socket.gethostname(), "card_env": os.environ.get("TT_VISIBLE_DEVICES"),
               "cases": res, "raw_fp32_y_rel": raw, "restored": before == after,
               "stats": dict(exactln.STATS), "worst_exact_rel": worst_exact,
               "reference": "torch float64 autograd on the same rounded inputs"}
        json.dump(rep, open(sys.argv[1], "w"), indent=1)
        print(json.dumps({k: rep[k] for k in ("raw_fp32_y_rel", "restored", "stats", "worst_exact_rel")}))
    finally:
        T.cleanup()


main()
