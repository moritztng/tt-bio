#!/usr/bin/env python3
"""AF2's Transition with its residual in the program (`af2.TRANSITION_RES`) against the shipped pair: `pair_relu`
then `rne_add`. Both arms are graded against float64 torch on the same bf16 input, forward (`x + f(x)`) and, through
the real tape under BindCraft 2's fast round, backward (the cotangent of x for a random bf16 seed). Also ms per
forward call, median of REPS synchronised calls, with the AICLK read before and after.

usage (one chip): TT_VISIBLE_DEVICES=N python perf/bc4_pairfwd/grade.py OUT.json [shape ...]   shape = 1x224x224x128
"""
import json
import sys
import time

import torch
import ttnn

import tt_bio.af2 as A
import tt_bio.autograd as ag
import tt_bio.bindcraft2 as B
import tt_bio.rne_add as K
import tt_bio.tenstorrent as T

OUT = sys.argv[1]
SHAPES = [tuple(int(v) for v in s.split("x")) for s in sys.argv[2:]] or [(1, 224, 224, 128), (1, 2, 224, 256)]
REPS = 20
dev = T.get_device()
# the node this process opened: TT_VISIBLE_DEVICES is an index, not the /dev/tenstorrent number
NODE = sorted({ln.rsplit("/", 1)[1].strip() for ln in open("/proc/self/maps") if "/dev/tenstorrent/" in ln})[0]
aiclk = lambda: int(open(f"/sys/class/tenstorrent/tenstorrent!{NODE}/tt_aiclk").read().split()[0])  # noqa: E731


def rel(a, ref):
    return float(((a.double() - ref).norm() / ref.norm()))


def run(shape):
    C = shape[-1]
    F = 4 * C
    torch.manual_seed(0)
    sd = {"norm.weight": 1 + 0.1 * torch.randn(C), "norm.bias": 0.1 * torch.randn(C),
          "fc1.weight": torch.randn(F, C) / C ** 0.5, "fc1.bias": 0.1 * torch.randn(F),
          "fc2.weight": torch.randn(C, F) / F ** 0.5, "fc2.bias": 0.1 * torch.randn(C)}
    mod = A.ReluTransition(sd, A.compute_kernel_config())
    # a residual stream several times its update, as the pair is by the middle of the trunk
    x_t = (4 * torch.randn(shape)).to(torch.bfloat16).float()
    dy_t = torch.randn(shape).to(torch.bfloat16).float()
    up = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)  # noqa: E731

    xr = x_t.double().requires_grad_(True)
    xn = torch.nn.functional.layer_norm(xr, (C,), sd["norm.weight"].double(), sd["norm.bias"].double(), 1e-5)
    y_ref = xr + torch.relu(xn @ sd["fc1.weight"].double().t() + sd["fc1.bias"].double()) \
        @ sd["fc2.weight"].double().t() + sd["fc2.bias"].double()
    y_ref.backward(dy_t.double())
    g_ref, y_ref = xr.grad, y_ref.detach()
    nearest = y_ref.to(torch.bfloat16).float()

    def composed(x):
        u = mod(x)
        out = K.rne_add(x, u, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(u)
        return out

    def fused(x):
        out = mod.residual(x)
        assert out is not None, "TRANSITION_RES declined"
        return out

    rec = {"shape": list(shape), "node": NODE}
    A.TRANSITION_RES = True
    with B.fast_round():
        for name, fn in (("composed", composed), ("fused", fused)):
            x = up(x_t)
            y = ttnn.to_torch(fn(x)).float()
            for _ in range(2):
                ttnn.deallocate(fn(x))
            ttnn.synchronize_device(dev)
            clk, ts = [aiclk()], []
            for _ in range(REPS):
                t0 = time.monotonic()
                o = fn(x)
                ttnn.synchronize_device(dev)
                ts.append(time.monotonic() - t0)
                ttnn.deallocate(o)
            clk.append(aiclk())
            ts.sort()
            with ag.tape():
                xt = ag.Tensor(up(x_t), requires_grad=True)
                out = fn(xt)
            out.backward(seed=up(dy_t))
            g = ttnn.to_torch(xt.grad).float()
            rec[name] = {"fwd_rel": rel(y, y_ref), "fwd_frac_nearest_bf16": float((y == nearest).float().mean()),
                         "bw_rel": rel(g, g_ref), "ms": 1e3 * ts[len(ts) // 2], "aiclk": clk,
                         "finite": bool(torch.isfinite(y).all() and torch.isfinite(g).all())}
    return rec


with open(OUT, "w") as f:
    for s in SHAPES:
        r = run(s)
        print(json.dumps(r), flush=True)
        f.write(json.dumps(r) + "\n")
