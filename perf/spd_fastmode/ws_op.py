"""transition_ws vs the shipped pair Transition, one call at the page shape, against float64.

    TT_VISIBLE_DEVICES=<card> python perf/spd_fastmode/ws_op.py [--n 512] [--c 128] [--hidden 512] [--reps 20]

Arms: shipped (`normal` set), shipped fast (`fast` set) and each + transition_ws, same weights and input, each at
the trunk's fidelity. Prints ms/call (median of --reps after 3 warm calls) and rel_rms / max_abs to a float64 torch
reference of LN -> fc3(silu(fc1) * fc2), and n_bad, the outputs off by more than 0.5 (the Wormhole fp32-dest
fault writes exact +-1 / 2 / 4). Output: one JSON line per arm.
"""
import argparse, json, os, statistics, time

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=512)
ap.add_argument("--c", type=int, default=128)
ap.add_argument("--hidden", type=int, default=512)
ap.add_argument("--reps", type=int, default=20)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--silu", type=int, default=None, help="pair_swiglu_ws silu: 0 metal, 1 degree-8 poly, 2 degree-6")
ap.add_argument("--arms", default="normal,normal+transition_ws,normal+transition_ws+ws_ln,fast,fast+transition_ws,"
                "fast+transition_ws+ws_ln", help="comma list of lever sets")
ap.add_argument("--resid", action="store_true",
                help="time and check the residual form x + transition(x) (Transition._transition, add_to_input), "
                     "the one the pair stack calls; each call gets a fresh device copy of x, made outside the clock")
a = ap.parse_args()

from tt_bio.main import ensure_p300_mesh_descriptor  # noqa: E402
ensure_p300_mesh_descriptor()
import ttnn  # noqa: E402
from tt_bio import tenstorrent as T  # noqa: E402
from tt_bio import pair_swiglu_ws as W  # noqa: E402

if a.silu is not None:
    W.SILU_POLY = a.silu

g = torch.Generator().manual_seed(a.seed)
c, h = a.c, a.hidden
sd = {"norm.weight": 1 + 0.1 * torch.randn(c, generator=g), "norm.bias": 0.1 * torch.randn(c, generator=g),
      "fc1.weight": torch.randn(h, c, generator=g) / c ** 0.5, "fc2.weight": torch.randn(h, c, generator=g) / c ** 0.5,
      "fc3.weight": torch.randn(c, h, generator=g) / h ** 0.5}
x = torch.randn(1, a.n, a.n, c, generator=g)
xb = x.to(torch.bfloat16).double()
ln = torch.nn.functional.layer_norm(xb, (c,), sd["norm.weight"].double(), sd["norm.bias"].double(), 1e-5)
ref = (torch.nn.functional.silu(ln @ sd["fc1.weight"].double().t()) * (ln @ sd["fc2.weight"].double().t())) \
    @ sd["fc3.weight"].double().t()
if a.resid:
    ref = ref + xb

dev = T.get_device()
base = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                        fp32_dest_acc_en=True, packer_l1_acc=True)
xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
for arm in a.arms.split(","):
    with T.levers(arm):
        mod = T.Transition(sd, T.trunk_compute_kernel_config(base))
        call = (lambda t: mod._transition(t, None, True)) if a.resid else mod
        fresh = (lambda: ttnn.clone(xt, memory_config=ttnn.DRAM_MEMORY_CONFIG)) if a.resid else (lambda: xt)
        for _ in range(3):
            ttnn.deallocate(call(fresh()))
        ttnn.synchronize_device(dev)
        ts = []
        for _ in range(a.reps):
            xi = fresh()
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            o = call(xi)
            ttnn.synchronize_device(dev)
            ts.append(time.perf_counter() - t0)
            last = o
        out = ttnn.to_torch(last).double()
    d = out - ref
    assert torch.isfinite(out).all(), arm
    print(json.dumps(dict(arm=arm, resid=a.resid, n=a.n, c=c, hidden=h, ms=round(1e3 * statistics.median(ts), 3),
                          rel_rms=float(d.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()),
                          max_abs=float(d.abs().max()), n_bad=int((d.abs() > 0.5).sum()), ws_fired=getattr(mod, "_ws", None) is not None, silu=W.SILU_POLY,
                          card=os.environ.get("TT_VISIBLE_DEVICES"))), flush=True)
