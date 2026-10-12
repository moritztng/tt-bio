"""triatt_tr's input side vs the shipped ending-node norm, one call, against float64.

    TT_VISIBLE_DEVICES=<card> python perf/spd_fastmode/ln_tr_op.py [--s 512] [--c 128] [--reps 20]

Shipped: `_pair_transpose(z)` then `ttnn.layer_norm` (the trunk's ln config). triatt_tr:
`pair_ln_transpose.ln_transpose(z)`, one program. Prints ms/call (median of --reps after 3 warm calls) and
rel_rms / max_abs to float64 LN(z)^T, one JSON line per arm, each arm twice.
"""
import argparse, json, os, statistics, time

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--s", type=int, default=512)
ap.add_argument("--c", type=int, default=128)
ap.add_argument("--reps", type=int, default=20)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

from tt_bio.main import ensure_p300_mesh_descriptor  # noqa: E402
ensure_p300_mesh_descriptor()
import ttnn  # noqa: E402
from tt_bio import tenstorrent as T  # noqa: E402
from tt_bio import pair_ln_transpose as L  # noqa: E402

g_ = torch.Generator().manual_seed(a.seed)
S, C = a.s, a.c
z = torch.randn(S, S, C, generator=g_).bfloat16()
gw = (1 + 0.1 * torch.randn(C, generator=g_)).bfloat16()
gb = (0.1 * torch.randn(C, generator=g_)).bfloat16()
ref = torch.nn.functional.layer_norm(z.double(), (C,), gw.double(), gb.double(), 1e-5).transpose(0, 1)

dev = T.get_device()
zt = ttnn.from_torch(z, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
wt, bt = (ttnn.from_torch(v.reshape(1, C), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev) for v in (gw, gb))
base = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                        fp32_dest_acc_en=True, packer_l1_acc=True)
ckc = T.ln_compute_kernel_config(T.trunk_compute_kernel_config(base))
assert L.eligible(zt, wt, bt), "ln_transpose declined"


def shipped():
    x = T._pair_transpose(zt, ttnn.DRAM_MEMORY_CONFIG)
    y = ttnn.layer_norm(x, weight=wt, bias=bt, epsilon=1e-5, compute_kernel_config=ckc,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(x)
    return y


def fused():
    return L.ln_transpose(zt, wt, bt, 1e-5)


for arm, fn in (("shipped", shipped), ("ln_transpose", fused), ("shipped2", shipped), ("ln_transpose2", fused)):
    ts = []
    for i in range(3 + a.reps):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        y = fn()
        ttnn.synchronize_device(dev)
        if i >= 3:
            ts.append(time.perf_counter() - t0)
        if i < 2 + a.reps:
            ttnn.deallocate(y)
    out = ttnn.to_torch(y).double().reshape(S, S, C)
    assert torch.isfinite(out).all(), arm
    d = out - ref
    print(json.dumps(dict(arm=arm, s=S, c=C, ms=round(1e3 * statistics.median(ts), 3),
                          rel_rms=float(d.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()),
                          max_abs=float(d.abs().max()), card=os.environ.get("TT_VISIBLE_DEVICES"))), flush=True)
