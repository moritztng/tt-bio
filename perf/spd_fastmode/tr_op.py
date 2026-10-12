"""triatt_tr vs the shipped ending-node tail, one call, against float64.

    TT_VISIBLE_DEVICES=<card> python perf/spd_fastmode/tr_op.py [--s 512] [--heads 4] [--reps 20]

Shipped: `gated_out_proj(o, g, Wo)` -> u [S, S, N], then `transpose_add(u, z)` (z += u^T).
triatt_tr: `gated_out_proj(o, g, Wo, resid=z, transposed=True)`, one program. Same o, g, Wo, z (z
re-uploaded per call, outside the timer). Prints ms/call (median of --reps after 3 warm calls) and rel_rms /
max_abs of the new z to a float64 reference, one JSON line per arm.
"""
import argparse, json, os, statistics, time

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--s", type=int, default=512)
ap.add_argument("--heads", type=int, default=4)
ap.add_argument("--reps", type=int, default=20)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

from tt_bio.main import ensure_p300_mesh_descriptor  # noqa: E402
ensure_p300_mesh_descriptor()
import ttnn  # noqa: E402
from tt_bio import tenstorrent as T  # noqa: E402
from tt_bio import triatt_qkv as Q  # noqa: E402
from tt_bio import pair_transpose_add as PTA  # noqa: E402

g_ = torch.Generator().manual_seed(a.seed)
S, H = a.s, a.heads
N = 32 * H
o = torch.randn(S, H, S, 32, generator=g_).bfloat16()
g = torch.randn(S, H, S, 32, generator=g_).bfloat16()
w = (torch.randn(H * 32, N, generator=g_) / (H * 32) ** 0.5).bfloat16()
z = torch.randn(1, S, S, N, generator=g_).bfloat16()
u = (o.double() * torch.sigmoid(g.double())).permute(0, 2, 1, 3).reshape(S, S, H * 32) @ w.double()
ref = z.double()[0] + u.transpose(0, 1)

dev = T.get_device()
up = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)  # noqa: E731
ot, gt, wt = up(o), up(g), up(w)
ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi3, math_approx_mode=False,
                                       fp32_dest_acc_en=True, packer_l1_acc=True)
Q.set_tail("1")


def shipped(zt):
    ut = Q.gated_out_proj(ot, gt, wt, ckc)
    assert ut is not None, "shipped tail declined"
    assert PTA.eligible(ut, zt), "transpose_add declined"
    PTA.transpose_add(ut, zt)
    ttnn.deallocate(ut)
    return zt


def fused(zt):
    out = Q.gated_out_proj(ot, gt, wt, ckc, resid=zt, transposed=True)
    assert out is zt, "triatt_tr declined"
    return zt


for arm, fn in (("shipped", shipped), ("triatt_tr", fused), ("shipped2", shipped), ("triatt_tr2", fused)):
    ts = []
    for i in range(3 + a.reps):
        zt = up(z)
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        fn(zt)
        ttnn.synchronize_device(dev)
        if i >= 3:
            ts.append(time.perf_counter() - t0)
        if i < 3 + a.reps - 1:
            ttnn.deallocate(zt)
    out = ttnn.to_torch(zt).double()[0]
    assert torch.isfinite(out).all(), arm
    d = out - ref
    print(json.dumps(dict(arm=arm, s=S, heads=H, ms=round(1e3 * statistics.median(ts), 3),
                          rel_rms=float(d.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()),
                          max_abs=float(d.abs().max()), card=os.environ.get("TT_VISIBLE_DEVICES"))), flush=True)
