"""gated_out_proj with a bf16 or a bfp8 gate (lever triatt_ln_b8g) at Protenix-v2's triangle-attention tail.

o, g head-major [N, 8, N, 32], w [256, 256], no residual (so the update is read unrounded by a pair add). Prints per arm ms/call (median of --reps
after 3 warm calls), how many calls the fused tail served, and the rel-RMS of the update against a float64
reference built from the bf16 host tensors (so the bfp8 arm's error includes the gate's own rounding).
"""
import argparse, json, statistics, time

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=512)
ap.add_argument("--reps", type=int, default=10)
a = ap.parse_args()

from tt_bio.main import ensure_p300_mesh_descriptor  # noqa: E402
ensure_p300_mesh_descriptor()
import ttnn  # noqa: E402
from tt_bio import tenstorrent as T  # noqa: E402
from tt_bio import triatt_qkv as Q  # noqa: E402

gen = torch.Generator().manual_seed(0)
N, H, c = a.n, 8, 256
dev = T.get_device()
o_h = torch.randn(N, H, N, 32, generator=gen).bfloat16()
g_h = torch.randn(N, H, N, 32, generator=gen).bfloat16()
w_h = (torch.randn(H * 32, c, generator=gen) / (H * 32) ** 0.5).bfloat16()
up = lambda t, dt=ttnn.bfloat16: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)  # noqa: E731
o, w = up(o_h), up(w_h)
ref = ((o_h.double() * torch.sigmoid(g_h.double())).permute(0, 2, 1, 3).reshape(N, N, H * 32) @ w_h.double())
ckc = (ttnn.MathFidelity.HiFi2, False, True, False)

for name, dt in (("bf16", ttnn.bfloat16), ("bfp8", ttnn.bfloat8_b)):
    g = up(g_h, dt)
    with T.lever_scope(["triatt_tail"] + ["triatt_ln_b8g"] * (dt == ttnn.bfloat8_b)):
        served0 = Q.GOP_STATS[0]
        ms = []
        for i in range(3 + a.reps):
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            r = Q.gated_out_proj(o, g, w, ckc)
            ttnn.synchronize_device(dev)
            if i >= 3:
                ms.append((time.perf_counter() - t0) * 1e3)
            assert r is not None, "gated_out_proj declined"
            if i < 3 + a.reps - 1:
                ttnn.deallocate(r)
        upd = ttnn.to_torch(r).double().reshape(N, N, c)
        rel = ((upd - ref).pow(2).mean() / ref.pow(2).mean()).sqrt().item()
        print(json.dumps({"arm": name, "ms": round(statistics.median(ms), 3), "served": Q.GOP_STATS[0] - served0,
                          "rel_rms_f64": round(rel, 5)}), flush=True)
    ttnn.deallocate(g)
