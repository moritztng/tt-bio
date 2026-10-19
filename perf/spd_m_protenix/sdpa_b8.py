"""Triangle attention's SDPA at Protenix-v2's shape with bf16 or bfp8 q, k, v, and a bf16 or bfp8 destination.

    TT_VISIBLE_DEVICES=<card> python perf/spd_m_protenix/sdpa_b8.py [--n 512] [--reps 10]

q, k, v [N, 8, N, 32], bias [1, 8, N, N] bf16, the fold's `_tri_att_sdpa_at`. Arms: bf16 (today), b8 (q, k, v
bfp8, destination follows q: what triatt_ln_b8 does now), b8_out16 (q, k, v bfp8, destination bf16 via
triatt_sdpa.OUT_DTYPE). Per arm: ms/call (median of --reps after 3 warm calls), the output's dtype, whether the
fused kernel served it, and its relative RMS against a float64 reference. One JSON line each.
"""
import argparse, json, os, statistics, time

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=512)
ap.add_argument("--reps", type=int, default=10)
a = ap.parse_args()

from tt_bio.main import ensure_p300_mesh_descriptor  # noqa: E402
ensure_p300_mesh_descriptor()
import ttnn  # noqa: E402
from tt_bio import tenstorrent as T  # noqa: E402
from tt_bio import triatt_sdpa as TS  # noqa: E402

g = torch.Generator().manual_seed(0)
N, H, D = a.n, 8, 32
dev = T.get_device()
qh, kh, vh = (torch.randn(N, H, N, D, generator=g) for _ in range(3))
bh = torch.randn(1, H, N, N, generator=g)
scale = D ** -0.5
up = lambda t, dt: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)  # noqa: E731

s = (qh.double() @ kh.double().transpose(-1, -2) + bh.double()) * scale
ref = torch.softmax(s, -1) @ vh.double()


def arm(name, dt, out_dt, bias_dt=ttnn.bfloat16):
    q, k, v = up(qh, dt), up(kh, dt), up(vh, dt)
    bias = up(bh, bias_dt)
    TS.OUT_DTYPE = out_dt
    served0 = TS.STATS[0] if hasattr(TS, "STATS") else None
    call = lambda: T._tri_att_sdpa_at(q, k, v, bias, scale)  # noqa: E731
    for _ in range(3):
        ttnn.deallocate(call())
    ttnn.synchronize_device(dev)
    ts = []
    for _ in range(a.reps):
        t0 = time.perf_counter()
        o = call()
        ttnn.synchronize_device(dev)
        ts.append(time.perf_counter() - t0)
        ttnn.deallocate(o)
    o = call()
    oh = ttnn.to_torch(o).double()
    rr = float((oh - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt())
    print(json.dumps(dict(arm=name, n=N, card=os.environ.get("TT_VISIBLE_DEVICES"), out_dtype=str(o.dtype),
                          ms=round(1e3 * statistics.median(ts), 3), rel_rms_f64=round(rr, 5),
                          served=(TS.STATS[0] - served0) if served0 is not None else None)), flush=True)
    for t in (q, k, v, o, bias):
        ttnn.deallocate(t)
    TS.OUT_DTYPE = None


arm("bf16", ttnn.bfloat16, None)
arm("b8", ttnn.bfloat8_b, None)
arm("b8_out16", ttnn.bfloat8_b, ttnn.bfloat16)
arm("b8all", ttnn.bfloat8_b, None, ttnn.bfloat8_b)
arm("b8all_out16", ttnn.bfloat8_b, ttnn.bfloat16, ttnn.bfloat8_b)
