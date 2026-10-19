"""gated_out_proj (the triangle-attention tail: gate, out projection, residual) at OpenDDE's shape, per fidelity and K block.

o, g head-major [N, 12, N, 32] bf16, w [384, 384], z += update in place (the fold's call). Per arm: ms/call (median of
--reps after 3 warm calls) and the update's rel-RMS against float64 from the same bf16 host tensors (one call on a zero z).
Arms: TAIL_FID HiFi3 (shipped) / HiFi2 / LoFi, each with TAIL_KB1 1 (shipped) and 0 (whole K in one DST block).
  TT_VISIBLE_DEVICES=<chip> python perf/spd_m_opendde/gop_sweep.py --n 512 640 --out gop.jsonl
"""
import argparse, json, statistics, time

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, nargs="+", default=[512, 640])
ap.add_argument("--heads", type=int, default=12)
ap.add_argument("--reps", type=int, default=10)
ap.add_argument("--out")
a = ap.parse_args()

from tt_bio.main import ensure_p300_mesh_descriptor  # noqa: E402
ensure_p300_mesh_descriptor()
import ttnn  # noqa: E402
from tt_bio import tenstorrent as T  # noqa: E402
from tt_bio import triatt_qkv as Q  # noqa: E402

dev = T.get_device()
F = ttnn.MathFidelity
ckc = (F.HiFi3, False, True, False)
up = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)  # noqa: E731
for N in a.n:
    gen = torch.Generator().manual_seed(N)
    H = a.heads
    c = H * 32
    o_h = torch.randn(N, H, N, 32, generator=gen).bfloat16()
    g_h = torch.randn(N, H, N, 32, generator=gen).bfloat16()
    w_h = (torch.randn(c, c, generator=gen) / c ** 0.5).bfloat16()
    ref = (o_h.double() * torch.sigmoid(g_h.double())).permute(0, 2, 1, 3).reshape(N, N, c) @ w_h.double()
    o, g, w = up(o_h), up(g_h), up(w_h)
    z = up(torch.zeros(1, N, N, c).bfloat16())
    for fid in ("HiFi3", "HiFi2", "LoFi"):
        for kb1 in (1, 0):
            Q.TAIL_FID, Q.TAIL_KB1 = getattr(F, fid), kb1
            with T.lever_scope(["triatt_tail"]):
                ms = []
                for i in range(3 + a.reps):
                    ttnn.synchronize_device(dev)
                    t0 = time.perf_counter()
                    r = Q.gated_out_proj(o, g, w, ckc, resid=z)
                    ttnn.synchronize_device(dev)
                    assert r is not None, "gated_out_proj declined"
                    if i >= 3:
                        ms.append((time.perf_counter() - t0) * 1e3)
                z0 = up(torch.zeros(1, N, N, c).bfloat16())
                Q.gated_out_proj(o, g, w, ckc, resid=z0)
                upd = ttnn.to_torch(z0).double().reshape(N, N, c)
                ttnn.deallocate(z0)
            rel = ((upd - ref).pow(2).mean() / ref.pow(2).mean()).sqrt().item()
            rec = {"n": N, "heads": H, "fid": fid, "kb1": kb1, "ms": round(statistics.median(ms), 3),
                   "ms_min": round(min(ms), 3), "rel": rel, "max_abs": float((upd - ref).abs().max())}
            print(json.dumps(rec), flush=True)
            if a.out:
                with open(a.out, "a") as f:
                    f.write(json.dumps(rec) + "\n")
    for t in (o, g, w, z):
        ttnn.deallocate(t)
