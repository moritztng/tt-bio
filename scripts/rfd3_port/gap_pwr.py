"""RFD3_PAIR_WS_RESID at the token pair shape: `z + transition(z)` as pair_swiglu_ws then ttnn.add (shipped) vs
one program that adds into z in place (resid=z, RMSNorm prologue), both against float64, plus ms/call.

    python scripts/rfd3_port/gap_pwr.py OUT.json [reps] [n] [hifi4|b8hifi2|b8lofi]

Random weights at RFdiffusion3's scale (c_z 128, hidden 256 and 512, n 704 rows of 704). Each in-place call gets a
fresh device copy of z made outside the clock. Writes one JSON with, per hidden width: ms of each arm, rel_rms and
max_abs to float64, and whether the in-place arm left a z outside the rows it owns untouched (it owns them all).
"""
import json
import os
import statistics
import sys
import time

import torch

from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import ttnn  # noqa: E402
from tt_bio import tenstorrent as T  # noqa: E402
from tt_bio import pair_swiglu_ws as W  # noqa: E402

out_path, reps = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 9
n = int(sys.argv[3]) if len(sys.argv) > 3 else 704
mode = sys.argv[4] if len(sys.argv) > 4 else "hifi4"     # weights bf16 HiFi4, or bfp8 at HiFi2 / LoFi (fast candidates)
c, eps = 128, 1e-6
dev = T.get_device()
g = torch.Generator().manual_seed(0)
z = (torch.randn(1, n, n, c, generator=g) * 2.0).to(torch.bfloat16)
gamma = (1.0 + 0.1 * torch.randn(c, generator=g)).to(torch.bfloat16)
zt = ttnn.from_torch(z, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
gt = ttnn.from_torch(gamma, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
ln = (gt, None, eps)
rec = {"mode": mode, "n": n, "c": c, "reps": reps, "card": os.environ.get("TT_VISIBLE_DEVICES"), "arch": str(dev.arch()), "cases": []}


def med(fn, fresh):
    ts = []
    for i in range(reps + 2):
        a = fresh()
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        o = fn(a)
        ttnn.synchronize_device(dev)
        if i >= 2:
            ts.append(time.perf_counter() - t0)
        ttnn.deallocate(o)
        if a is not o and a is not zt:
            ttnn.deallocate(a)
    return round(1e3 * statistics.median(ts), 3)


def err(t, ref):
    d = ttnn.to_torch(t).double().reshape(ref.shape) - ref
    return {"rel_rms": float(d.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()), "max_abs": float(d.abs().max())}


for h in (256, 512):
    w1, w2 = (torch.randn(c, h, generator=g) / c ** 0.5 for _ in range(2))
    w3 = torch.randn(h, c, generator=g) / h ** 0.5
    (w12, w3t), = W.weights(w1, w2, w3, dev, ttnn.bfloat16 if mode == "hifi4" else ttnn.bfloat8_b)
    zd = z.double()
    zn = zd * torch.rsqrt(zd.pow(2).mean(-1, keepdim=True) + eps) * gamma.double()
    ref = zd + (torch.nn.functional.silu(zn @ w1.to(torch.bfloat16).double()) * (zn @ w2.to(torch.bfloat16).double())) \
        @ w3.to(torch.bfloat16).double()
    fid = {"hifi4": ttnn.MathFidelity.HiFi4, "b8hifi2": ttnn.MathFidelity.HiFi2, "b8lofi": ttnn.MathFidelity.LoFi}[mode]
    ship = lambda a: ttnn.add(a, W.pair_swiglu(a, w12, w3t, fid, ln=ln))  # noqa: E731
    fused = lambda a: W.pair_swiglu(a, w12, w3t, fid, ln=ln, resid=a)  # noqa: E731
    copy = lambda: ttnn.clone(zt)  # noqa: E731
    o_ship = ship(zt)
    zc = copy()
    o_fused = fused(zc)
    case = {"hidden": h, "eligible": W.eligible(zt, w12, w3t, ln=(gt, None), resid=zt),
            "inplace": o_fused.buffer_address() == zc.buffer_address(),
            "ship": err(o_ship, ref), "fused": err(o_fused, ref),
            "fused_vs_ship_max_abs": float((ttnn.to_torch(o_fused).double() - ttnn.to_torch(o_ship).double()).abs().max()),
            "z_untouched_by_ship": bool(torch.equal(ttnn.to_torch(zt), z))}
    ttnn.deallocate(o_ship), ttnn.deallocate(o_fused)
    case["ship_ms"] = med(ship, lambda: zt)
    case["fused_ms"] = med(fused, copy)
    case["clone_ms"] = med(lambda a: ttnn.clone(a), lambda: zt)
    rec["cases"].append(case)
    print(json.dumps(case), flush=True)
json.dump(rec, open(out_path, "w"), indent=1)
