"""Gradient check for the fused triangle-attention backward, against a float64 reference.

Small shapes by default so the JIT compile loop is short. The reference is torch in float64 on the
host, not another device path: grading an approximation against an approximation is how a wrong
transform passes.
"""
import argparse, json, os, pathlib, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--b", type=int, default=4)
ap.add_argument("--h", type=int, default=2)
ap.add_argument("--n", type=int, default=64)
ap.add_argument("--d", type=int, default=32)
ap.add_argument("--grid", type=int, nargs=2, default=None)
ap.add_argument("--out", default=None)
a = ap.parse_args()

os.environ.setdefault("TT_VISIBLE_DEVICES", "2")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import torch
import ttnn
from tt_bio import triatt_bw as T

torch.manual_seed(0)
B, H, N, D = a.b, a.h, a.n, a.d
scale = D ** -0.5

q = torch.randn(B, H, N, D, dtype=torch.float64)
k = torch.randn(B, H, N, D, dtype=torch.float64)
v = torch.randn(B, H, N, D, dtype=torch.float64)
bias = torch.randn(1, H, N, N, dtype=torch.float64) * 0.5
g = torch.randn(B, H, N, D, dtype=torch.float64)

# float64 reference VJP. This is the thing being graded against; nothing on the card touches it.
qr, kr, vr, br = (t.clone().requires_grad_(True) for t in (q, k, v, bias))
p_ref = torch.softmax(qr @ kr.transpose(-1, -2) * scale + br, dim=-1)
(p_ref @ vr).backward(g)
ref = {"dq": qr.grad, "dk": kr.grad, "dv": vr.grad, "dbias": br.grad}

dev = ttnn.open_device(device_id=0)
try:
    def up(t):
        return ttnn.from_torch(t.to(torch.bfloat16), dtype=ttnn.bfloat16,
                               layout=ttnn.TILE_LAYOUT, device=dev)
    tq, tk, tv, tb, tg = (up(t) for t in (q, k, v, bias, g))
    # The compute grid, off the device. A fixed 13x10 reaches this part's dispatch cores and
    # tt-metal refuses the program with "Kernels cannot be placed on dispatch cores".
    cg = dev.compute_with_storage_grid_size()
    grid = tuple(a.grid) if a.grid else (cg.x, cg.y)
    print("compute grid:", grid)
    p = T.plan(B, H, N, D, grid)
    if not T.fits_l1(p):
        print(json.dumps({"skipped": "does not fit L1", "l1_bytes": p["l1_bytes"]})); raise SystemExit(0)

    z = torch.zeros(B, H, N, D)
    dq, dk, dv = (up(z.to(torch.float64)) for _ in range(3))
    part = ttnn.from_torch(torch.zeros(*T.partial_shape(p)), dtype=ttnn.float32,
                           layout=ttnn.TILE_LAYOUT, device=dev)

    ckc = (ttnn.MathFidelity.HiFi4,)
    t0 = time.time()
    e = T.build(dev, tq, tk, tv, tg, tb, dq, dk, dv, part, p, ckc, scale)
    ttnn.generic_op([tq, tk, tv, tg, tb, dq, dk, dv, part], e["pd"])
    ttnn.synchronize_device(dev)
    print(f"ran in {time.time()-t0:.2f}s (includes JIT compile)")

    got = {"dq": ttnn.to_torch(dq).double(), "dk": ttnn.to_torch(dk).double(),
           "dv": ttnn.to_torch(dv).double(),
           "dbias": ttnn.to_torch(part).double().sum(0, keepdim=True)}

    res = {"shape": [B, H, N, D], "l1_bytes": p["l1_bytes"], "cores": p["num_cores"]}
    for name, r in ref.items():
        e_ = got[name] - r
        res[name] = {"rel_l2": float(e_.norm() / r.norm()), "max_abs": float(e_.abs().max()),
                     "ref_norm": float(r.norm())}
    print(json.dumps(res, indent=2))
    if a.out:
        open(a.out, "w").write(json.dumps(res, indent=2))
finally:
    ttnn.close_device(dev)
