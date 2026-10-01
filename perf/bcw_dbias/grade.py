"""The query-chunked triangle-attention backward against float64, and against the arm it replaces.

`triatt_bw.run` at a forced query chunk (or the serving plan), the chunked-recompute fallback
(`autograd.triangle_attention` with the fused path off) on the same bf16 operands and card, and a
float64 torch VJP. Reported per gradient: rel L2 of each against float64 and the ratio of the two,
which is the number the campaign grades against its 1.1-1.2x bar.
"""
import argparse, json, os, pathlib, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--b", type=int, default=16)
ap.add_argument("--h", type=int, default=4)
ap.add_argument("--n", type=int, default=288)
ap.add_argument("--d", type=int, default=32)
ap.add_argument("--qt", type=int, default=None, help="force this query chunk, in tiles")
ap.add_argument("--no-fallback", action="store_true")
ap.add_argument("--out", default=None)
a = ap.parse_args()

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()

import torch
import ttnn
from tt_bio import triatt_bw as T
import tt_bio.autograd as ag

torch.manual_seed(0)
B, H, N, D = a.b, a.h, a.n, a.d
scale = D ** -0.5
q, k, v, g = (torch.randn(B, H, N, D, dtype=torch.float64) for _ in range(4))
bias = torch.randn(1, H, N, N, dtype=torch.float64) * 0.5
# The reference sees the SAME bf16-rounded operands the card sees, so what is graded is the
# kernel's arithmetic, not the input cast.
q, k, v, g, bias = (t.to(torch.bfloat16).double() for t in (q, k, v, g, bias))
qr, kr, vr, br = (t.clone().requires_grad_(True) for t in (q, k, v, bias))
(torch.softmax(qr @ kr.transpose(-1, -2) * scale + br, dim=-1) @ vr).backward(g)
ref = {"dq": qr.grad, "dk": kr.grad, "dv": vr.grad, "dbias": br.grad}

dev = ttnn.open_device(device_id=0)
try:
    def up(t):
        return ttnn.from_torch(t.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                               device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tq, tk, tv, tb, tg = (up(t) for t in (q, k, v, bias, g))
    cg = dev.compute_with_storage_grid_size()
    p = (T.serving_plan(B, H, N, D, (cg.x, cg.y)) if a.qt is None
         else T.plan(B, H, N, D, (cg.x, cg.y), q_chunk_tiles=a.qt))
    ttnn.synchronize_device(dev)
    t0 = time.time()
    dq, dk, dv, db = T.run(dev, tq, tk, tv, tb, tg, scale, (ttnn.MathFidelity.HiFi4,),
                           q_chunk_tiles=a.qt)
    ttnn.synchronize_device(dev)
    first = time.time() - t0
    got = {"dq": dq, "dk": dk, "dv": dv, "dbias": db}
    got = {n: ttnn.to_torch(x).double() for n, x in got.items()}
    res = {"shape": [B, H, N, D], "grid": [cg.x, cg.y], "Qt": p["Qt"], "Nt": p["Nt"],
           "groups": p["groups"], "rows_per_core": p["rows_per_core"], "l1_bytes": p["l1_bytes"],
           "first_call_s_incl_jit": round(first, 2), "fused": {}}
    for n, r in ref.items():
        res["fused"][n] = float((got[n] - r).norm() / r.norm())
    if not a.no_fallback:
        T.FUSED = False
        aq, ak, av, ab = (ag.Tensor(up(x), requires_grad=True) for x in (q, k, v, bias))
        out = ag.triangle_attention(aq, ak, av, ab, scale=scale, chunk=max(1, B // 4))
        ag.backward([out], [up(g)])
        comp = {"dq": aq.grad, "dk": ak.grad, "dv": av.grad, "dbias": ab.grad}
        res["fallback"], res["ratio"] = {}, {}
        for n, r in ref.items():
            e = float((ttnn.to_torch(comp[n]).double() - r).norm() / r.norm())
            res["fallback"][n] = e
            res["ratio"][n] = round(res["fused"][n] / e, 4)
    print(json.dumps(res, indent=1))
    if a.out:
        pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        open(a.out, "w").write(json.dumps(res, indent=1))
finally:
    ttnn.close_device(dev)
