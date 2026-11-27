"""B1 (r4t-perf) site float64 grade for tt_bio.mm_layout.DW_2D (`transpose_a=True` long-K bf16->fp32 dW).

Pre-registered bar (2026-11-27 18:28Z, before any full-step run):
  PASS iff every RFD4 dW shape (K=65536, (M,N) in the 7 trunk/pair shapes) is served (served == 7, declined == 0),
           rel_L2(dw2d, f64) <= 1.05 * rel_L2(stock, f64),
           (1 - cos(dw2d, f64)) <= 1.21 * max(1 - cos(stock, f64), 1e-12) (D54 clause),
           and BREAK=1 (scales output by 1.10) FAILS.
"""
import json, os, sys, time
from tt_bio.main import ensure_p300_mesh_descriptor; ensure_p300_mesh_descriptor()
import torch, ttnn
from tt_bio import tenstorrent as T, autograd as ag, mm_layout as mml

BREAK = os.environ.get("BREAK") == "1"
dev = T.get_device(); cfg = ag.precise_config()
tile = dict(layout=ttnn.TILE_LAYOUT, device=dev)
K = 65536
SHAPES = [(256, 384), (384, 384), (384, 768), (384, 1024), (768, 384), (1024, 384), (384, 256)]

def timed(fn, reps=6):
    o = fn(); ttnn.synchronize_device(dev); ttnn.deallocate(o)
    ts = []
    for i in range(reps):
        t0 = time.perf_counter(); o = fn(); ttnn.synchronize_device(dev)
        ts.append((time.perf_counter() - t0) * 1e3)
        if i < reps - 1: ttnn.deallocate(o)
    ts.sort()
    return ts[len(ts) // 2], o

def metrics(tt_t, ref64):
    h = ttnn.to_torch(tt_t).float().double()
    if BREAK:
        h = h * 1.10
    rel = float((h - ref64).norm() / ref64.norm())
    omc = float(1.0 - (h * ref64).sum() / (h.norm() * ref64.norm()))
    return rel, omc

mml.DW_2D_REACH.clear()
rows = []
for M, N in SHAPES:
    g = torch.Generator().manual_seed(101 + M + N)
    ah = torch.randn(K, M, generator=g).bfloat16()
    bh = torch.randn(K, N, generator=g).bfloat16()
    ref64 = ah.double().t() @ bh.double()
    A = ttnn.from_torch(ah, dtype=ttnn.bfloat16, **tile)
    B = ttnn.from_torch(bh, dtype=ttnn.bfloat16, **tile)

    mml.DW_2D = False
    t_stk, o_stk = timed(lambda: ag._matmul(A, B, transpose_a=True, dtype=ttnn.float32, compute_kernel_config=cfg))
    r_stk, c_stk = metrics(o_stk, ref64) if not BREAK else (
        float((ttnn.to_torch(o_stk).float().double() - ref64).norm() / ref64.norm()),
        float(1.0 - (ttnn.to_torch(o_stk).float().double() * ref64).sum() / (ttnn.to_torch(o_stk).float().double().norm() * ref64.norm()))
    )
    ttnn.deallocate(o_stk)

    mml.DW_2D = True
    t_2d, o_2d = timed(lambda: ag._matmul(A, B, transpose_a=True, dtype=ttnn.float32, compute_kernel_config=cfg))
    r_2d, c_2d = metrics(o_2d, ref64)
    ttnn.deallocate(o_2d)
    ttnn.deallocate(A); ttnn.deallocate(B)

    ok = (r_2d <= 1.05 * r_stk) and (c_2d <= 1.21 * max(c_stk, 1e-12))
    rows.append(dict(M=M, N=N, stock_ms=round(t_stk, 3), dw2d_ms=round(t_2d, 3),
                     speedup=round(t_stk / t_2d, 2), rel_stk=r_stk, rel_2d=r_2d,
                     omc_stk=c_stk, omc_2d=c_2d, ok=ok))

served = mml.DW_2D_REACH.get("served", 0)
declined = sum(v for k, v in mml.DW_2D_REACH.items() if k != "served")
verdict = "PASS" if (all(r["ok"] for r in rows) and served >= len(SHAPES) and declined == 0) else "FAIL"
print("DW2D_GRADE " + json.dumps(dict(brk=BREAK, verdict=verdict, served=served, declined=declined,
                                      speedup_min=min(r["speedup"] for r in rows),
                                      speedup_max=max(r["speedup"] for r in rows), rows=rows)), flush=True)
