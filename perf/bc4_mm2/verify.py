#!/usr/bin/env python3
"""bc4-bet-mm2: autograd._matmul with the split-K rule on and off, at BindCraft 2's two shapes and its neighbours."""
import argparse, json, os, statistics, time
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); A = ap.parse_args()
from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch, ttnn
from tt_bio import tenstorrent as T, autograd as AG
dev = T.get_device(); cfg = AG.precise_config(); DRAM = ttnn.DRAM_MEMORY_CONFIG
CHIP = os.environ.get("TT_VISIBLE_DEVICES", "0")
def timed(f, burst=20, reps=9):
    ttnn.deallocate(f()); ttnn.synchronize_device(dev)
    outs = []; tid = ttnn.begin_trace_capture(dev, cq_id=0)
    for _ in range(burst): outs.append(f())
    ttnn.end_trace_capture(dev, tid, cq_id=0); ttnn.synchronize_device(dev)
    ms = []
    for _ in range(reps):
        t0 = time.perf_counter(); ttnn.execute_trace(dev, tid, cq_id=0, blocking=False); ttnn.synchronize_device(dev)
        ms.append((time.perf_counter() - t0) * 1e3 / burst)
    ttnn.release_trace(dev, tid); [ttnn.deallocate(o) for o in outs]
    return round(min(ms), 5)
rows = []
g = torch.Generator().manual_seed(1)
for K, M, N, od in ((28672, 64, 224, ttnn.float32), (8192, 64, 224, None), (50176, 128, 128, None), (448, 256, 768, None), (4096, 32, 32, None)):
    ha = torch.randn(K, M, generator=g).bfloat16(); hb = (torch.randn(K, N, generator=g) / K ** 0.5).bfloat16()
    ref = ha.double().T @ hb.double()
    a = ttnn.from_torch(ha, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
    b = ttnn.from_torch(hb, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
    kw = dict(transpose_a=True, compute_kernel_config=cfg, **({"dtype": od} if od else {}))
    for on in (False, True):
        AG.MM_SPLIT_K = on; s0 = AG.SPLIT_K["served"]
        o = AG._matmul(a, b, **kw); served = AG.SPLIT_K["served"] - s0
        got = ttnn.to_torch(o).double()
        r = dict(K=K, M=M, N=N, split=on, served=served, shape=list(o.shape), dtype=str(o.dtype),
                 slabs=AG._split_k_slabs(a, b, kw) if on else None,
                 rel_f64=round(float((got.reshape(M, N) - ref).norm() / ref.norm()), 6))
        ttnn.deallocate(o)
        r["dev_min_ms"] = timed(lambda: AG._matmul(a, b, **kw))
        r["aiclk"] = int(open(f"/sys/class/tenstorrent/tenstorrent!{CHIP}/tt_aiclk").read())
        print(json.dumps(r), flush=True); rows.append(r)
json.dump(dict(rows=rows, loadavg=open("/proc/loadavg").read().split()[:3]), open(A.out, "w"), indent=1)
