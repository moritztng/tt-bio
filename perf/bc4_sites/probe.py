#!/usr/bin/env python3
"""bc4-bet-sites per-call probe: each lever at its BindCraft 2 shape, stock against the change.

    TT_VISIBLE_DEVICES=<chip> python perf/bc4_sites/probe.py --out probe.jsonl

Time per call: median of 7 bursts of 20 back-to-back calls with one synchronize per burst (eager), and the
same burst captured once as a trace and replayed (device-only, host enqueue removed). Accuracy: rel L2
against float64 on the same bf16 operands; data movement must be torch.equal.
"""
import argparse, json, os, statistics, time

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--burst", type=int, default=20)
ap.add_argument("--reps", type=int, default=7)
A = ap.parse_args()

from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch
import ttnn
from tt_bio import tenstorrent as T
from tt_bio import autograd as ag
from tt_bio import taped_ttnn as tt

dev = T.get_device()
CHIP = os.environ.get("TT_VISIBLE_DEVICES", "?")


def aiclk():
    try:
        return int(open(f"/sys/class/tenstorrent/tenstorrent!{CHIP}/tt_aiclk").read().split()[0])
    except Exception:
        return None


def eager(f):
    ttnn.deallocate(f()); ttnn.synchronize_device(dev)
    ms = []
    for _ in range(A.reps):
        outs = []
        t0 = time.perf_counter()
        for _ in range(A.burst):
            outs.append(f())
        ttnn.synchronize_device(dev)
        ms.append((time.perf_counter() - t0) * 1e3 / A.burst)
        for o in outs:
            ttnn.deallocate(o)
    return statistics.median(ms), max(ms) - min(ms)


def traced(f):
    try:
        ttnn.deallocate(f()); ttnn.synchronize_device(dev)
        tid = ttnn.begin_trace_capture(dev, cq_id=0)
        outs = [f() for _ in range(A.burst)]
        ttnn.end_trace_capture(dev, tid, cq_id=0)
        ms = []
        for _ in range(A.reps):
            t0 = time.perf_counter()
            ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
            ms.append((time.perf_counter() - t0) * 1e3 / A.burst)
        ttnn.release_trace(dev, tid)
        for o in outs:
            ttnn.deallocate(o)
        return statistics.median(ms), max(ms) - min(ms)
    except Exception as e:  # noqa: BLE001
        return None, repr(e)[:200]


def dt(t, dtype=ttnn.bfloat16):
    return ttnn.from_torch(t.contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=dev,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)


fo = open(A.out, "a")


def emit(r):
    r.update(aiclk=aiclk(), load=os.getloadavg()[0], at=time.time())
    print(json.dumps(r), flush=True)
    fo.write(json.dumps(r) + "\n"); fo.flush()


cfg = ag.precise_config()

# 1. split-K: the two transpose_a products of the masked OPM backward
for K, out_dtype in ((28672, ttnn.float32), (8192, None)):
    torch.manual_seed(K)
    a, b = dt(torch.randn(K, 64)), dt(torch.randn(K, 224))
    ref = ttnn.to_torch(a).double().T @ ttnn.to_torch(b).double()
    rel = lambda y: float((ttnn.to_torch(y).double() - ref).norm() / ref.norm())
    kw = dict(transpose_a=True, compute_kernel_config=cfg)
    if out_dtype is not None:
        kw["dtype"] = out_dtype
    stock = lambda: ttnn.matmul(a, b, **kw)
    split = lambda: ag._split_k(a, b, dict(kw))
    for arm, f in (("stock", stock), ("split", split), ("stock2", stock)):
        e, es = eager(f)
        t, ts = traced(f)
        emit(dict(lever="split_k", shape=[K, 64, 224], arm=arm, eager_ms=e, eager_spread=es, trace_ms=t,
                  trace_spread=ts, rel_l2=rel(f())))

# 2. the OPM permute pair: forward (2,0,1) on [128,64,224], VJP (1,2,0) on [224,128,64]
torch.manual_seed(1)
for dims, shape in (((1, 2, 0), (224, 128, 64)), ((2, 0, 1), (128, 64, 224))):
    x = dt(torch.randn(*shape))
    ref = ttnn.to_torch(x).permute(*dims)
    stock = lambda: ttnn.permute(x, dims)
    if dims == (1, 2, 0):
        new = lambda: tt._permute_back(x, list(dims))
    else:
        def new():
            m = ttnn.transpose(x, -2, -1)
            o = ttnn.transpose(m, -3, -2)
            ttnn.deallocate(m)
            return o
    for arm, f in (("stock", stock), ("transposes", new), ("stock2", stock)):
        e, es = eager(f)
        t, ts = traced(f)
        emit(dict(lever="permute", dims=list(dims), shape=list(shape), arm=arm, eager_ms=e, eager_spread=es,
                  trace_ms=t, trace_spread=ts, equal=bool(torch.equal(ttnn.to_torch(f()), ref))))
emit(dict(lever="done"))
