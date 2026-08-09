#!/usr/bin/env python3
"""Atom-window attention microbenchmark, qb2 card 1.

Reproduces the exact sub-graph of AtomTransformer._attention (tt_bio/protenix.py) at the 298 aa
production shapes: nb=75 windows, H=4 heads, n_queries=32, n_keys=128, head_dim=32.

  sc = Qb @ Kb^T            protenix.py:414   [75,4,32,32] . [75,4,32,128] -> [75,4,32,128]
  sc = sc * dh^-0.5
  sc = (sc + z) + pad_bias
  o  = softmax(sc) @ Vb     protenix.py:417   [75,4,32,128] . [75,4,128,32] -> [75,4,32,32]

Modes:
  pipeline   per-op timing of the whole sub-graph, fp32 and bf16
  batch      nb sweep on both matmuls -- is the batch dim parallelised across cores or serial?
  rank       does the shape of the batch dims ([75,4] vs [300,1] vs [1,300]) change anything?
  mshape     grow M (n_queries) at constant total work -- is per-core work the limit?
"""
import argparse, json, statistics as st, sys, time
import torch, ttnn
from tt_bio.tenstorrent import get_device, CORE_GRID_MAIN

NB, H, NQ, NK, DH = 75, 4, 32, 128, 32
dev = get_device()
CKC = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                             fp32_dest_acc_en=True, packer_l1_acc=True)


def timed(fn, warm=3, reps=8, trials=5):
    for _ in range(warm):
        fn()
    ttnn.synchronize_device(dev)
    o = []
    for _ in range(trials):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        ttnn.synchronize_device(dev)
        o.append((time.perf_counter() - t0) / reps)
    return st.median(o)


def tt(x, dt):
    return ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=dt)


def mk(dt, nb=NB, h=H, nq=NQ, nk=NK, dh=DH):
    g = torch.Generator().manual_seed(0)
    q = tt(torch.randn(nb, h, nq, dh, generator=g), dt)
    k = tt(torch.randn(nb, h, nk, dh, generator=g), dt)
    v = tt(torch.randn(nb, h, nk, dh, generator=g), dt)
    z = tt(torch.randn(nb, h, nq, nk, generator=g), dt)
    pb = tt(torch.zeros(nb, 1, nq, nk), dt)
    return q, k, v, z, pb


def pipeline(dt, name, out):
    q, k, v, z, pb = mk(dt)
    kt = ttnn.permute(k, (0, 1, 3, 2))
    sc0 = ttnn.matmul(q, kt, compute_kernel_config=CKC)
    sc1 = ttnn.multiply(sc0, DH ** -0.5)
    sc2 = ttnn.add(sc1, z)
    sc3 = ttnn.add(sc2, pb)
    sm = ttnn.softmax(sc3, dim=-1)
    rows = [
        ("permute k (:413)", lambda: ttnn.deallocate(ttnn.permute(k, (0, 1, 3, 2)))),
        ("matmul QK^T (:414)", lambda: ttnn.deallocate(ttnn.matmul(q, kt, compute_kernel_config=CKC))),
        ("multiply scale (:415)", lambda: ttnn.deallocate(ttnn.multiply(sc0, DH ** -0.5))),
        ("add z (:416)", lambda: ttnn.deallocate(ttnn.add(sc1, z))),
        ("add pad_bias (:416)", lambda: ttnn.deallocate(ttnn.add(sc2, pb))),
        ("softmax (:417)", lambda: ttnn.deallocate(ttnn.softmax(sc3, dim=-1))),
        ("matmul A@V (:417)", lambda: ttnn.deallocate(ttnn.matmul(sm, v, compute_kernel_config=CKC))),
    ]
    tot = 0.0
    res = []
    print(f"--- pipeline {name} (nb={NB} H={H} nq={NQ} nk={NK} dh={DH})", flush=True)
    for lbl, fn in rows:
        s = timed(fn)
        tot += s
        res.append({"op": lbl, "us": round(s * 1e6, 2)})
        print(f"  {lbl:24s} {s*1e6:9.2f} us", flush=True)
    print(f"  {'TOTAL':24s} {tot*1e6:9.2f} us   -> x1200 calls/fold = {tot*1200*1e3:.1f} ms/fold "
          f"(2 matmuls only: {(res[1]['us']+res[6]['us'])*1200/1e3:.1f} ms/fold)", flush=True)
    out[f"pipeline_{name}"] = {"ops": res, "total_us": round(tot * 1e6, 2)}


def batch(dt, name, out):
    print(f"--- batch sweep {name}: sc = [nb,4,32,32] . [nb,4,32,128]", flush=True)
    rows = []
    for nb in (1, 2, 4, 8, 16, 32, 64, 75, 128, 256):
        q, k, v, z, pb = mk(dt, nb=nb)
        kt = ttnn.permute(k, (0, 1, 3, 2))
        s1 = timed(lambda: ttnn.deallocate(ttnn.matmul(q, kt, compute_kernel_config=CKC)))
        a = tt(torch.rand(nb, H, NQ, NK), dt)
        s2 = timed(lambda: ttnn.deallocate(ttnn.matmul(a, v, compute_kernel_config=CKC)))
        units = nb * H
        rows.append({"nb": nb, "units": units, "qk_us": round(s1 * 1e6, 2),
                     "qk_us_per_unit": round(s1 * 1e6 / units, 3),
                     "av_us": round(s2 * 1e6, 2), "av_us_per_unit": round(s2 * 1e6 / units, 3)})
        print(f"  nb={nb:<4} units={units:<5} QK^T {s1*1e6:8.2f} us ({s1*1e6/units:6.3f}/unit)   "
              f"A@V {s2*1e6:8.2f} us ({s2*1e6/units:6.3f}/unit)", flush=True)
        for t in (q, k, v, z, pb, kt, a):
            ttnn.deallocate(t)
    out[f"batch_{name}"] = rows


def rank(dt, name, out):
    print(f"--- batch-dim shape {name}: same 300 (32x32)@(32x128) problems, different rank/split", flush=True)
    rows = []
    for lbl, qs, ks in (("[75,4]", (75, 4, NQ, DH), (75, 4, DH, NK)),
                        ("[300,1]", (300, 1, NQ, DH), (300, 1, DH, NK)),
                        ("[1,300]", (1, 300, NQ, DH), (1, 300, DH, NK)),
                        ("[300] 3D", (300, NQ, DH), (300, DH, NK))):
        g = torch.Generator().manual_seed(0)
        a = tt(torch.randn(*qs, generator=g), dt)
        b = tt(torch.randn(*ks, generator=g), dt)
        try:
            s = timed(lambda: ttnn.deallocate(ttnn.matmul(a, b, compute_kernel_config=CKC)))
            rows.append({"shape": lbl, "us": round(s * 1e6, 2)})
            print(f"  {lbl:10s} {s*1e6:9.2f} us", flush=True)
        except Exception as e:
            print(f"  {lbl:10s} ERR {str(e)[:90]}", flush=True)
        ttnn.deallocate(a); ttnn.deallocate(b)
    out[f"rank_{name}"] = rows


def mshape(dt, name, out):
    """Constant total FLOPs, moving work from the batch dim into M."""
    print(f"--- M-shape sweep {name}: 300*32 query rows total, nq per matmul varied", flush=True)
    rows = []
    for nq in (32, 64, 128, 256, 512, 1200, 2400, 9600):
        units = 9600 // nq
        g = torch.Generator().manual_seed(0)
        a = tt(torch.randn(units, 1, nq, DH, generator=g), dt)
        b = tt(torch.randn(units, 1, DH, NK, generator=g), dt)
        s = timed(lambda: ttnn.deallocate(ttnn.matmul(a, b, compute_kernel_config=CKC)))
        rows.append({"nq": nq, "units": units, "us": round(s * 1e6, 2)})
        print(f"  nq={nq:<6} units={units:<5} {s*1e6:9.2f} us", flush=True)
        ttnn.deallocate(a); ttnn.deallocate(b)
    out[f"mshape_{name}"] = rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="pipeline,batch,rank,mshape")
    ap.add_argument("--dtypes", default="fp32,bf16")
    ap.add_argument("--out", default="perf/atom_window/micro_card1.json")
    a = ap.parse_args()
    out = {"card": "qb2-card1", "ttnn": getattr(ttnn, "__version__", "?"),
           "grid": f"{CORE_GRID_MAIN.x}x{CORE_GRID_MAIN.y}"}
    fns = {"pipeline": pipeline, "batch": batch, "rank": rank, "mshape": mshape}
    for dn in a.dtypes.split(","):
        dt = ttnn.float32 if dn == "fp32" else ttnn.bfloat16
        for m in a.modes.split(","):
            fns[m](dt, dn, out)
    json.dump(out, open(a.out, "w"), indent=2)
    print("wrote", a.out, flush=True)
