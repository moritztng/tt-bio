"""RFD4 ceiling microbenchmarks (ARCH.md section 8): the five rates only a device answers.

1. pair-matmul rate at RFD4's pair shapes, per fidelity / weight dtype / n
2. triangle contraction: ttnn batched matmul at c_hidden 256, and tt-bio's whole TriangleMultiplication
   at c_z 384 / hidden 256
3. whether the pair track fits in L1 at n=128, and up to what batch
4. per-op floor under trace (small op chains, and one RFD4 trunk block at n=128)
5. whether `dit_pair_bias` takes c_z 384

Each result is one JSON line on stdout and in --out. AICLK is sampled by the caller during the run (tt-smi),
since reading it from inside the process that holds the card is not possible on this stack.

Run: TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3 python scripts/rfd4_ceiling_bench.py --out /tmp/ceiling.jsonl
"""
from __future__ import annotations

import argparse
import json
import time

import torch
import ttnn

from tt_bio import tenstorrent as T

OUT = None


def emit(**rec):
    rec["t"] = time.strftime("%H:%M:%S")
    line = json.dumps(rec)
    print(line, flush=True)
    if OUT:
        with open(OUT, "a") as f:
            f.write(line + "\n")


def ckc(dev, fid):
    return ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=getattr(ttnn.MathFidelity, fid), fp32_dest_acc_en=True,
        packer_l1_acc=True)


def timed(fn, dev, iters):
    fn()
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn()
        if out is not None:
            ttnn.deallocate(out)
    ttnn.synchronize_device(dev)
    return (time.perf_counter() - t0) / iters


def up(t, dev, dtype=ttnn.bfloat16, mem=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mem)


def q1_pair_matmul(dev, ns):
    shapes = [(384, 1024), (384, 768), (768, 384), (256, 384)]
    for n in ns:
        for k, m in shapes:
            x = up(torch.randn(1, 1, n * n, k), dev)
            for wd in ("bfloat16", "bfloat8_b"):
                w = up(torch.randn(1, 1, k, m) / k ** 0.5, dev, getattr(ttnn, wd))
                for fid in ("HiFi4", "HiFi2", "LoFi"):
                    cfg = ckc(dev, fid)
                    try:
                        s = timed(lambda: ttnn.linear(x, w, compute_kernel_config=cfg, dtype=ttnn.bfloat16),
                                  dev, 10)
                        emit(q=1, op="linear", shape=f"[{n}^2,{k}]x[{k},{m}]", n=n, k=k, m=m, fidelity=fid,
                             act="bfloat16", weight=wd, s=s, tflops=2 * n * n * k * m / s / 1e12)
                    except Exception as e:  # noqa: BLE001
                        emit(q=1, op="linear", n=n, k=k, m=m, fidelity=fid, weight=wd, error=str(e)[:200])
                ttnn.deallocate(w)
            ttnn.deallocate(x)


def q2_tri_contraction(dev, ns):
    c = 256
    for n in ns:
        a = up(torch.randn(1, c, n, n), dev)
        b = up(torch.randn(1, c, n, n), dev)
        for fid in ("HiFi4", "HiFi2", "LoFi"):
            cfg = ckc(dev, fid)
            try:
                s = timed(lambda: ttnn.matmul(a, b, compute_kernel_config=cfg, dtype=ttnn.bfloat16), dev, 5)
                emit(q=2, op="batched matmul (outgoing a @ b^T form)", c_hidden=c, n=n, fidelity=fid, s=s,
                     tflops=2 * c * n ** 3 / s / 1e12)
            except Exception as e:  # noqa: BLE001
                emit(q=2, op="batched matmul", c_hidden=c, n=n, fidelity=fid, error=str(e)[:200])
        ttnn.deallocate(a)
        ttnn.deallocate(b)
    # the whole module at RFD4's widths, through tt-bio's TriangleMultiplication (whatever T1/T2 it picks)
    import sys
    sys.path.insert(0, "tests")
    import rfd4_reference as REF
    from tt_bio.rfd4 import trunk as TR
    w = REF.pair_update_weights(1024, 384, 256, seed=0, prefix="p")
    scope = T.WeightScope({k: v.float() for k, v in w.items()}).child("p")
    for n in ns:
        for ending, name in ((False, "tri_mult_out"), (True, "tri_mult_in")):
            mod = T.TriangleMultiplication(ending, TR._trimul_weights(scope.child(name)), TR.compute_kernel_config())
            z = up(torch.randn(1, n, n, 384), dev)
            pm = up(torch.ones(1, n, n), dev)
            try:
                s = timed(lambda: mod(z, pm), dev, 5)
                # in-proj 4 x [n^2,384]x[384,256], contraction 2 c n^3, out-proj [n^2,256]x[256,384], gate [n^2,384]^2
                fl = 2 * n * n * (4 * 384 * 256 + 256 * 384 + 384 * 384) + 2 * 256 * n ** 3
                emit(q=2, op=f"tt-bio TriangleMultiplication ({name})", c_z=384, c_hidden=256, n=n, s=s,
                     tflops=fl / s / 1e12)
            except Exception as e:  # noqa: BLE001
                emit(q=2, op=f"tt-bio TriangleMultiplication ({name})", c_z=384, n=n, error=str(e)[:300])
            ttnn.deallocate(z)


def q3_l1_fit(dev):
    n = 128
    l1 = ttnn.L1_MEMORY_CONFIG
    g = dev.compute_with_storage_grid_size()
    emit(q=3, grid=f"{g.x}x{g.y}", note="L1-interleaved z [b, n^2, 384] bf16 and a linear to [n^2, 768] in L1")
    w = up(torch.randn(1, 1, 384, 768) / 20, dev)
    cfg = ckc(dev, "HiFi2")
    for b in (1, 2, 4, 6, 8, 12, 16):
        try:
            z = up(torch.randn(1, b, n * n, 384), dev, mem=l1)
            s = timed(lambda: ttnn.linear(z, w, compute_kernel_config=cfg, dtype=ttnn.bfloat16, memory_config=l1),
                      dev, 5)
            zd = up(torch.randn(1, b, n * n, 384), dev)
            sd = timed(lambda: ttnn.linear(zd, w, compute_kernel_config=cfg, dtype=ttnn.bfloat16), dev, 5)
            emit(q=3, n=n, batch=b, z_mb=b * n * n * 384 * 2 / 1e6, fits=True, s_l1=s, s_dram=sd,
                 tflops_l1=2 * b * n * n * 384 * 768 / s / 1e12, tflops_dram=2 * b * n * n * 384 * 768 / sd / 1e12)
            ttnn.deallocate(z)
            ttnn.deallocate(zd)
        except Exception as e:  # noqa: BLE001
            emit(q=3, n=n, batch=b, z_mb=b * n * n * 384 * 2 / 1e6, fits=False, error=str(e)[:200])
            break


def q4_trace_floor(dev):
    x = up(torch.randn(1, 1, 32, 32), dev)
    for nops in (100, 1000):
        def chain():
            y = x
            for _ in range(nops):
                y2 = ttnn.add(y, 1.0)
                if y is not x:
                    ttnn.deallocate(y)
                y = y2
            return y
        eager = timed(chain, dev, 3)
        tid = ttnn.begin_trace_capture(dev, cq_id=0)
        out = chain()
        ttnn.end_trace_capture(dev, tid, cq_id=0)
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(10):
            ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(dev)
        tr = (time.perf_counter() - t0) / 10
        ttnn.release_trace(dev, tid)
        ttnn.deallocate(out)
        emit(q=4, op="ttnn.add on one tile, chained", ops=nops, eager_us_per_op=eager / nops * 1e6,
             trace_us_per_op=tr / nops * 1e6)
    # one RFD4 trunk block + one pair update at n=128, eager vs trace
    import sys
    sys.path.insert(0, "tests")
    import rfd4_reference as REF
    from tt_bio.rfd4 import trunk as TR
    w = REF.trunk_weights(2, seed=0)
    mod = TR.Trunk(T.WeightScope({k: v.float() for k, v in w.items()}), TR.compute_kernel_config(), 2, 16,
                   residual_dtype=ttnn.float32)
    mask = torch.ones(1, 128)
    masks = TR.trunk_masks(mask, dev, residual_dtype=ttnn.float32)
    s0 = up(torch.randn(1, 128, 1024), dev, ttnn.float32)
    z0 = up(torch.randn(1, 128, 128, 384), dev, ttnn.float32)
    c = up(torch.randn(1, 128, 256), dev)

    def one():
        s, z = mod.run_pass(ttnn.clone(s0), ttnn.clone(z0), c, masks)
        ttnn.deallocate(z)
        return s
    eager = timed(one, dev, 3)
    try:
        tid = ttnn.begin_trace_capture(dev, cq_id=0)
        out = one()
        ttnn.end_trace_capture(dev, tid, cq_id=0)
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(10):
            ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(dev)
        tr = (time.perf_counter() - t0) / 10
        ttnn.release_trace(dev, tid)
        ttnn.deallocate(out)
    except Exception as e:  # noqa: BLE001
        tr = None
        emit(q=4, op="trace capture of 2 blocks + 1 pair update", error=str(e)[:300])
    emit(q=4, op="RFD4 2 trunk blocks + 1 pair update, n=128, fp32 streams", eager_ms=eager * 1e3,
         trace_ms=None if tr is None else tr * 1e3)


def q5_dit_pair_bias(dev):
    from tt_bio import dit_pair_bias as D
    for cz in (128, 384):
        z = up(torch.randn(1, 128, 128, cz), dev)
        wt = up(torch.randn(1, 1, 32, cz), dev)  # 16 heads pad to one 32-row tile
        m = up(torch.zeros(1, 1, 128, 128), dev)
        kt, ht = cz // 32, 1
        emit(q=5, c_z=cz, eligible=bool(D.eligible(z, wt, m)), l1_bytes=D._l1_bytes(kt, ht),
             budget=D._L1_BUDGET, z_cb_bytes=2 * 32 * kt * D._TILE)


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--only", default="1,2,3,4,5")
    ap.add_argument("--ns", default="128,256,512")
    a = ap.parse_args()
    OUT = a.out
    ns = [int(x) for x in a.ns.split(",")]
    dev = T.get_device(trace="rf3")
    emit(q=0, arch=str(dev.arch()), grid=str(dev.compute_with_storage_grid_size()))
    for q in a.only.split(","):
        try:
            {"1": lambda: q1_pair_matmul(dev, ns), "2": lambda: q2_tri_contraction(dev, ns),
             "3": lambda: q3_l1_fit(dev), "4": lambda: q4_trace_floor(dev), "5": lambda: q5_dit_pair_bias(dev)}[q]()
        except Exception as e:  # noqa: BLE001
            emit(q=int(q), fatal=str(e)[:400])


if __name__ == "__main__":
    main()
