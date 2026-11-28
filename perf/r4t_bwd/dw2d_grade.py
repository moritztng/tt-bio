#!/usr/bin/env python3
"""B2 site float64 grade for `tt_bio.mm_layout.DW_2D` + ragged `_split_k` on Blackhole.

Pre-registered bar:
  - Every RFD4 main dW shape (K=65536, 7 shapes) served by DW_2D (`served >= 7`, `declined == 0`),
    `rel_L2(dw2d, f64) <= 1.05 * rel_L2(stock, f64)`,
    `(1 - cos(dw2d, f64)) <= 1.21 * max(1 - cos(stock, f64), 1e-12)` (D54 clause),
    and `BREAK=1` FAILS.
  - Every ragged/small dW shape (`384x24`, `384x12`) served by `_split_k`, same rel/cos bar,
    and `TT_BIO_SPLIT_K_BROKEN=1` FAILS.
  - Samples AICLK DURING the run.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def sample_aiclk(card: int, stop: threading.Event, out: list):
    path = f"/sys/class/tenstorrent/tenstorrent!{card}/tt_aiclk"
    while not stop.is_set():
        try:
            out.append(int(open(path).read().split()[0]))
        except Exception:
            pass
        time.sleep(0.05)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=6)
    a = ap.parse_args()

    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import torch
    import ttnn
    from tt_bio import autograd as ag, mm_layout as mml, tenstorrent as T

    brk = os.environ.get("BREAK") == "1"
    dev = T.get_device()
    card = int(os.environ["TT_VISIBLE_DEVICES"].split(",")[0])
    cfg = ag.precise_config()
    tile = dict(layout=ttnn.TILE_LAYOUT, device=dev)
    sha = subprocess.run(["git", "rev-parse", "--short=9", "HEAD"], capture_output=True, text=True).stdout.strip()
    clk, stop = [], threading.Event()
    thr = threading.Thread(target=sample_aiclk, args=(card, stop, clk), daemon=True)
    thr.start()

    K = 65536
    MAIN_SHAPES = [(256, 384), (384, 384), (384, 768), (384, 1024), (768, 384), (1024, 384), (384, 256)]
    RAGGED_SHAPES = [(384, 24), (384, 12)]

    def timed(fn):
        o = fn()
        ttnn.synchronize_device(dev)
        ttnn.deallocate(o)
        ts = []
        for i in range(a.reps):
            t0 = time.perf_counter()
            o = fn()
            ttnn.synchronize_device(dev)
            ts.append((time.perf_counter() - t0) * 1e3)
            if i < a.reps - 1:
                ttnn.deallocate(o)
        ts.sort()
        return ts[len(ts) // 2], o

    def metrics(tt_t, ref64, scale=1.0):
        h = ttnn.to_torch(tt_t).float().double().reshape(ref64.shape) * scale
        rel = float((h - ref64).norm() / ref64.norm())
        omc = float(1.0 - (h * ref64).sum() / (h.norm() * ref64.norm()))
        return rel, omc

    mml.DW_2D_REACH.clear()
    rows = []
    for M, N in MAIN_SHAPES:
        g = torch.Generator().manual_seed(101 + M + N)
        ah = torch.randn(K, M, generator=g).bfloat16()
        bh = torch.randn(K, N, generator=g).bfloat16()
        ref64 = ah.double().t() @ bh.double()
        A = ttnn.from_torch(ah, dtype=ttnn.bfloat16, **tile)
        B = ttnn.from_torch(bh, dtype=ttnn.bfloat16, **tile)

        mml.DW_2D = False
        old_sk = ag.SPLIT_K_OUT_TILES
        ag.SPLIT_K_OUT_TILES = 0
        t_stk, o_stk = timed(lambda: ag._matmul(A, B, transpose_a=True, dtype=ttnn.float32, compute_kernel_config=cfg))
        r_stk, c_stk = metrics(o_stk, ref64)
        ttnn.deallocate(o_stk)

        mml.DW_2D = True
        ag.SPLIT_K_OUT_TILES = old_sk
        t_2d, o_2d = timed(lambda: ag._matmul(A, B, transpose_a=True, dtype=ttnn.float32, compute_kernel_config=cfg))
        r_2d, c_2d = metrics(o_2d, ref64, scale=1.10 if brk else 1.0)
        ttnn.deallocate(o_2d)
        ttnn.deallocate(A)
        ttnn.deallocate(B)

        ok = (r_2d <= 1.05 * r_stk) and (c_2d <= 1.21 * max(c_stk, 1e-12))
        rows.append(dict(kind="dw2d", M=M, N=N, stock_ms=round(t_stk, 3), fast_ms=round(t_2d, 3),
                         speedup=round(t_stk / t_2d, 2), rel_stk=r_stk, rel_fast=r_2d,
                         omc_stk=c_stk, omc_fast=c_2d, ok=ok))

    ag.SPLIT_K_STATS["served"] = 0
    ragged_rows = []
    for M, N in RAGGED_SHAPES:
        g = torch.Generator().manual_seed(201 + M + N)
        ah = torch.randn(K, M, generator=g).bfloat16()
        bh = torch.randn(K, N, generator=g).bfloat16()
        ref64 = ah.double().t() @ bh.double()
        A = ttnn.from_torch(ah, dtype=ttnn.bfloat16, **tile)
        B = ttnn.from_torch(bh, dtype=ttnn.bfloat16, **tile)

        ag.SPLIT_K_OUT_TILES = 0
        t_stk, o_stk = timed(lambda: ag._matmul(A, B, transpose_a=True, dtype=ttnn.float32, compute_kernel_config=cfg))
        r_stk, c_stk = metrics(o_stk, ref64)
        ttnn.deallocate(o_stk)

        ag.SPLIT_K_OUT_TILES = 32
        ag.SPLIT_K_BROKEN = brk
        t_sk, o_sk = timed(lambda: ag._matmul(A, B, transpose_a=True, dtype=ttnn.float32, compute_kernel_config=cfg))
        r_sk, c_sk = metrics(o_sk, ref64)
        ttnn.deallocate(o_sk)
        ttnn.deallocate(A)
        ttnn.deallocate(B)

        ok = (r_sk <= 1.05 * r_stk) and (c_sk <= 1.21 * max(c_stk, 1e-12))
        ragged_rows.append(dict(kind="split_k", M=M, N=N, stock_ms=round(t_stk, 3), fast_ms=round(t_sk, 3),
                                speedup=round(t_stk / t_sk, 2), rel_stk=r_stk, rel_fast=r_sk,
                                omc_stk=c_stk, omc_fast=c_sk, ok=ok))
    ag.SPLIT_K_BROKEN = False

    stop.set()
    thr.join(timeout=2)
    served = mml.DW_2D_REACH.get("served", 0)
    declined = sum(v for k, v in mml.DW_2D_REACH.items() if k != "served")
    sk_served = ag.SPLIT_K_STATS.get("served", 0)
    verdict = "PASS" if (all(r["ok"] for r in rows + ragged_rows) and served >= len(MAIN_SHAPES)
                         and declined == 0 and sk_served >= len(RAGGED_SHAPES)) else "FAIL"
    doc = dict(host=os.uname().nodename, chip=card, sha=sha,
               aiclk=[min(clk or [0]), max(clk or [0])],
               aiclk_med=sorted(clk)[len(clk) // 2] if clk else 0,
               brk=brk, verdict=verdict, dw2d_served=served, dw2d_declined=declined,
               split_k_served=sk_served,
               speedup_min=min(r["speedup"] for r in rows),
               speedup_max=max(r["speedup"] for r in rows),
               rows=rows, ragged_rows=ragged_rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(doc, indent=1))
    print("DW2D_GRADE " + json.dumps({k: v for k, v in doc.items() if k not in ("rows", "ragged_rows")}), flush=True)
    for r in rows + ragged_rows:
        print(f"  {r['kind']:7s} {r['M']:4d}x{r['N']:4d}: {r['stock_ms']:6.3f} -> {r['fast_ms']:6.3f} ms "
              f"({r['speedup']:4.2f}x) rel {r['rel_stk']:.5f}->{r['rel_fast']:.5f} ok={r['ok']}", flush=True)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
