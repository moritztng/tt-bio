#!/usr/bin/env python3
"""z-silu -- the production number: Transition block wall + fold wall + plDDT, one arm per process.

The arm IS the JIT header, so it cannot be flipped inside a live process the way `_UNFUSED_SILU`
could. Each process measures one arm; the driver alternates A, P2a, A, P2a so cross-process drift
is bracketed rather than assumed away.

    TT_VISIBLE_DEVICES=2 ... python3 z_fold.py --arm A --out a.json
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))

import numpy as np, torch, ttnn


def load():
    return [round(v, 2) for v in os.getloadavg()]


def med(v):
    return sorted(v)[len(v) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--block-reps", type=int, default=9)
    ap.add_argument("--folds", type=int, default=2)
    ap.add_argument("--census", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import tt_bio.tenstorrent as T
    import tt_baseline as B

    res = dict(arm=a.arm, host="qb2", card="physical 2", ttnn="0.68.0", load_start=load())

    grab = {}
    _orig_tr = T.Transition.__call__

    def _grab(self, x):
        if "inst" not in grab and len(x.shape) == 4 and int(x.shape[-1]) == 256:
            grab["inst"], grab["x"] = self, ttnn.to_torch(x).clone()
        return _orig_tr(self, x)

    T.Transition.__call__ = _grab

    t0 = time.perf_counter()
    one_fold, meta, state = B.build_fold(
        "protenix-v2", ROOT / ".msa_zsilu", ROOT / "examples/prot300.yaml",
        ROOT / "scripts/gpu_vs_tt/fixtures/prot300.a3m")
    print(f"model loaded {time.perf_counter()-t0:.1f}s", flush=True)

    cold_s, cold_m = one_fold()
    assert cold_m.get("msa"), "fold ran without an MSA"
    res["cold_s"] = round(cold_s, 3)
    res["n_tokens"] = cold_m.get("n_tokens")
    res["plddt_cold"] = cold_m.get("plddt")
    T.Transition.__call__ = _orig_tr
    print("cold", res["cold_s"], "plddt", res["plddt_cold"], flush=True)

    # --- fold wall, untimed-instrumentation-free ----------------------------------------------
    fw, pl = [], []
    for _ in range(a.folds):
        t, m = one_fold()
        fw.append(round(t, 4))
        pl.append(m.get("plddt"))
        print("fold", fw[-1], m.get("plddt"), load(), flush=True)
    res["fold_s"] = fw
    res["fold_med_s"] = med(fw)
    res["plddt"] = pl

    # --- Transition block wall at the fold's own pair shape --------------------------------------
    dev = T.get_device()
    inst = grab["inst"]
    xz = ttnn.from_torch(grab["x"], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                         memory_config=ttnn.DRAM_MEMORY_CONFIG)
    res["block_shape"] = list(grab["x"].shape)

    def block():
        ttnn.synchronize_device(dev)
        t = time.perf_counter()
        y = _orig_tr(inst, xz)
        ttnn.synchronize_device(dev)
        dt = time.perf_counter() - t
        ttnn.deallocate(y)
        return dt

    block(); block()
    bw = [round(block() * 1e6, 2) for _ in range(a.block_reps)]
    res["block_wall_us"] = bw
    res["block_med_us"] = round(med(bw), 2)
    print("block us", res["block_med_us"], flush=True)

    # --- the census: how many fc1 silu calls a real fold issues, counted not assumed ------------
    if a.census:
        cnt = Counter()
        _lin = ttnn.linear

        def _count(x, w, *ar, **kw):
            if kw.get("activation") == "silu":
                cnt[(tuple(int(v) for v in x.shape), tuple(int(v) for v in w.shape))] += 1
            return _lin(x, w, *ar, **kw)

        ttnn.linear = _count
        T.ttnn.linear = _count
        try:
            one_fold()
        finally:
            ttnn.linear = _lin
            T.ttnn.linear = _lin
        res["silu_linear_census"] = sorted(
            ({"x": list(k[0]), "w": list(k[1]), "calls": v} for k, v in cnt.items()),
            key=lambda r: -r["calls"])
        res["silu_linear_calls_total"] = sum(cnt.values())
        print("census", res["silu_linear_calls_total"], flush=True)

    res["load_end"] = load()
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, default=str)
    print("wrote", a.out, flush=True)


main()
