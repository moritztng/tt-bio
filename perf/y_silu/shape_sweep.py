#!/usr/bin/env python3
"""y-silu -- the other five Transition construction sites, at their censused shapes.

The Transition block wall prices the c_z=256 pair track only. The census found `activation="silu"`
at five more (site, shape) classes, and the same unfuse applies to every one of them. Each is priced
here as arm A minus arm B at its own shape times its own censused call count -- an estimate, labelled
as one, and cross-checked against the fold wall rather than substituted for it.

    TT_VISIBLE_DEVICES=0 python3 perf/y_silu/shape_sweep.py --out perf/y_silu/shapes.json
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import numpy as np, torch, ttnn


def med(v):
    return sorted(v)[len(v) // 2]


def timed(dev, fn, k, reps=7, warm=2):
    for _ in range(warm):
        fn()
    ttnn.synchronize_device(dev)
    out = []
    for _ in range(reps):
        ttnn.synchronize_device(dev)
        t = time.perf_counter()
        for _ in range(k):
            fn()
        ttnn.synchronize_device(dev)
        out.append((time.perf_counter() - t) / k * 1e6)
    return round(med(out), 3)


# (label, x shape, weight shape, calls/fold from the live census)
CASES = [
    ("transition_z c_z=256 h=30  (tenstorrent.py:2520)", (1, 30, 298, 256), (256, 1024), 4716),
    ("transition_z c_z=256 h=28  (tenstorrent.py:2520)", (1, 28, 298, 256), (256, 1024), 524),
    ("transition_z template c=64 (tenstorrent.py:2520)", (1, 30, 298, 64), (64, 128), 800),
    ("transition_s c_s=384       (tenstorrent.py:2536)", (1, 298, 384), (384, 1536), 484),
    ("protenix.py:758  c_s=384",                          (1, 298, 384), (384, 768), 400),
    ("protenix.py:2035 msa c=128",                        (1, 18, 298, 128), (128, 512), 60),
    ("protenix.py:1766 c=256",                            (1, 30, 298, 256), (256, 512), 20),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "shapes.json"))
    ap.add_argument("--k", type=int, default=10)
    a = ap.parse_args()

    import tt_bio.tenstorrent as T
    dev = T.get_device()
    L1 = ttnn.L1_MEMORY_CONFIG
    CG = T.CORE_GRID_MAIN
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
        packer_l1_acc=True)
    torch.manual_seed(0)
    rows = []
    for label, xs, ws, calls in CASES:
        r = dict(label=label, x=list(xs), w=list(ws), calls_per_fold=calls)
        try:
            x = ttnn.from_torch(torch.randn(*xs, dtype=torch.bfloat16), dtype=ttnn.bfloat16,
                                layout=ttnn.TILE_LAYOUT, device=dev, memory_config=L1)
            w = ttnn.from_torch(torch.randn(*ws, dtype=torch.bfloat16) * 0.05, dtype=ttnn.bfloat16,
                                layout=ttnn.TILE_LAYOUT, device=dev, memory_config=L1)

            def lin(act):
                return ttnn.linear(x, w, activation=act, compute_kernel_config=ckc,
                                   memory_config=L1, dtype=ttnn.bfloat16, core_grid=CG)

            def arm_a():
                y = lin("silu"); ttnn.deallocate(y)

            def arm_b():
                y = lin(None); z = ttnn.silu(y, memory_config=L1, output_tensor=y)
                ttnn.deallocate(z)

            def arm_d():
                y = lin(None); ttnn.deallocate(y)

            r["A_fused_us"] = timed(dev, arm_a, a.k)
            r["B_unfused_us"] = timed(dev, arm_b, a.k)
            r["D_bare_us"] = timed(dev, arm_d, a.k)
            r["win_us"] = round(r["A_fused_us"] - r["B_unfused_us"], 3)
            r["fused_silu_cost_us"] = round(r["A_fused_us"] - r["D_bare_us"], 3)
            r["standalone_silu_us"] = round(r["B_unfused_us"] - r["D_bare_us"], 3)
            r["ms_per_fold"] = round(r["win_us"] * calls / 1e3, 2)
            ttnn.deallocate(x); ttnn.deallocate(w)
        except Exception as e:
            r["error"] = repr(e)[:300]
        print(json.dumps(r), flush=True)
        rows.append(r)
    total = round(sum(r.get("ms_per_fold", 0.0) for r in rows), 1)
    print("TOTAL ms/fold across all sites:", total, flush=True)
    with open(a.out, "w") as f:
        json.dump(dict(rows=rows, total_ms_per_fold=total, load=os.getloadavg()), f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
