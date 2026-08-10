#!/usr/bin/env python3
"""y-silu-lowering round 3 -- settle the gelu arm, which disagreed between rounds 1 and 2.

Round 1 measured the fused-gelu penalty at 138.6 us/call (matching y-silu's 141.3); round 2 measured
221.5 at the same config, while every other arm agreed to 3 %. Arms are alternated here rather than
run in blocks, three passes, so a drift or an ordering effect shows up as spread rather than as a
number.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import torch, ttnn


def med(v):
    return sorted(v)[len(v) // 2]


def timed(dev, fn, k, reps=5, warm=2):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "lowering3.json"))
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--passes", type=int, default=3)
    a = ap.parse_args()
    import tt_bio.tenstorrent as T
    dev = T.get_device()
    L1, CG = ttnn.L1_MEMORY_CONFIG, T.CORE_GRID_MAIN
    ckc = {"on": ttnn.init_device_compute_kernel_config(
               dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
               packer_l1_acc=True),
           "off": ttnn.init_device_compute_kernel_config(
               dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=False,
               packer_l1_acc=True)}
    torch.manual_seed(0)
    xt = torch.randn(1, 30, 298, 256, dtype=torch.bfloat16)
    wt = (torch.randn(256, 1024) * 0.05).to(torch.bfloat16)
    x = ttnn.from_torch(xt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=L1)
    w = ttnn.from_torch(wt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=L1)

    arms = [(acc, act) for acc in ("on", "off") for act in (None, "silu", "gelu")]
    rows = {f"{acc}_{act}": [] for acc, act in arms}
    for p in range(a.passes):
        for acc, act in arms:
            def fn(act=act, acc=acc):
                y = ttnn.linear(x, w, activation=act, compute_kernel_config=ckc[acc],
                                memory_config=L1, dtype=ttnn.bfloat16, core_grid=CG)
                ttnn.deallocate(y)
            rows[f"{acc}_{act}"].append(timed(dev, fn, a.k))
        print("pass", p, {k: v[-1] for k, v in rows.items()},
              "load", [round(v, 2) for v in os.getloadavg()], flush=True)
    res = {"raw": rows, "median": {k: med(v) for k, v in rows.items()},
           "load_end": [round(v, 2) for v in os.getloadavg()]}
    m = res["median"]
    res["penalty"] = {f"{acc}_{act}": round(m[f"{acc}_{act}"] - m[f"{acc}_None"], 3)
                      for acc in ("on", "off") for act in ("silu", "gelu")}
    print("median", json.dumps(m), flush=True)
    print("penalty", json.dumps(res["penalty"]), flush=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    print("wrote", a.out, flush=True)


main()
