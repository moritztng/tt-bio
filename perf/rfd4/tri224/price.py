"""rfd4-perf-tri224 run1: price the batched triangle's move back and the contraction's output write at main's widths.

    TT_VISIBLE_DEVICES=<chip> python perf/rfd4/tri224/price.py --out <file.json> [--shapes 224x10,192x14] [--reps 30]

Per shape [B, 256, N, N]: `permute_back_pk` (bf16 in and out) with its stage ablations (ABL 1 no DRAM read, 2 no re-tile,
4 no DRAM write), and the contraction as RFD4 runs it (ttnn.matmul, bfp8 a / b, HiFi4) writing bf16 vs bfp8. Medians of
synced calls, AICLK sampled during.
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # perf/ for clocksample

from tt_bio.main import ensure_p300_mesh_descriptor

ensure_p300_mesh_descriptor()

import clocksample  # noqa: E402
import torch  # noqa: E402
import ttnn  # noqa: E402
from tt_bio import permute_back_pk as PK  # noqa: E402
from tt_bio import tenstorrent as T  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--shapes", default="224x10,192x14")
ap.add_argument("--reps", type=int, default=30)
a = ap.parse_args()

dev = T.get_device()
grid = tuple(T.COMPUTE_GRID_MAIN)
DRAM = ttnn.DRAM_MEMORY_CONFIG
ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                       fp32_dest_acc_en=False, packer_l1_acc=True)


def timed(f):
    f()
    ttnn.synchronize_device(dev)
    ms = []
    for _ in range(a.reps):
        t0 = time.perf_counter()
        f()
        ttnn.synchronize_device(dev)
        ms.append((time.perf_counter() - t0) * 1e3)
    return round(statistics.median(ms), 4)


recs = []
gen = torch.Generator().manual_seed(0)
with clocksample.during(period=0.5) as clk:
    for spec in a.shapes.split(","):
        n, B = (int(v) for v in spec.split("x"))
        rec = {"n": n, "batch": B}
        xt = torch.randn(B, 256, n, n, generator=gen)
        x = ttnn.from_torch(xt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
        out = ttnn.allocate_tensor_on_device(ttnn.Shape([B, n, n, 256]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, DRAM)
        for abl in (0, 1, 2, 4, 5):
            PK.ABL = abl
            rec[f"move_abl{abl}_ms"] = timed(lambda: PK.permute_back_pk(x, out, grid))
        PK.ABL = 0
        PK.permute_back_pk(x, out, grid)
        rec["move_exact"] = bool(torch.equal(ttnn.to_torch(out), xt.to(torch.bfloat16).permute(0, 2, 3, 1)))
        ttnn.deallocate(x)
        ttnn.deallocate(out)
        at = torch.randn(B, 256, n, n, generator=gen)
        bt = torch.randn(B, 256, n, n, generator=gen)
        ad = ttnn.from_torch(at, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
        bd = ttnn.from_torch(bt, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
        for name, dt in (("bf16", ttnn.bfloat16), ("bfp8", ttnn.bfloat8_b)):
            def mm(dt=dt):
                ttnn.deallocate(ttnn.matmul(ad, bd, transpose_b=True, dtype=dt, memory_config=DRAM,
                                            compute_kernel_config=ckc))
            rec[f"mm_out_{name}_ms"] = timed(mm)
        ttnn.deallocate(ad)
        ttnn.deallocate(bd)
        print(json.dumps(rec), flush=True)
        recs.append(rec)
json.dump({"records": recs, "aiclk": clk.summary(), "grid": grid}, open(a.out, "w"), indent=1)
print(json.dumps({"aiclk": clk.summary()}), flush=True)
