"""ln_split vs ttnn.layer_norm on the ESMC-6B's [544, 2560] (and nearby few-row shapes): ms a call and error vs
float64, with and without the affine.

    python3 perf/spd_m_esmc/lns_bench.py --out lns.json
"""
import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import ttnn

from tt_bio import ln_split
from tt_bio.tenstorrent import get_device

SHAPES = [(1, 544, 2560), (1, 256, 2560), (1, 1024, 2560), (1, 544, 1152)]


def ms(fn, dev, n=50):
    for _ in range(3):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(dev)
    t = []
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(n):
            ttnn.deallocate(fn())
        ttnn.synchronize_device(dev)
        t.append((time.perf_counter() - t0) * 1000 / n)
    return round(statistics.median(t), 4)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = get_device()
    ck = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                          fp32_dest_acc_en=True, packer_l1_acc=False)
    torch.manual_seed(0)
    rows = []
    for shp in SHAPES:
        k = shp[-1]
        xs = (torch.randn(shp) * 2.0 + torch.randn(1, 1, k) * 0.5 + 0.3).to(torch.bfloat16)
        gs, bs = (torch.randn(k) * 0.3 + 1).to(torch.bfloat16), (torch.randn(k) * 0.1).to(torch.bfloat16)
        x = ttnn.from_torch(xs, device=dev, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        tt = lambda v: ttnn.from_torch(v.view(1, k), device=dev, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)  # noqa: E731
        gw, gb = tt(gs), tt(bs)
        for aff in ("wb", "w", "none"):
            w = gw if aff in ("wb", "w") else None
            b = gb if aff == "wb" else None
            ref = torch.nn.functional.layer_norm(xs.double(), (k,), gs.double() if w is not None else None,
                                                 bs.double() if b is not None else None, 1e-5)
            row = {"shape": shp, "affine": aff}
            kw = {"weight": w, "bias": b}
            stock = ttnn.to_torch(ttnn.layer_norm(x, epsilon=1e-5, compute_kernel_config=ck, **kw)).double()
            got = ln_split.layer_norm(x, w, b, 1e-5)
            row["g"] = ln_split.plan(int(x.volume()) // k // 32, k // 32,
                                     dev.compute_with_storage_grid_size().x * dev.compute_with_storage_grid_size().y)
            err = lambda a: round(float((a - ref).norm() / ref.norm()), 6)  # noqa: E731
            row["rel_l2_stock"] = err(stock)
            row["ms_stock"] = ms(lambda: ttnn.layer_norm(x, epsilon=1e-5, compute_kernel_config=ck, **kw), dev)
            if got is not None:
                g = ttnn.to_torch(got).double()
                row["rel_l2_split"] = err(g)
                row["max_abs_split_vs_stock"] = round(float((g - stock).abs().max()), 5)
                row["ms_split"] = ms(lambda: ln_split.layer_norm(x, w, b, 1e-5), dev)
                # repeat-launch consistency: the semaphores reset between launches
                again = ttnn.to_torch(ln_split.layer_norm(x, w, b, 1e-5)).double()
                row["repeat_equal"] = bool(torch.equal(again, g))
            rows.append(row)
            print(json.dumps(row), flush=True)
    Path(args.out).write_text(json.dumps(rows, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
