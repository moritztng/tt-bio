"""RFD4 pair recycle projection: the shipped `Module._lin` call against `tok_recycle_mm`
(`_pair_proj_minimal_matmul`), torch.equal and wall time per call (synchronised, warm), with AICLK.

    TT_VISIBLE_DEVICES=<card> python3 perf/rfd4/tok512/rec_check.py --out <json> [--shapes 1x512 10x224 4x256 16x128]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from tt_bio.main import ensure_p300_mesh_descriptor

ensure_p300_mesh_descriptor()

import ttnn  # noqa: E402

import clocksample  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--shapes", nargs="+", default=["1x512", "10x224", "4x256", "16x128"])
    ap.add_argument("--reps", type=int, default=20)
    a = ap.parse_args()

    from tt_bio import ops
    from tt_bio.rfd4.trunk import compute_kernel_config
    from tt_bio.tenstorrent import CORE_GRID_MAIN, _pair_proj_minimal_matmul, get_device

    dev = get_device()
    ckc = compute_kernel_config()
    g = torch.Generator().manual_seed(0)
    up = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,  # noqa: E731
                                   memory_config=ttnn.DRAM_MEMORY_CONFIG)
    wt = up(torch.randn(384, 384, generator=g) / 384 ** 0.5)
    arms = {"linear": lambda x: ops.linear(x, wt, bias=None, compute_kernel_config=ckc, dtype=ttnn.bfloat16,
                                           core_grid=CORE_GRID_MAIN),
            "mm": lambda x: _pair_proj_minimal_matmul(x, wt, ckc, ttnn.bfloat16)}
    res = []
    with clocksample.during() as clk:
        for sh in a.shapes:
            b, n = map(int, sh.split("x"))
            xt = up(torch.randn(b, n, n, 384, generator=g))
            out, us = {}, {}
            for name, fn in arms.items():
                y = fn(xt)
                if y is None:
                    out[name] = None
                    continue
                out[name] = ttnn.to_torch(y)
                ttnn.deallocate(y)
            for _ in range(2):   # interleaved rounds
                for name, fn in arms.items():
                    if out[name] is None:
                        continue
                    ttnn.synchronize_device(dev)
                    t0 = time.perf_counter()
                    for _ in range(a.reps):
                        ttnn.deallocate(fn(xt))
                    ttnn.synchronize_device(dev)
                    us.setdefault(name, []).append((time.perf_counter() - t0) / a.reps * 1e6)
            r = {"shape": sh, "us": {k: min(v) for k, v in us.items()}, "mm_declined": out["mm"] is None,
                 "equal": out["mm"] is not None and bool(torch.equal(out["mm"], out["linear"]))}
            res.append(r)
            print(json.dumps(r), flush=True)
            ttnn.deallocate(xt)
    Path(a.out).write_text(json.dumps(dict(rows=res, aiclk=clk.summary(), load=clk.load and max(clk.load)), indent=1))
    print(clk.line())


if __name__ == "__main__":
    main()
