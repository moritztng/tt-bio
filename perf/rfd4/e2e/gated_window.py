"""The gated channel move against the four-way split it replaces, on an L1 projection, across N and C.

    TT_VISIBLE_DEVICES=<card> python perf/rfd4/e2e/gated_window.py --out <file.json>

Arm A is what `TriangleMultiplication` runs where `reblock_permute.eligible_gated` declines: chunk the
[1, N, N, 4C] projection four ways, `p * sigmoid(g)` twice, and two `_channel_move`s to [1, C, N, N].
Arm B is two `reblock_permute_gated` calls. Both on L1, interleaved A/B per repeat, synced per arm, median
over --repeats; B's two outputs are checked `torch.equal` against A's. Shapes whose L1 footprint does
not fit are skipped. AICLK is sampled throughout.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # perf/ for clocksample

import torch

from tt_bio.main import ensure_p300_mesh_descriptor

ensure_p300_mesh_descriptor()

import ttnn  # noqa: E402

import clocksample  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ns", default="64,96,128,160,192,224,256,288")
    ap.add_argument("--cs", default="32,64,128,256")
    ap.add_argument("--repeats", type=int, default=20)
    a = ap.parse_args()

    from tt_bio import reblock_permute as RP
    from tt_bio import tenstorrent as T

    dev = T.get_device()
    L1 = ttnn.L1_MEMORY_CONFIG
    sig = [ttnn.UnaryOpType.SIGMOID]

    def split(xw, C):
        q = ttnn.chunk(xw, 4, dim=-1)
        outs = []
        for p, g in ((q[0], q[1]), (q[2], q[3])):
            ab = ttnn.multiply_(p, g, input_tensor_b_activations=sig)
            mv = T._channel_move(ab, L1)
            ttnn.deallocate(ab)
            outs.append(mv)
        for t in q:
            ttnn.deallocate(t)
        return outs

    def gated(xw, C):
        return [RP.reblock_permute_gated(xw, 0, C, C, memory_config=L1),
                RP.reblock_permute_gated(xw, 2 * C, 3 * C, C, memory_config=L1)]

    rows = []
    with clocksample.during(period=1.0) as clk:
        for n in map(int, a.ns.split(",")):
            for C in map(int, a.cs.split(",")):
                row = {"n": n, "c": C}
                # projection + two outputs + split's four slices
                if 2 * n * n * C * (4 + 2 + 4) > 120e6:
                    row["skip"] = "l1"
                    rows.append(row)
                    continue
                torch.manual_seed(n * 1000 + C)
                xw = ttnn.from_torch(torch.randn(1, n, n, 4 * C), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                     device=dev, memory_config=L1)
                try:
                    ra, rb = split(xw, C), gated(xw, C)
                    ttnn.synchronize_device(dev)
                    row["bit_identical"] = all(torch.equal(ttnn.to_torch(x), ttnn.to_torch(y)) for x, y in zip(ra, rb))
                    for t in ra + rb:
                        ttnn.deallocate(t)
                    ts = {"split": [], "gated": []}
                    for _ in range(a.repeats):
                        for name, fn in (("split", split), ("gated", gated)):
                            ttnn.synchronize_device(dev)
                            t0 = time.perf_counter()
                            outs = fn(xw, C)
                            ttnn.synchronize_device(dev)
                            ts[name].append(time.perf_counter() - t0)
                            for t in outs:
                                ttnn.deallocate(t)
                    for k, v in ts.items():
                        row[f"{k}_ms"] = round(statistics.median(v) * 1e3, 4)
                    row["speedup"] = round(row["split_ms"] / row["gated_ms"], 3)
                    row["eligible_today"] = RP.eligible_gated(xw, C, L1)
                except Exception as exc:  # a shape that does not run is a result
                    row["error"] = str(exc)[:300]
                ttnn.deallocate(xw)
                row["load"] = os.getloadavg()[0]
                print(json.dumps(row), flush=True)
                rows.append(row)
    json.dump({"card": os.environ.get("TT_VISIBLE_DEVICES"), "grid": list(T.COMPUTE_GRID_MAIN),
               "aiclk": clk.summary(), "rows": rows}, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
