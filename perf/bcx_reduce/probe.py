#!/usr/bin/env python3
"""bcx-reduce: the three backward levers, each graded against float64 on its own tensors.

  reduce  `autograd._reduce_to` over a leading axis, `ttnn.sum` (TREE_REDUCE off) against
          `_tree_sum` (on), plus `_sum_leading`'s own 2-D reduce, on fp32 inputs
  permute `taped_ttnn._permute_back` against `ttnn.permute` for the two channel-move inverses,
          bit for bit, at the trunk's shapes and at one the gate declines
  fanin   k bf16 contributions accumulated by `Tensor.add_grad`, FANIN_MIXED off against on

Every arm's rel L2 is against a float64 computation on the same inputs, never against the other
arm. Timings are synced wall medians per call on one card, AICLK sampled during them.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "perf" / "bcx_reduce"


def rel(x, ref):
    return float((x.double() - ref).norm() / ref.norm())


def main():
    import ttnn
    from perf.bcx_stack.stack import Clock, sysfs_node
    from tt_bio import autograd as ag, taped_ttnn as T
    from tt_bio.tenstorrent import get_device
    dev = get_device()
    clock = Clock(dt=0.02)
    f32, bf = ttnn.float32, ttnn.bfloat16

    def up(t, dtype=f32):
        return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=dev)

    def timed(fn, reps=12):
        ttnn.deallocate(fn())
        ttnn.synchronize_device(dev)
        ts, t_a = [], time.time()
        for _ in range(reps):
            t0 = time.perf_counter()
            y = fn()
            ttnn.synchronize_device(dev)
            ts.append(time.perf_counter() - t0)
            ttnn.deallocate(y)
        span = [(t_a, time.time())]
        return {"min_ms": min(ts) * 1e3, "median_ms": float(np.median(ts)) * 1e3,
                "aiclk": clock.window(span), "load1": os.getloadavg()[0]}

    blob = {"pci": sysfs_node()[1], "ttnn": getattr(ttnn, "__version__", "?"),
            "reduce": [], "sum_leading": [], "permute": [], "fanin": []}
    gen = torch.Generator().manual_seed(0)

    # ---------------------------------------------------------------- reduce
    for shape, want in (([256, 4, 256, 256], [1, 4, 256, 256]), ([81, 4, 256, 256], [1, 4, 256, 256]),
                        ([128, 4, 128, 128], [1, 4, 128, 128]), ([13, 4, 256, 256], [1, 4, 256, 256]),
                        ([1, 96, 256, 256], [1, 1, 256, 256]), ([96, 1, 256, 256], [256, 256])):
        x = torch.randn(shape, generator=gen) * 1e-3
        pad = [1] * (len(shape) - len(want)) + want
        ref = x.double().sum(dim=[i for i in range(len(shape)) if pad[i] == 1 and shape[i] != 1],
                             keepdim=True).reshape(want)
        g = up(x)
        row = {"shape": shape, "to": want,
               "torch_f32": rel(x.sum(dim=[i for i in range(len(shape)) if pad[i] == 1 and shape[i] != 1]).reshape(want), ref)}
        outs = {}
        for arm, on in (("sum", False), ("tree", True)):
            ag.TREE_REDUCE = on
            y = ag._reduce_to(g, want)
            outs[arm] = ttnn.to_torch(y).float().reshape(want)
            ttnn.deallocate(y)
            row[arm] = {"rel_l2_vs_f64": rel(outs[arm], ref),
                        **timed(lambda: ag._reduce_to(g, want))}
        row["tree_vs_sum_rel_l2"] = rel(outs["tree"], outs["sum"].double())
        ag.TREE_REDUCE = True
        ttnn.deallocate(g)
        print("reduce", json.dumps(row), flush=True)
        blob["reduce"].append(row)

    # ---------------------------------------------------------------- _sum_leading
    for shape in ([256, 256, 128], [256, 256, 64], [64, 64, 32], [256, 1, 256, 384], [1, 32, 128],
                  [497, 128]):
        x = torch.randn(shape, generator=gen) * 1e-2
        ref = x.double().reshape(-1, shape[-1]).sum(0)
        g = up(x)
        row = {"shape": shape, "torch_f32": rel(x.reshape(-1, shape[-1]).sum(0), ref)}
        outs = {}
        for arm, on in (("sum", False), ("tree", True)):
            ag.TREE_REDUCE = on
            y = ag._sum_leading(g, [shape[-1]])
            outs[arm] = ttnn.to_torch(y).float().reshape(-1)
            ttnn.deallocate(y)
            row[arm] = {"rel_l2_vs_f64": rel(outs[arm], ref), **timed(lambda: ag._sum_leading(g, [shape[-1]]))}
        ag.TREE_REDUCE = True
        # the last-axis reduce, for the record: the same ttnn.sum on the axis inside the tile
        y = ttnn.sum(g, dim=-1, keepdim=True, compute_kernel_config=ag.precise_config())
        row["sum_last_axis_rel_l2_vs_f64"] = rel(ttnn.to_torch(y).float().reshape(x.sum(-1, keepdim=True).shape),
                                                 x.double().sum(-1, keepdim=True))
        ttnn.deallocate(y)
        ttnn.deallocate(g)
        print("sum_leading", json.dumps(row), flush=True)
        blob["sum_leading"].append(row)

    # ---------------------------------------------------------------- permute backward
    from tt_bio import reblock_permute as R
    for shape, inv in (([1, 64, 256, 256], [0, 2, 3, 1]), ([1, 256, 256, 64], [0, 3, 1, 2]),
                       ([1, 128, 256, 256], [0, 2, 3, 1]), ([1, 64, 128, 128], [0, 2, 3, 1]),
                       ([1, 256, 256, 4], [0, 3, 1, 2])):
        x = torch.randn(shape, generator=gen).bfloat16()
        g = up(x, bf)
        s0, s1 = R.STATS[0], R.STATS_BACK[0]
        T.REBLOCK_PERMUTE_BW = True
        a = ttnn.to_torch(T._permute_back(g, inv))
        served = (R.STATS[0] - s0) + (R.STATS_BACK[0] - s1)
        T.REBLOCK_PERMUTE_BW = False
        b = ttnn.to_torch(T._permute_back(g, inv))
        ref = x.permute(inv)
        row = {"shape": shape, "inv": inv, "reblock_served": bool(served),
               "bit_identical_to_permute": bool(torch.equal(a, b)),
               "bit_identical_to_torch": bool(torch.equal(a, ref))}
        for arm, on in (("permute", False), ("reblock", True)):
            T.REBLOCK_PERMUTE_BW = on
            row[arm] = timed(lambda: T._permute_back(g, inv))
        T.REBLOCK_PERMUTE_BW = True
        ttnn.deallocate(g)
        print("permute", json.dumps(row), flush=True)
        blob["permute"].append(row)

    # ---------------------------------------------------------------- fan-in
    for shape in ([1, 256, 256, 128], [256, 4, 256, 256]):
        for k in (2, 4, 16):
            parts = [(torch.randn(shape, generator=gen) * 1e-2).bfloat16() for _ in range(k)]
            ref = sum(p.double() for p in parts)
            devs = [up(p, bf) for p in parts]
            row = {"shape": shape, "k": k}
            for arm, on in (("widened", False), ("mixed", True)):
                ag.FANIN_MIXED = on

                def acc():
                    t = ag.Tensor(devs[0], requires_grad=True)
                    for p in devs:
                        t.add_grad(p)
                    return t._grad

                y = acc()
                out = ttnn.to_torch(y).float()
                ttnn.deallocate(y)
                row[arm] = {"rel_l2_vs_f64": rel(out, ref), "dtype": "fp32", **timed(acc, reps=6)}
            ag.FANIN_MIXED = False
            for d in devs:
                ttnn.deallocate(d)
            print("fanin", json.dumps(row), flush=True)
            blob["fanin"].append(row)

    clock.stop()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "probe.json").write_text(json.dumps(blob, indent=1))
    print("wrote", OUT / "probe.json")


if __name__ == "__main__":
    main()
