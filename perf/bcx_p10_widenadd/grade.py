#!/usr/bin/env python3
"""bcx-p10-widenadd: the fan-in kernel graded against float64, and priced against the path it
replaces, on one card.

`Tensor.add_grad` accumulates k contributions exactly as `perf/bcx_reduce/probe.py`'s fan-in
section does, with `rne_add.WIDEN_ADD` the only difference between two arms:

  widened   today: typecast the accumulator and the contribution to float32, `ttnn.add`
  widen     `rne_add.widen_add`, one kernel, both operands widened in the unpacker

Both arms are graded against the float64 sum of the same bfloat16 contributions, never against
each other, and `torch.equal` between them is reported beside it. Contributions are generated
from a per-part seed and streamed, so k = 64 at `288x1x288x384` needs one part on the host at a
time. The inputs are read back after every accumulation to prove the kernel wrote nothing but
its own output.

The micro-bench times one `add_grad` per case, first promotion (bf16 + bf16) and later ones
(f32 + bf16), synced medians with AICLK sampled during.

  TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=1 timeout 1500 python3 perf/bcx_p10_widenadd/grade.py
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
OUT = ROOT / "perf" / "bcx_p10_widenadd" / "out"

#: (shape, contributions, scale spread). The first two shapes are the round's own fan-in
#: tensors (`bcx-p10-l1fuse` chains 1-3); the rest are not tile-aligned in the last two dims, so
#: the kernel adds tile padding like any other element.
CASES = [([1, 288, 288, 128], k, 1.0) for k in (2, 4, 16, 64)] + \
        [([288, 1, 288, 384], k, 1.0) for k in (2, 4, 16, 64)] + \
        [([1, 288, 288, 128], 16, 256.0),       # contributions spread over 256x in magnitude
         ([3, 50, 70], 4, 1.0), ([1, 17, 33, 100], 16, 1.0), ([7, 1, 45, 31], 64, 1.0)]


def rel(x, ref):
    return float((x.double() - ref).norm() / ref.norm())


def part(shape, i, spread):
    g = torch.Generator().manual_seed(1000 + i)
    scale = 1e-2 * (spread ** (i / 7 % 1.0))
    return (torch.randn(shape, generator=g) * scale).bfloat16()


def main():
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import ttnn
    from perf.bcx_stack.stack import Clock, sysfs_node
    from tt_bio import autograd as ag, rne_add as R
    from tt_bio.tenstorrent import get_device
    dev = get_device()
    clock = Clock(dt=0.02)
    bf, f32 = ttnn.bfloat16, ttnn.float32

    def up(t, dtype=bf):
        return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=dev)

    blob = {"pci": sysfs_node()[1], "card": os.environ.get("TT_VISIBLE_DEVICES"),
            "loadavg_start": os.getloadavg(), "grade": [], "bench": []}

    for shape, k, spread in CASES:
        ref = torch.zeros(shape, dtype=torch.float64)
        row = {"shape": shape, "k": k, "spread": spread}
        outs = {}
        for arm, on in (("widened", False), ("widen", True)):
            R.WIDEN_ADD = on
            before = R.widen_reach()
            t, moved = None, 0.0
            span = [time.time()]
            for i in range(k):
                p = part(shape, i, spread)
                if arm == "widened":
                    ref += p.double()
                d = up(p)
                if t is None:
                    t = ag.Tensor(d, requires_grad=True)
                t.add_grad(d)
                if i:
                    # The contribution must come back as it went in.
                    moved = max(moved, float((ttnn.to_torch(d).float() - p.float()).abs().max()))
                    ttnn.deallocate(d)
            outs[arm] = ttnn.to_torch(t._grad).float().reshape(shape)
            span.append(time.time())
            after = R.widen_reach()
            row[arm] = {"rel_l2_vs_f64": rel(outs[arm], ref),
                        "max_abs_vs_f64": float((outs[arm].double() - ref).abs().max()),
                        "dtype": str(t._grad.dtype),
                        "input_max_abs_moved": moved,
                        "reach": {kk: after.get(kk, 0) - before.get(kk, 0)
                                  for kk in set(after) | set(before)},
                        "aiclk": clock.window([tuple(span)])}
            ttnn.deallocate(t._grad)
            ttnn.deallocate(t.value)
        R.WIDEN_ADD = False
        row["torch_equal_widen_vs_widened"] = bool(torch.equal(outs["widen"], outs["widened"]))
        row["max_abs_widen_vs_widened"] = float((outs["widen"] - outs["widened"]).abs().max())
        print("grade", json.dumps(row), flush=True)
        blob["grade"].append(row)

    # ---------------------------------------------------------------- one add_grad, priced
    for shape in ([1, 288, 288, 128], [288, 1, 288, 384]):
        a16, b16 = up(part(shape, 0, 1.0)), up(part(shape, 1, 1.0))
        a32 = ttnn.typecast(a16, f32)
        n = int(np.prod(shape))
        for case, acc in (("first bf16+bf16", a16), ("later f32+bf16", a32)):
            row = {"shape": shape, "case": case}
            for arm, on in (("widened", False), ("widen", True)):
                R.WIDEN_ADD = on

                def one():
                    t = ag.Tensor(a16, requires_grad=True)
                    t._grad = acc
                    t.add_grad(b16)
                    return t._grad

                ttnn.deallocate(one())
                ttnn.synchronize_device(dev)
                ts, t_a = [], time.time()
                for _ in range(20):
                    t0 = time.perf_counter()
                    y = one()
                    ttnn.synchronize_device(dev)
                    ts.append(time.perf_counter() - t0)
                    ttnn.deallocate(y)
                med = float(np.median(ts))
                row[arm] = {"median_ms": med * 1e3, "min_ms": min(ts) * 1e3,
                            "aiclk": clock.window([(t_a, time.time())]),
                            "load1": os.getloadavg()[0]}
            R.WIDEN_ADD = False
            # The kernel's own bytes: both operands read once, the float32 result written once.
            kb = n * ((4 if acc.dtype == f32 else 2) + 2 + 4)
            row["widen"]["kernel_bytes"] = kb
            row["widen"]["achieved_GBps"] = kb / (row["widen"]["min_ms"] * 1e-3) / 1e9
            row["speedup_median"] = row["widened"]["median_ms"] / row["widen"]["median_ms"]
            print("bench", json.dumps(row), flush=True)
            blob["bench"].append(row)
        for x in (a16, b16, a32):
            ttnn.deallocate(x)

    clock.stop()
    blob["loadavg_end"] = os.getloadavg()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "grade.json").write_text(json.dumps(blob, indent=1))
    print("wrote", OUT / "grade.json")


if __name__ == "__main__":
    main()
