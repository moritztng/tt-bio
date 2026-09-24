#!/usr/bin/env python3
"""bcx-bytes: the broadcast-bias gradient of triangle attention, summed over the batch rows.

`autograd._reduce_to` hands `ttnn.sum(g, dim=0, keepdim=True)` the fp32 score gradient
[rows, heads, q, k]. On the 0.68 wheel ttnn serves a dim-0 reduce by permuting the whole tensor
(dims 2,1,0,3) and reducing the result: on the frontier block at n=256 the permute alone ran at
85 GB/s, 6.3 ms per call, next to a 2.1 ms reduce (`psum_prof_arms.json`). The sum itself only
has to read the tensor once.

Arms, each against a float64 sum of the same fp32 input, and against `ttnn.sum` bit for bit:
  sum    `ttnn.sum(dim=0)` at `precise_config()`, what ships
  moreh  `ttnn.moreh_sum(dim=0)`, which reduces a leading dim in place
  tree   pairwise `ttnn.add` over row halves, fp32 throughout
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "perf" / "bcx_bytes"


def aiclk():
    from perf.bcx_stack.stack import sysfs_node
    node, _ = sysfs_node()
    return int(open(f"{node}/tt_aiclk").read().split()[0])


def main():
    import ttnn
    from tt_bio import autograd as ag
    from tt_bio.tenstorrent import get_device
    dev = get_device()
    cfg = ag.precise_config()

    def tree(g):
        rows = int(g.shape[0])
        parts = [g[i:i + 1] for i in range(rows)] if rows <= 1 else None
        cur, own = g, False
        while int(cur.shape[0]) > 1:
            r = int(cur.shape[0])
            h = r // 2
            a, b = cur[0:h], cur[h:2 * h]
            nxt = ttnn.add(a, b)
            ttnn.deallocate(a)
            ttnn.deallocate(b)
            if r % 2:
                t = cur[2 * h:r]
                head = nxt[0:1]
                rest = nxt[1:h] if h > 1 else None
                head2 = ttnn.add(head, t)
                ttnn.deallocate(head)
                ttnn.deallocate(t)
                pieces = [head2] + ([rest] if rest is not None else [])
                n2 = ttnn.concat(pieces, dim=0) if len(pieces) > 1 else head2
                if len(pieces) > 1:
                    for p in pieces:
                        ttnn.deallocate(p)
                ttnn.deallocate(nxt)
                nxt = n2
            if own:
                ttnn.deallocate(cur)
            cur, own = nxt, True
        return cur

    arms = {
        "sum": lambda g: ttnn.sum(g, dim=0, keepdim=True, compute_kernel_config=cfg),
        "moreh": lambda g: ttnn.moreh_sum(g, dim=0, keepdim=True, compute_kernel_config=cfg),
        "tree": tree,
    }

    def timed(fn, g, reps=12):
        y = fn(g)
        ttnn.synchronize_device(dev)
        ttnn.deallocate(y)
        ts = []
        for _ in range(reps):
            t0 = time.time()
            y = fn(g)
            ttnn.synchronize_device(dev)
            ts.append(time.time() - t0)
            ttnn.deallocate(y)
        ts.sort()
        return ts[0] * 1e3, ts[len(ts) // 2] * 1e3

    rows = []
    gen = torch.Generator().manual_seed(0)
    for shape in ([256, 4, 256, 256], [81, 4, 256, 256], [13, 4, 256, 256], [128, 4, 128, 128]):
        x = torch.randn(shape, generator=gen) * 1e-3
        ref = x.double().sum(0, keepdim=True)
        g = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
        out, base = {"shape": shape, "aiclk_before": aiclk()}, None
        for name, fn in arms.items():
            try:
                y = fn(g)
                yt = ttnn.to_torch(y).reshape(ref.shape)
                ttnn.deallocate(y)
            except Exception as e:  # an arm the wheel refuses is a result, not a crash
                out[name] = {"error": str(e).splitlines()[0][:200]}
                continue
            if base is None:
                base = yt
            mn, md = timed(fn, g)
            out[name] = {"rel_l2_vs_f64": float((yt.double() - ref).norm() / ref.norm()),
                         "bits_eq_sum": bool((yt == base).all()), "min_ms": mn, "median_ms": md}
        out["torch_f32_rel_l2_vs_f64"] = float((x.sum(0, keepdim=True).double() - ref).norm() / ref.norm())
        out["aiclk_after"] = aiclk()
        print(json.dumps(out), flush=True)
        rows.append(out)
        ttnn.deallocate(g)
    (OUT / "reduce_probe.json").write_text(json.dumps({"rows": rows}, indent=1))


if __name__ == "__main__":
    main()
