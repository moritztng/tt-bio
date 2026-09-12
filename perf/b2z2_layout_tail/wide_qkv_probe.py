#!/usr/bin/env python3
"""The head split written by the matmul, at the token-DiT shape: does it land in the right place?

`qkv_heads_wide` makes the qkv projection emit `[batch, head, seq, head_dim]` directly, so
`nlp_create_qkv_heads` never runs. The head a column tile belongs to is `tidx / HEAD_MAJOR_HD_T`,
which is new -- the shipped macro assumed a one-tile head. A wrong mapping does not fail, it
silently permutes heads, so the check that matters is not "is it close" but "is it closer to the
right head than to the wrong one".

  mapping     fused q,k,v against the stock pair's, and against the same tensors with the heads
              rolled by one. The roll is the negative control and it must be far worse.
  cost        the two arms interleaved in one process, n=5 blocks each, alternating the lead.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

S, C, H, HD = 512, 768, 16, 64


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--blocks", type=int, default=5)
    a = ap.parse_args()

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    import tt_bio.tenstorrent as T
    from tt_bio import triatt_qkv as Q

    dev = T.get_device(trace_region_size=512 << 20)
    out = {"env": {"card": os.environ.get("TT_VISIBLE_DEVICES"),
                   "commit": os.popen(f"git -C {ROOT} rev-parse --short HEAD").read().strip(),
                   "blk": list(Q.DIT_QKV_BLOCK), "shape": [S, C, 3 * H * HD],
                   "loadavg": open("/proc/loadavg").read().split()[:3]}}
    a.out.parent.mkdir(parents=True, exist_ok=True)

    def dump():
        a.out.write_text(json.dumps(out, indent=1))

    tt = lambda x: ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16,   # noqa: E731
                                   device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    N = 3 * H * HD
    s, w, b = tt(torch.randn(1, S, C)), tt(torch.randn(C, N)), tt(torch.randn(1, N))
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False,
        fp32_dest_acc_en=False, packer_l1_acc=True)

    def shipped():
        qkv = ttnn.linear(s, w, bias=b, compute_kernel_config=ckc, core_grid=T.CORE_GRID_MAIN)
        qkv = ttnn.unsqueeze(qkv, 1)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv, num_heads=H, num_kv_heads=H, transpose_k_heads=False)
        ttnn.deallocate(qkv)
        return q, k, v

    def fused():
        Q._WIDE_ENABLED = True
        r = Q.qkv_heads_wide(s, w, b, ckc, H, HD, ttnn.bfloat16)
        assert r is not None, f"declined: {Q.WIDE_REJECTS}"
        return r

    ref = [ttnn.to_torch(t).clone() for t in shipped()]
    got = [ttnn.to_torch(t).clone() for t in fused()]
    out["eligible"] = {"served": Q.WIDE_STATS[0], "declined": Q.WIDE_STATS[1],
                       "rejects": {str(k): v for k, v in Q.WIDE_REJECTS.items()}}

    def cmp(x, y):
        d = (x.float() - y.float()).abs()
        return {"max_abs": round(float(d.max()), 5), "mean_abs": round(float(d.mean()), 6),
                "rms_ref": round(float(y.float().pow(2).mean().sqrt()), 4)}

    out["mapping"] = {}
    for name, r, g in zip("qkv", ref, got):
        out["mapping"][name] = {
            "shape": list(g.shape),
            "vs_stock": cmp(g, r),
            "vs_stock_heads_rolled_by_one": cmp(g, torch.roll(r, 1, dims=1)),
        }
    dump()
    print("  mapping:", json.dumps(out["mapping"], indent=1), flush=True)

    walls = {"stock": [], "fused": []}
    order = []
    for blk in range(a.blocks):
        for tag in ("stock", "fused") if blk % 2 == 0 else ("fused", "stock"):
            fn = shipped if tag == "stock" else fused
            for _ in range(3):
                fn()
            ttnn.synchronize_device(dev)
            t = time.perf_counter()
            for _ in range(a.reps):
                fn()
            ttnn.synchronize_device(dev)
            us = 1e6 * (time.perf_counter() - t) / a.reps
            walls[tag].append(us)
            order.append((tag, round(us, 2)))
    med = {k: st.median(v) for k, v in walls.items()}
    off = walls["stock"]
    h = len(off) // 2
    out["ab"] = {"us_stock": round(med["stock"], 2), "us_fused": round(med["fused"], 2),
                 "ratio": round(med["stock"] / med["fused"], 5),
                 "delta_us_per_call": round(med["stock"] - med["fused"], 2),
                 "aa_floor": round(st.median(off[:h]) / st.median(off[h:]), 5),
                 "order": order}
    dump()
    print(f"  stock {med['stock']:.2f} us  fused {med['fused']:.2f} us  "
          f"ratio {out['ab']['ratio']:.5f}x  A/A {out['ab']['aa_floor']}", flush=True)
    print("DONE", a.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
