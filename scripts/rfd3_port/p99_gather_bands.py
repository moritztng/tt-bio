#!/usr/bin/env python3
"""p99 -- the gathered atom softmax against the dense one, swept across the atom-count bands.

p98 answers the production shape. This asks where, if anywhere, the gathered route wins, because
`ttnn.gather` changes program factory at a key axis of 1920: `GATHER_WT_THRESHOLD = 60` tiles, and
above it the multi-core factory splits work over Wt_index alone, which for RFD3 is K/32 = 4 tiles
no matter how many atoms there are.

One process, one device context, every band interleaved dense/gathered with an A/A partner.
"""
import json
import os
import pathlib
import statistics
import sys
import time

import torch
import ttnn

sys.path.insert(0, os.getcwd())
from tt_bio.tenstorrent import get_device  # noqa: E402
from tt_bio import softmax_generic  # noqa: E402

ARM = sys.argv[1] if len(sys.argv) > 1 else "patched"
OUT = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "perf/p99/bands_%s.json" % ARM)
BANDS = [int(x) for x in os.environ.get("P99_BANDS", "512,1024,1920,2048,3040,6080").split(",")]
ROUNDS = int(os.environ.get("P99_ROUNDS", "3"))
H, K = 4, 128

DEV = get_device()
torch.manual_seed(0)


def band(L):
    centre = torch.linspace(0, L - 1, L)
    rows = [torch.arange(int(max(0, min(L - K, centre[r] - K // 2))),
                         int(max(0, min(L - K, centre[r] - K // 2))) + K) for r in range(L)]
    idx = torch.stack(rows).unsqueeze(0).unsqueeze(0).expand(1, H, L, K).contiguous()
    scores = ttnn.from_torch(torch.randn(1, H, L, L, dtype=torch.float32),
                             layout=ttnn.TILE_LAYOUT, device=DEV, dtype=ttnn.float32)
    gi = ttnn.from_torch(idx.to(torch.int32), layout=ttnn.TILE_LAYOUT, device=DEV,
                         dtype=ttnn.uint32)
    zeros = ttnn.zeros((1, H, L, L), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=DEV)

    def dense():
        a = softmax_generic.softmax_bf16(scores, ttnn.bfloat16)
        ttnn.synchronize_device(DEV)
        ttnn.deallocate(a)

    def gathered():
        c = ttnn.gather(scores, 3, gi)
        w = softmax_generic.softmax_bf16(c, ttnn.bfloat16)
        ttnn.deallocate(c)
        a = ttnn.scatter(zeros, 3, gi, w)
        ttnn.deallocate(w)
        ttnn.synchronize_device(DEV)
        ttnn.deallocate(a)

    def gather_only():
        c = ttnn.gather(scores, 3, gi)
        ttnn.synchronize_device(DEV)
        ttnn.deallocate(c)

    def timed(fn):
        t0 = time.perf_counter()
        fn()
        return 1000 * (time.perf_counter() - t0)

    dense()
    gathered()
    d, g, aa, go = [], [], [], []
    for _ in range(ROUNDS):
        d.append(timed(dense))
        g.append(timed(gathered))
        aa.append(timed(dense))
        go.append(timed(gather_only))
    ttnn.deallocate(scores)
    ttnn.deallocate(gi)
    ttnn.deallocate(zeros)
    md, mg = statistics.median(d), statistics.median(g)
    return dict(L=L, Wt_input=L // 32, factory="single-core" if L // 32 <= 60 else "multi-core",
                dense_ms=md, gathered_ms=mg, gather_only_ms=statistics.median(go),
                aa_ms=statistics.median(aa), aa_floor_ms=abs(statistics.median(aa) - md),
                ratio=mg / md, raw=dict(dense=d, gathered=g, aa=aa, gather_only=go))


rows = []
print("%-7s %-6s %-12s %10s %12s %12s %9s %9s"
      % ("atoms", "Wt_in", "factory", "dense ms", "gathered ms", "gather ms", "A/A", "ratio"),
      flush=True)
for L in BANDS:
    try:
        r = band(L)
        print("%-7d %-6d %-12s %10.2f %12.2f %12.2f %9.2f %8.2fx"
              % (r["L"], r["Wt_input"], r["factory"], r["dense_ms"], r["gathered_ms"],
                 r["gather_only_ms"], r["aa_floor_ms"], r["ratio"]), flush=True)
        rows.append(r)
    except Exception as e:
        print("%-7d  EXC %s" % (L, str(e)[:90]), flush=True)
        rows.append(dict(L=L, exc=str(e)[:300]))

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps({"arm": ARM, "H": H, "K": K, "rounds": ROUNDS, "rows": rows,
                           "host": "qb2", "card": os.environ.get("TT_VISIBLE_DEVICES"),
                           "runtime_root": os.environ.get("TT_METAL_RUNTIME_ROOT")},
                          indent=2) + "\n")
print("\nwrote", OUT)
