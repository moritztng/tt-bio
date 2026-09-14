#!/usr/bin/env python3
"""p98 -- the gathered atom softmax against the dense one it replaces, on the patched gather.

`RFD3_GATHERED_SOFTMAX` swaps one thing and only one thing: the dense
`softmax_bf16(scores)` over the full key axis becomes
`gather -> softmax_bf16(compact) -> scatter`. Both arms hand the same `attention` tensor to the
same `attn_value_matmul`, so timing exactly those two chains is the whole comparison.

The recorded "gathered is 7.2x slower" verdict was measured against a gather that returned wrong
data for every tile-row after the first AND rescanned all 1024 index values per input tile. Both
are fixed here, so this re-measures it honestly.

Arms are interleaved round by round, both warmed before the first timed round, and an A/A round
(dense against dense) runs in the same session to floor the noise.
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

ARM = sys.argv[1] if len(sys.argv) > 1 else "unknown"
OUT = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "perf/p98/route_%s.json" % ARM)
ROUNDS = int(os.environ.get("P98_ROUNDS", "5"))
L = int(os.environ.get("P98_L", "6080"))
H = int(os.environ.get("P98_H", "4"))
K = 128

DEV = get_device()
torch.manual_seed(0)

# RFD3's neighbour index: K sorted neighbours in a band around the query row.
centre = torch.linspace(0, L - 1, L)
rows = []
for r in range(L):
    lo = int(max(0, min(L - K, centre[r] - K // 2)))
    rows.append(torch.arange(lo, lo + K))
idx = torch.stack(rows).unsqueeze(0).unsqueeze(0).expand(1, H, L, K).contiguous()

scores_t = torch.randn(1, H, L, L, dtype=torch.float32)
scores = ttnn.from_torch(scores_t, layout=ttnn.TILE_LAYOUT, device=DEV, dtype=ttnn.float32)
gather_idx = ttnn.from_torch(idx.to(torch.int32), layout=ttnn.TILE_LAYOUT, device=DEV,
                             dtype=ttnn.uint32)
zeros = ttnn.zeros((1, H, L, L), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=DEV)
print("scores %s fp32 = %.1f MB" % (list(scores.shape), scores_t.numel() * 4 / 1e6), flush=True)


def dense():
    a = softmax_generic.softmax_bf16(scores, ttnn.bfloat16)
    ttnn.synchronize_device(DEV)
    ttnn.deallocate(a)


def gathered():
    compact = ttnn.gather(scores, 3, gather_idx)
    w = softmax_generic.softmax_bf16(compact, ttnn.bfloat16)
    ttnn.deallocate(compact)
    a = ttnn.scatter(zeros, 3, gather_idx, w)
    ttnn.deallocate(w)
    ttnn.synchronize_device(DEV)
    ttnn.deallocate(a)


def gather_only():
    c = ttnn.gather(scores, 3, gather_idx)
    ttnn.synchronize_device(DEV)
    ttnn.deallocate(c)


def scatter_only():
    w = ttnn.from_torch(torch.randn(1, H, L, K, dtype=torch.float32), layout=ttnn.TILE_LAYOUT,
                        device=DEV, dtype=ttnn.bfloat16)
    t0 = time.perf_counter()
    a = ttnn.scatter(zeros, 3, gather_idx, w)
    ttnn.synchronize_device(DEV)
    ms = 1000 * (time.perf_counter() - t0)
    ttnn.deallocate(a)
    ttnn.deallocate(w)
    return ms


def timed(fn):
    t0 = time.perf_counter()
    fn()
    return 1000 * (time.perf_counter() - t0)


print("warming both arms (compile + program cache) ...", flush=True)
dense()
gathered()

res = {"dense": [], "gathered": [], "aa": [], "gather_only": []}
print("\n%-7s %12s %12s %12s %12s" % ("round", "dense ms", "gathered ms", "A/A ms", "gather ms"),
      flush=True)
for r in range(ROUNDS):
    d = timed(dense)
    g = timed(gathered)
    a = timed(dense)          # A/A partner for the dense arm, same round, same order distance
    go = timed(gather_only)
    res["dense"].append(d)
    res["gathered"].append(g)
    res["aa"].append(a)
    res["gather_only"].append(go)
    print("%-7d %12.2f %12.2f %12.2f %12.2f" % (r, d, g, a, go), flush=True)

sc = statistics.median([scatter_only() for _ in range(3)])
med = {k: statistics.median(v) for k, v in res.items()}
aa_spread = abs(med["aa"] - med["dense"])
print("\nmedian dense    %.2f ms   (A/A partner %.2f ms, |floor| %.2f ms)"
      % (med["dense"], med["aa"], aa_spread), flush=True)
print("median gathered %.2f ms   = gather %.2f + softmax/scatter %.2f (scatter alone %.2f)"
      % (med["gathered"], med["gather_only"], med["gathered"] - med["gather_only"], sc), flush=True)
print("gathered / dense = %.2fx  %s"
      % (med["gathered"] / med["dense"],
         "SLOWER" if med["gathered"] > med["dense"] else "FASTER"), flush=True)

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps({"arm": ARM, "L": L, "H": H, "K": K, "rounds": ROUNDS,
                           "ms": res, "median_ms": med, "scatter_only_ms": sc,
                           "aa_floor_ms": aa_spread,
                           "ratio_gathered_over_dense": med["gathered"] / med["dense"],
                           "host": "qb2", "card": os.environ.get("TT_VISIBLE_DEVICES"),
                           "runtime_root": os.environ.get("TT_METAL_RUNTIME_ROOT")},
                          indent=2) + "\n")
print("\nwrote", OUT)
