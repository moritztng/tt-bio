#!/usr/bin/env python3
"""p97 -- does the backported ttnn.gather still have a correctness threshold, and what does it cost?

p81b/p81c found stock ttnn 0.68.0's `ttnn.gather` on dim 3 exact up to NK=1920 (fp32) and wrong
for every tile-row after the first above it, which is what put `_TTNN_GATHER_MAX_KEY_AXIS = 1920`
in `tt_bio/rfd3/model.py` and retired `RFD3_GATHERED_SOFTMAX`. Upstream fixed that one day after
our tag. This re-runs the same question against a locally patched kernel and carries the sweep up
to the production key axis, 6080.

Arm is named on the command line and only labels the output; which kernel actually runs is set by
TT_METAL_RUNTIME_ROOT.
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

ARM = sys.argv[1] if len(sys.argv) > 1 else "unknown"
OUT = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "perf/p97/gather_%s.json" % ARM)
DEV = get_device()
torch.manual_seed(0)


def neighbour_index(L, NK, K, sequential):
    """`sequential=True` mimics RFD3: K sorted neighbours drawn from a band around the row."""
    if not sequential:
        return torch.stack([torch.randperm(NK)[:K].sort().values for _ in range(L)])
    rows = []
    centre = torch.linspace(0, NK - 1, L)
    for r in range(L):
        lo = int(max(0, min(NK - K, centre[r] - K // 2)))
        rows.append(torch.arange(lo, lo + K))
    return torch.stack(rows)


def trial(L, NK, K, H, dtype, tdt, sequential, reps=3):
    idx = neighbour_index(L, NK, K, sequential)
    idx = idx.unsqueeze(0).unsqueeze(0).expand(1, H, L, K).contiguous()
    src = torch.randn(1, H, L, NK).to(tdt)
    ref = src.gather(3, idx)
    s_dev = ttnn.from_torch(src, layout=ttnn.TILE_LAYOUT, device=DEV, dtype=dtype)
    i_dev = ttnn.from_torch(idx.to(torch.int32), layout=ttnn.TILE_LAYOUT, device=DEV,
                            dtype=ttnn.uint32)
    got = ttnn.to_torch(ttnn.gather(s_dev, 3, i_dev)).to(tdt)
    bad = (got != ref)
    rows_bad = bad.any(3)[0, 0]
    first_bad = int(rows_bad.nonzero()[0]) if int(rows_bad.sum()) else -1
    ms = []
    for _ in range(reps):
        t0 = time.perf_counter()
        o = ttnn.gather(s_dev, 3, i_dev)
        ttnn.synchronize_device(DEV)
        ms.append(1000 * (time.perf_counter() - t0))
        ttnn.deallocate(o)
    ttnn.deallocate(s_dev)
    ttnn.deallocate(i_dev)
    # How much of the inner scan the needed[] bitmap can skip, measured on this index itself:
    # per output index tile, the fraction of the Wt_input input tiles any of its 1024 values hits.
    wt_in, wt_idx, ht = NK // 32, K // 32, (H * L) // 32
    flat = idx[0, 0]
    hit = []
    for h in range(0, min(ht, L // 32)):
        for w in range(wt_idx):
            block = flat[h * 32:(h + 1) * 32, w * 32:(w + 1) * 32]
            hit.append(len(torch.unique(block // 32)))
    return dict(arm=ARM, L=L, n_key=NK, K=K, H=H, dtype=str(tdt).replace("torch.", ""),
                index="banded" if sequential else "random", wrong=int(bad.sum()),
                out_elems=int(ref.numel()), first_bad_row=first_bad,
                median_ms=statistics.median(ms), all_ms=ms, Wt_input=wt_in,
                needed_tiles_mean=round(sum(hit) / len(hit), 1),
                needed_tiles_max=max(hit), needed_frac=round(sum(hit) / len(hit) / wt_in, 4))


rows = []
print("=== dim-3 gather, L=512 K=128 H=1, key axis swept to the production 6080 ===", flush=True)
hdr = "%-6s %-8s %-7s %10s %8s %11s %12s %9s"
print(hdr % ("dtype", "index", "NK", "wrong", "%", "1st bad row", "median ms", "need/Wt"),
      flush=True)
for label, dtype, tdt in (("fp32", ttnn.float32, torch.float32),
                          ("bf16", ttnn.bfloat16, torch.bfloat16)):
    for seq in (False, True):
        for NK in (1920, 2048, 3072, 4096, 6080):
            try:
                r = trial(512, NK, 128, 1, dtype, tdt, seq)
                print(hdr % (label, r["index"], NK, r["wrong"],
                             "%.2f%%" % (100.0 * r["wrong"] / r["out_elems"]),
                             r["first_bad_row"], "%.3f" % r["median_ms"],
                             "%.0f/%d" % (r["needed_tiles_mean"], r["Wt_input"])), flush=True)
                rows.append(r)
            except Exception as e:
                print("%-6s %-8s %-7d  EXC %s" % (label, "banded" if seq else "random", NK,
                                                  str(e)[:70]), flush=True)
                rows.append(dict(arm=ARM, dtype=label, n_key=NK, exc=str(e)[:300]))

print("\n=== production RFD3 atom-attention shape: [1,4,6080,6080] fp32, K=128 banded ===",
      flush=True)
try:
    r = trial(6080, 6080, 128, 4, ttnn.float32, torch.float32, True, reps=3)
    print("wrong %d / %d (%.4f%%), first bad row %d, median %.1f ms, needed %.0f/%d tiles"
          % (r["wrong"], r["out_elems"], 100.0 * r["wrong"] / r["out_elems"], r["first_bad_row"],
             r["median_ms"], r["needed_tiles_mean"], r["Wt_input"]), flush=True)
    rows.append(r)
except Exception as e:
    print("production shape EXC %s" % str(e)[:200], flush=True)
    rows.append(dict(arm=ARM, shape="production", exc=str(e)[:300]))

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps({"arm": ARM, "rows": rows, "host": "qb2",
                           "card": os.environ.get("TT_VISIBLE_DEVICES"),
                           "runtime_root": os.environ.get("TT_METAL_RUNTIME_ROOT")},
                          indent=2) + "\n")
print("\nwrote", OUT)
