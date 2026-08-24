#!/usr/bin/env python3
"""p81c -- the exact NK boundary of the ttnn.gather bug, and whether it is a BYTE cap.

p81b: at L=512, K=128, dim 3, ttnn.gather is exact for NK <= 1536 and 93.75 % wrong at NK=2048,
where "93.75 %" is every tile-row except the first. Cost jumps 0.162 -> 18.95 ms at the same step.

If the trigger is a byte capacity, halving the element size doubles the boundary. That
discriminates a byte cap from a tile-count/element-count limit, and it decides whether ANY dtype
gets us to the production NK=6080.
"""
import statistics, sys, os, time, json, pathlib
import torch, ttnn
sys.path.insert(0, os.getcwd())
from tt_bio.tenstorrent import get_device

DEV = get_device()
OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "perf/p81/gather_boundary.json")
torch.manual_seed(0)


def trial(L, NK, K, dtype, tdt, reps=3):
    idx = torch.stack([torch.randperm(NK)[:K].sort().values for _ in range(L)])
    idx = idx.unsqueeze(0).unsqueeze(0).contiguous()
    src = torch.randn(1, 1, L, NK).to(tdt)
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
    return int(bad.sum()), idx.numel(), first_bad, statistics.median(ms)


rows = []
for label, dtype, tdt, esz, sweep in (
        ("fp32", ttnn.float32, torch.float32, 4,
         (1536, 1568, 1600, 1664, 1728, 1792, 1920, 2048)),
        ("bf16", ttnn.bfloat16, torch.bfloat16, 2,
         (1536, 2048, 2560, 3072, 3200, 3328, 3584, 4096))):
    print("\n=== %s source, L=512, K=128, dim 3 ===" % label, flush=True)
    print("%-10s %6s %10s %8s %12s %11s"
          % ("NK", "KB/row", "wrong", "%", "1st bad row", "median ms"), flush=True)
    for NK in sweep:
        try:
            bad, n, first, ms = trial(512, NK, 128, dtype, tdt)
            print("%-10d %6.1f %10d %7.2f%% %12d %11.3f"
                  % (NK, NK * esz / 1024.0, bad, 100.0 * bad / n, first, ms), flush=True)
            rows.append(dict(dtype=label, L=512, n_key=NK, K=128, kb_per_row=NK * esz / 1024.0,
                             wrong=bad, out_elems=n, first_bad_row=first, median_ms=ms))
        except Exception as e:
            print("%-10d  EXC %s" % (NK, str(e)[:70]), flush=True)
            rows.append(dict(dtype=label, n_key=NK, exc=str(e)[:300]))

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps({"rows": rows, "host": "qb2",
                           "card": os.environ.get("TT_VISIBLE_DEVICES")}, indent=2) + "\n")
print("\nwrote", OUT)
