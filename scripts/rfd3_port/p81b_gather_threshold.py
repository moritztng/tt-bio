#!/usr/bin/env python3
"""p81b -- where exactly does ttnn.gather stop being correct on dim 3?

p81 found ttnn.gather (ttnn 0.68.0) exact at L=512/NK=1024 and 99.2 % WRONG at L=1024/NK=2048,
with the per-call cost jumping from 0.181 ms to 145 ms at the same step. One threshold, two
symptoms. This separates the source width NK from the row count L so the bug report names the
actual trigger, and checks whether the wrongness is the padding tail or the whole tensor.
"""
import statistics, sys, os, time, json, pathlib
import torch, ttnn
sys.path.insert(0, os.getcwd())
from tt_bio.tenstorrent import get_device

DEV = get_device()
OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "perf/p81/gather_threshold.json")
torch.manual_seed(0)


def trial(L, NK, K, H=1, reps=3):
    idx = torch.stack([torch.randperm(NK)[:K].sort().values for _ in range(L)])
    idx = idx.unsqueeze(0).unsqueeze(0).expand(1, H, L, K).contiguous()
    src = torch.randn(1, H, L, NK, dtype=torch.float32)
    ref = src.gather(3, idx)
    s_dev = ttnn.from_torch(src, layout=ttnn.TILE_LAYOUT, device=DEV, dtype=ttnn.float32)
    i_dev = ttnn.from_torch(idx.to(torch.int32), layout=ttnn.TILE_LAYOUT, device=DEV,
                            dtype=ttnn.uint32)
    got = ttnn.to_torch(ttnn.gather(s_dev, 3, i_dev)).to(torch.float32)
    bad = (got != ref)
    # Where are the wrong elements? First wrong row tells us if it is a tail or the whole tensor.
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
    return int(bad.sum()), idx.numel(), int(rows_bad.sum()), L, first_bad, statistics.median(ms)


rows = []
print("=== source width NK swept at fixed L=512, K=128, H=1 ===", flush=True)
print("%-28s %10s %8s %12s %11s" % ("case", "wrong", "%", "1st bad row", "median ms"), flush=True)
for NK in (1024, 1056, 1088, 1152, 1280, 1536, 2048):
    try:
        bad, n, rbad, L, first, ms = trial(512, NK, 128)
        print("L=512  NK=%-6d           %10d %7.2f%% %12d %11.3f"
              % (NK, bad, 100.0 * bad / n, first, ms), flush=True)
        rows.append(dict(sweep="NK", L=512, n_key=NK, K=128, wrong=bad, out_elems=n,
                         bad_rows=rbad, first_bad_row=first, median_ms=ms))
    except Exception as e:
        print("L=512  NK=%-6d           EXC %s" % (NK, str(e)[:70]), flush=True)

print("\n=== row count L swept at fixed NK=1024, K=128, H=1 ===", flush=True)
for L in (512, 544, 576, 640, 768, 1024):
    try:
        bad, n, rbad, LL, first, ms = trial(L, 1024, 128)
        print("L=%-6d NK=1024           %10d %7.2f%% %12d %11.3f"
              % (L, bad, 100.0 * bad / n, first, ms), flush=True)
        rows.append(dict(sweep="L", L=L, n_key=1024, K=128, wrong=bad, out_elems=n,
                         bad_rows=rbad, first_bad_row=first, median_ms=ms))
    except Exception as e:
        print("L=%-6d NK=1024           EXC %s" % (L, str(e)[:70]), flush=True)

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps({"rows": rows, "ttnn": ttnn.__version__ if hasattr(ttnn, "__version__")
                           else "unknown", "host": "qb2",
                           "card": os.environ.get("TT_VISIBLE_DEVICES")}, indent=2) + "\n")
print("\nwrote", OUT)
