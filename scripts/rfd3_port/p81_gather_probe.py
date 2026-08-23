#!/usr/bin/env python3
"""p81 -- is ttnn.gather usable at all for the gathered atom softmax?

p79 gate 1a failed at the production shape: maxabs 1.001e+04, which is the magnitude of the
-1e4 mask, so the op returned MASKED columns where the reference returned neighbour values.
And the same call took 2553.97 ms against a 4.83 ms dense softmax.

Two questions, and the plan for Job 1 turns on both:
  A  does ttnn.gather have torch.gather semantics on dim 3 at ALL, or only below some shape?
  B  how does its cost scale, and does layout/dtype change either answer?
"""
import statistics, sys, os, time, json, pathlib
import torch, ttnn
sys.path.insert(0, os.getcwd())
from tt_bio.tenstorrent import get_device

DEV = get_device()
OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "perf/p81/gather_probe.json")
torch.manual_seed(0)


def trial(L, NK, K, H, dtype=ttnn.float32, tdt=torch.float32, reps=3,
          idx_layout=ttnn.TILE_LAYOUT):
    """Exact torch.gather reference on dim 3, plus a median ms."""
    idx = torch.stack([torch.randperm(NK)[:K].sort().values for _ in range(L)])
    idx = idx.unsqueeze(0).unsqueeze(0).expand(1, H, L, K).contiguous()
    src = torch.randn(1, H, L, NK, dtype=tdt)
    ref = src.gather(3, idx)
    s_dev = ttnn.from_torch(src, layout=ttnn.TILE_LAYOUT, device=DEV, dtype=dtype)
    i_dev = ttnn.from_torch(idx.to(torch.int32), layout=idx_layout, device=DEV,
                            dtype=ttnn.uint32)
    got = ttnn.to_torch(ttnn.gather(s_dev, 3, i_dev))
    exact = bool(torch.equal(got.to(tdt), ref))
    bad = int((got.to(tdt) != ref).sum())
    ms = []
    for _ in range(reps):
        t0 = time.perf_counter()
        o = ttnn.gather(s_dev, 3, i_dev)
        ttnn.synchronize_device(DEV)
        ms.append(1000 * (time.perf_counter() - t0))
        ttnn.deallocate(o)
    ttnn.deallocate(s_dev)
    ttnn.deallocate(i_dev)
    return exact, bad, statistics.median(ms), idx.numel()


rows = []
print("%-34s %7s %10s %11s %9s" % ("case", "exact", "wrong", "median ms", "us/elem"), flush=True)
CASES = [
    ("L=32   NK=64   K=32  H=1", 32, 64, 32, 1),
    ("L=64   NK=128  K=32  H=1", 64, 128, 32, 1),
    ("L=128  NK=256  K=64  H=1", 128, 256, 64, 1),
    ("L=128  NK=256  K=128 H=4", 128, 256, 128, 4),
    ("L=512  NK=1024 K=128 H=4", 512, 1024, 128, 4),
    ("L=1024 NK=2048 K=128 H=4", 1024, 2048, 128, 4),
    ("L=2048 NK=6080 K=128 H=4", 2048, 6080, 128, 4),
    ("L=6051 NK=6080 K=128 H=4", 6051, 6080, 128, 4),
]
for name, L, NK, K, H in CASES:
    try:
        exact, bad, ms, n = trial(L, NK, K, H)
        print("%-34s %7s %10d %11.3f %9.3f" % (name, exact, bad, ms, 1000 * ms / n), flush=True)
        rows.append(dict(case=name, L=L, n_key=NK, K=K, H=H, exact=exact, wrong=bad,
                         median_ms=ms, us_per_elem=1000 * ms / n, out_elems=n))
    except Exception as e:
        print("%-34s  EXC %s" % (name, str(e)[:90]), flush=True)
        rows.append(dict(case=name, L=L, n_key=NK, K=K, H=H, exc=str(e)[:300]))

# Does a ROW_MAJOR index or a bf16 source change the verdict at the production shape?
print(flush=True)
for label, kw in (("prod, ROW_MAJOR index", dict(idx_layout=ttnn.ROW_MAJOR_LAYOUT)),
                  ("prod, bf16 source", dict(dtype=ttnn.bfloat16, tdt=torch.bfloat16))):
    try:
        exact, bad, ms, n = trial(6051, 6080, 128, 4, **kw)
        print("%-34s %7s %10d %11.3f %9.3f" % (label, exact, bad, ms, 1000 * ms / n), flush=True)
        rows.append(dict(case=label, exact=exact, wrong=bad, median_ms=ms,
                         us_per_elem=1000 * ms / n))
    except Exception as e:
        print("%-34s  EXC %s" % (label, str(e)[:90]), flush=True)
        rows.append(dict(case=label, exc=str(e)[:300]))

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps({"rows": rows, "host": "qb2",
                           "card": os.environ.get("TT_VISIBLE_DEVICES")}, indent=2) + "\n")
print("\nwrote", OUT)
