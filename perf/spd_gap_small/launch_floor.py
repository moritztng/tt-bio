"""Device time per program for short ops, traced and eager, at the Boltz-2 diffusion step's shapes.

A Boltz-2 WH --fast fold runs 198,845 programs, ~180k of them in the 200 diffusion steps, and the unprofiled
diffusion takes ~1.65 s longer than its kernels (cen15). The host is not the limit there (b2spy, gq), so the question
is what the device charges per program on top of the kernel. A traced chain of N in-place ops has no host in it at
all: `traced_us` per op minus the op's own data time is that floor. The 1-tile case is the floor alone.
`eager_us` is the same chain issued from Python with one sync at the end.
"""
import argparse
import json
import sys
import time

import torch

sys.path.insert(0, ".")
from tt_bio.main import ensure_p300_mesh_descriptor  # noqa: E402

ensure_p300_mesh_descriptor()
import tt_bio.tenstorrent as T  # noqa: E402
import ttnn  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=400)
ap.add_argument("--reps", type=int, default=5)
args = ap.parse_args()

dev = T.get_device(trace="diffusion")
up = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
torch.manual_seed(0)

SHAPES = {"tile": (1, 1, 32, 32), "8tiles": (1, 1, 256, 128), "atom": (1, 140, 32, 128),
          "token": (1, 1, 512, 768), "token2x": (1, 1, 512, 1536)}
res = {"n": args.n, "arch": str(dev.arch()), "cases": {}}


def bench(name, step):
    for _ in range(20):
        step()
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(args.n):
        step()
    ttnn.synchronize_device(dev)
    eager = (time.perf_counter() - t0) / args.n * 1e6
    tid, _ = T.capture_trace(dev, lambda: [step() for _ in range(args.n)])
    ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
    ts = []
    for _ in range(args.reps):
        t0 = time.perf_counter()
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
        ts.append((time.perf_counter() - t0) / args.n * 1e6)
    ttnn.release_trace(dev, tid)
    ts.sort()
    res["cases"][name] = {"eager_us": round(eager, 2), "traced_us": round(ts[len(ts) // 2], 2),
                          "traced_min_us": round(ts[0], 2)}
    print(name, res["cases"][name], flush=True)


for tag, shp in SHAPES.items():
    x = up(torch.randn(shp) * 1e-3)
    w = up(torch.randn(shp) * 1e-3)
    bench(f"add_/{tag}", lambda x=x, w=w: ttnn.add_(x, w))
    bench(f"multiply_/{tag}", lambda x=x, w=w: ttnn.multiply_(x, w))
    ttnn.deallocate(x)
    ttnn.deallocate(w)

# The token DiT's cond-linear shape, output freed inside the chain (no eager tensor is made after a capture).
a = up(torch.randn(1, 512, 768))
wt = up(torch.randn(768, 1536) * 0.02)
bench("linear/token_768x1536", lambda: ttnn.deallocate(ttnn.linear(a, wt)))
json.dump(res, open(args.out, "w"), indent=1)
print(json.dumps(res))
