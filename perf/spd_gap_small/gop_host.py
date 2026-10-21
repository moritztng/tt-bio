"""Host cost per call of a generic_op fused kernel (addcmul_at) against stock ttnn ops at the token DiT's shapes.

cen15 (Boltz-2 WH --fast) shows ~110-125 us of idle before each addcmul_at against ~35-46 us before stock ops, 19,200
calls a fold. This separates the two halves: `host_us` is the Python+dispatch time per call with no sync (what the
host spends enqueuing), `wall_us` is per call over the whole batch including the final sync (what the device ran at).
host_us > wall_us of the device-only ops means the op is host-bound when issued back to back.
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
from tt_bio.addcmul_at import addcmul_at  # noqa: E402
import ttnn  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=2000)
args = ap.parse_args()

dev = T.get_device()
up = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
torch.manual_seed(0)
x, y, z = up(torch.randn(1, 512, 1536)), up(torch.randn(1, 512, 768)), up(torch.randn(1, 512, 1536))
z7 = up(torch.randn(1, 512, 768))

cases = {
    "addcmul_at": lambda: addcmul_at(x, y, z, 768, x_off=768, z_off=0),
    "ttnn.addcmul": lambda: ttnn.addcmul(y, y, z7),
    "ttnn.multiply": lambda: ttnn.multiply(y, z7),
    "ttnn.add": lambda: ttnn.add(y, z7),
}
ref = ttnn.to_torch(cases["addcmul_at"]()).float()
check = (ttnn.to_torch(x).float()[..., 768:] + ttnn.to_torch(y).float() * ttnn.to_torch(z).float()[..., :768])
res = {"rel_err_vs_host": float((ref - check).norm() / check.norm()), "n": args.n, "cases": {}}
for name, f in cases.items():
    for _ in range(50):
        ttnn.deallocate(f())
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for i in range(args.n):
        o = f()
        ttnn.deallocate(o)
    t1 = time.perf_counter()
    ttnn.synchronize_device(dev)
    t2 = time.perf_counter()
    res["cases"][name] = {"host_us": round((t1 - t0) / args.n * 1e6, 2), "wall_us": round((t2 - t0) / args.n * 1e6, 2)}
    print(name, res["cases"][name], flush=True)
json.dump(res, open(args.out, "w"), indent=1)
print(json.dumps(res))
