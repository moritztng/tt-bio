#!/usr/bin/env python3
"""permute_fwd_pk (channel-last -> channel-major, packer re-tiling) against torch and the gathering reblock_permute.

Parity is torch.equal against torch's permute (a bf16 copy must be exact), whole tensor and a channel slice at
c_off. Then ms/call (median of --reps, synced per call) for pk, reblock_permute (gather), ttnn.permute and clone.

    TT_VISIBLE_DEVICES=<card> python perf/bc4_block/fwd_pk_bench.py --ns 224,288 --cs 128 --out x.json
"""
import argparse, json, statistics, time

ap = argparse.ArgumentParser()
ap.add_argument("--ns", default="224,288,384,512")
ap.add_argument("--cs", default="128,256")
ap.add_argument("--reps", type=int, default=50)
ap.add_argument("--out")
A = ap.parse_args()

from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch, ttnn
from tt_bio import tenstorrent as T, reblock_permute as RB, permute_back_pk as PK

dev = T.get_device()
grid = tuple(T.COMPUTE_GRID_MAIN)


def timed(f):
    f(); ttnn.synchronize_device(dev)
    ms = []
    for _ in range(A.reps):
        t0 = time.perf_counter(); f(); ttnn.synchronize_device(dev); ms.append((time.perf_counter() - t0) * 1e3)
    return round(statistics.median(ms), 4)


gen = torch.Generator().manual_seed(0)
recs = []
for N in map(int, A.ns.split(",")):
    for C in map(int, A.cs.split(",")):
        h = torch.randn(1, N, N, C, generator=gen).bfloat16()
        x = ttnn.from_torch(h, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        ref = h.permute(0, 3, 1, 2)
        exact = torch.equal(ttnn.to_torch(PK.permute_fwd_pk(x, grid)), ref)
        # a channel slice from a 4x wider tensor, the trimul in-projection's shape
        hw = torch.randn(1, N, N, 4 * C, generator=gen).bfloat16()
        xw = ttnn.from_torch(hw, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        exact_slice = torch.equal(ttnn.to_torch(PK.permute_fwd_pk(xw, grid, C=C, c_off=2 * C)),
                                  hw[..., 2 * C:3 * C].permute(0, 3, 1, 2))
        r = dict(n=N, c=C, exact=exact, exact_slice=exact_slice,
                 pk_ms=timed(lambda: PK.permute_fwd_pk(x, grid)),
                 gather_ms=timed(lambda: RB.reblock_permute(x)),
                 pk_ms_again=timed(lambda: PK.permute_fwd_pk(x, grid)),
                 ttnn_permute_ms=timed(lambda: ttnn.permute(x, (0, 3, 1, 2))),
                 clone_ms=timed(lambda: ttnn.clone(x)))
        r["gather_exact"] = torch.equal(ttnn.to_torch(RB.reblock_permute(x)), ref)
        print(json.dumps(r), flush=True); recs.append(r)
        for t in (x, xw): ttnn.deallocate(t)
if A.out:
    open(A.out, "w").write("\n".join(json.dumps(r) for r in recs) + "\n")
