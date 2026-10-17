#!/usr/bin/env python3
"""reblock_permute_back at a ragged N (not a multiple of 32): parity, padding and ms/call against the stock path.

AF2-IG and BindCraft 2 run their pair track at 195, where the trimul's back move is still `ttnn.permute` into L1
plus a clone to DRAM plus the chunks' concat. For each (N, C, source buffer) this checks the kernel's logical output
`torch.equal` against torch, that the j-axis tile padding of the output is zero, the same for the in-place form
(each C-chunk written into its channel slice of one wide tensor), then times the kernel against the stock pair.
Run under `timeout` with TT_VISIBLE_DEVICES set; writes --out as JSON.
"""
import argparse, json, statistics, time

ap = argparse.ArgumentParser()
ap.add_argument("--ns", default="195,100,298,33,256")
ap.add_argument("--cs", default="32,64,128")
ap.add_argument("--reps", type=int, default=20)
ap.add_argument("--out", required=True)
A = ap.parse_args()

from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch, ttnn
from tt_bio import tenstorrent as T, reblock_permute as RB

dev = T.get_device()
RB.set_back_ragged(True)
L1, DRAM = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG


def timed(f, reps):
    f(); ttnn.synchronize_device(dev)
    ms = []
    for _ in range(reps):
        t0 = time.perf_counter()
        f(); ttnn.synchronize_device(dev)
        ms.append((time.perf_counter() - t0) * 1e3)
    return round(statistics.median(ms), 4)


def pad_zero(out, n):
    """Whether rows j >= n of the physical [1, n, Np, C] buffer are all zero."""
    p = out.cpu().to_torch_with_padded_shape()
    return bool((p[:, :n, n:, :] == 0).all()) if p.shape[2] > n else True


def stock(x):
    y = ttnn.permute(x, (0, 2, 3, 1), memory_config=L1)
    z = ttnn.clone(y, memory_config=DRAM)
    ttnn.deallocate(y)
    return z


gen = torch.Generator().manual_seed(0)
recs = []
for n in (int(v) for v in A.ns.split(",")):
    for c in (int(v) for v in A.cs.split(",")):
        xt = torch.randn(1, c, n, n, generator=gen).to(torch.bfloat16)
        ref = xt.permute(0, 2, 3, 1)
        for src in ("L1", "DRAM"):
            rec = dict(n=n, c=c, src=src)
            try:
                x = ttnn.from_torch(xt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                                    memory_config=L1 if src == "L1" else DRAM)
                rec["eligible"] = RB.eligible_back(x, DRAM)
                out = RB.reblock_permute_back(x, DRAM)
                ttnn.synchronize_device(dev)
                rec["exact"] = bool(torch.equal(ttnn.to_torch(out), ref))
                rec["pad_zero"] = pad_zero(out, n)
                ttnn.deallocate(out)
                # in place: two chunks of c into one [1, n, n, 2c]
                wide = ttnn.allocate_tensor_on_device(ttnn.Shape([1, n, n, 2 * c]), ttnn.bfloat16, ttnn.TILE_LAYOUT,
                                                      dev, DRAM)
                x2t = torch.randn(1, c, n, n, generator=gen).to(torch.bfloat16)
                x2 = ttnn.from_torch(x2t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                                     memory_config=L1 if src == "L1" else DRAM)
                RB.reblock_permute_back(x, DRAM, out=wide, c_off=0)
                RB.reblock_permute_back(x2, DRAM, out=wide, c_off=c)
                ttnn.synchronize_device(dev)
                rec["into_exact"] = bool(torch.equal(ttnn.to_torch(wide),
                                                     torch.cat([ref, x2t.permute(0, 2, 3, 1)], dim=-1)))
                rec["into_pad_zero"] = pad_zero(wide, n)
                s = stock(x); ttnn.synchronize_device(dev)
                rec["stock_exact"] = bool(torch.equal(ttnn.to_torch(s), ref))
                rec["stock_pad_zero"] = pad_zero(s, n)
                ttnn.deallocate(s)
                rec["kernel_ms"] = timed(lambda: ttnn.deallocate(RB.reblock_permute_back(x, DRAM)), A.reps)
                rec["into_ms"] = timed(lambda: RB.reblock_permute_back(x, DRAM, out=wide, c_off=0), A.reps)
                rec["stock_ms"] = timed(lambda: ttnn.deallocate(stock(x)), A.reps)
                for t in (x, x2, wide):
                    ttnn.deallocate(t)
            except Exception as e:  # noqa: BLE001  one shape failing must not hide the others
                rec["error"] = repr(e)[:400]
            print(json.dumps(rec), flush=True)
            recs.append(rec)
json.dump(recs, open(A.out, "w"), indent=1)
