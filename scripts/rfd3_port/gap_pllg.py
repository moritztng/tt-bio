#!/usr/bin/env python3
"""gap_pllg: the encoder/decoder neighbour pair gather off the resident P_LL table, embedding + to_layout (shipped
until RFD3_PLL_GATHER) against gather_tiled with 16-channel rows. Must be bit-equal; times both.
Page cell: 6051 atoms x 128 neighbours off a 6051^2 x 16 table. Also a partial-width case (K=100).
    python scripts/rfd3_port/gap_pllg.py OUT.json [reps]
"""
import json, os, pathlib, statistics, sys, time

import torch

sys.path.insert(0, os.getcwd())
from tt_bio.main import ensure_p300_mesh_descriptor                      # noqa: E402
ensure_p300_mesh_descriptor()
import ttnn                                                               # noqa: E402
from tt_bio.rfd3 import model as M                                       # noqa: E402
from tt_bio.tenstorrent import get_device                                # noqa: E402

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "out/gap_pllg.json")
N = int(sys.argv[2]) if len(sys.argv) > 2 else 10


def timeit(fn, dev):
    o = fn()
    ttnn.synchronize_device(dev)
    ts = []
    for _ in range(N):
        ttnn.deallocate(o)
        t0 = time.perf_counter()
        o = fn()
        ttnn.synchronize_device(dev)
        ts.append((time.perf_counter() - t0) * 1e3)
    return o, statistics.median(ts), min(ts)


def case(dev, L, K):
    g = torch.Generator().manual_seed(L + K)
    p = torch.randn(1, L, L, 16, generator=g).bfloat16().float()
    ind = torch.randint(0, L, (1, L, K), generator=g, dtype=torch.int32)
    res = {"L": L, "K": K}
    outs = {}
    for arm in ("embedding", "gather_tiled"):
        M._PLL_GATHER = arm == "gather_tiled"
        cache = {}
        o, med, mn = timeit(lambda: M._sparse_pair_gather(cache, p, ind, dev, ttnn.bfloat16), dev)
        outs[arm] = ttnn.to_torch(o).float()
        res[arm] = {"ms_median": round(med, 3), "ms_min": round(mn, 3), "shape": list(o.shape),
                    "padded": [int(d) for d in o.padded_shape]}
        ttnn.deallocate(o)
        for _, t in cache.get("tables", {}).values():
            ttnn.deallocate(t)
    ref = p[0].reshape(L * L, 16)[(ind[0].long() + torch.arange(L).reshape(L, 1) * L)].unsqueeze(0)
    res["bit_equal"] = bool(torch.equal(outs["embedding"], outs["gather_tiled"]))
    res["equal_host"] = bool(torch.equal(outs["gather_tiled"], ref))
    return res


def main():
    dev = get_device()
    rows = [case(dev, 6051, 128), case(dev, 1000, 100)]
    for r in rows:
        print(r, flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"arch": str(dev.arch()), "cases": rows}, indent=1))


main()
