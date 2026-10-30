#!/usr/bin/env python3
"""gap_mmgate: RFD3's transition silu(x @ fc1) * (x @ fc2) @ fc3 through mm_gate (RFD3_MM_GATE) against the three-op
path, both graded against float64. Times both. DiT page shape 704 x 768 -> 1536 and an atom-stream shape.
    python scripts/rfd3_port/gap_mmgate.py OUT.json [reps]
"""
import json, os, pathlib, statistics, sys, time, types

import torch

sys.path.insert(0, os.getcwd())
from tt_bio.main import ensure_p300_mesh_descriptor                      # noqa: E402
ensure_p300_mesh_descriptor()
import ttnn                                                               # noqa: E402
from tt_bio.rfd3 import model as M                                       # noqa: E402
from tt_bio.tenstorrent import get_device                                # noqa: E402

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "out/gap_mmgate.json")
N = int(sys.argv[2]) if len(sys.argv) > 2 else 20


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
    return o, statistics.median(ts)


def case(dev, ckc, rows, c, n, wdt):
    g = torch.Generator().manual_seed(rows + c + n)
    x = torch.randn(1, rows, c, generator=g).bfloat16()
    ws = [(torch.randn(k, m, generator=g) / k ** 0.5).bfloat16() for k, m in ((c, n), (c, n), (n, c))]
    dt = lambda t: ttnn.from_torch(t, dtype=wdt, layout=ttnn.TILE_LAYOUT, device=dev)
    blk = types.SimpleNamespace(compute_kernel_config=ckc, dtype=ttnn.bfloat16,
                                t_fc1=dt(ws[0]), t_fc2=dt(ws[1]), t_fc3=dt(ws[2]))
    blk._mm_gate = types.MethodType(M.RFD3AtomBlock._mm_gate, blk)
    tr = types.MethodType(M.RFD3AtomBlock._transition, blk)
    w64 = [ttnn.to_torch(getattr(blk, k)).double() for k in ("t_fc1", "t_fc2", "t_fc3")]
    x64 = x.double()
    ref = (torch.nn.functional.silu(x64 @ w64[0]) * (x64 @ w64[1])) @ w64[2]
    xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    res = {"rows": rows, "c": c, "n": n, "wdt": str(wdt)}
    for arm, on in (("three_op", False), ("mm_gate", True)):
        M._MM_GATE = on
        served = blk._mm_gate(xt) is not None
        o, ms = timeit(lambda: tr(xt), dev)
        y = ttnn.to_torch(o).double()
        res[arm] = {"served": served, "ms": round(ms, 4),
                    "rel_l2": float((y - ref).norm() / ref.norm()), "max_abs": float((y - ref).abs().max())}
    return res


def main():
    dev = get_device()
    ckc = M._default_compute_kernel_config()
    rows = [case(dev, ckc, r, c, n, w) for r, c, n in ((704, 768, 1536), (6080, 128, 256))
            for w in (ttnn.bfloat16, ttnn.bfloat8_b)]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"arch": str(dev.arch()), "cases": rows}, indent=1))
    for r in rows:
        print(json.dumps(r))


if __name__ == "__main__":
    main()
