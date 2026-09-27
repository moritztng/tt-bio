#!/usr/bin/env python3
"""Which ttnn binary broadcasts are a function of their inputs? Fixed operands, dirtied DRAM between.

`lnbw_det.py` pinned the layer-norm backward's run-to-run divergence on one op: a bf16 full
tensor times an fp32 [...,1] column, where every same-dtype op in the closure repeats bit-exact.
This sweeps the class: op x dtype pair x broadcast kind, each repeated with garbage written to
freshly freed DRAM before it.

    TT_VISIBLE_DEVICES=<card> bcast_det.py --out <json>
"""
import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tt_bio.main import ensure_p300_mesh_descriptor                  # noqa: E402
ensure_p300_mesh_descriptor()

import numpy as np                                                    # noqa: E402
import torch                                                          # noqa: E402
import ttnn                                                           # noqa: E402
from tt_bio.tenstorrent import get_device                             # noqa: E402


def h(v):
    a = ttnn.to_torch(v).float().numpy()
    return hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()[:12]


def dirty(dev, n=8):
    ts = [ttnn.from_torch(torch.randn(1, 1024, 1024, 64) * 1e3, dtype=ttnn.bfloat16,
                          layout=ttnn.TILE_LAYOUT, device=dev) for _ in range(n)]
    for t in ts:
        ttnn.deallocate(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="1,64,384,128")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    shape = [int(s) for s in a.shape.split(",")]
    dev = get_device()
    torch.manual_seed(0)
    DT = {"bf16": ttnn.bfloat16, "fp32": ttnn.float32}
    kinds = {"full": shape, "col": shape[:-1] + [1], "row": [1] * (len(shape) - 1) + [shape[-1]],
             "vec": [shape[-1]]}
    host = {k: torch.randn(s) for k, s in kinds.items()}
    res = []
    for op, kind, (da, db) in itertools.product(
            ("multiply", "subtract", "add"), ("col", "row", "vec", "full"),
            itertools.product(DT, DT)):
        A = ttnn.from_torch(host["full"], dtype=DT[da], layout=ttnn.TILE_LAYOUT, device=dev)
        B = ttnn.from_torch(host[kind], dtype=DT[db], layout=ttnn.TILE_LAYOUT, device=dev)
        hs = []
        for r in range(a.reps):
            if r:
                dirty(dev)
            y = getattr(ttnn, op)(A, B)
            hs.append(h(y))
            ttnn.deallocate(y)
        ok = len(set(hs)) == 1
        res.append({"op": op, "b": kind, "a_dtype": da, "b_dtype": db, "repeat_bitexact": ok,
                    "hashes": hs})
        print(f"{op:9s} a={da} b={db}:{kind:4s} {'ok' if ok else 'NONDETERMINISTIC'}", flush=True)
        ttnn.deallocate(A); ttnn.deallocate(B)
    if a.out:
        a.out.write_text(json.dumps({"shape": shape, "rows": res}, indent=1))


if __name__ == "__main__":
    main()
