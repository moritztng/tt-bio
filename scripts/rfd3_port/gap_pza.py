#!/usr/bin/env python3
"""process_z tail microbench: z = em * inv + ainv as multiply + add (shipped) vs one ttnn.addcmul.

    gap_pza.py OUT.json [I=685] [REPS=30]

em, ainv [1,I,I,128] bf16 TILE, inv [1,I,I,1] bf16 TILE (the rms scale, broadcast over channels), the shapes and
value ranges _process_z_collapsed sees. Per arm: ms/call (synchronized), max/rms error against the float64 result
of the same bf16 inputs, and how many elements differ from the shipped chain.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.getcwd())
from tt_bio.main import ensure_p300_mesh_descriptor                       # noqa: E402

ensure_p300_mesh_descriptor()
import torch                                                              # noqa: E402
import ttnn                                                               # noqa: E402
from tt_bio import tenstorrent as T                                       # noqa: E402
from tt_bio import eltwise_fusion as EF                                   # noqa: E402

OUT = sys.argv[1]
I = int(sys.argv[2]) if len(sys.argv) > 2 else 685
REPS = int(sys.argv[3]) if len(sys.argv) > 3 else 30
C = 128
torch.manual_seed(0)
dev = T.get_device()
em_h = torch.randn(1, I, I, C).bfloat16()
inv_h = (0.5 + torch.rand(1, I, I, 1)).bfloat16()
ainv_h = torch.randn(1, I, I, C).bfloat16()
up = lambda t: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)  # noqa: E731
em, inv, ainv = up(em_h), up(inv_h), up(ainv_h)
ref = em_h.double() * inv_h.double() + ainv_h.double()


def chain():
    sc = ttnn.multiply(em, inv)
    out = ttnn.add(sc, ainv)
    ttnn.deallocate(sc)
    return out


def fused():
    EF.FUSE_MASK_ADD = True
    return EF.mask_add(ainv, em, inv)


res = {"I": I, "reps": REPS, "card": os.environ.get("TT_VISIBLE_DEVICES"), "arch": str(dev.arch())}
outs = {}
for name, fn in (("chain", chain), ("addcmul", fused)):
    outs[name] = ttnn.to_torch(fn())
    for _ in range(3):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(REPS):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(dev)
    err = (outs[name].double() - ref).abs()
    res[name] = {"ms_per_call": 1e3 * (time.perf_counter() - t0) / REPS, "max_abs_f64": float(err.max()),
                 "rms_f64": float(err.pow(2).mean().sqrt()),
                 "neq_chain": int((outs[name] != outs["chain"]).sum())}
    print(name, res[name], flush=True)
json.dump(res, open(OUT, "w"), indent=1)
