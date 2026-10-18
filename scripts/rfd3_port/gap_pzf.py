#!/usr/bin/env python3
"""process_z gather with the tail fused in (gather_tiled scale/shift) vs gather_tiled + multiply + add (shipped).

    gap_pzf.py OUT.json [I=685] [N=4225] [REPS=30]

table [N,128] bf16 row-major, flat index [1,I,I] in [0,N), inv [1,I,I,1] and ainv [1,I,I,128] bf16 TILE: the shapes
_process_z_collapsed sees (N 65 at recycle 0, 4225 with the self bins). Per arm: ms/call (synchronized), error
against float64 of the same bf16 inputs, elements differing from the shipped chain, and whether every repeat of
the fused call (interleaved with the chain, so it runs after other programs) returns the same bits.
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
from tt_bio import gather_tiled as G                                      # noqa: E402

OUT = sys.argv[1]
I = int(sys.argv[2]) if len(sys.argv) > 2 else 685
N = int(sys.argv[3]) if len(sys.argv) > 3 else 4225
REPS = int(sys.argv[4]) if len(sys.argv) > 4 else 30
C = 128
torch.manual_seed(0)
dev = T.get_device()
tab_h = torch.randn(N, C).bfloat16()
flat = torch.randint(0, N, (1, I, I), dtype=torch.int32)
inv_h = (0.05 + torch.rand(1, I, I, 1)).bfloat16()
ainv_h = torch.randn(1, I, I, C).bfloat16()
up = lambda t: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)  # noqa: E731
inv, ainv = up(inv_h), up(ainv_h)
rm = ttnn.from_torch(tab_h, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, dtype=ttnn.bfloat16)
idx = ttnn.from_torch(G.pad_index(flat, I, I), layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, dtype=ttnn.uint32)
em_h = tab_h[flat.long()]
ref = em_h.double() * inv_h.double() + ainv_h.double()


def chain():
    em = G.gather_tiled(idx, rm, 1, I)
    sc = ttnn.multiply(em, inv)
    ttnn.deallocate(em)
    out = ttnn.add(sc, ainv)
    ttnn.deallocate(sc)
    return out


def fused():
    return G.gather_tiled(idx, rm, 1, I, scale=inv, shift=ainv)


res = {"I": I, "N": N, "reps": REPS, "card": os.environ.get("TT_VISIBLE_DEVICES"), "arch": str(dev.arch())}
outs = {}
for name, fn in (("chain", chain), ("fused", fused)):
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
stable = 0
for _ in range(5):
    ttnn.deallocate(chain())
    stable += int((ttnn.to_torch(fused()) != outs["fused"]).sum())
res["fused_repeat_neq"] = stable
print("fused_repeat_neq", stable, flush=True)
json.dump(res, open(OUT, "w"), indent=1)
