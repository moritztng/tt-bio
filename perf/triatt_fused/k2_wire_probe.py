#!/usr/bin/env python3
"""Does the wired K2 gate actually serve the fold's SDPA call, and at the WIDE q_chunk?

The first wiring passed `fits[-1]`, which is the production fallback, not the widest candidate --
`_tri_att_q_chunks` is documented "widest first, production pick last". That would have measured K2
with the wide-q lever thrown away. This drives `_tri_att_sdpa` exactly as the fold does and reports
which chunk was served, before any locked timing run is spent on it.
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch
import ttnn
from tt_bio import tenstorrent as T
from tt_bio import triatt_sdpa as PM

S, H, D = 512, 8, 32
dev = T.get_device()


def dram(t):
    return ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)


torch.manual_seed(0)
q = dram(torch.randn(S, H, S, D).to(torch.bfloat16))
k = dram(torch.randn(S, H, S, D).to(torch.bfloat16))
v = dram(torch.randn(S, H, S, D).to(torch.bfloat16))
bias = dram(torch.randn(1, H, S, S).to(torch.bfloat16))

fits = T._tri_att_q_chunks(S, S)
k_chunk = T._sdpa_chunks_shipped(S, S)[1]
print(json.dumps({"fits": list(fits), "k_chunk": k_chunk,
                  "shipped_pick": T._sdpa_chunks_shipped(S, S)[0]}), flush=True)

PM._ENABLED = False
ref = ttnn.to_torch(T._tri_att_sdpa(q, k, v, bias, D ** -0.5))
print(json.dumps({"stock_ok": True}), flush=True)

PM._ENABLED = True
PM.STATS[0] = PM.STATS[1] = 0
PM.REJECTS.clear()
got = ttnn.to_torch(T._tri_att_sdpa(q, k, v, bias, D ** -0.5))
print(json.dumps({"served": PM.STATS[0], "declined": PM.STATS[1],
                  "rejects": {f"{r}:{sh}": n for (r, sh), n in PM.REJECTS.items()},
                  "equal_to_stock": bool(torch.equal(got, ref))}), flush=True)
