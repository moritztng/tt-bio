#!/usr/bin/env python3
"""Is `_fc1_full()`'s concat byte-identical to the shipped `w.t()` fc1 weight?

The fold-level A/B's control arm reads `_fc1_full()`, so if that concat is not the same bytes as
main's `torch_to_tt("1.weight", transform=w.t())`, the whole base-vs-armA comparison is against a
model main does not ship. Checked at both dtypes the FFN can load: bf16 (default) and bfloat8_b
(fast mode), since bf8 shares an exponent per tile row and a column split could in principle
land inside a group.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import ttnn
import tt_bio.tenstorrent as T

from tt_bio.main import _detect_p300_devices, _find_ttnn_mesh_graph_descriptor

if _detect_p300_devices() and not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
    mgd = _find_ttnn_mesh_graph_descriptor("p150_mesh_graph_descriptor.textproto")
    if mgd:
        os.environ["TT_MESH_GRAPH_DESC_PATH"] = mgd

dev = T.get_device()
ok = True
for cz, d_ff in ((256, 1024), (2560, 6832)):
    for dt in (ttnn.bfloat16, ttnn.bfloat8_b):
        torch.manual_seed(0)
        w = torch.randn(2 * d_ff, cz) * 0.02          # checkpoint layout [2*d_ff, cz]
        f = lambda t: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=dt)
        full = f(w.t())
        a, b = f(w[:d_ff].t()), f(w[d_ff:].t())
        cat = ttnn.concat([a, b], dim=-1)
        same = torch.equal(ttnn.to_torch(full), ttnn.to_torch(cat))
        ok &= same
        print(f"cz={cz:5d} d_ff={d_ff:5d} {str(dt):28s} concat==w.t(): {same}", flush=True)
        for t in (full, a, b, cat):
            ttnn.deallocate(t)
print("ALL BYTE-IDENTICAL" if ok else "MISMATCH")
sys.exit(0 if ok else 1)
