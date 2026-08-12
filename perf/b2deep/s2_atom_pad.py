#!/usr/bin/env python3
"""S2 off-fold screen: the atom-level q pad, three ways.

`AttentionPairBias.__call__`'s atom-level branch (tenstorrent.py) widens q from ATOM_WINDOW=32 to
ATOM_DIM=128 on dim -2 by leaving TILE layout, padding in ROW_MAJOR, and tilizing back -- 1200 times
per 512 aa fold. The pad is tile-aligned (32 -> 128 adds exactly 3 tiles of zeros), so it needs no
row-granular work at all, and `ttnn-untilize-single-core-fallback` records untilize silently falling
back to one core for a 36x degradation that differs per chip.

Three forms at the production shape, on the real card. The true window count K is whatever the fold
uses; a ladder brackets it so this screen does not depend on the census landing first.

  A  to_layout(ROW_MAJOR) -> pad -> to_layout(TILE)      what ships
  B  concat([q, cached_zeros], dim=-2) in TILE            no layout change
  C  pad(q) directly in TILE                             if ttnn allows it

torch.equal against A decides correctness; median of 7 after 2 warm decides the number.
Kill rule from the plan: if A is under 100 us/call the whole class is worth <0.12 s and S2 stops.
"""
import json, os, statistics as st, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch, ttnn                                                            # noqa: E402
import tt_bio.tenstorrent as T                                                # noqa: E402
from tt_bio.main import _detect_p300_devices, _find_ttnn_mesh_graph_descriptor  # noqa: E402

if _detect_p300_devices() and not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
    mgd = _find_ttnn_mesh_graph_descriptor("p150_mesh_graph_descriptor.textproto")
    if mgd:
        os.environ["TT_MESH_GRAPH_DESC_PATH"] = mgd

dev = T.get_device()
g = dev.compute_with_storage_grid_size()
W, D = T.ATOM_WINDOW, T.ATOM_DIM
PAD = D - W
out = {"host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
       "grid": [g.x, g.y], "atom_window": W, "atom_dim": D, "rows": []}


def bench(fn, n=7, warm=2):
    for _ in range(warm):
        r = fn(); ttnn.synchronize_device(dev); ttnn.deallocate(r)
    ts = []
    for _ in range(n):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        r = fn()
        ttnn.synchronize_device(dev)
        ts.append(time.perf_counter() - t0)
        ttnn.deallocate(r)
    return st.median(ts) * 1e6                                   # us


# K=224 is the production window count at 512 aa, MEASURED by the S0 split census
# (body:AttentionPairBias|atom|1x224x32x128, 1200 calls, 1723.8 ms). The rest of the ladder is
# there to show whether the chain scales with K or is dominated by a fixed cost.
for K in (224, 64, 448):
    qt = torch.randn(1, K, W, D, dtype=torch.bfloat16)
    q = ttnn.from_torch(qt, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=dev,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG)
    zeros = ttnn.from_torch(torch.zeros(1, K, PAD, D, dtype=torch.bfloat16),
                            layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=dev,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def form_a():
        x = ttnn.to_layout(q, ttnn.ROW_MAJOR_LAYOUT)
        y = ttnn.pad(x, [[0, 0], [0, 0], [0, PAD], [0, 0]], 0.0)
        ttnn.deallocate(x)
        z = ttnn.to_layout(y, ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        ttnn.deallocate(y)
        return z

    def form_b():
        return ttnn.concat([q, zeros], dim=-2)

    def form_c():
        return ttnn.pad(q, [[0, 0], [0, 0], [0, PAD], [0, 0]], 0.0)

    row = {"K": K, "shape": [1, K, W, D], "MiB": round(K * W * D * 2 / 2**20, 3)}
    ref = None
    for nm, fn in (("A_rowmajor_chain", form_a), ("B_tile_concat", form_b), ("C_tile_pad", form_c)):
        try:
            r = fn()
            t = ttnn.to_torch(r)
            ttnn.deallocate(r)
            if ref is None:
                ref = t
                row[f"{nm}_equal_to_A"] = True
            else:
                row[f"{nm}_equal_to_A"] = bool(t.shape == ref.shape and torch.equal(t, ref))
            row[f"{nm}_us"] = round(bench(fn), 2)
        except Exception as e:                                                 # noqa: BLE001
            row[f"{nm}_error"] = f"{type(e).__name__}: {e}"[:200]
    if row.get("A_rowmajor_chain_us") and row.get("B_tile_concat_us"):
        row["speedup_B_over_A"] = round(row["A_rowmajor_chain_us"] / row["B_tile_concat_us"], 4)
    if row.get("A_rowmajor_chain_us") and row.get("C_tile_pad_us"):
        row["speedup_C_over_A"] = round(row["A_rowmajor_chain_us"] / row["C_tile_pad_us"], 4)
    out["rows"].append(row)
    print(json.dumps(row), flush=True)
    ttnn.deallocate(q)
    ttnn.deallocate(zeros)

Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
print("wrote", sys.argv[1], flush=True)
T.cleanup()
