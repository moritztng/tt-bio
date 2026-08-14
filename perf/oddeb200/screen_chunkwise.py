#!/usr/bin/env python3
"""Which chunks lose bit-exactness when the row slice moves DRAM -> L1, and why it matters.

Pass 10 measured the L1 slice torch.equal True on chunk 0 and NOT bit-exact at the fold. This
localises it: run all 52 chunks a 512-row call issues, compare swiglu(DRAM slice) against
swiglu(L1 slice) per chunk, and report exactly which ones differ.

The answer decides the multi-day fusion in section 8.4. That fusion's premise is "delete the slice
with a custom reader and stay bit-exact because the matmul is untouched". Pass 10 showed that not
touching the matmul is NOT sufficient: moving only where the slice LANDS moved plDDT at the fold. So

  - if ONLY the ragged tail chunk (h=2, rows 510-512) differs, the mechanism is the odd shape and a
    fused reader that keeps the full-height blocks intact can still be bit-exact. Fusion premise
    survives.
  - if full-height h=10 chunks differ too, then feeding the same values through a different
    page-to-core mapping is enough to change the result on its own, and a reader that re-indexes
    rows cannot be assumed bit-exact by construction. Fusion premise is broken and section 8.4's
    torch.equal clause is at risk before any code is written.

Values are identical by construction in both arms (a slice is a copy), so anything this finds is
the downstream op folding differently, not the slice being wrong.
"""
import json, sys
from pathlib import Path

WT = Path("/home/ttuser/.coworker/wt/opendde-beat-b200")
sys.path.insert(0, str(WT))

import torch, ttnn
import tt_bio.tenstorrent as T

dev = T.get_device()
ckc = (ttnn.types.WormholeComputeKernelConfig
       if dev.arch() == ttnn.Arch.WORMHOLE_B0
       else ttnn.types.BlackholeComputeKernelConfig)(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
    fp32_dest_acc_en=True, packer_l1_acc=True)

B, H, W, C = 1, 512, 512, 384
HID = 4 * C
h = max(1, int(T.TRANSITION_H_CHUNK_SIZE * min(1.0, (1024 * 128) / (W * C))))
CG, L1, DRAM = T.CORE_GRID_MAIN, ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG
kw = dict(compute_kernel_config=ckc)

tt = lambda x: ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
g = torch.Generator().manual_seed(0)
x_dev = tt(torch.randn(B, H, W, C, generator=g, dtype=torch.float32).bfloat16().float())
nw, nb = tt(torch.randn(C, generator=g).float()), tt(torch.randn(C, generator=g).float())
w1 = tt(torch.randn(C, HID, generator=g).float())
w2 = tt(torch.randn(C, HID, generator=g).float())
w3 = tt(torch.randn(HID, C, generator=g).float())


def swiglu(c):
    n = ttnn.layer_norm(c, weight=nw, bias=nb, epsilon=1e-5, memory_config=L1, **kw)
    p = ttnn.linear(n, w1, activation="silu", memory_config=L1, dtype=ttnn.bfloat16, core_grid=CG, **kw)
    q = ttnn.linear(n, w2, memory_config=L1, dtype=ttnn.bfloat16, core_grid=CG, **kw)
    ttnn.deallocate(n)
    ttnn.multiply_(p, q)
    ttnn.deallocate(q)
    o = ttnn.linear(p, w3, dtype=ttnn.bfloat16, core_grid=CG, memory_config=DRAM, **kw)
    ttnn.deallocate(p)
    return o


rows = []
for s in range(0, H, h):
    e = min(s + h, H)
    cA = x_dev[:, s:e]
    oA = swiglu(cA); ttnn.deallocate(cA)
    tA = torch.Tensor(ttnn.to_torch(oA)).float(); ttnn.deallocate(oA)

    cB = ttnn.slice(x_dev, [0, s, 0, 0], [B, e, W, C], memory_config=L1)
    oB = swiglu(cB); ttnn.deallocate(cB)
    tB = torch.Tensor(ttnn.to_torch(oB)).float(); ttnn.deallocate(oB)

    eq = bool(torch.equal(tA, tB))
    md = float((tA - tB).abs().max())
    rows.append({"start": s, "rows": e - s, "ragged": (e - s) != h,
                 "torch_equal": eq, "max_abs_diff": md})
    if not eq or (e - s) != h:
        print(f"chunk s={s:4d} rows={e-s:3d} ragged={(e-s)!=h} equal={eq} maxdiff={md:.3e}", flush=True)

full = [r for r in rows if not r["ragged"]]
rag = [r for r in rows if r["ragged"]]
out = {
    "shape_full": [B, h, W, C], "n_chunks": len(rows),
    "full_height_chunks": len(full), "ragged_chunks": len(rag),
    "full_height_differing": sum(1 for r in full if not r["torch_equal"]),
    "ragged_differing": sum(1 for r in rag if not r["torch_equal"]),
    "max_abs_diff_over_all": max(r["max_abs_diff"] for r in rows),
    "verdict": None, "chunks": rows,
}
if out["full_height_differing"] == 0 and out["ragged_differing"] == 0:
    out["verdict"] = ("every chunk bit-exact in isolation -- the fold-level divergence comes from "
                      "the surrounding allocation sequence, not from any single chunk")
elif out["full_height_differing"] == 0:
    out["verdict"] = ("only the ragged tail differs -- the fusion premise survives if the reader "
                      "keeps full-height blocks intact")
else:
    out["verdict"] = ("full-height chunks differ: the same values through a different page-to-core "
                      "mapping change the result on their own, so a re-indexing reader is NOT "
                      "bit-exact by construction and section 8.4's torch.equal clause is at risk")
p = WT / "perf" / "oddeb200" / "screen_chunkwise.json"
p.write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps({k: out[k] for k in
                  ("n_chunks", "full_height_chunks", "ragged_chunks", "full_height_differing",
                   "ragged_differing", "max_abs_diff_over_all", "verdict")}, indent=1))
T.cleanup()
