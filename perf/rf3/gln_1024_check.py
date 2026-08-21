#!/usr/bin/env python3
"""Does the one-row flatten actually stop a 1024 aa confidence head, or is that a stale comment?

`confidence_head.global_layer_norm` claims the row fold is what makes 1024 aa run at all: the
one-row flatten pads a `(1, 1, 1, n)` tensor up to a full 32-row tile, so it asks the allocator
for 32x the tensor. That claim is load-bearing for flipping the default, so it gets measured on
the real pair shape rather than quoted. Both arms, same tensor, on the card.
"""
from __future__ import annotations
import enum, json, os, sys, time
from pathlib import Path

if not hasattr(enum, "StrEnum"):
    class StrEnum(str, enum.Enum):
        def __str__(self): return str(self.value)
    enum.StrEnum = StrEnum

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

import torch
import ttnn
from tt_bio.tenstorrent import get_device
import tt_bio.rf3.confidence_head as CH

AA = int(os.environ.get("GLN_AA", "1024"))
C = 128
device = get_device()
kcfg = ttnn.init_device_compute_kernel_config(
    device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
    fp32_dest_acc_en=True, packer_l1_acc=True)

nbytes = AA * AA * C * 2
print(f"pair rep [1, {AA}, {AA}, {C}] bf16 = {nbytes/1e9:.3f} GB")
print(f"one-row flatten (1,1,1,{AA*AA*C}) padded to 32 rows = {nbytes*32/1e9:.3f} GB")

out = {"aa": AA, "C": C, "pair_gb": nbytes / 1e9, "arms": {}}
for arm, rowfold in (("row_fold_off", False), ("row_fold_on", True)):
    CH._GLN_ROW_FOLD = rowfold
    x = ttnn.from_torch(torch.zeros(1, AA, AA, C, dtype=torch.bfloat16),
                        layout=ttnn.TILE_LAYOUT, device=device)
    try:
        t0 = time.perf_counter()
        y = CH.global_layer_norm(x, kcfg)
        ttnn.synchronize_device(device)
        dt = time.perf_counter() - t0
        out["arms"][arm] = {"ok": True, "s": round(dt, 4), "out_shape": list(y.shape)}
        print(f"  {arm:14s} OK   {dt:7.3f} s  out {list(y.shape)}")
        ttnn.deallocate(y)
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        out["arms"][arm] = {"ok": False, "error": msg[:400]}
        print(f"  {arm:14s} FAILED  {msg[:300]}")
    try:
        ttnn.deallocate(x)
    except Exception:
        pass

dst = REPO / "perf/rf3/accuracy" / f"gln_{AA}_check.json"
dst.write_text(json.dumps(out, indent=1) + "\n")
print("wrote", dst)
