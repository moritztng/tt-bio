#!/usr/bin/env python3
"""Phase 4 -- size the next blocker: Fourier-slice gather and backprojection scatter.

Not a fix, a sizing. Even a perfect FFT leaves RELION's refinement inner loop doing two things this
card is known to be bad at: extracting a 2D central slice from a 3D Fourier volume (gather) and
accumulating a slice back into it (scatter-add). The inherited rates were measured on a contiguous
45.1M-element case, and an interpolation index pattern is a different access shape in a way that
could go either direction, so they are re-measured here at RELION's real shapes with a real
trilinear index tensor.

Shapes. A central slice of a box-N half-volume is N x (N/2+1) complex pixels, each trilinearly
interpolated from 8 voxels, so one slice is 4*N^2 gathers (8 voxels x N^2/2 complex pixels) and
backprojection is the same count of scatter-adds.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import ttnn

BOXES = (256, 384, 512)
REPS = 5


def timed(fn, dev, reps=REPS):
    fn()
    ttnn.synchronize_device(dev)
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ttnn.synchronize_device(dev)
        best = min(best, time.perf_counter() - t0)
    return best


def main():
    dev = ttnn.open_device(device_id=0)
    out = {"boxes": {}}
    try:
        torch.manual_seed(0)
        for N in BOXES:
            nvox = N * N * (N // 2 + 1)
            nidx = 4 * N * N            # 8 voxels x N*(N/2+1) pixels, rounded to 4N^2
            rec = {"n_voxels": nvox, "n_indices": nidx}

            # The 3D Fourier half-volume, flattened. Real component only: a complex volume is two
            # of these and doubles every number below, which is stated rather than measured twice.
            vol = ttnn.from_torch(
                torch.randn(1, 1, 1, nvox, dtype=torch.float32),
                dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
            # A real trilinear index pattern: 8 neighbours of a rotated plane, not a contiguous run.
            base = torch.randint(0, max(nvox - 8, 1), (1, 1, 1, nidx // 8), dtype=torch.int32)
            idx_t = (base.repeat_interleave(8, dim=-1)
                     + torch.arange(8, dtype=torch.int32).repeat(nidx // 8)).clamp(0, nvox - 1)
            # ttnn.gather requires UINT32 or UINT16 indices; INT32 is refused outright.
            idx = ttnn.from_torch(idx_t.to(torch.int64), dtype=ttnn.uint32,
                                  layout=ttnn.TILE_LAYOUT, device=dev)
            src = ttnn.from_torch(
                torch.randn(1, 1, 1, nidx, dtype=torch.float32),
                dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)

            # elementwise reference on the same element count: the rate everything is measured against
            a = ttnn.from_torch(torch.randn(1, 1, 1, nidx, dtype=torch.float32),
                                dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
            try:
                s = timed(lambda: ttnn.add(a, src), dev)
                rec["add"] = {"s": s, "g_elem_per_s": nidx / s / 1e9}
            except Exception as e:                                      # noqa: BLE001
                rec["add"] = {"error": str(e)[:200]}

            # ttnn.scatter refuses fp32 TILE outright -- "Scatter doesn't work for fp32 tiled
            # tensors yet" -- so the scatter arms run in bf16. That is not a free substitution for
            # backprojection, which accumulates hundreds of thousands of slices into one volume and
            # needs fp32 to do it. Recorded as a capability gap, not as a measurement choice.
            volb = ttnn.from_torch(torch.randn(1, 1, 1, nvox, dtype=torch.float32),
                                   dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
            srcb = ttnn.from_torch(torch.randn(1, 1, 1, nidx, dtype=torch.float32),
                                   dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
            for name, fn in (
                ("gather", lambda: ttnn.gather(vol, dim=-1, index=idx)),
                ("scatter_bf16", lambda: ttnn.scatter(volb, dim=-1, index=idx, src=srcb)),
                ("scatter_add_bf16", lambda: ttnn.scatter_add(volb, dim=-1, index=idx, src=srcb)),
            ):
                try:
                    s = timed(fn, dev)
                    rec[name] = {
                        "s": s,
                        "g_elem_per_s": nidx / s / 1e9,
                        "us_per_slice": s * 1e6,
                    }
                    print(f"N={N} {name:12s} {s*1e6:9.1f} us  {nidx/s/1e9:6.2f} G elem/s", flush=True)
                except Exception as e:                                  # noqa: BLE001
                    rec[name] = {"error": str(e)[:300]}
                    print(f"N={N} {name:12s} ERROR {str(e)[:120]}", flush=True)

            out["boxes"][N] = rec
            json.dump(out, open(Path(__file__).resolve().parent / "screen_s5.json", "w"), indent=1)
            for t in (vol, idx, src, a, volb, srcb):
                ttnn.deallocate(t)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
