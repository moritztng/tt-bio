#!/usr/bin/env python3
"""Scatter arm only. ttnn.scatter refuses fp32 TILE inputs and refuses i32/uint32 TILE indices, so
this sweeps the layout/dtype combinations to find one it accepts and reports the rate for it."""
import json, time, itertools
from pathlib import Path
import torch, ttnn

BOXES = (256, 384, 512)
ROWS = 32


def timed(fn, dev, reps=3):
    fn(); ttnn.synchronize_device(dev)
    b = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ttnn.synchronize_device(dev)
        b = min(b, time.perf_counter() - t0)
    return b


def main():
    dev = ttnn.open_device(device_id=0)
    out = {}
    try:
        torch.manual_seed(0)
        for N in BOXES:
            nvox = (N * N * (N // 2 + 1) // 1024) * 1024
            nidx = 4 * N * N
            dom, nper = nvox // ROWS, nidx // ROWS
            rec = {"n_voxels": nvox, "n_indices": nidx, "domain_per_row": dom}
            it = torch.randint(0, dom - 8, (1, 1, ROWS, nper // 8), dtype=torch.int64)
            it = (it.repeat_interleave(8, -1)
                  + torch.arange(8, dtype=torch.int64).repeat(nper // 8)).clamp(0, dom - 1)
            for vdt, idt, ilay in itertools.product(
                    (ttnn.bfloat16, ttnn.float32), (ttnn.uint32, ttnn.int32),
                    (ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT)):
                key = f"{vdt}/{idt}/{'RM' if ilay == ttnn.ROW_MAJOR_LAYOUT else 'TILE'}"
                try:
                    vol = ttnn.from_torch(torch.randn(1, 1, ROWS, dom), dtype=vdt,
                                          layout=ttnn.TILE_LAYOUT, device=dev)
                    src = ttnn.from_torch(torch.randn(1, 1, ROWS, nper), dtype=vdt,
                                          layout=ttnn.TILE_LAYOUT, device=dev)
                    idx = ttnn.from_torch(it, dtype=idt, layout=ilay, device=dev)
                    s = timed(lambda: ttnn.scatter(vol, dim=-1, index=idx, src=src), dev)
                    rec[key] = {"s": s, "g_elem_per_s": nidx / s / 1e9, "us": s * 1e6}
                    print(f"N={N} scatter {key:34s} {s*1e6:11.1f} us  {nidx/s/1e9:8.5f} G elem/s",
                          flush=True)
                except Exception as e:                                   # noqa: BLE001
                    rec[key] = {"error": str(e).split("\n")[0][:180]}
                    print(f"N={N} scatter {key:34s} REFUSED", flush=True)
            out[N] = rec
            json.dump(out, open(Path(__file__).resolve().parent / "screen_s5_scatter.json", "w"),
                      indent=1)
    finally:
        ttnn.close_device(dev)


main()
