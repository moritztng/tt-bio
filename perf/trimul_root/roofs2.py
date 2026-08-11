#!/usr/bin/env python3
"""The two roofs the first probe could not allocate: DRAM read and L1->L1 copy, swept small."""
import json, statistics as st, time
from pathlib import Path
import torch, ttnn
from tt_bio.tenstorrent import get_device

dev = get_device()
out = {"cores": 110}


def pipe(fn, warm=3, n=6):
    for _ in range(warm):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    rs = [fn() for _ in range(n)]
    ttnn.synchronize_device(dev)
    ms = (time.perf_counter() - t0) * 1e3 / n
    for r in rs:
        ttnn.deallocate(r)
    return ms


for mb in (8, 16, 24, 32):
    rows = int(mb * 1e6 / 2) // 4096
    nb = rows * 4096 * 2
    for name, src, dst in (("read", ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG),
                           ("l1copy", ttnn.L1_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG)):
        try:
            x = ttnn.from_torch(torch.randn(rows, 4096), dtype=ttnn.bfloat16,
                                layout=ttnn.TILE_LAYOUT, device=dev, memory_config=src)
            ms = pipe(lambda: ttnn.clone(x, memory_config=dst))
            g = nb / (ms * 1e-3) / 1e9
            out[f"{name}_{mb}MB_gbs"] = round(g, 1)
            print(f"{name} {nb/1e6:.1f} MB: {g:.1f} GB/s each way ({ms:.4f} ms)", flush=True)
            ttnn.deallocate(x)
        except Exception as e:
            out[f"{name}_{mb}MB_err"] = str(e)[:120]
            print(f"{name} {mb} MB: ERR {str(e)[:110]}", flush=True)

# the contraction's own operand shape, L1 vs DRAM, as a copy: what a layout op would pay
for shape in [(1, 32, 512, 512)],:
    pass
s = (1, 32, 512, 512)
nb = 32 * 512 * 512 * 2
for name, src, dst in (("chunk_dram2dram", ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
                       ("chunk_l1_2_l1", ttnn.L1_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG),
                       ("chunk_dram2l1", ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG)):
    try:
        x = ttnn.from_torch(torch.randn(*s), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                            device=dev, memory_config=src)
        ms = pipe(lambda: ttnn.clone(x, memory_config=dst))
        out[f"{name}_gbs_each_way"] = round(nb / (ms * 1e-3) / 1e9, 1)
        out[f"{name}_ms"] = round(ms, 4)
        print(f"{name} [1,32,512,512] {nb/2**20:.1f} MiB: {ms:.4f} ms = "
              f"{nb / (ms * 1e-3) / 1e9:.1f} GB/s each way", flush=True)
        ttnn.deallocate(x)
    except Exception as e:
        out[f"{name}_err"] = str(e)[:140]
        print(f"{name}: ERR {str(e)[:130]}", flush=True)

# and the permute the channel loop really runs, DRAM vs L1
for name, mem in (("permute_dram", ttnn.DRAM_MEMORY_CONFIG), ("permute_l1", ttnn.L1_MEMORY_CONFIG)):
    try:
        x = ttnn.from_torch(torch.randn(1, 512, 512, 32), dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mem)
        ms = pipe(lambda: ttnn.permute(x, (0, 3, 1, 2), memory_config=mem))
        out[f"{name}_ms"] = round(ms, 4)
        out[f"{name}_gbs_each_way"] = round(nb / (ms * 1e-3) / 1e9, 1)
        print(f"{name} (0,3,1,2) on [1,512,512,32]: {ms:.4f} ms = "
              f"{nb / (ms * 1e-3) / 1e9:.1f} GB/s each way", flush=True)
        ttnn.deallocate(x)
    except Exception as e:
        out[f"{name}_err"] = str(e)[:140]
        print(f"{name}: ERR {str(e)[:130]}", flush=True)

Path("perf/trimul_root/roofs2_qb2c0.json").write_text(json.dumps(out, indent=2))
print("RESULT_JSON " + json.dumps(out), flush=True)
