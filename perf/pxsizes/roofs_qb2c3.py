#!/usr/bin/env python3
"""The copy and matmul roofs of qb2 card 3, measured, because an asserted roof has been wrong here.

This lineage once published 668 GB/s on a ~400 GB/s card, and separately made an op sitting at 96 %
of its read roof read as 13 % of roof by leaving traffic out of the byte model. So every byte model
in state/protenix-v2-sizes-perf.md is checked against these two numbers in BOTH directions: implied
GB/s against the copy roof, and implied TFLOP/s against the matmul roof.

Roofs are per card. Card 3 is not card 0 and the grid is read off the device, not hardcoded.
"""
import json, os, statistics as st, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def bench(fn, warm=2, n=9):
    import ttnn
    for _ in range(warm):
        fn()
    ttnn.synchronize_device(DEV)
    ts = []
    for _ in range(n):
        ttnn.synchronize_device(DEV)
        t0 = time.perf_counter()
        fn()
        ttnn.synchronize_device(DEV)
        ts.append(time.perf_counter() - t0)
    return st.median(ts)


import ttnn
import tt_bio.tenstorrent as T

DEV = T.get_device()
grid = DEV.compute_with_storage_grid_size()
res = {"host": "qb2", "chip": os.environ.get("TT_VISIBLE_DEVICES", "?"),
       "ttnn": __import__("importlib.metadata", fromlist=["x"]).version("ttnn"),
       "grid": [grid.x, grid.y], "cores": grid.x * grid.y,
       "l1_size_bytes": DEV.l1_size_per_core(), "dram_channels": DEV.num_dram_channels(),
       "copy": [], "matmul": []}

# --- copy roof: DRAM -> DRAM, the traffic every residency argument is priced against -------------
for N, C in [(128, 256), (256, 256), (512, 256), (768, 256), (1024, 256), (1024, 384)]:
    x = ttnn.from_torch(__import__("torch").zeros(1, N, N, C, dtype=__import__("torch").bfloat16),
                        layout=ttnn.TILE_LAYOUT, device=DEV, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    try:
        s = bench(lambda: ttnn.deallocate(ttnn.clone(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)))
        by = N * N * C * 2
        res["copy"].append({"shape": [N, N, C], "ms": round(s * 1e3, 4),
                            "bytes_each_way": by,
                            "read_plus_write_GBps": round(2 * by / s / 1e9, 1)})
        print(res["copy"][-1], flush=True)
    except Exception as e:                                                       # noqa: BLE001
        res["copy"].append({"shape": [N, N, C], "error": f"{type(e).__name__}: {str(e)[:200]}"})
        print(res["copy"][-1], flush=True)
    ttnn.deallocate(x)

# --- matmul roof: HiFi4 bf16 at a production pair shape ------------------------------------------
# The trimul triangle contraction is [c, N, N] x [c, N, N] batched over the channel chunk, so the
# rate that matters is a batched bf16 HiFi4 matmul at the real per-chunk shape, not a square GEMM.
ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4,
                                       math_approx_mode=False, fp32_dest_acc_en=False,
                                       packer_l1_acc=True)
import torch
for B, M, K, Nn in [(32, 512, 512, 512), (32, 1024, 1024, 1024), (1, 4096, 256, 256)]:
    try:
        aa = ttnn.from_torch(torch.zeros(1, B, M, K, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                             device=DEV, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        bb = ttnn.from_torch(torch.zeros(1, B, K, Nn, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                             device=DEV, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        s = bench(lambda: ttnn.deallocate(ttnn.matmul(aa, bb, compute_kernel_config=ckc,
                                                      core_grid=ttnn.CoreGrid(y=grid.y, x=grid.x),
                                                      memory_config=ttnn.DRAM_MEMORY_CONFIG)))
        fl = 2 * B * M * K * Nn
        res["matmul"].append({"shape": [B, M, K, Nn], "ms": round(s * 1e3, 4),
                              "flop": fl, "TFLOPs": round(fl / s / 1e12, 2)})
        print(res["matmul"][-1], flush=True)
        ttnn.deallocate(aa); ttnn.deallocate(bb)
    except Exception as e:                                                       # noqa: BLE001
        res["matmul"].append({"shape": [B, M, K, Nn], "error": f"{type(e).__name__}: {str(e)[:200]}"})
        print(res["matmul"][-1], flush=True)

out = ROOT / "perf" / "pxsizes" / "roofs_qb2c3.json"
out.write_text(json.dumps(res, indent=1))
print("wrote", out, flush=True)
