#!/usr/bin/env python3
"""fp32 compute roof on THIS card, plus the fp32 DRAM read/write roofs.

The 298 aa diffusion stage runs entirely in float32 on device (every tensor in the per-op record
is FLOAT32), so scoring its matmuls against the bf16 HiFi4 roof puts them on the wrong side of the
knee. Same method as roofs_card.py's compute leg -- square matmul, HiFi4, the production compute
kernel config (fp32_dest_acc_en + packer_l1_acc) and CORE_GRID_MAIN -- with fp32 operands, plus a
bf16 arm at the same shapes so the fp32/bf16 ratio is measured on one card in one process.
"""
import json, sys, time, statistics as st
import torch, ttnn
from tt_bio.tenstorrent import get_device, CORE_GRID_MAIN

DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
dev = get_device()
ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                            fp32_dest_acc_en=True, packer_l1_acc=True)


def timed(fn, warm=4, pipe=4, reps=5):
    for _ in range(warm):
        fn()
    ttnn.synchronize_device(dev)
    o = []
    for _ in range(reps):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(pipe):
            fn()
        ttnn.synchronize_device(dev)
        o.append((time.perf_counter() - t0) / pipe)
    return st.median(o)


res = {"compute": {}, "dram": {}}
for dt, name in ((ttnn.float32, "fp32"), (ttnn.bfloat16, "bf16")):
    for N in (1024, 2048, 4096):
        try:
            a = ttnn.from_torch(torch.randn(N, N), dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev,
                                memory_config=DRAM)
            b = ttnn.from_torch(torch.randn(N, N), dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev,
                                memory_config=DRAM)
            t = timed(lambda: ttnn.deallocate(ttnn.matmul(a, b, compute_kernel_config=ckc,
                                                          core_grid=CORE_GRID_MAIN,
                                                          memory_config=DRAM)))
            res["compute"][f"{name}_{N}"] = {"ms": round(t * 1e3, 4),
                                             "tflops": round(2 * N ** 3 / t / 1e12, 2)}
            ttnn.deallocate(a); ttnn.deallocate(b)
        except Exception as e:                                       # noqa: BLE001
            res["compute"][f"{name}_{N}"] = {"err": str(e)[:120]}
        print(f"  {name} N={N}: {res['compute'][f'{name}_{N}']}", flush=True)

# DRAM roofs with fp32 tensors: bytes are bytes, but measure rather than assume.
for dt, name in ((ttnn.float32, "fp32"), (ttnn.bfloat16, "bf16")):
    nrow, ncol = 8192, 4096
    nbytes = nrow * ncol * (4 if dt == ttnn.float32 else 2)
    xd = ttnn.from_torch(torch.randn(nrow, ncol), dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev,
                         memory_config=DRAM)
    tr = timed(lambda: ttnn.deallocate(ttnn.clone(xd, memory_config=L1)))
    res["dram"][f"{name}_read_GBs"] = round(nbytes / tr / 1e9, 1)
    ttnn.deallocate(xd)
    xl = ttnn.from_torch(torch.randn(nrow, ncol), dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev,
                         memory_config=L1)
    tw = timed(lambda: ttnn.deallocate(ttnn.clone(xl, memory_config=DRAM)))
    res["dram"][f"{name}_write_GBs"] = round(nbytes / tw / 1e9, 1)
    ttnn.deallocate(xl)
    print(f"  {name} MB={nbytes/1e6:.1f} read={res['dram'][f'{name}_read_GBs']} "
          f"write={res['dram'][f'{name}_write_GBs']}", flush=True)

peak_fp32 = max((v.get("tflops", 0) for k, v in res["compute"].items() if k.startswith("fp32")),
                default=0.0)
peak_bf16 = max((v.get("tflops", 0) for k, v in res["compute"].items() if k.startswith("bf16")),
                default=0.0)
res["fp32_peak_TFLOPs"] = peak_fp32
res["bf16_peak_TFLOPs"] = peak_bf16
res["fp32_over_bf16"] = round(peak_fp32 / peak_bf16, 3) if peak_bf16 else None
print(f"FP32_COMPUTE_ROOF {peak_fp32} TFLOP/s   BF16 {peak_bf16} TFLOP/s   "
      f"ratio {res['fp32_over_bf16']}", flush=True)
json.dump(res, open(sys.argv[1], "w"), indent=2)
