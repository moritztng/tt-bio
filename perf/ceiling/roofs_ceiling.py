#!/usr/bin/env python3
"""Roofs on THIS card, re-measured. W8 (ceiling model) inherits nothing.

Three roofs, each measured in the direction the ceiling model needs:
  compute  : square bf16 matmul, HiFi4, result in L1 (a DRAM result makes it a
             memory-bound measurement, which is how 100.6 TFLOP/s got mislabelled).
  DRAM read: DRAM -> L1 clone, DRAM sees reads only.
  DRAM write: L1 -> DRAM clone, DRAM sees writes only.

    TT_VISIBLE_DEVICES=3 python3 perf/ceiling/roofs_ceiling.py out.json
"""
import json
import statistics as st
import sys
import time

import torch

import ttnn
from tt_bio.tenstorrent import get_device

DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG


def timed(dev, fn, warm=5, pipe=6, reps=5):
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
        o.append((time.perf_counter() - t0) * 1e3 / pipe)
    return st.median(o)


def main():
    dev = get_device()
    res = {"ttnn": ttnn.__version__ if hasattr(ttnn, "__version__") else "?"}
    torch.manual_seed(0)

    ckc_op = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)
    ckc_plain = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=False, packer_l1_acc=False)

    print("=== compute roof ===", flush=True)
    roof = {}
    for n in (2048, 3072, 4096):
        a = ttnn.from_torch(torch.randn(1, 1, n, n), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        b = ttnn.from_torch(torch.randn(1, 1, n, n), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        gf = 2 * n ** 3 / 1e9
        for lbl, kc, omem in (("op_ckc_oL1", ckc_op, L1), ("op_ckc_oDRAM", ckc_op, DRAM),
                              ("plain_ckc_oL1", ckc_plain, L1)):
            try:
                ms = timed(dev, lambda: ttnn.deallocate(
                    ttnn.matmul(a, b, compute_kernel_config=kc, memory_config=omem)))
            except Exception as e:
                print(f"  N={n} {lbl} ERR {str(e)[:70]}", flush=True)
                continue
            tf = gf / (ms / 1e3) / 1e3
            roof[f"{n}_{lbl}"] = {"ms": round(ms, 4), "tflops": round(tf, 2)}
            print(f"  N={n:<5} {lbl:16s} {ms:9.4f} ms {tf:8.2f} TFLOP/s", flush=True)
        ttnn.deallocate(a)
        ttnn.deallocate(b)
    res["compute"] = {"runs": roof, "peak_TFLOPs": max(v["tflops"] for v in roof.values())}

    print("=== DRAM roofs ===", flush=True)
    band = {}
    for mb in (16, 32, 64):
        rows = int(mb * 1e6 / 2) // 4096
        nbytes = rows * 4096 * 2
        xd = ttnn.from_torch(torch.randn(rows, 4096), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                             device=dev, memory_config=DRAM)
        ms = timed(dev, lambda: ttnn.deallocate(ttnn.clone(xd, memory_config=L1)), warm=3, pipe=4, reps=5)
        rd = nbytes / (ms / 1e3) / 1e9
        xl = ttnn.clone(xd, memory_config=L1)
        ms2 = timed(dev, lambda: ttnn.deallocate(ttnn.clone(xl, memory_config=DRAM)), warm=3, pipe=4, reps=5)
        wr = nbytes / (ms2 / 1e3) / 1e9
        band[f"{mb}MB"] = {"read_GBs": round(rd, 1), "write_GBs": round(wr, 1)}
        print(f"  {nbytes/1e6:6.1f} MB  read {rd:7.1f} GB/s   write {wr:7.1f} GB/s", flush=True)
        ttnn.deallocate(xl)
        ttnn.deallocate(xd)
    res["dram"] = {"runs": band,
                   "read_GBs": max(v["read_GBs"] for v in band.values()),
                   "write_GBs": max(v["write_GBs"] for v in band.values())}
    print(f"ROOFS compute {res['compute']['peak_TFLOPs']:.1f} TFLOP/s  "
          f"read {res['dram']['read_GBs']:.1f} GB/s  write {res['dram']['write_GBs']:.1f} GB/s", flush=True)
    json.dump(res, open(sys.argv[1], "w"), indent=2)
    print("wrote", sys.argv[1], flush=True)


main()
