#!/usr/bin/env python3
"""Re-measure the three roofs on THIS card (card 3), with the fp32 compute leg and the
128 MB read ladder the protenix-v2 diffusion per-op table needs.

protenix-v2 diffusion runs entirely in fp32 on device, so its matmuls must be scored
against the fp32 compute roof, not the bf16 one. Method follows the two scripts that
produced the card-0 / card-2 numbers (perf/ledger_298/roofs_card.py and roof_fp32.py),
so a delta against them is a card delta and not a method delta:
  compute : square matmul, HiFi4, fp32_dest_acc_en + packer_l1_acc, CORE_GRID_MAIN, DRAM out
  read    : DRAM-interleaved -> L1 clone, DRAM sees reads only
  write   : L1 -> DRAM clone, DRAM sees writes only
Every timed region synchronises immediately before the clock starts and before it stops.
"""
import json, sys, time, statistics as st
import torch, ttnn
import tt_bio.tenstorrent as T
from tt_bio.tenstorrent import get_device

DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
dev = get_device()
# Read CORE_GRID_MAIN off the module AFTER device open: _configure_active_compute_grid
# rebinds the module global to 13x10 on qb1, but an `from ... import CORE_GRID_MAIN` name
# stays pinned to the 11x10 module-load value (D3 §1 trap). Production call sites read the
# module global, so they run on 13x10=130; the roof must be measured on the same grid.
CORE_GRID_MAIN = T.CORE_GRID_MAIN
dg = dev.compute_with_storage_grid_size()
res = {"device": {"compute_grid": f"{dg.x}x{dg.y}",
                  "core_grid_main": f"{CORE_GRID_MAIN.x}x{CORE_GRID_MAIN.y}",
                  "arch": str(dev.arch())}}
print(f"compute_grid={dg.x}x{dg.y} core_grid_main={CORE_GRID_MAIN.x}x{CORE_GRID_MAIN.y}", flush=True)

ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                              fp32_dest_acc_en=True, packer_l1_acc=True)


def timed(fn, warm=5, pipe=6, reps=7):
    for _ in range(warm):
        fn()
    ttnn.synchronize_device(dev)
    out = []
    for _ in range(reps):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(pipe):
            fn()
        ttnn.synchronize_device(dev)
        out.append((time.perf_counter() - t0) / pipe)
    return st.median(out)


print("=== compute roof: square matmul, HiFi4, fp32_dest_acc + packer_l1_acc ===", flush=True)
compute = {}
for dt, name in ((ttnn.bfloat16, "bf16"), (ttnn.float32, "fp32")):
    for N in (2048, 4096, 6144):
        try:
            a = ttnn.from_torch(torch.randn(1, 1, N, N), dtype=dt, layout=ttnn.TILE_LAYOUT,
                                device=dev, memory_config=DRAM)
            b = ttnn.from_torch(torch.randn(1, 1, N, N), dtype=dt, layout=ttnn.TILE_LAYOUT,
                                device=dev, memory_config=DRAM)
            t = timed(lambda: ttnn.deallocate(
                ttnn.matmul(a, b, compute_kernel_config=ckc, core_grid=CORE_GRID_MAIN,
                            memory_config=DRAM)), warm=4, pipe=4, reps=5)
            tf = 2 * N ** 3 / t / 1e12
            compute[f"{name}_{N}"] = {"ms": round(t * 1e3, 4), "tflops": round(tf, 2)}
            print(f"  {name} N={N:<5} {t*1e3:9.4f} ms {tf:8.2f} TFLOP/s", flush=True)
            ttnn.deallocate(a); ttnn.deallocate(b)
        except Exception as e:  # noqa: BLE001
            compute[f"{name}_{N}"] = {"err": str(e)[:120]}
            print(f"  {name} N={N} ERR {str(e)[:80]}", flush=True)

peak_bf16 = max((v.get("tflops", 0) for k, v in compute.items() if k.startswith("bf16")), default=0.0)
peak_fp32 = max((v.get("tflops", 0) for k, v in compute.items() if k.startswith("fp32")), default=0.0)
res["compute_roof"] = {"runs": compute, "bf16_peak_TFLOPs": peak_bf16,
                       "fp32_peak_TFLOPs": peak_fp32,
                       "fp32_over_bf16": round(peak_fp32 / peak_bf16, 3) if peak_bf16 else None}
print(f"COMPUTE_ROOF bf16={peak_bf16} fp32={peak_fp32} ratio={res['compute_roof']['fp32_over_bf16']}", flush=True)

print("\n=== directional DRAM roofs (bf16 ladder to 128 MB) ===", flush=True)
rows = []
for mb in (8, 16, 32, 48, 64, 96, 128):
    nrow = int(mb * 1e6 / 2) // 4096
    shape, nbytes = (nrow, 4096), nrow * 4096 * 2
    r = {"MB": round(nbytes / 1e6, 2)}
    try:
        xd = ttnn.from_torch(torch.randn(*shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                             device=dev, memory_config=DRAM)
        t = timed(lambda: ttnn.deallocate(ttnn.clone(xd, memory_config=L1)), warm=3, pipe=4, reps=7)
        r["read_GBs"] = round(nbytes / t / 1e9, 1)
        ttnn.deallocate(xd)
    except Exception as e:  # noqa: BLE001
        r["read_err"] = str(e)[:100]
    try:
        xl = ttnn.from_torch(torch.randn(*shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                             device=dev, memory_config=L1)
        t = timed(lambda: ttnn.deallocate(ttnn.clone(xl, memory_config=DRAM)), warm=3, pipe=4, reps=7)
        r["write_GBs"] = round(nbytes / t / 1e9, 1)
        ttnn.deallocate(xl)
    except Exception as e:  # noqa: BLE001
        r["write_err"] = str(e)[:100]
    rows.append(r)
    print("  " + json.dumps(r), flush=True)

peak_r = max((r.get("read_GBs", 0) for r in rows), default=0.0)
peak_w = max((r.get("write_GBs", 0) for r in rows), default=0.0)
res["dram_roofs"] = {"runs": rows, "read_peak_GBs": peak_r, "write_peak_GBs": peak_w}
res["machine_balance_bf16"] = round(peak_bf16 * 1e12 / (peak_r * 1e9), 1) if peak_r else None
res["machine_balance_fp32"] = round(peak_fp32 * 1e12 / (peak_r * 1e9), 1) if peak_r else None
print(f"DRAM_READ_ROOF {peak_r} GB/s  DRAM_WRITE_ROOF {peak_w} GB/s", flush=True)
print(f"MACHINE_BALANCE bf16={res['machine_balance_bf16']} fp32={res['machine_balance_fp32']} FLOP/byte (read)", flush=True)

json.dump(res, open(sys.argv[1], "w"), indent=2)
print("wrote", sys.argv[1], flush=True)
