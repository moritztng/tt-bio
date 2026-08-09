#!/usr/bin/env python3
"""Re-measure the three roofs on THIS card. Roofs are per-card; card 3 read
137.1 TFLOP/s HiFi4 / 410.9 GB/s read / 277.6 GB/s write and the ledger must not inherit them.

Methods follow the two scripts that produced the card-3 numbers, so a delta is a card delta and not
a method delta:
  compute : perf/qkv_rootcause/compute_roof.py  -- square bf16 matmul, HiFi4, 4 ckc/placement legs
  read    : perf/dram_write/roofs.py            -- DRAM-interleaved -> L1 clone, DRAM sees reads only
  write   : perf/dram_write/roofs.py            -- L1 -> DRAM clone, DRAM sees writes only
Every timed region synchronises immediately before the clock starts and before it stops.
"""
import json, sys, time, statistics as st
import torch, ttnn
from tt_bio.tenstorrent import get_device, CORE_GRID_MAIN

DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
dev = get_device()
dg = dev.compute_with_storage_grid_size()
res = {"device": {"compute_grid": f"{dg.x}x{dg.y}",
                  "core_grid_main": f"{CORE_GRID_MAIN.x}x{CORE_GRID_MAIN.y}",
                  "ttnn": getattr(ttnn, "__version__", "?")}}
print(f"compute_grid={dg.x}x{dg.y} core_grid_main={CORE_GRID_MAIN.x}x{CORE_GRID_MAIN.y}", flush=True)


def timed(fn, warm=5, pipe=6, reps=7):
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


print("=== compute roof: square bf16 matmul, HiFi4 ===", flush=True)
ckc_op = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                               fp32_dest_acc_en=True, packer_l1_acc=True)
ckc_plain = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                                   fp32_dest_acc_en=False, packer_l1_acc=False)
roof = {}
for n in (2048, 4096, 6144):
    a = ttnn.from_torch(torch.randn(1, 1, n, n), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    b = ttnn.from_torch(torch.randn(1, 1, n, n), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    gf = 2 * n ** 3 / 1e9
    for lbl, kc, omem in (("op_ckc_oDRAM", ckc_op, DRAM), ("op_ckc_oL1", ckc_op, L1),
                          ("plain_ckc_oDRAM", ckc_plain, DRAM), ("plain_ckc_oL1", ckc_plain, L1)):
        try:
            s = timed(lambda: ttnn.deallocate(ttnn.matmul(a, b, compute_kernel_config=kc, memory_config=omem)),
                      warm=4, pipe=4, reps=5)
        except Exception as e:
            print(f"  N={n} {lbl:16s} ERR {str(e)[:70]}", flush=True)
            continue
        tf = gf / s / 1e3
        roof[f"{n}_{lbl}"] = {"ms": round(s * 1e3, 4), "tflops": round(tf, 2)}
        print(f"  N={n:<5} {lbl:16s} {s*1e3:9.4f} ms {tf:8.2f} TFLOP/s", flush=True)
    ttnn.deallocate(a)
    ttnn.deallocate(b)
peak_c = max((v["tflops"] for v in roof.values()), default=0.0)
res["compute_roof"] = {"runs": roof, "peak_TFLOPs": peak_c}
print(f"MEASURED_HiFi4_COMPUTE_ROOF {peak_c:.2f} TFLOP/s", flush=True)

print("\n=== directional DRAM roofs ===", flush=True)
rows = []
for mb in (8, 16, 32, 48, 64):
    nrow = int(mb * 1e6 / 2) // 4096
    shape, nbytes = (nrow, 4096), nrow * 4096 * 2
    r = {"MB": round(nbytes / 1e6, 2)}
    try:
        xd = ttnn.from_torch(torch.randn(*shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                             device=dev, memory_config=DRAM)
        t = timed(lambda: ttnn.deallocate(ttnn.clone(xd, memory_config=L1)), warm=3, pipe=4, reps=7)
        r["read_GBs"] = round(nbytes / t / 1e9, 1)
        ttnn.deallocate(xd)
    except Exception as e:
        r["read_err"] = str(e)[:100]
    try:
        xl = ttnn.from_torch(torch.randn(*shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                             device=dev, memory_config=L1)
        t = timed(lambda: ttnn.deallocate(ttnn.clone(xl, memory_config=DRAM)), warm=3, pipe=4, reps=7)
        r["write_GBs"] = round(nbytes / t / 1e9, 1)
        ttnn.deallocate(xl)
    except Exception as e:
        r["write_err"] = str(e)[:100]
    try:
        xd = ttnn.from_torch(torch.randn(*shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                             device=dev, memory_config=DRAM)
        t = timed(lambda: ttnn.deallocate(ttnn.clone(xd, memory_config=DRAM)), warm=3, pipe=4, reps=7)
        r["dram2dram_rw_GBs"] = round(2 * nbytes / t / 1e9, 1)
        ttnn.deallocate(xd)
    except Exception as e:
        r["rw_err"] = str(e)[:100]
    rows.append(r)
    print("  " + json.dumps(r), flush=True)
peak_r = max((r.get("read_GBs", 0) for r in rows), default=0.0)
peak_w = max((r.get("write_GBs", 0) for r in rows), default=0.0)
mb_bal = round(peak_c * 1e12 / (peak_r * 1e9), 1) if peak_r else None
res["dram_roofs"] = {"runs": rows, "read_peak_GBs": peak_r, "write_peak_GBs": peak_w}
res["machine_balance_FLOP_per_byte_read"] = mb_bal
print(f"MEASURED_DRAM_READ_ROOF {peak_r:.1f} GB/s   MEASURED_DRAM_WRITE_ROOF {peak_w:.1f} GB/s", flush=True)
print(f"MACHINE_BALANCE {mb_bal} FLOP/byte (read)", flush=True)

json.dump(res, open(sys.argv[1], "w"), indent=2)
print("wrote", sys.argv[1], flush=True)
