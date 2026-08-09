#!/usr/bin/env python3
"""Roofs on THIS card (qb2 card 1) for the atom-window leg. Method is copied from
perf/ledger_298/roofs_card.py so a delta against W1/W10 is a card delta, not a method delta.

Adds an fp32 compute leg and takes the DRAM read roof off a 64->128 MB sweep: W1 and W10 both
found the 8->64 MB ladder under-reads the read roof.
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


ckc_op = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                                fp32_dest_acc_en=True, packer_l1_acc=True)
ckc_plain = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                                   fp32_dest_acc_en=False, packer_l1_acc=False)
for dt, name in ((ttnn.bfloat16, "bf16"), (ttnn.float32, "fp32")):
    print(f"=== compute roof: square {name} matmul, HiFi4 ===", flush=True)
    roof = {}
    for n in (2048, 4096):
        a = ttnn.from_torch(torch.randn(1, 1, n, n), layout=ttnn.TILE_LAYOUT, device=dev, dtype=dt)
        b = ttnn.from_torch(torch.randn(1, 1, n, n), layout=ttnn.TILE_LAYOUT, device=dev, dtype=dt)
        gf = 2 * n ** 3 / 1e9
        for lbl, kc, omem in (("op_ckc_oDRAM", ckc_op, DRAM), ("plain_ckc_oDRAM", ckc_plain, DRAM)):
            try:
                s = timed(lambda: ttnn.deallocate(ttnn.matmul(a, b, compute_kernel_config=kc,
                                                              memory_config=omem, dtype=dt)))
            except Exception as e:
                print(f"  N={n} {lbl:16s} ERR {str(e)[:70]}", flush=True)
                continue
            tf = gf / s / 1e3
            roof[f"{n}_{lbl}"] = {"ms": round(s * 1e3, 4), "tflops": round(tf, 2)}
            print(f"  N={n:<5} {lbl:16s} {s*1e3:9.4f} ms {tf:8.2f} TFLOP/s", flush=True)
        ttnn.deallocate(a); ttnn.deallocate(b)
    peak = max((v["tflops"] for v in roof.values()), default=0.0)
    res[f"compute_roof_{name}"] = {"runs": roof, "peak_TFLOPs": peak}
    print(f"MEASURED_HiFi4_COMPUTE_ROOF_{name} {peak:.2f} TFLOP/s", flush=True)

print("\n=== directional DRAM roofs, 64->128 MB ===", flush=True)
for dt, name, esz in ((ttnn.bfloat16, "bf16", 2), (ttnn.float32, "fp32", 4)):
    rows = []
    for mb in (64, 96, 128):
        nrow = int(mb * 1e6 / esz) // 4096
        shape, nbytes = (nrow, 4096), nrow * 4096 * esz
        r = {"dtype": name, "MB": round(nbytes / 1e6, 2)}
        xd = ttnn.from_torch(torch.randn(*shape), dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev,
                             memory_config=DRAM)
        t = timed(lambda: ttnn.deallocate(ttnn.clone(xd, memory_config=L1)), warm=2, pipe=3, reps=5)
        r["read_GBs"] = round(nbytes / t / 1e9, 1)
        ttnn.deallocate(xd)
        xl = ttnn.from_torch(torch.randn(*shape), dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev,
                             memory_config=L1)
        t = timed(lambda: ttnn.deallocate(ttnn.clone(xl, memory_config=DRAM)), warm=2, pipe=3, reps=5)
        r["write_GBs"] = round(nbytes / t / 1e9, 1)
        ttnn.deallocate(xl)
        rows.append(r)
        print("  " + json.dumps(r), flush=True)
    pr = max(r["read_GBs"] for r in rows)
    pw = max(r["write_GBs"] for r in rows)
    bal = round(res[f"compute_roof_{name}"]["peak_TFLOPs"] * 1e12 / (pr * 1e9), 1)
    res[f"dram_roofs_{name}"] = {"runs": rows, "read_peak_GBs": pr, "write_peak_GBs": pw,
                                 "machine_balance_FLOP_per_byte": bal}
    print(f"MEASURED_{name}_READ {pr:.1f} GB/s  WRITE {pw:.1f} GB/s  BALANCE {bal} FLOP/byte", flush=True)

json.dump(res, open(sys.argv[1], "w"), indent=2)
print("wrote", sys.argv[1], flush=True)
