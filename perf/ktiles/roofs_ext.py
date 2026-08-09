#!/usr/bin/env python3
"""Extend this card's roofs where the planning pass left them open: the DRAM read ladder was still
rising at 64 MB, and there was no fp32 compute roof (protenix-v2 runs its whole diffusion in fp32).
Same method as perf/ledger_298/roofs_card.py -- synchronise immediately before the clock starts and
immediately before it stops, warm, median of ITERS.
"""
import json, statistics, sys, time
import torch, ttnn

ITERS = 7


def timeit(fn, dev, reps=5):
    fn(); ttnn.synchronize_device(dev)
    out = []
    for _ in range(ITERS):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        ttnn.synchronize_device(dev)
        out.append((time.perf_counter() - t0) / reps)
    return statistics.median(out)


def main(out_path):
    dev = ttnn.open_device(device_id=0)
    g = dev.compute_with_storage_grid_size()
    res = {"grid_cores": g.x * g.y, "read_ladder_GBs": {}, "write_ladder_GBs": {}, "compute": {}}
    for mb in (64, 96, 128):
        n = mb * 1024 * 1024 // 2 // 1024          # bf16 elements / 1024
        a = ttnn.from_torch(torch.zeros(1024, n), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                            device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        nbytes = 1024 * n * 2
        t = timeit(lambda: ttnn.deallocate(ttnn.to_memory_config(a, ttnn.L1_MEMORY_CONFIG)), dev, 3)
        res["read_ladder_GBs"][mb] = round(nbytes / t / 1e9, 1)
        print(f"  read  {mb:4d} MB  {nbytes / t / 1e9:6.1f} GB/s", flush=True)
        ttnn.deallocate(a)
    for mb in ():
        n = mb * 1024 * 1024 // 2 // 1024
        a = ttnn.from_torch(torch.zeros(1024, n), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                            device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        nbytes = 1024 * n * 2
        t = timeit(lambda: ttnn.deallocate(ttnn.clone(a, memory_config=ttnn.DRAM_MEMORY_CONFIG)),
                   dev, 3)
        # A DRAM->DRAM clone moves the bytes twice; the write leg is the half we want.
        res["write_ladder_GBs"][mb] = round(nbytes / t / 1e9, 1)
        print(f"  clone {mb:4d} MB  {nbytes / t / 1e9:6.1f} GB/s each way", flush=True)
        ttnn.deallocate(a)
    ckc = ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    for name, dt in (("bfloat16", ttnn.bfloat16), ("float32", ttnn.float32)):
        N = 6144 if dt == ttnn.bfloat16 else 4096
        x = ttnn.from_torch(torch.zeros(N, N), dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
        t = timeit(lambda: ttnn.deallocate(ttnn.matmul(x, x, compute_kernel_config=ckc)), dev, 3)
        tf = 2 * N ** 3 / t / 1e12
        res["compute"][name] = {"N": N, "TFLOPs": round(tf, 2)}
        print(f"  compute {name:9s} N={N}  {tf:6.2f} TFLOP/s", flush=True)
        ttnn.deallocate(x)
    ttnn.close_device(dev)
    rd = max(res["read_ladder_GBs"].values())
    res["machine_balance_FLOP_per_byte"] = round(res["compute"]["bfloat16"]["TFLOPs"] * 1e12
                                                 / (rd * 1e9), 1)
    print(f"\n  read roof {rd} GB/s, machine balance {res['machine_balance_FLOP_per_byte']} FLOP/byte")
    json.dump(res, open(out_path, "w"), indent=1)
    print(f"wrote {out_path}")


main(sys.argv[1])
