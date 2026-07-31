#!/usr/bin/env python3
"""Blackhole roofline micro-benchmarks: achievable DRAM bandwidth and BF16 matmul FLOP/s.

MEASUREMENT RULE encoded here: every timed region calls ttnn.synchronize_device() both
before the clock starts (drain the warmup) and before the clock stops (drain the work we
are timing).  ttnn op calls are asynchronous enqueues -- without the trailing sync the
host clock stops while the device is still working and the number is fiction.
"""
import argparse
import time

import ttnn

BF16 = ttnn.bfloat16


def timed(fn, device, iters, warmup):
    for _ in range(warmup):
        fn()
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    ttnn.synchronize_device(device)
    return (time.perf_counter() - t0) / iters


def fidelity_cfg(name):
    if name is None:
        return None
    fid = {"LoFi": ttnn.MathFidelity.LoFi, "HiFi2": ttnn.MathFidelity.HiFi2, "HiFi4": ttnn.MathFidelity.HiFi4}[name]
    return ttnn.WormholeComputeKernelConfig(math_fidelity=fid, fp32_dest_acc_en=False, packer_l1_acc=False)


def bandwidth(device, iters, warmup):
    print("\n=== DRAM bandwidth (bf16, DRAM-interleaved, whole-chip) ===")
    print(f"{'op':<12} {'shape':<20} {'bytes_moved_MB':>15} {'ms/iter':>10} {'GB/s':>9}")
    rows = []
    for r, c in [(4096, 4096), (8192, 4096), (8192, 8192)]:
        a = ttnn.ones((1, 1, r, c), dtype=BF16, layout=ttnn.TILE_LAYOUT, device=device,
                      memory_config=ttnn.DRAM_MEMORY_CONFIG)
        b = ttnn.ones((1, 1, r, c), dtype=BF16, layout=ttnn.TILE_LAYOUT, device=device,
                      memory_config=ttnn.DRAM_MEMORY_CONFIG)
        n_bytes = r * c * 2
        for opname, fn, traffic in [
            ("add", lambda: ttnn.add(a, b, memory_config=ttnn.DRAM_MEMORY_CONFIG), 3 * n_bytes),
            ("clone", lambda: ttnn.clone(a, memory_config=ttnn.DRAM_MEMORY_CONFIG), 2 * n_bytes),
        ]:
            try:
                t = timed(fn, device, iters, warmup)
            except Exception as e:  # noqa: BLE001
                print(f"{opname:<12} {f'{r}x{c}':<20} {'-':>15} {'FAIL':>10} {str(e)[:40]:>9}")
                continue
            gbs = traffic / t / 1e9
            print(f"{opname:<12} {f'{r}x{c}':<20} {traffic/1e6:>15.1f} {t*1e3:>10.3f} {gbs:>9.1f}")
            rows.append((opname, r, c, traffic, t, gbs))
        ttnn.deallocate(a)
        ttnn.deallocate(b)
    if rows:
        best = max(rows, key=lambda x: x[5])
        print(f"PEAK_DRAM_BW_GBs {best[5]:.1f}  (op={best[0]} {best[1]}x{best[2]})")
    return rows


def flops(device, iters, warmup, sizes=(512, 1024, 2048, 4096)):
    print("\n=== BF16 square matmul FLOP/s ===")
    print(f"{'N':<7} {'fidelity':<9} {'GFLOP':>10} {'ms/iter':>10} {'TFLOP/s':>9}")
    rows = []
    for n in sizes:
        a = ttnn.ones((1, 1, n, n), dtype=BF16, layout=ttnn.TILE_LAYOUT, device=device,
                      memory_config=ttnn.DRAM_MEMORY_CONFIG)
        b = ttnn.ones((1, 1, n, n), dtype=BF16, layout=ttnn.TILE_LAYOUT, device=device,
                      memory_config=ttnn.DRAM_MEMORY_CONFIG)
        fl = 2.0 * n ** 3
        for fname in ["LoFi", "HiFi2", "HiFi4"]:
            cfg = fidelity_cfg(fname)
            try:
                t = timed(lambda: ttnn.matmul(a, b, compute_kernel_config=cfg,
                                              memory_config=ttnn.DRAM_MEMORY_CONFIG),
                          device, iters, warmup)
            except Exception as e:  # noqa: BLE001
                print(f"{n:<7} {fname:<9} {'-':>10} {'FAIL':>10} {str(e)[:40]}")
                continue
            tf = fl / t / 1e12
            print(f"{n:<7} {fname:<9} {fl/1e9:>10.2f} {t*1e3:>10.3f} {tf:>9.2f}")
            rows.append((n, fname, fl, t, tf))
        ttnn.deallocate(a)
        ttnn.deallocate(b)
    if rows:
        best = max(rows, key=lambda x: x[4])
        print(f"PEAK_BF16_TFLOPs {best[4]:.2f}  (N={best[0]} fidelity={best[1]})")
    return rows


def unsynced_demo(device, iters, warmup):
    """Show what dropping the trailing sync does to the same measurement."""
    print("\n=== sync vs no-sync on the SAME workload ===")
    n = 2048
    a = ttnn.ones((1, 1, n, n), dtype=BF16, layout=ttnn.TILE_LAYOUT, device=device,
                  memory_config=ttnn.DRAM_MEMORY_CONFIG)
    b = ttnn.ones((1, 1, n, n), dtype=BF16, layout=ttnn.TILE_LAYOUT, device=device,
                  memory_config=ttnn.DRAM_MEMORY_CONFIG)
    fn = lambda: ttnn.matmul(a, b, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # noqa: E731
    for _ in range(warmup):
        fn()
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    t_nosync = (time.perf_counter() - t0) / iters
    ttnn.synchronize_device(device)
    t_sync = timed(fn, device, iters, warmup)
    print(f"matmul {n}^3  no trailing sync: {t_nosync*1e3:.3f} ms/iter")
    print(f"matmul {n}^3  with sync       : {t_sync*1e3:.3f} ms/iter")
    print(f"UNSYNCED_UNDERREPORT_FACTOR {t_sync/t_nosync:.2f}x")
    ttnn.deallocate(a)
    ttnn.deallocate(b)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device_id", type=int, default=0)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--only", choices=["bw", "flops", "sync"], default=None)
    p.add_argument("--mm_sizes", type=int, nargs="+", default=[512, 1024, 2048, 4096],
                   help="square matmul N values; extend upward until TFLOP/s saturates")
    args = p.parse_args()

    device = ttnn.open_device(device_id=args.device_id)
    try:
        g = device.compute_with_storage_grid_size()
        dg = device.dram_grid_size()
        # ttnn.open_device returns a MeshDevice on recent tt-metal; it exposes neither
        # num_dram_channels() nor l1_size_per_core(), so report the grids it does expose.
        print(f"DEVICE arch={device.arch()} compute_grid={g.x}x{g.y} "
              f"dram_grid={dg.x}x{dg.y} num_devices={device.get_num_devices()}")
        if args.only in (None, "bw"):
            bandwidth(device, args.iters, args.warmup)
        if args.only in (None, "flops"):
            flops(device, args.iters, args.warmup, sizes=args.mm_sizes)
        if args.only in (None, "sync"):
            unsynced_demo(device, args.iters, args.warmup)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
