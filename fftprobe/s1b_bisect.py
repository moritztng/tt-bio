#!/usr/bin/env python3
"""Bisect the S1b hang: plumbing first, then the loop, on one core."""
import sys, time
from pathlib import Path
import torch, ttnn

KDIR = Path(__file__).resolve().parent / "s1b_kernels"
IN_CB, OUT_CB = 0, 16
TB = 32 * 32 * 4


def run(dev, x, out, mode, outer, nx, ny):
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(nx - 1, ny - 1))])

    def cb(idx, depth):
        f = ttnn.CBFormatDescriptor(buffer_index=idx, data_format=ttnn.float32, page_size=TB)
        return ttnn.CBDescriptor(total_size=depth * TB, core_ranges=core_grid, format_descriptors=[f])

    rct = [IN_CB, TB, 2] + list(ttnn.TensorAccessorArgs(x).get_compile_time_args())
    wct = [OUT_CB, TB] + list(ttnn.TensorAccessorArgs(out).get_compile_time_args())
    rrt, crt, wrt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    c = 0
    for cy in range(ny):
        for cx in range(nx):
            rrt[cx][cy] = [x.buffer_address(), 2 * c]
            crt[cx][cy] = [outer]
            wrt[cx][cy] = [out.buffer_address(), c]
            c += 1
    k = lambda src, ct, rt, cfg: ttnn.KernelDescriptor(
        kernel_source=str(KDIR / src), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid, compile_time_args=ct, runtime_args=rt, config=cfg)
    pd = ttnn.ProgramDescriptor(kernels=[
        k("reader_s1b.cpp", rct, rrt, ttnn.ReaderConfigDescriptor()),
        k("writer_s1b.cpp", wct, wrt, ttnn.WriterConfigDescriptor()),
        ttnn.KernelDescriptor(
            kernel_source=str(KDIR / "compute_min.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=core_grid, compile_time_args=[IN_CB, OUT_CB, mode], runtime_args=crt,
            config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4,
                                                fp32_dest_acc_en=True)),
    ], semaphores=[], cbs=[cb(IN_CB, 2), cb(OUT_CB, 2)])
    t0 = time.perf_counter()
    ttnn.generic_op([x, out], pd)
    ttnn.synchronize_device(dev)
    return time.perf_counter() - t0, ttnn.to_torch(out).clone()


def main():
    dev = ttnn.open_device(device_id=0)
    try:
        for (nx, ny) in ((1, 1), (13, 10)):
            n = nx * ny
            xt = 1.0 + 0.01 * torch.randn(1, 1, 64 * n, 32, dtype=torch.float32)
            ot = torch.zeros(1, 1, 32 * n, 32, dtype=torch.float32)
            x = ttnn.from_torch(xt, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
            out = ttnn.from_torch(ot, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
            for mode, outer in ((0, 0), (1, 1), (1, 1000), (1, 20000)):
                try:
                    s, got = run(dev, x, out, mode, outer, nx, ny)
                    # core c must receive input tile 2c
                    ref = xt.reshape(-1, 32, 32, 32)[:, 0] if False else None
                    ok = bool(torch.isfinite(got).all()) and float(got.abs().sum()) > 0
                    print(f"grid={nx}x{ny} mode={mode} outer={outer:6d}  {s*1e3:9.3f} ms  nonzero={ok}", flush=True)
                except Exception as e:
                    print(f"grid={nx}x{ny} mode={mode} outer={outer:6d}  FAIL {str(e)[:130]}", flush=True)
                    return
            ttnn.deallocate(x); ttnn.deallocate(out)
    finally:
        ttnn.close_device(dev)


main()
