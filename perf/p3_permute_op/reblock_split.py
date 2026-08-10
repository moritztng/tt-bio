#!/usr/bin/env python3
"""Split the generic_op cost: python descriptor construction vs the device program itself.

The first run of reblock_generic.py timed descriptor-build + generic_op together and lost to
ttnn.permute. This separates them, because only one of the two is a real device cost.
"""
import json, sys, time
from pathlib import Path
import torch, ttnn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reblock_generic import reblock_permute_generic, KDIR, TILE_H, GROUP_TILES, IN_CB, OUT_CB, STAGE_CB


def build_descriptor(x, out, device):
    N = int(x.shape[1]); Nt = N // TILE_H; num_groups = Nt * Nt
    g = device.compute_with_storage_grid_size()
    all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))])
    (_, core_grid, cg1, cg2, work1, work2) = ttnn.split_work_to_cores(all_cores, num_groups)
    tile_bytes = 32 * 32 * 2

    def cb(idx, depth):
        f = ttnn.CBFormatDescriptor(buffer_index=idx, data_format=ttnn.bfloat16, page_size=tile_bytes)
        return ttnn.CBDescriptor(total_size=depth * tile_bytes, core_ranges=core_grid, format_descriptors=[f])

    reader_ct = list(ttnn.TensorAccessorArgs(x).get_compile_time_args())
    writer_ct = [2, OUT_CB, 32, 32, 16, 16, STAGE_CB] + list(ttnn.TensorAccessorArgs(out).get_compile_time_args())
    rrt, crt, wrt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    src, dst = x.buffer_address(), out.buffer_address()
    start = 0
    for group, per_core in ((cg1, work1), (cg2, work2)):
        for cr in group.ranges():
            for cx in range(cr.start.x, cr.end.x + 1):
                for cy in range(cr.start.y, cr.end.y + 1):
                    rrt[cx][cy] = [src, start, per_core, Nt]
                    crt[cx][cy] = [per_core * GROUP_TILES]
                    wrt[cx][cy] = [dst, start, per_core, Nt]
                    start += per_core
    ks = [
        ttnn.KernelDescriptor(kernel_source=str(KDIR / "reader_reblock_permute.cpp"), core_ranges=core_grid,
                              compile_time_args=reader_ct, runtime_args=rrt, config=ttnn.ReaderConfigDescriptor()),
        ttnn.KernelDescriptor(kernel_source=str(KDIR / "writer_reblock_permute.cpp"), core_ranges=core_grid,
                              compile_time_args=writer_ct, runtime_args=wrt, config=ttnn.WriterConfigDescriptor()),
        ttnn.KernelDescriptor(kernel_source=str(KDIR / "compute_reblock_permute.cpp"), core_ranges=core_grid,
                              compile_time_args=[IN_CB, OUT_CB], runtime_args=crt,
                              config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi2,
                                                                  fp32_dest_acc_en=True)),
    ]
    return ttnn.ProgramDescriptor(kernels=ks, semaphores=[], cbs=[cb(IN_CB, 2), cb(OUT_CB, GROUP_TILES * 2), cb(STAGE_CB, 2)])


def timeit(device, fn, reps=15, warmup=3):
    for _ in range(warmup):
        fn()
    ttnn.synchronize_device(device)
    ts = []
    for _ in range(reps):
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        fn()
        ttnn.synchronize_device(device)
        ts.append((time.perf_counter() - t0) * 1e6)
    ts.sort()
    return ts[len(ts) // 2]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default="320")
    ap.add_argument("--out", default="perf/p3_permute_op/reblock_split.json")
    a = ap.parse_args()
    device = ttnn.open_device(device_id=0)
    R = {"rows": []}
    for N in [int(v) for v in a.ns.split(",")]:
      ref = torch.randn(1, N, N, 32, dtype=torch.bfloat16)
      golden = ref.permute(0, 3, 1, 2).contiguous()
      for where in ("l1", "dram"):
        mc = ttnn.L1_MEMORY_CONFIG if where == "l1" else ttnn.DRAM_MEMORY_CONFIG
        ntiles = N * N * 32 // 1024
        x = ttnn.from_torch(ref, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device, memory_config=mc)
        out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 32, N, N]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, mc)
        pd = build_descriptor(x, out, device)
        r = ttnn.generic_op([x, out], pd)
        eq = torch.equal(ttnn.to_torch(r), golden)
        us_dev = timeit(device, lambda: ttnn.generic_op([x, out], pd))
        t0 = time.perf_counter()
        for _ in range(10):
            build_descriptor(x, out, device)
        us_build = (time.perf_counter() - t0) * 1e5
        us_full = timeit(device, lambda: ttnn.generic_op([x, out], build_descriptor(x, out, device)))
        us_ref = timeit(device, lambda: ttnn.permute(x, (0, 3, 1, 2), memory_config=mc))
        row = {"N": N, "buf": where, "generic_device_only_us": round(us_dev, 2),
               "python_descriptor_build_us": round(us_build, 2),
               "generic_rebuilt_each_call_us": round(us_full, 2),
               "ttnn_permute_us": round(us_ref, 2),
               "ratio_device_only": round(us_ref / us_dev, 3),
               "tiles": ntiles, "us_per_tile_per_engaged_core": round(us_dev / ntiles * min(100, (N // 32) ** 2), 3),
               "torch_equal": bool(eq)}
        R["rows"].append(row); print(row, flush=True)
        ttnn.deallocate(x); ttnn.deallocate(out)
    ttnn.close_device(device)
    Path(a.out).write_text(json.dumps(R, indent=2))


if __name__ == "__main__":
    main()
