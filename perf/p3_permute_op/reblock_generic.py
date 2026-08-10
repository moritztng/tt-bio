#!/usr/bin/env python3
"""reblock_permute on the PRODUCTION ttnn 0.68.0 wheel, via ttnn.generic_op.

The prototype lives as a C++ `ttnn.experimental` op in a v0.74.0-dev source build. That build
cannot carry a tt-bio fold: it is six minors ahead of the 0.68.0 wheel tt-bio ships against, so
pointing tt-bio at it is a dependency bump, not a delivery.

`ttnn.generic_op` is the route that does not need a build. It takes a ProgramDescriptor with
KernelDescriptors that name kernel SOURCE FILES, JIT-compiles them, and runs them against
pre-allocated io_tensors. So the C++ program factory is the only thing that has to be ported; the
three kernels are used verbatim. This is that port.

    permute(x, (0,3,1,2)) for x [1, N, N, 32] bf16 TILE -> [1, 32, N, N]
"""
import argparse, json, time
from pathlib import Path

import torch
import ttnn

KDIR = Path(__file__).resolve().parents[2] / "tt_bio" / "kernels" / "reblock_permute"
TILE_H = TILE_W = 32
FACE_H = FACE_W = 16
GROUP_TILES = 32
IN_CB, OUT_CB, STAGE_CB = 0, 16, 24


def reblock_permute_generic(x, memory_config=None, device=None):
    """(0,3,1,2) on [1,N,N,32] bf16 TILE, as a generic_op. Port of
    reblock_permute_program_factory.cpp:create()."""
    device = device or x.device()
    shape = [int(d) for d in x.shape]
    assert len(shape) == 4 and shape[0] == 1 and shape[3] == 32 and shape[1] == shape[2], shape
    N = shape[1]
    assert N % TILE_H == 0
    Nt = N // TILE_H
    num_groups = Nt * Nt
    mc = memory_config or x.memory_config()

    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 32, N, N]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, mc)

    g = device.compute_with_storage_grid_size()
    all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))])
    (_, core_grid, cg1, cg2, work1, work2) = ttnn.split_work_to_cores(all_cores, num_groups)

    tile_bytes = TILE_H * TILE_W * 2  # bf16
    def cb(idx, depth):
        fmt = ttnn.CBFormatDescriptor(buffer_index=idx, data_format=ttnn.bfloat16, page_size=tile_bytes)
        return ttnn.CBDescriptor(total_size=depth * tile_bytes, core_ranges=core_grid,
                                 format_descriptors=[fmt])

    # c_16 depth MUST be a multiple of 32 or the writer's 32-tile L1 window wraps mid-group.
    cbs = [cb(IN_CB, 2), cb(OUT_CB, GROUP_TILES * 2), cb(STAGE_CB, 2)]

    reader_ct = list(ttnn.TensorAccessorArgs(x).get_compile_time_args())
    writer_ct = [2, OUT_CB, TILE_H, TILE_W, FACE_H, FACE_W, STAGE_CB]
    writer_ct.extend(ttnn.TensorAccessorArgs(out).get_compile_time_args())
    compute_ct = [IN_CB, OUT_CB]

    reader_rt, compute_rt, writer_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    src, dst = x.buffer_address(), out.buffer_address()
    start = 0
    for group, per_core in ((cg1, work1), (cg2, work2)):
        for cr in group.ranges():
            for cx in range(cr.start.x, cr.end.x + 1):
                for cy in range(cr.start.y, cr.end.y + 1):
                    reader_rt[cx][cy] = [src, start, per_core, Nt]
                    compute_rt[cx][cy] = [per_core * GROUP_TILES]
                    writer_rt[cx][cy] = [dst, start, per_core, Nt]
                    start += per_core
    assert start == num_groups, (start, num_groups)

    kernels = [
        ttnn.KernelDescriptor(kernel_source=str(KDIR / "reader_reblock_permute.cpp"),
                              source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                              core_ranges=core_grid, compile_time_args=reader_ct,
                              runtime_args=reader_rt, config=ttnn.ReaderConfigDescriptor()),
        ttnn.KernelDescriptor(kernel_source=str(KDIR / "writer_reblock_permute.cpp"),
                              source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                              core_ranges=core_grid, compile_time_args=writer_ct,
                              runtime_args=writer_rt, config=ttnn.WriterConfigDescriptor()),
        ttnn.KernelDescriptor(kernel_source=str(KDIR / "compute_reblock_permute.cpp"),
                              source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                              core_ranges=core_grid, compile_time_args=compute_ct,
                              runtime_args=compute_rt,
                              config=ttnn.ComputeConfigDescriptor(
                                  math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True)),
    ]
    pd = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
    return ttnn.generic_op([x, out], pd), core_grid


def timeit(device, fn, reps=9, warmup=2):
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default="192,256,320,384")
    ap.add_argument("--out", default="perf/p3_permute_op/reblock_generic.json")
    a = ap.parse_args()
    device = ttnn.open_device(device_id=0)
    g = device.compute_with_storage_grid_size()
    R = {"cores": g.x * g.y, "rows": []}
    print(f"grid {g.x}x{g.y} = {g.x*g.y} cores", flush=True)

    for N in [int(s) for s in a.ns.split(",")]:
        ref = torch.randn(1, N, N, 32, dtype=torch.bfloat16)
        golden = ref.permute(0, 3, 1, 2).contiguous()
        for where in ("l1", "dram"):
            mc = ttnn.L1_MEMORY_CONFIG if where == "l1" else ttnn.DRAM_MEMORY_CONFIG
            try:
                x = ttnn.from_torch(ref, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16,
                                    device=device, memory_config=mc)
                out, core_grid = reblock_permute_generic(x, mc, device)
                ok = torch.equal(ttnn.to_torch(out), golden)
                ncores = sum((cr.end.x - cr.start.x + 1) * (cr.end.y - cr.start.y + 1)
                             for cr in core_grid.ranges())
                us_gen = timeit(device, lambda x=x, mc=mc: reblock_permute_generic(x, mc, device)[0])
                us_ref = timeit(device, lambda x=x, mc=mc: ttnn.permute(x, (0, 3, 1, 2), memory_config=mc))
                ntiles = N * N * 32 // 1024
                row = {"N": N, "buf": where, "tiles": ntiles, "cores_engaged": ncores,
                       "generic_us": round(us_gen, 2), "ttnn_permute_us": round(us_ref, 2),
                       "ratio": round(us_ref / us_gen, 3), "torch_equal": bool(ok),
                       "us_per_tile_per_engaged_core": round(us_gen / ntiles * ncores, 3)}
                R["rows"].append(row)
                print(row, flush=True)
                ttnn.deallocate(x)
            except Exception as e:
                print(f"N={N} {where} FAILED: {type(e).__name__}: {e}", flush=True)
                R["rows"].append({"N": N, "buf": where, "error": f"{type(e).__name__}: {e}"[:400]})

    ttnn.close_device(device)
    Path(a.out).write_text(json.dumps(R, indent=2))
    print(json.dumps(R, indent=2))


if __name__ == "__main__":
    main()
