"""``ttnn.sum(x, dim=0, keepdim=True)`` for a float32 TILE tensor, as one Tensix kernel.

`triatt_bw.run` reduces its float32 ``[groups, H, N, N]`` dbias partial this way, and ttnn runs
it as permute, reduce, permute: 0.78 ms at the BindCraft 2 round's ``[27, 4, 288, 288]``. The
leading axis is the slowest one in the page order, so output tile t is the sum of pages k*T + t:
this reads them straight, adds in a float32 DST with the SFPU and packs once. The program is
rne_add's (genq compact split, cached descriptor, common-arg addresses, rne_add's writer).
"""

from __future__ import annotations

from pathlib import Path

import ttnn

from . import genq
from .envflags import env_flag
from .rne_add import _NUM_CBS, _split_plan, _tile_grid

KERNEL_DIR = Path(__file__).resolve().parent / "kernels" / "lead_sum"
WRITER = Path(__file__).resolve().parent / "kernels" / "rne_add" / "writer_rne_add.cpp"

IN_CB, OUT_CB = 0, 16


def _gran(G):
    """Reader block size: the largest of 4, 3, 2, 1 that divides G. The reader walks a block from
    one `get_write_ptr` in a CB two blocks deep, so a ragged last block would leave the next
    tiles block straddling the CBs end (measured: rel L2 1.6 at G=27 with GRAN=4)."""
    return next(g for g in (4, 3, 2, 1) if G % g == 0)
TILE_BYTES = 32 * 32 * 4

#: The lever. Default OFF and release-gated; armed by `bindcraft2.fast_round()`.
LEAD_SUM_FUSED = env_flag("TT_BIO_LEAD_SUM_FUSED", False)

#: (calls served, calls declined), cumulative; sample at a round boundary.
STATS = [0, 0]
_CACHE: dict = {}


def eligible(x) -> bool:
    """Float32 TILE interleaved, rank 4, a leading axis worth summing."""
    if not LEAD_SUM_FUSED:
        return False
    ok = (x.dtype == ttnn.float32 and x.layout == ttnn.TILE_LAYOUT and len(x.shape) == 4
          and int(x.shape[0]) > 1
          and x.memory_config().memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED)
    if not ok:
        STATS[1] += 1
    return ok


def _cb(idx, depth, core_grid):
    fmt = ttnn.CBFormatDescriptor(buffer_index=idx, data_format=ttnn.float32,
                                  page_size=TILE_BYTES)
    return ttnn.CBDescriptor(total_size=depth * TILE_BYTES, core_ranges=core_grid,
                             format_descriptors=[fmt])


def _build(x, out, device, G, T):
    num_cores, core_grid, cg1, cg2, work1, work2 = _split_plan(device, T)
    assign, placed = {}, 0
    for group, per_core in ((cg1, work1), (cg2, work2)):
        for cr in group.ranges():
            for cx in range(cr.start.x, cr.end.x + 1):
                for cy in range(cr.start.y, cr.end.y + 1):
                    assign[(cx, cy)] = (placed, per_core)
                    placed += per_core
    assert (len(assign), placed) == (num_cores, T), (len(assign), placed, T)
    plan_ct = genq.compact_plan(assign, core_grid) if genq.compact() else None
    genq_ct = [1] + plan_ct if plan_ct else [0] * 8
    reader_rt, compute_rt, writer_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    if plan_ct is None:
        for (cx, cy), (fst, per_core) in assign.items():
            reader_rt[cx][cy] = [fst, per_core]
            compute_rt[cx][cy] = [per_core]
            writer_rt[cx][cy] = [fst, per_core]
    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4,
                                       fp32_dest_acc_en=True)
    modes = [ttnn.UnpackToDestMode.Default] * _NUM_CBS
    modes[IN_CB] = ttnn.UnpackToDestMode.UnpackToDestFp32
    cfg.unpack_to_dest_mode = modes
    src = ttnn.KernelDescriptor.SourceType.FILE_PATH
    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "reader_lead_sum.cpp"), source_type=src,
        core_ranges=core_grid,
        compile_time_args=[_gran(G), G, T] + list(ttnn.TensorAccessorArgs(x).get_compile_time_args())
        + genq_ct, runtime_args=reader_rt, common_runtime_args=[0],
        config=ttnn.ReaderConfigDescriptor())
    # rne_add's writer reads its block size from arg 0: one tile, the rate compute produces.
    writer = ttnn.KernelDescriptor(
        kernel_source=str(WRITER), source_type=src, core_ranges=core_grid,
        compile_time_args=[1] + list(ttnn.TensorAccessorArgs(out).get_compile_time_args())
        + genq_ct, runtime_args=writer_rt, common_runtime_args=[0],
        config=ttnn.WriterConfigDescriptor())
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "compute_lead_sum.cpp"), source_type=src,
        core_ranges=core_grid, compile_time_args=[IN_CB, OUT_CB, G] + genq_ct,
        runtime_args=compute_rt, config=cfg)
    return {"kernels": [reader, writer, compute],
            "cbs": [_cb(IN_CB, 2 * _gran(G), core_grid), _cb(OUT_CB, 2, core_grid)]}


def lead_sum(x):
    """``sum(x, dim=0, keepdim=True)``, float32, in DRAM."""
    device = x.device()
    G = int(x.shape[0])
    ht, wt = _tile_grid(x)
    T = ht * wt // G
    shape = [1] + [int(d) for d in list(x.shape)[1:]]
    out = ttnn.empty(ttnn.Shape(shape), ttnn.float32, ttnn.TILE_LAYOUT, device,
                     ttnn.DRAM_MEMORY_CONFIG)
    g_ = device.compute_with_storage_grid_size()
    key = (device.id(), G, T, x.memory_config().buffer_type, g_.x, g_.y, genq.compact())
    entry = _CACHE.get(key)
    if entry is None:
        entry = _CACHE[key] = _build(x, out, device, G, T)
    reader, writer, compute = entry["kernels"]
    reader.common_runtime_args = [x.buffer_address()]
    writer.common_runtime_args = [out.buffer_address()]
    pd = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[],
                                cbs=entry["cbs"])
    STATS[0] += 1
    ttnn.generic_op([x, out], pd)
    return out
