"""The trimul's back channel move with the packer doing the re-tiling.

`reblock_permute_back` moves the einsum's channel-major `[1, C, N, N]` result to `[1, N, N, H]` with a
reader-side L1 gather of 64 NOC packets per output tile: ~2.2 ms a call at 512 c256 on Wormhole, ~116 GB/s.
Here compute does it (kernels/permute_back_pk): untilize 32 channel tiles as one row, tilize each 2 KB page,
transpose in the unpacker. Bit-exact by construction (a copy through a 16-bit dest). `reblock_permute_back` routes
here on Wormhole.

`tile_rows` is the same compute without the transpose: it turns each 32x32 tile of a matmul product into one row of
1024, the move ESMFold2's outer-product mean needs between its outer product and Wout.
"""
from __future__ import annotations

from pathlib import Path

import ttnn

from . import mm_generic as MG

KERNEL_DIR = Path(__file__).resolve().parent / "kernels" / "permute_back_pk"
ABL = 0            # stage ablation bits (see each .cpp). Diagnostic only.
# out_cb tiles and writer tiles per flush: 16/8 through 128/32 all read 1.61-1.65 ms at 512 c256 (bpk7), so the smallest.
OUT_TILES = 16
WB = 8
# Units run channel tile fastest. Channel tile slowest put a unit's 32 reads (stride N^2/1024 pages) and 32 writes
# (stride N*H/1024) on 3 of Wormhole's 12 DRAM banks: 1.62 vs 1.48 ms at 512 c256, 6.18 vs 5.67 at 1024 (bpk8).
ORD = 1
_CACHE: dict = {}
TILE = 32


def _cb(idx, core_grid, tiles):
    fmt = ttnn.CBFormatDescriptor(buffer_index=idx, data_format=ttnn.bfloat16,
                                  page_size=MG.tile_bytes(ttnn.bfloat16))
    return ttnn.CBDescriptor(total_size=tiles * MG.tile_bytes(ttnn.bfloat16), core_ranges=core_grid,
                             format_descriptors=[fmt])


def ok(x, out, c_off=0):
    xs, os_ = [int(d) for d in x.shape], [int(d) for d in out.shape]
    return (len(xs) == 4 and len(os_) == 4 and xs[0] == os_[0] == 1 and xs[2] == xs[3] == os_[1] == os_[2]
            and xs[2] % TILE == 0 and xs[1] % TILE == 0 and c_off % TILE == 0 and c_off + xs[1] <= os_[3]
            and os_[3] % TILE == 0 and x.dtype == out.dtype == ttnn.bfloat16
            and x.layout == out.layout == ttnn.TILE_LAYOUT)


def permute_back_pk(x, out, grid, c_off=0):
    """out[0, i, j, c_off + c] = x[0, c, i, j]."""
    assert ok(x, out, c_off)
    C, N, H = int(x.shape[1]), int(x.shape[2]), int(out.shape[3])
    St, CT, Ht = N // TILE, C // TILE, H // TILE
    # The accessor args differ by buffer type (DRAM / L1) and bank layout, so the memory configs are part of the key.
    key = (C, N, H, c_off, tuple(grid), str(x.memory_config()), str(out.memory_config()), ABL, OUT_TILES, WB, ORD)
    entry = _CACHE.get(key)
    if entry is None:
        gx, gy = grid
        ncores = gx * gy
        core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))])
        nu = CT * St * St
        rd, wr, cp = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        u0 = 0
        for c in range(ncores):
            n = nu // ncores + (c < nu % ncores)
            rd[c % gx][c // gx] = [u0, n]
            wr[c % gx][c // gx] = [u0, n]
            cp[c % gx][c // gx] = [n]
            u0 += n
        acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        src = ttnn.KernelDescriptor.SourceType.FILE_PATH
        reader = ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "reader.cpp"), source_type=src, core_ranges=core_grid,
            compile_time_args=[St, ABL, CT, ORD] + acc(x), runtime_args=rd, common_runtime_args=[0],
            config=ttnn.ReaderConfigDescriptor())
        writer = ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "writer.cpp"), source_type=src, core_ranges=core_grid,
            compile_time_args=[St, Ht, c_off // TILE, ABL, WB, CT, ORD] + acc(out), runtime_args=wr, common_runtime_args=[0],
            config=ttnn.WriterConfigDescriptor())
        compute = ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "compute.cpp"), source_type=src, core_ranges=core_grid,
            compile_time_args=[ABL, 1], runtime_args=cp,
            config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=False))
        cbs = [_cb(0, core_grid, 16), _cb(1, core_grid, 32), _cb(2, core_grid, 32), _cb(16, core_grid, OUT_TILES)]
        entry = _CACHE[key] = ([reader, writer, compute], cbs)
    kernels, cbs = entry
    kernels[0].common_runtime_args = [x.buffer_address()]
    kernels[1].common_runtime_args = [out.buffer_address()]
    ttnn.generic_op([x, out], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
    return out


def tile_rows_ok(x, out):
    xs, os_ = [int(d) for d in x.shape], [int(d) for d in out.shape]
    return (len(xs) == 3 and len(os_) == 4 and xs[0] == os_[0] == 1 and os_[3] == TILE * TILE
            and xs[1] == os_[1] * TILE and xs[2] == os_[2] * TILE and os_[2] % TILE == 0
            and x.dtype == out.dtype == ttnn.bfloat16 and x.layout == out.layout == ttnn.TILE_LAYOUT
            and x.memory_config().memory_layout == out.memory_config().memory_layout
            == ttnn.TensorMemoryLayout.INTERLEAVED)


def tile_rows(x, out, grid):
    """out[0, i, j, p*32 + q] = x[0, i*32 + p, j*32 + q]: every 32x32 tile of x becomes one row of 1024."""
    assert tile_rows_ok(x, out)
    R, L = int(out.shape[1]), int(out.shape[2])
    key = ("rows", R, L, tuple(grid), str(x.memory_config()), str(out.memory_config()), OUT_TILES, WB)
    entry = _CACHE.get(key)
    if entry is None:
        gx, gy = grid
        ncores = gx * gy
        core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))])
        nu = R * (L // TILE)  # unit (i, jb): 32 input tiles in, 32 output tiles out, both consecutive pages
        rd, wr, cp = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        u0 = 0
        for c in range(ncores):
            n = nu // ncores + (c < nu % ncores)
            rd[c % gx][c // gx] = [u0, n]
            wr[c % gx][c // gx] = [u0, n]
            cp[c % gx][c // gx] = [n]
            u0 += n
        acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        src = ttnn.KernelDescriptor.SourceType.FILE_PATH
        reader = ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "reader_rows.cpp"), source_type=src, core_ranges=core_grid,
            compile_time_args=acc(x), runtime_args=rd, common_runtime_args=[0],
            config=ttnn.ReaderConfigDescriptor())
        writer = ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "writer_rows.cpp"), source_type=src, core_ranges=core_grid,
            compile_time_args=[WB] + acc(out), runtime_args=wr, common_runtime_args=[0],
            config=ttnn.WriterConfigDescriptor())
        compute = ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "compute.cpp"), source_type=src, core_ranges=core_grid,
            compile_time_args=[0, 0], runtime_args=cp,
            config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=False))
        cbs = [_cb(0, core_grid, 16), _cb(1, core_grid, 32), _cb(16, core_grid, OUT_TILES)]
        entry = _CACHE[key] = ([reader, writer, compute], cbs)
    kernels, cbs = entry
    kernels[0].common_runtime_args = [x.buffer_address()]
    kernels[1].common_runtime_args = [out.buffer_address()]
    ttnn.generic_op([x, out], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
    return out
