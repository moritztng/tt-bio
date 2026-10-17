"""Table rows gathered by index straight into TILE layout, as one Tensix program.

``out[b, r, w, :] = table[idx[b * R + r, w], :]`` for a row-major uint32 index [B * R, Wp] (Wp = W rounded up
to 8, so a row is a whole number of 32-byte DRAM words; the pad entries are never read) and a row-major bf16
table [N, C]. It replaces ``ttnn.embedding`` (row-major out) + ``ttnn.to_layout``: on Wormhole the embedding of
469225 rows of 128 runs ~4.8 ms against a 0.5 ms write roof and the tilize adds ~1.4 ms (RFD3 process_z, WH fast
census). Here each core takes (r, 32-column block) units, issues the 32 row reads of a block behind one barrier,
tilizes the block on the unpacker and writes its C / 32 tiles. Columns past W in the last block are zero, the
pad ttnn's tilize writes. Data movement only: every gathered value is the table's bit pattern.
"""

from __future__ import annotations

from pathlib import Path

import ttnn

from . import core_split

KERNEL_DIR = Path(__file__).resolve().parent / "kernels" / "gather_tiled"
RM_CB, IDX_CB, ZERO_CB, OUT_CB = 0, 1, 2, 16
_TILE = 32 * 32 * 2
_CACHE: dict = {}


def _align(n, a=32):
    return (n + a - 1) // a * a


def _build(idx, table, out, device, units, W, Wp, JT, CT):
    g = device.compute_with_storage_grid_size()
    _, grid, cg1, cg2, work1, work2 = core_split.split_work_to_cores(g, units, 0)

    def cb(i, page, depth):
        fmt = ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16, page_size=page)
        return ttnn.CBDescriptor(total_size=depth * page, core_ranges=grid, format_descriptors=[fmt])

    cbs = [cb(RM_CB, _TILE, 2 * CT), cb(OUT_CB, _TILE, 2 * CT), cb(IDX_CB, Wp * 4, 1),
           cb(ZERO_CB, CT * 64, 1)]
    rt = ttnn.RuntimeArgs()
    first = 0
    for group, per in ((cg1, work1), (cg2, work2)):
        for cr in group.ranges():
            for cx in range(cr.start.x, cr.end.x + 1):
                for cy in range(cr.start.y, cr.end.y + 1):
                    rt[cx][cy] = [first, per]
                    first += per
    assert first == units, (first, units)
    rct = [W, JT, CT, Wp * 4] + list(ttnn.TensorAccessorArgs(idx).get_compile_time_args()) \
        + list(ttnn.TensorAccessorArgs(table).get_compile_time_args())
    wct = [CT] + list(ttnn.TensorAccessorArgs(out).get_compile_time_args())
    src = lambda n: str(KERNEL_DIR / n)  # noqa: E731
    path = ttnn.KernelDescriptor.SourceType.FILE_PATH
    reader = ttnn.KernelDescriptor(kernel_source=src("reader.cpp"), source_type=path, core_ranges=grid,
                                   compile_time_args=rct, runtime_args=rt, common_runtime_args=[0, 0],
                                   config=ttnn.ReaderConfigDescriptor())
    writer = ttnn.KernelDescriptor(kernel_source=src("writer.cpp"), source_type=path, core_ranges=grid,
                                   compile_time_args=wct, runtime_args=rt, common_runtime_args=[0],
                                   config=ttnn.WriterConfigDescriptor())
    compute = ttnn.KernelDescriptor(kernel_source=src("compute.cpp"), source_type=path, core_ranges=grid,
                                    compile_time_args=[CT], runtime_args=rt,
                                    config=ttnn.ComputeConfigDescriptor())
    return [reader, writer, compute], cbs


def eligible(idx, table) -> bool:
    """idx uint32 ROW_MAJOR [B * R, Wp], Wp a multiple of 8, and table bf16 ROW_MAJOR [N, C], C a tile multiple, both in DRAM."""
    ok = (idx.dtype == ttnn.uint32 and table.dtype == ttnn.bfloat16
          and idx.layout == table.layout == ttnn.ROW_MAJOR_LAYOUT and len(idx.shape) == 2 and len(table.shape) == 2)
    return ok and int(table.shape[1]) % 32 == 0 and int(idx.shape[1]) % 8 == 0 and all(
        t.memory_config().buffer_type == ttnn.BufferType.DRAM for t in (idx, table))


def pad_index(flat, rows, width):
    """A flat host index of rows x width as the [rows, Wp] uint32 tensor gather_tiled takes (torch, not uploaded)."""
    import torch
    out = torch.zeros(rows, _align(width, 8), dtype=torch.int32)
    out[:, :width] = flat.reshape(rows, width)
    return out


def gather_tiled(idx, table, batch, width):
    """[batch, R, width, C] bf16 TILE with out[b, r, w] = table[idx[b * R + r, w]]; idx is [batch * R, Wp]."""
    device = idx.device()
    BR, W, Wp = int(idx.shape[0]), width, int(idx.shape[1])
    assert Wp == _align(W, 8), (W, Wp)
    C = int(table.shape[1])
    R, JT, CT = BR // batch, _align(W) // 32, C // 32
    out = ttnn.allocate_tensor_on_device(ttnn.Shape([batch, R, W, C]), ttnn.bfloat16, ttnn.TILE_LAYOUT,
                                         device, ttnn.DRAM_MEMORY_CONFIG)
    key = (device.id(), BR, W, C, int(table.shape[0]))
    if key not in _CACHE:
        _CACHE[key] = _build(idx, table, out, device, BR * JT, W, Wp, JT, CT)
    kernels, cbs = _CACHE[key]
    kernels[0].common_runtime_args = [idx.buffer_address(), table.buffer_address()]
    kernels[1].common_runtime_args = [out.buffer_address()]
    ttnn.generic_op([idx, table, out], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
    return out
