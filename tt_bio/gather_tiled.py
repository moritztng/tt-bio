"""Table rows gathered by index straight into TILE layout, as one Tensix program.

``out[b, r, w, :] = table[idx[b * R + r, w], :]`` for a row-major uint32 index [B * R, Wp] (Wp = W rounded up
to 8, so a row is a whole number of 32-byte DRAM words; the pad entries are never read) and a row-major bf16
table [N, C]. It replaces ``ttnn.embedding`` (row-major out) + ``ttnn.to_layout``: on Wormhole the embedding of
469225 rows of 128 runs ~4.8 ms against a 0.5 ms write roof and the tilize adds ~1.4 ms (RFD3 process_z, WH fast
census). Here each core takes (r, 32-column block) units, issues the 32 row reads of a block behind one barrier,
tilizes the block on the unpacker and writes its C / 32 tiles. Columns past W in the last block are zero, the
pad ttnn's tilize writes. Data movement only: every gathered value is the table's bit pattern.

An index that names the same row over and over (RFD3's CA distance bins: every pair past the top edge takes the last row)
queues every read on the one DRAM bank holding that row: 4.1-4.3 ms/call in-model on Wormhole against 1.28 ms for a
random index of the same shape. ``spread`` lays the table out as COPIES replicas whose copies of a row sit in
different banks; pass ``copies`` to ``gather_tiled`` and successive reads rotate over them.

``fma=(a, inv)`` returns ``a + gathered * inv`` instead (a [.., R, W, C] TILE, inv its [.., R, W, 1] column scale):
RFD3's process_z without the gathered tensor's DRAM round trip. ``rows`` = ``pair_swiglu_ws.row_list(...)`` over the
output computes only those row tiles (32 columns of one index row) and leaves the rest of ``out`` unwritten.
"""

from __future__ import annotations

from pathlib import Path

import ttnn

from . import core_split

KERNEL_DIR = Path(__file__).resolve().parent / "kernels" / "gather_tiled"
RM_CB, IDX_CB, ZERO_CB, A_CB, INV_CB, LIST_CB_R, LIST_CB_W, OUT_CB, EM_CB = 0, 1, 2, 3, 4, 5, 6, 16, 24
_TILE = 32 * 32 * 2
_CACHE: dict = {}


def _align(n, a=32):
    return (n + a - 1) // a * a


def _build(idx, table, out, device, units, W, Wp, JT, CT, copies, fma=None, rows=None):
    g = device.compute_with_storage_grid_size()
    ncores, grid, cg1, cg2, work1, work2 = core_split.split_work_to_cores(g, units, 0)

    def cb(i, page, depth):
        fmt = ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16, page_size=page)
        return ttnn.CBDescriptor(total_size=depth * page, core_ranges=grid, format_descriptors=[fmt])

    cbs = [cb(RM_CB, _TILE, 2 * CT), cb(OUT_CB, _TILE, 2 * CT), cb(IDX_CB, Wp * 4, 1),
           cb(ZERO_CB, CT * 64, 1)]
    rt = ttnn.RuntimeArgs()
    first = ordinal = 0
    for group, per in ((cg1, work1), (cg2, work2)):
        for cr in group.ranges():
            for cx in range(cr.start.x, cr.end.x + 1):
                for cy in range(cr.start.y, cr.end.y + 1):
                    rt[cx][cy] = [first, per, ordinal]
                    first += per
                    ordinal += 1
    assert first == units, (first, units)
    rct = [W, JT, CT, Wp * 4, copies, int(table.shape[0]) // copies] + list(ttnn.TensorAccessorArgs(idx).get_compile_time_args()) \
        + list(ttnn.TensorAccessorArgs(table).get_compile_time_args())
    wct = [CT] + list(ttnn.TensorAccessorArgs(out).get_compile_time_args())
    rdef, wdef, cdef = [], [], []
    if fma is not None:
        cbs += [cb(A_CB, _TILE, 2 * CT), cb(INV_CB, _TILE, 2), cb(EM_CB, _TILE, CT)]
        rdef.append(("GT_FMA", str(len(rct))))
        cdef.append(("GT_FMA", "1"))
        for t in fma:
            rct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    if rows is not None:
        assert int(rows.shape[0]) == ncores, (rows.shape, ncores)
        page = int(rows.shape[1]) * 4
        lfmt = lambda i: ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.uint32, page_size=page)  # noqa: E731
        cbs += [ttnn.CBDescriptor(total_size=page, core_ranges=grid, format_descriptors=[lfmt(i)])
                for i in (LIST_CB_R, LIST_CB_W)]
        lb = [("GT_LIST_BYTES", str(page))]
        rdef += lb + [("GT_ROWLIST", str(len(rct)))]
        wdef += lb + [("GT_ROWLIST", str(len(wct)))]
        cdef.append(("GT_ROWLIST", "1"))
        rct.extend(ttnn.TensorAccessorArgs(rows).get_compile_time_args())
        wct.extend(ttnn.TensorAccessorArgs(rows).get_compile_time_args())
    src = lambda n: str(KERNEL_DIR / n)  # noqa: E731
    path = ttnn.KernelDescriptor.SourceType.FILE_PATH
    reader = ttnn.KernelDescriptor(kernel_source=src("reader.cpp"), source_type=path, core_ranges=grid,
                                   compile_time_args=rct, runtime_args=rt, common_runtime_args=[0] * 7,
                                   defines=rdef, config=ttnn.ReaderConfigDescriptor())
    writer = ttnn.KernelDescriptor(kernel_source=src("writer.cpp"), source_type=path, core_ranges=grid,
                                   compile_time_args=wct, runtime_args=rt, common_runtime_args=[0] * 4,
                                   defines=wdef, config=ttnn.WriterConfigDescriptor())
    compute = ttnn.KernelDescriptor(kernel_source=src("compute.cpp"), source_type=path, core_ranges=grid,
                                    compile_time_args=[CT], runtime_args=rt, common_runtime_args=[0, 0],
                                    defines=cdef, config=ttnn.ComputeConfigDescriptor(
                                        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=fma is not None))
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


SPREAD_COPIES = 12   # Wormhole's DRAM bank count; on Blackhole's 8 the 12 replicas still cover every bank


def spread(table, copies=SPREAD_COPIES):
    """``copies`` replicas of a row-major [N, C] table, STRIDE >= N rows apart with STRIDE coprime to 24, so the
    replicas of one row land in distinct DRAM banks under round-robin page interleaving (8 or 12 banks)."""
    import math

    import torch
    t = ttnn.to_torch(table)
    stride = next(s for s in range(t.shape[0], t.shape[0] + 24) if math.gcd(s, 24) == 1)
    rep = torch.zeros(copies * stride, t.shape[1], dtype=t.dtype)
    for c in range(copies):
        rep[c * stride:c * stride + t.shape[0]] = t
    return ttnn.from_torch(rep, layout=ttnn.ROW_MAJOR_LAYOUT, device=table.device(), dtype=ttnn.bfloat16,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)


def gather_tiled(idx, table, batch, width, copies=1, fma=None, rows=None):
    """[batch, R, width, C] bf16 TILE with out[b, r, w] = table[idx[b * R + r, w]]; idx is [batch * R, Wp].
    ``copies`` > 1 reads a ``spread(table, copies)`` table. ``fma=(a, inv)``: out = a + gathered * inv. ``rows`` =
    ``(list, n)`` from ``pair_swiglu_ws.row_list`` over out: only those units are written (the rest is undefined)."""
    device = idx.device()
    BR, W, Wp = int(idx.shape[0]), width, int(idx.shape[1])
    assert Wp == _align(W, 8), (W, Wp)
    C = int(table.shape[1])
    R, JT, CT = BR // batch, _align(W) // 32, C // 32
    out = ttnn.allocate_tensor_on_device(ttnn.Shape([batch, R, W, C]), ttnn.bfloat16, ttnn.TILE_LAYOUT,
                                         device, ttnn.DRAM_MEMORY_CONFIG)
    key = (device.id(), BR, W, C, int(table.shape[0]), copies, fma is not None,
           None if rows is None else tuple(rows[0].shape))
    if key not in _CACHE:
        _CACHE[key] = _build(idx, table, out, device, BR * JT, W, Wp, JT, CT, copies, fma,
                             None if rows is None else rows[0])
    kernels, cbs = _CACHE[key]
    ra = [idx.buffer_address(), table.buffer_address()]
    ra += [0, 0, 0] if rows is None else [rows[0].buffer_address(), rows[1], int(rows[0].shape[0])]
    ra += [0, 0] if fma is None else [fma[0].buffer_address(), fma[1].buffer_address()]
    kernels[0].common_runtime_args = ra
    kernels[1].common_runtime_args = [out.buffer_address()] + ra[2:5]
    kernels[2].common_runtime_args = [0, 0] if rows is None else [rows[1], int(rows[0].shape[0])]
    io = [idx, table] + list(fma or ()) + ([] if rows is None else [rows[0]]) + [out]
    ttnn.generic_op(io, ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
    return out
