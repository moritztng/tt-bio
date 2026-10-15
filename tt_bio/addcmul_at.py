"""``x + y * z`` where any operand may be a column window of a wider tensor, as one Tensix kernel.

The OF3 DiT conditions every one of its 24 blocks on the same `s` within a diffusion step: two
adaLN scale/shift pairs and two sigmoid gates a block, six `[N, 384] x [384, 768]` linears, 144 a
step and 28,800 a 512-token fold at 16.7 us each on a p150a (cen68), ten times the matmul's byte
time. `OF3DiffusionTransformer` computes them as three wide linears a step instead, and the blocks
read their 768 columns out of the wide outputs here, where they were consumed anyway: the adaLN's
`sb + LN(a) * sc` and the two gated residuals `a + o * g` are each this op.

A window is described by the operand's own tile width (compile time) and a column tile offset (a
common runtime arg), so one cached program serves all 24 blocks' windows of the same tensor and a
call costs the dispatch of a cached descriptor with seven rewritten scalars.

Declines (returns None) on anything but bfloat16 or float32 (all alike) TILE interleaved operands whose row tiles match;
the caller then runs the ttnn ops. x and z may instead be one sample of y's rows (leading dims 1, same row count),
which every sample of y reads: the DiT's conditioning is one row block for all diffusion samples.
"""

from __future__ import annotations

from pathlib import Path

import ttnn

from . import core_split
from . import ops as _ops

KERNEL_DIR = Path(__file__).resolve().parent / "kernels" / "addcmul_at"

TILE = 32
X_CB, Y_CB, Z_CB, OUT_CB = 0, 1, 2, 16
_GRAN = 4  # tiles per DEST acquire: four fp32 tiles fill Blackhole's 32-bit DEST

_TILE_BYTES = {ttnn.float32: 4096, ttnn.bfloat16: 2048, ttnn.bfloat8_b: 1088}
_CACHE: dict = {}
ADDR_WRITE_MODE = None
#: (served, declined), cumulative.
STATS = [0, 0]


def _grid(t):
    """(row tiles over all leading dims, column tiles) of a TILE tensor."""
    shape = [int(d) for d in t.shape]
    lead = 1
    for d in shape[:-2]:
        lead *= d
    return lead * ((shape[-2] + TILE - 1) // TILE), (shape[-1] + TILE - 1) // TILE


def _build(x, y, z, out, device, ht, wt):
    f32 = y.dtype == ttnn.float32
    num_tiles = ht * wt
    g = device.compute_with_storage_grid_size()
    num_cores, core_grid, cg1, cg2, work1, work2 = core_split.split_work_to_cores(g, num_tiles, 0)

    def cb(idx, dt=y.dtype):
        b = _TILE_BYTES[dt]
        fmt = ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dt, page_size=b)
        return ttnn.CBDescriptor(total_size=2 * _GRAN * b, core_ranges=core_grid, format_descriptors=[fmt])

    reader_rt, compute_rt, writer_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    first = 0
    for group, per_core in ((cg1, work1), (cg2, work2)):
        for cr in group.ranges():
            for cx in range(cr.start.x, cr.end.x + 1):
                for cy in range(cr.start.y, cr.end.y + 1):
                    reader_rt[cx][cy] = [first, per_core]
                    compute_rt[cx][cy] = [per_core]
                    writer_rt[cx][cy] = [first, per_core]
                    first += per_core
    assert first == num_tiles, (first, num_tiles)

    reader_ct = [_GRAN, wt] + [_grid(t)[1] for t in (x, y, z)] + [_grid(t)[0] for t in (x, y, z)]
    for t in (x, y, z):
        reader_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    writer_ct = [_GRAN] + list(ttnn.TensorAccessorArgs(out).get_compile_time_args())
    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "reader_addcmul_at.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=core_grid,
        compile_time_args=reader_ct, runtime_args=reader_rt, common_runtime_args=[0] * 6,
        config=ttnn.ReaderConfigDescriptor())
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "writer_addcmul_at.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=core_grid,
        compile_time_args=writer_ct, runtime_args=writer_rt, common_runtime_args=[0],
        config=ttnn.WriterConfigDescriptor())
    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)
    if f32:
        from .rne_add import _NUM_CBS
        modes = [ttnn.UnpackToDestMode.Default] * _NUM_CBS
        for i in (X_CB, Y_CB, Z_CB):
            modes[i] = ttnn.UnpackToDestMode.UnpackToDestFp32
        cfg.unpack_to_dest_mode = modes
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "compute_addcmul_at.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=core_grid,
        compile_time_args=[_GRAN], defines=[("ADDCMUL_SFPU", "1")] if f32 else [],
        runtime_args=compute_rt, config=cfg)
    cbs = [cb(X_CB), cb(Y_CB), cb(Z_CB), cb(OUT_CB, out.dtype)]
    pd = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)

    global ADDR_WRITE_MODE
    if ADDR_WRITE_MODE is None:
        probe = 0xABCD1234
        pd.kernels[0].common_runtime_args = [probe] + [0] * 5
        ADDR_WRITE_MODE = ("in_place" if list(pd.kernels[0].common_runtime_args)[0] == probe
                           else "rebuild_pd")
    return {"pd": pd, "kernels": [reader, writer, compute], "cbs": cbs}


def _ok(t, y, ht, dtype=ttnn.bfloat16):
    """`t` has y's row tiles, or is one sample of them (leading dims 1, y's row count) when `t` is not y."""
    rows = _grid(t)[0] == ht or (t is not y and all(int(d) == 1 for d in t.shape[:-2])
                                 and int(t.shape[-2]) == int(y.shape[-2]))
    return (t.dtype == dtype and t.layout == ttnn.TILE_LAYOUT and rows
            and t.memory_config().memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED)


@_ops.fused_kernel("addcmul_at")
def addcmul_at(x, y, z, width, x_off=0, y_off=0, z_off=0, out_dtype=None):
    """``x[.., off_x:off_x+width] + y[..] * z[..]`` (offsets and width in elements, tile-aligned).

    The result is a dense tensor of `out_dtype` (default `y`'s; the packer converts), `y`'s leading
    shape and `width` columns: bfloat16 on the
    FPU, or float32 on the SFPU at full precision (all three operands the same dtype). None when
    the kernel declines; no operand is deallocated.
    """
    ht = _grid(y)[0]
    if (width % TILE or any(o % TILE for o in (x_off, y_off, z_off))
            or y.dtype not in (ttnn.bfloat16, ttnn.float32)
            or (out_dtype or y.dtype) not in _TILE_BYTES
            or not all(_ok(t, y, ht, y.dtype) for t in (x, y, z))
            or any(o + width > int(t.shape[-1]) for t, o in ((x, x_off), (y, y_off), (z, z_off)))
            or _ops.declines_under_tape("addcmul_at")):
        STATS[1] += 1
        return None
    STATS[0] += 1
    wt = width // TILE
    device = y.device()
    shape = list(y.shape)[:-1] + [width]
    out = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), out_dtype or y.dtype, ttnn.TILE_LAYOUT, device,
                                         ttnn.DRAM_MEMORY_CONFIG)
    key = (device.id(), ht, wt, _grid(x), _grid(y)[1], _grid(z), str(y.dtype), str(out.dtype))
    entry = _CACHE.get(key)
    if entry is None:
        entry = _CACHE[key] = _build(x, y, z, out, device, ht, wt)
    common_r = [x.buffer_address(), x_off // TILE, y.buffer_address(), y_off // TILE,
                z.buffer_address(), z_off // TILE]
    common_w = [out.buffer_address()]
    if ADDR_WRITE_MODE == "in_place":
        pd = entry["pd"]
        pd.kernels[0].common_runtime_args = common_r
        pd.kernels[1].common_runtime_args = common_w
    else:
        reader, writer, compute = entry["kernels"]
        reader.common_runtime_args = common_r
        writer.common_runtime_args = common_w
        pd = entry["pd"] = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[],
                                                  cbs=entry["cbs"])
    return ttnn.generic_op([x, y, z, out], pd)
