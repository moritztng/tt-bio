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
which every sample of y reads: the DiT's conditioning is one row block for all diffusion samples. Such an x or z may
also be row block `x_row` / `z_row` of a tensor that stacks several (leading dims G): `dit_cond_steps` keeps G
diffusion steps' conditioning in one.
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


def _facts(t):
    """What the checks and the cache key read off an operand, each read once (a ttnn attribute read is a host call):
    (shape, row tiles over all leading dims, column tiles, leading blocks, dtype, layout, memory config)."""
    shape = [int(d) for d in t.shape]
    lead = 1
    for d in shape[:-2]:
        lead *= d
    return (shape, lead * ((shape[-2] + TILE - 1) // TILE), (shape[-1] + TILE - 1) // TILE, lead, t.dtype, t.layout,
            t.memory_config())


def _build(x, y, z, out, device, ht, wt, x_row, z_row, fx, fy, fz):
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

    reader_ct = [_GRAN, wt] + [f[2] for f in (fx, fy, fz)] + [_rows(f, ht, r) for f, r in ((fx, x_row), (fy, None), (fz, z_row))]
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


def _rows(f, ht, row=None):
    """The row tiles an operand (its `_facts`) contributes to one sample of the output: all of y's, or one block."""
    return ht if f[1] == ht and row is None else -(-f[0][-2] // TILE)


def _ok(f, fy, is_y, ht, row=None):
    """An operand (its `_facts`) has y's row tiles, or (when it is not y) holds blocks of one sample's rows (y's row
    count a block): one (leading dims 1), which every sample of y reads, or several, of which it is read at block `row`."""
    rows = (f[1] == ht and row is None) or (not is_y and f[0][-2] == fy[0][-2] and 0 <= (row or 0) < f[3])
    return (f[4] == fy[4] and f[5] == ttnn.TILE_LAYOUT and rows
            and f[6].memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED)


@_ops.fused_kernel("addcmul_at")
def addcmul_at(x, y, z, width, x_off=0, y_off=0, z_off=0, out_dtype=None, x_row=None, z_row=None):
    """``x[.., off_x:off_x+width] + y[..] * z[..]`` (offsets and width in elements, tile-aligned; `x_row` / `z_row`
    the row block of a stacked x / z, see the module docstring).

    The result is a dense tensor of `out_dtype` (default `y`'s; the packer converts), `y`'s leading
    shape and `width` columns: bfloat16 on the
    FPU, or float32 on the SFPU at full precision (all three operands the same dtype). None when
    the kernel declines; no operand is deallocated.
    """
    fx, fy, fz = _facts(x), _facts(y), _facts(z)
    ht = fy[1]
    if (width % TILE or any(o % TILE for o in (x_off, y_off, z_off))
            or fy[4] not in (ttnn.bfloat16, ttnn.float32)
            or (out_dtype or fy[4]) not in _TILE_BYTES
            or not all(_ok(f, fy, t is y, ht, r) for t, f, r in ((x, fx, x_row), (y, fy, None), (z, fz, z_row)))
            or any(o + width > f[0][-1] for f, o in ((fx, x_off), (fy, y_off), (fz, z_off)))
            or _ops.declines_under_tape("addcmul_at")):
        STATS[1] += 1
        return None
    STATS[0] += 1
    wt = width // TILE
    device = y.device()
    out_dtype = out_dtype or fy[4]
    out = ttnn.allocate_tensor_on_device(ttnn.Shape(fy[0][:-1] + [width]), out_dtype, ttnn.TILE_LAYOUT, device,
                                         ttnn.DRAM_MEMORY_CONFIG)
    rx, rz = _rows(fx, ht, x_row), _rows(fz, ht, z_row)
    # The buffer types are compile-time args of the TensorAccessors: an L1 operand must not reuse a DRAM program.
    key = (device.id(), ht, wt, fx[1:3], fy[2], fz[1:3], rx, rz, fy[4], out_dtype, *(f[6].buffer_type for f in (fx, fy, fz)))
    entry = _CACHE.get(key)
    if entry is None:
        entry = _CACHE[key] = _build(x, y, z, out, device, ht, wt, x_row, z_row, fx, fy, fz)
    # A row block is a page offset: block b of a stacked operand starts b * (its rows) * (its width) pages in.
    common_r = [x.buffer_address(), x_off // TILE + (x_row or 0) * rx * fx[2], y.buffer_address(),
                y_off // TILE, z.buffer_address(), z_off // TILE + (z_row or 0) * rz * fz[2]]
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
