"""OPM relayout in one pass (`kernels/opm_relayout`).

The outer-product-mean contraction gives z with token rows i and c on M, j and d on N. Today b's
rows are (d j), so z = [(i c), (d j)], and four ops (TILE -> RM, reshape, RM -> TILE, permute) turn
it into the projection's [i, j, (c d)]: 11.5 ms a call at 512 tokens on a p150a, 16 calls a fold
(cen68). With b's rows in (j d) order, z = [(i c), (j d)] and its tile (i, j) is exactly the [c, d]
block of pair (i, j). For C = D = 32 the projection input is then one in-group row transpose of 32
consecutive tiles, which this op does in one DRAM read and one write of z. It moves bytes only, so it
is bit-exact against the chain.
"""
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).resolve().parent / "kernels" / "opm_relayout"
SCRATCH_CB, OUT_CB = 24, 16
TILES, TILE_BYTES = 32, 2048
_CACHE = {}


def eligible(z, C, D):
    """None when `opm_relayout` serves z = [(rows C), (J D)], else the reason."""
    if C != 32 or D != 32:
        return "cd"
    if z.dtype != ttnn.bfloat16 or z.layout != ttnn.TILE_LAYOUT or z.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
        return "dtype"
    s = [int(d) for d in z.padded_shape]
    if len(s) != 2 or s != [int(d) for d in z.shape] or s[0] % 32 or s[1] % (32 * TILES):
        return "shape"
    return None


def split(groups, ncores):
    """[(first group, count)] per core, contiguous blocks."""
    out, g0 = [], 0
    for c in range(ncores):
        n = groups // ncores + (c < groups % ncores)
        out.append((g0, n))
        g0 += n
    return out


def opm_relayout(z, rows, J):
    """z [(rows 32), (J 32)] bf16 tiled DRAM -> y [rows, J, 1024] with y[i, j, 32 c + d] =
    z[32 i + c, 32 j + d]. `z` is not deallocated."""
    device = z.device()
    y = ttnn.allocate_tensor_on_device(ttnn.Shape([rows * J, TILES * 32]), ttnn.bfloat16,
                                       ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    g = device.compute_with_storage_grid_size()
    key = (str(z.padded_shape), g.x, g.y)
    entry = _CACHE.get(key)
    if entry is None:
        entry = _CACHE[key] = _build(z, y, rows * J // TILES, g.x, g.y)
    reader, writer = entry["kernels"]
    reader.common_runtime_args = [z.buffer_address()]
    writer.common_runtime_args = [y.buffer_address()]
    pd = ttnn.ProgramDescriptor(kernels=entry["kernels"], semaphores=[], cbs=entry["cbs"])
    ttnn.generic_op([z, y], pd)
    return ttnn.reshape(y, (rows, J, TILES * 32))


def _build(z, y, groups, gx, gy):
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))])

    def cb(idx, depth):
        fmt = ttnn.CBFormatDescriptor(buffer_index=idx, data_format=ttnn.bfloat16, page_size=TILE_BYTES)
        return ttnn.CBDescriptor(total_size=depth * TILE_BYTES, core_ranges=core_grid, format_descriptors=[fmt])

    # Depths are multiples of 32 so a group never wraps the ring (reblock_permute's CB note).
    cbs = [cb(SCRATCH_CB, TILES), cb(OUT_CB, 2 * TILES)]
    rd, wr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for c, (g0, n) in enumerate(split(groups, gx * gy)):
        rd[c % gx][c // gx] = [g0, n]
        wr[c % gx][c // gx] = [g0, n]
    src = ttnn.KernelDescriptor.SourceType.FILE_PATH
    acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())  # noqa: E731
    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "reader.cpp"), source_type=src, core_ranges=core_grid,
        compile_time_args=[SCRATCH_CB, OUT_CB] + acc(z), runtime_args=rd,
        common_runtime_args=[0], config=ttnn.ReaderConfigDescriptor())
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "writer.cpp"), source_type=src, core_ranges=core_grid,
        compile_time_args=[OUT_CB] + acc(y), runtime_args=wr,
        common_runtime_args=[0], config=ttnn.WriterConfigDescriptor())
    return {"kernels": [reader, writer], "cbs": cbs}
