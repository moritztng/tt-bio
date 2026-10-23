"""RFD4's initial pair, built on the card as one Tensix program.

`tt_bio.rfd4.pair.PairOnCard` computes z = (C + W_x[bin]) * scale * (w gamma) + beta for every pair: a one-hot
embedding, a matmul that gathers the weight rows, an add, a column-broadcast multiply by the per-pair scale and two
row-broadcast ops for the AdaLN. At 512 tokens that is ~29 ms of kernels a sampler step plus 9.4 ms the card waits for
the scale's upload (state/rfd4/bets/host2.md). Here one program reads the constant projection C once and writes z
once: the reader gathers W_x's float32 rows from a copy of the table in its own L1 straight into tile faces, so the
gather is exact (the matmul kept ~10 mantissa bits), and builds the scale tile from a row-major [B * N, N] upload;
the compute runs the same four float32 ops in the same order on the SFPU.

`pair_build(...)` returns z or None, and None means the caller keeps its own path.
"""

from __future__ import annotations

from pathlib import Path

import ttnn

from . import core_split

KERNEL_DIR = Path(__file__).resolve().parent / "kernels" / "pair_build"
C_CB, G_CB, S_CB, WG_CB, BETA_CB, TAB_CB, BINS_CB, SROW_CB, OUT_CB = 0, 1, 2, 3, 4, 5, 6, 7, 16
_NUM_CBS = 64
_TILE = 32 * 32 * 4
_CACHE: dict = {}


def eligible(const, table, bins, scale, wg, beta) -> bool:
    """const [B, N, N, C] float32 TILE; table [NB, C] float32 ROW_MAJOR; bins uint32 and scale float32 ROW_MAJOR
    [B * N, N]; wg and beta [B, 32, C] float32 TILE (each row the same vector). N and C tile multiples, all in DRAM."""
    ts = [const, table, bins, scale, wg, beta]
    if [t.dtype for t in ts] != [ttnn.float32, ttnn.float32, ttnn.uint32, ttnn.float32, ttnn.float32, ttnn.float32]:
        return False
    if [t.layout for t in ts] != [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT, ttnn.ROW_MAJOR_LAYOUT,
                                  ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT]:
        return False
    if any(t.memory_config().buffer_type != ttnn.BufferType.DRAM for t in ts):
        return False
    cs = [int(d) for d in const.shape]
    if len(cs) != 4:
        return False
    B, n, m, c = cs
    return (n == m and n % 32 == 0 and c % 32 == 0 and int(table.shape[0]) <= 32 and list(table.shape) == [table.shape[0], c]
            and [int(d) for d in bins.shape] == [B * n, n] == [int(d) for d in scale.shape]
            and [int(d) for d in wg.shape] == [B, 32, c] == [int(d) for d in beta.shape])


def _build(const, table, bins, scale, wg, beta, out, device):
    B, n, _, c = (int(d) for d in const.shape)
    nt, kt, nb = n // 32, c // 32, int(table.shape[0])
    units = B * n * nt
    g = device.compute_with_storage_grid_size()
    _, grid, cg1, cg2, work1, work2 = core_split.split_work_to_cores(g, units, 0)

    def cb(idx, page, depth):
        fmt = ttnn.CBFormatDescriptor(buffer_index=idx, data_format=ttnn.float32, page_size=page)
        return ttnn.CBDescriptor(total_size=depth * page, core_ranges=grid, format_descriptors=[fmt])

    cbs = [cb(C_CB, _TILE, 2 * kt), cb(G_CB, _TILE, 2 * kt), cb(S_CB, _TILE, 2), cb(WG_CB, _TILE, kt),
           cb(BETA_CB, _TILE, kt), cb(OUT_CB, _TILE, 2 * kt), cb(TAB_CB, nb * c * 4, 1), cb(BINS_CB, n * 4, 1),
           cb(SROW_CB, n * 4, 1)]
    rt = ttnn.RuntimeArgs()
    first = 0
    for group, per in ((cg1, work1), (cg2, work2)):
        for cr in group.ranges():
            for cx in range(cr.start.x, cr.end.x + 1):
                for cy in range(cr.start.y, cr.end.y + 1):
                    rt[cx][cy] = [first, per]
                    first += per
    assert first == units, (first, units)
    acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())  # noqa: E731
    path = ttnn.KernelDescriptor.SourceType.FILE_PATH
    src = lambda name: str(KERNEL_DIR / name)  # noqa: E731
    reader = ttnn.KernelDescriptor(kernel_source=src("reader.cpp"), source_type=path, core_ranges=grid,
                                   compile_time_args=[n, nt, kt, nb] + acc(const) + acc(table) + acc(bins) + acc(scale)
                                   + acc(wg) + acc(beta),
                                   runtime_args=rt, common_runtime_args=[0] * 6, config=ttnn.ReaderConfigDescriptor())
    writer = ttnn.KernelDescriptor(kernel_source=src("writer.cpp"), source_type=path, core_ranges=grid,
                                   compile_time_args=[kt] + acc(out), runtime_args=rt, common_runtime_args=[0],
                                   config=ttnn.WriterConfigDescriptor())
    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)
    modes = [ttnn.UnpackToDestMode.Default] * _NUM_CBS
    for i in (C_CB, G_CB, S_CB, WG_CB, BETA_CB):
        modes[i] = ttnn.UnpackToDestMode.UnpackToDestFp32
    cfg.unpack_to_dest_mode = modes
    compute = ttnn.KernelDescriptor(kernel_source=src("compute.cpp"), source_type=path, core_ranges=grid,
                                    compile_time_args=[n, nt, kt], runtime_args=rt, config=cfg)
    return [reader, writer, compute], cbs


def pair_build(const, table, bins, scale, wg, beta):
    """z [B, N, N, C] float32 TILE = (const + table[bins]) * scale * wg + beta, or None when `eligible` declines.
    Nothing is freed."""
    if not eligible(const, table, bins, scale, wg, beta):
        return None
    device = const.device()
    out = ttnn.allocate_tensor_on_device(const.shape, ttnn.float32, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    key = (device.id(), tuple(int(d) for d in const.shape), int(table.shape[0]))
    if key not in _CACHE:
        _CACHE[key] = _build(const, table, bins, scale, wg, beta, out, device)
    kernels, cbs = _CACHE[key]
    kernels[0].common_runtime_args = [t.buffer_address() for t in (const, table, bins, scale, wg, beta)]
    kernels[1].common_runtime_args = [out.buffer_address()]
    ttnn.generic_op([const, table, bins, scale, wg, beta, out], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[],
                                                                                     cbs=cbs))
    return out
