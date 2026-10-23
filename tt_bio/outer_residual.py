"""RFD4's Hadamard pair write and its residual add into the float32 pair stream as one Tensix program.

`PairUpdate._write` makes the update pj[b, i, :] * pi[b, j, :] for every pair (i, j) and `_residual` adds it to z:
a broadcast multiply that writes the [n, n, pair] update in bfloat16, then `rne_add.residual_dual` reading it back
with z and writing z and its bfloat16 copy. Here the product is formed in DEST from one pi tile and one row of pj
and added to z there, so the update never touches DRAM: at 512 tokens 1.0 GB a call against 1.41 GB plus the
multiply's pass (state/rfd4/bets/pairbias.md). The program walks z in (b, jt, i, ct) order so a core holds one
run's pi tiles and reads only row 0 of each pj4 tile: z is the only tensor it streams (state/rfd4/bets/ptrans.md;
1.41 -> ~1.02 GB a call at 512 tokens, every bit the same). The product is exact in the 32-bit DEST and the sum is rounded once
to float32, so the result is closer to float64 than the two-op path, whose update is rounded to bfloat16 first;
the bfloat16 copy is that sum rounded to even, as `residual_dual` writes it.

`outer_residual` returns `(z', zb')` as new tensors or None, and None means the caller keeps its own path. A bfloat16
z (RFD4's `pair_bf16z` stream) is its own copy: the sum rounded once to even is the one output, `(z', z')`.
"""

from __future__ import annotations

from pathlib import Path

import ttnn

from . import core_split

KERNEL_DIR = Path(__file__).resolve().parent / "kernels" / "outer_residual"
Z_CB, I_CB, J_CB, OUT_CB, LO_CB = 0, 1, 2, 16, 17
GRAN = 4
_NUM_CBS = 64
_CACHE: dict = {}


def eligible(z, pi, pj4) -> bool:
    """z [B, n, n, C] float32, bfloat16 or bfloat8_b (`pair_bfp8z`), pi [B, n, C] and pj4 [B, n, 1, C] bfloat16, all TILE interleaved, n and C
    tile multiples (no padding: every tile of z has a whole pi tile)."""
    if z.dtype not in (ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b) or pi.dtype != ttnn.bfloat16 or pj4.dtype != ttnn.bfloat16:
        return False
    if any(t.layout != ttnn.TILE_LAYOUT or t.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED
           for t in (z, pi, pj4)):
        return False
    zs, ps, js = ([int(d) for d in t.shape] for t in (z, pi, pj4))
    if len(zs) != 4 or len(ps) != 3 or len(js) != 4:
        return False
    B, n, m, c = zs
    return n == m and n % 32 == 0 and c % 32 == 0 and ps == [B, n, c] and js == [B, n, 1, c]


def _build(z, pi, pj4, out, lo, device):
    B, n, _, c = (int(d) for d in z.shape)
    b16z = z.dtype in (ttnn.bfloat16, ttnn.bfloat8_b)   # a narrow z: one rounded output, no float32 master
    b8z = z.dtype == ttnn.bfloat8_b
    kt, nt = c // 32, n // 32
    pages = B * n * nt * kt
    g = device.compute_with_storage_grid_size()
    _, grid, cg1, cg2, work1, work2 = core_split.split_work_to_cores(g, pages, 0)

    def cb(idx, dtype, size, tiles=2 * GRAN):
        fmt = ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dtype, page_size=size)
        return ttnn.CBDescriptor(total_size=tiles * size, core_ranges=grid, format_descriptors=[fmt])

    f32, b16, b8 = 32 * 32 * 4, 32 * 32 * 2, 32 * 32 + 64
    if b16z:
        zf, zp = (ttnn.bfloat8_b, b8) if b8z else (ttnn.bfloat16, b16)
        cbs = [cb(Z_CB, zf, zp), cb(I_CB, ttnn.bfloat16, b16, 2 * kt), cb(J_CB, ttnn.bfloat16, b16),
               cb(OUT_CB, zf, zp)]
    else:
        cbs = [cb(Z_CB, ttnn.float32, f32), cb(I_CB, ttnn.bfloat16, b16, 2 * kt), cb(J_CB, ttnn.bfloat16, b16),
               cb(OUT_CB, ttnn.float32, f32), cb(LO_CB, ttnn.bfloat16, b16)]
    defines = ([("OR_BF16", "1")] if b16z else []) + ([("OR_BFP8", "1")] if b8z else [])
    rd, cp, wr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    first = 0
    for group, per in ((cg1, work1), (cg2, work2)):
        for cr in group.ranges():
            for cx in range(cr.start.x, cr.end.x + 1):
                for cy in range(cr.start.y, cr.end.y + 1):
                    rd[cx][cy] = cp[cx][cy] = wr[cx][cy] = [first, per]
                    first += per
    assert first == pages, (first, pages)
    acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())  # noqa: E731
    path = ttnn.KernelDescriptor.SourceType.FILE_PATH
    reader = ttnn.KernelDescriptor(kernel_source=str(KERNEL_DIR / "reader.cpp"), source_type=path, core_ranges=grid,
                                   compile_time_args=[kt, nt, n, GRAN] + acc(z) + acc(pi) + acc(pj4),
                                   runtime_args=rd, common_runtime_args=[0] * 3, defines=defines,
                                   config=ttnn.ReaderConfigDescriptor())
    writer = ttnn.KernelDescriptor(kernel_source=str(KERNEL_DIR / "writer.cpp"), source_type=path, core_ranges=grid,
                                   compile_time_args=[GRAN, kt, nt, n] + acc(out) + ([] if b16z else acc(lo)),
                                   runtime_args=wr, common_runtime_args=[0] * 2, defines=defines,
                                   config=ttnn.WriterConfigDescriptor())
    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)
    modes = [ttnn.UnpackToDestMode.Default] * _NUM_CBS
    if not b16z:    # a bf16 or bfp8 z goes through srcA exactly
        modes[Z_CB] = ttnn.UnpackToDestMode.UnpackToDestFp32
    cfg.unpack_to_dest_mode = modes
    compute = ttnn.KernelDescriptor(kernel_source=str(KERNEL_DIR / "compute.cpp"), source_type=path,
                                    core_ranges=grid, compile_time_args=[GRAN, n * kt, kt], runtime_args=cp,
                                    defines=defines, config=cfg)
    return [reader, writer, compute], cbs


def outer_residual(z, pi, pj4):
    """`(z + pj4 * pi4, the same rounded to even in bfloat16)` for the float32 pair stream z [B, n, n, C], pi [B, n, C]
    and pj4 [B, n, 1, C] (pi4 = pi as [B, 1, n, C]), or None when `eligible` declines. For a bfloat16 or bfloat8_b z,
    `(out, out)` with out the sum packed in z's format. Nothing is freed."""
    if not eligible(z, pi, pj4):
        return None
    device = z.device()
    alloc = lambda dt: ttnn.allocate_tensor_on_device(z.shape, dt, ttnn.TILE_LAYOUT, device,  # noqa: E731
                                                      ttnn.DRAM_MEMORY_CONFIG)
    out = alloc(z.dtype)
    lo = out if z.dtype != ttnn.float32 else alloc(ttnn.bfloat16)
    key = (device.id(), tuple(int(d) for d in z.shape), str(z.dtype))
    if key not in _CACHE:
        _CACHE[key] = _build(z, pi, pj4, out, lo, device)
    kernels, cbs = _CACHE[key]
    kernels[0].common_runtime_args = [z.buffer_address(), pi.buffer_address(), pj4.buffer_address()]
    kernels[1].common_runtime_args = [out.buffer_address(), lo.buffer_address()]
    ttnn.generic_op([z, pi, pj4] + ([out] if lo is out else [out, lo]),
                    ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
    return out, lo
