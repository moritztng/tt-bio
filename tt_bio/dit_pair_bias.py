"""Every DiT block's SDPA mask in one Tensix program: ``(z @ W) * scale + mask``, head-first.

RFdiffusion3's token DiT projects the pair tensor z [B, I, n, 128] to all 18 blocks' 16 heads at once and needs
them as [B, 288, n, n] SDPA masks. The ttnn chain (linear, permute (0, 3, 1, 2), multiply, add, pad) writes and
re-reads the 285 MB stack four times and its permute transposes inside every tile: ~24 ms a call on Wormhole
against ~2 ms of DRAM traffic for z in and the mask out.

Here a core takes a (b, ib, jb) tile of the output's (i, j) plane, computes ``W^T[hb] @ z(i, jb)^T`` for its 32
rows i (tile i is [h x j]), packs the 32 results untilized as one 32-tile row so that row h of that buffer is head
h's [i x j] block in row-major order, and tilizes each 2 KB row back into head h's output tile. The scale and the
mask add run on the dest before the pack (kernels/dit_pair_bias). The mask comes in already padded to n x n with
the pad value, so rows i >= I come out exactly as the pad.
"""

from __future__ import annotations

import struct
from pathlib import Path

import ttnn

from . import core_split

KERNEL_DIR = Path(__file__).resolve().parent / "kernels" / "dit_pair_bias"
W_CB, Z_CB, M_CB, RM_CB, T_CB, OUT_CB = 0, 1, 2, 3, 4, 16
_TILE = 32 * 32 * 2
_SW = 4
_L1_BUDGET = 1100 * 1024
_CACHE: dict = {}


def _l1_bytes(kt, ht) -> int:
    return (ht * kt + 2 * 32 * kt + 2 + 32 + 32 + 2 * 32) * _TILE


def eligible(z, wt, mask) -> bool:
    """z [B, I, n, C] and mask [1 or B, 1, n, n] bf16 TILE in DRAM, W^T [1, 1, H, C]; H, C, n tile multiples."""
    ts = [z, wt, mask]
    if any(t.dtype != ttnn.bfloat16 or t.layout != ttnn.TILE_LAYOUT
           or t.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED
           or t.memory_config().buffer_type != ttnn.BufferType.DRAM for t in ts):
        return False
    zs, ws, ms = ([int(d) for d in t.padded_shape] for t in ts)
    if len(zs) != 4 or len(ws) != 4 or len(ms) != 4:
        return False
    n, c, h = zs[2], zs[3], ws[2]
    return (ws[3] == c and h % 32 == 0 and ms[2] == ms[3] == n and ms[1] == 1 and ms[0] in (1, zs[0])
            and zs[1] <= n and _l1_bytes(c // 32, h // 32) <= _L1_BUDGET)


def _build(z, wt, mask, out, device, units, kt, ht, nt, rows, cols, mb, scale, fidelity, raw=False, k1=True,
           n_out=1):
    g = device.compute_with_storage_grid_size()
    _, grid, cg1, cg2, work1, work2 = core_split.split_work_to_cores(g, units, 0)

    def cb(idx, tiles):
        fmt = ttnn.CBFormatDescriptor(buffer_index=idx, data_format=ttnn.bfloat16, page_size=_TILE)
        return ttnn.CBDescriptor(total_size=tiles * _TILE, core_ranges=grid, format_descriptors=[fmt])

    cbs = [cb(W_CB, ht * kt), cb(Z_CB, 2 * 32 * kt), cb(M_CB, 2), cb(RM_CB, 32), cb(T_CB, 32), cb(OUT_CB, 2 * 32)]
    rd, cp, wr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    first = 0
    for group, per in ((cg1, work1), (cg2, work2)):
        for cr in group.ranges():
            for cx in range(cr.start.x, cr.end.x + 1):
                for cy in range(cr.start.y, cr.end.y + 1):
                    rd[cx][cy] = cp[cx][cy] = wr[cx][cy] = [first, per]
                    first += per
    assert first == units, (first, units)
    rct = [kt, ht, nt, rows, mb, cols]
    for t in (z, wt, mask):
        rct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    wct = [ht, nt, _SW, ht * 32 // n_out] + list(ttnn.TensorAccessorArgs(out).get_compile_time_args())
    sbits = struct.unpack("<I", struct.pack("<f", scale))[0]
    # Wormhole's fp32 dest writes -2/-4/-8 where a second K tile nearly cancels the first (tt_bio/dest_guard.py):
    # sum the K tiles on the SFPU there. Blackhole does not have the fault.
    defines = [("DPB_K1", "1")] if device.arch() == ttnn.Arch.WORMHOLE_B0 and kt <= 4 and k1 else []
    if raw:
        defines.append(("DPB_RAW", "1"))
    src = lambda n: str(KERNEL_DIR / n)  # noqa: E731
    path = ttnn.KernelDescriptor.SourceType.FILE_PATH
    reader = ttnn.KernelDescriptor(kernel_source=src("reader.cpp"), source_type=path, core_ranges=grid,
                                   compile_time_args=rct, runtime_args=rd, common_runtime_args=[0] * 3,
                                   config=ttnn.ReaderConfigDescriptor())
    writer = ttnn.KernelDescriptor(kernel_source=src("writer.cpp"), source_type=path, core_ranges=grid,
                                   compile_time_args=wct, runtime_args=wr, common_runtime_args=[0] * n_out,
                                   config=ttnn.WriterConfigDescriptor())
    compute = ttnn.KernelDescriptor(
        kernel_source=src("compute.cpp"), source_type=path, core_ranges=grid,
        compile_time_args=[kt, ht, nt, rows, sbits], runtime_args=cp, defines=defines,
        config=ttnn.ComputeConfigDescriptor(math_fidelity=fidelity, fp32_dest_acc_en=True))
    return [reader, writer, compute], cbs


def dit_pair_bias(z, wt, mask, scale, fidelity=ttnn.MathFidelity.HiFi4, raw=False, k1=True, n_out=1):
    """[B, H, n, n] bf16 = (z @ W) * scale + mask, head-first, for z [B, I, n, C], wt = W^T as [1, 1, H, C] and
    mask [1 or B, 1, n, n] already padded (rows i >= I take the mask alone). With `n_out` > 1 the heads come back
    as a list of n_out tensors [B, H / n_out, n, n], one per consumer, written in place of slicing the stack.
    `raw` (bench only) skips the scale and the mask: the head-first projection with zero padding. `k1=False`
    (bench only) keeps the K sum in the FPU on Wormhole too."""
    device = z.device()
    zs = [int(d) for d in z.padded_shape]
    B, rows, n, c = zs
    h = int(wt.padded_shape[2])
    kt, ht, nt = c // 32, h // 32, n // 32
    cols = int(z.shape[2])
    mb = int(int(mask.padded_shape[0]) == B and B > 1)
    assert h % n_out == 0, (h, n_out)
    outs = [ttnn.allocate_tensor_on_device(ttnn.Shape([B, h // n_out, n, n]), ttnn.bfloat16, ttnn.TILE_LAYOUT,
                                           device, ttnn.DRAM_MEMORY_CONFIG) for _ in range(n_out)]
    key = (device.id(), tuple(zs), cols, h, mb, float(scale), fidelity, raw, k1, n_out)
    if key not in _CACHE:
        _CACHE[key] = _build(z, wt, mask, outs[0], device, B * nt * nt, kt, ht, nt, rows, cols, mb, float(scale),
                             fidelity, raw, k1, n_out)
    kernels, cbs = _CACHE[key]
    kernels[0].common_runtime_args = [z.buffer_address(), wt.buffer_address(), mask.buffer_address()]
    kernels[1].common_runtime_args = [o.buffer_address() for o in outs]
    pd = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
    ttnn.generic_op([z, wt, mask, *outs], pd)
    return outs if n_out > 1 else outs[0]
