"""ESMC's q/k LayerNorm, NeoX RoPE and head split as one Tensix program.

The protein LMs (ESMC 300M / 600M / 6B, and the ESMC inside ESMFold2) run, between the qkv matmul and SDPA,
``layer_norm(q)``, ``layer_norm(k)``, a concat of q, k, v, ``nlp_create_qkv_heads`` and two ``rotary_embedding``
calls: six ops and about 6x the activation's bytes through DRAM. On a Wormhole chip at the ESMC 300M page batch
(7 x 544 tokens) that chain was ~35 ms of a 134 ms forward. Every one of these models has head_dim 64, two tiles,
so rotate_half is a swap of a head's two tiles with a sign and the head split is only where a tile is written:
this program reads q, k and v once and writes the per-head [B, H, L, 64] tensors once (kernels/qk_rope_heads).

Units are row tiles of q, k and v (3 * B * L / 32 of them), spread over the whole grid. The LayerNorm is
kernels/common/ln_prologue.hpp's two-pass fp32 statistics without the beta; RoPE's cos products go through an
fp32 CB and the sin products are added in dest.
"""

from __future__ import annotations

import struct
from pathlib import Path

import ttnn

from . import core_split

KERNEL_DIR = Path(__file__).resolve().parent / "kernels" / "qk_rope_heads"
_TILE = 32 * 32 * 2
_L1_BUDGET = 1100 * 1024
_CACHE: dict = {}
#: [calls served, declined], read by the A/B harness.
STATS = [0, 0]


def _l1_bytes(kt: int) -> int:
    bf16 = (2 * kt + 2 + 2 + kt + kt + 1 + kt + 4 + 2 * kt) * _TILE      # x, cos, sin, gammas, scaler, n, out, v
    fp32 = (4 + 2 * kt + 2) * 2 * _TILE                                 # eps, mean, var, rstd, xc, tmp, p
    return bf16 + fp32


def eligible(q, k, v, cos) -> bool:
    """bf16 TILE interleaved [B, L, D] q, k, v of one shape, D / 64 heads, a cos table covering L, CBs in budget."""
    shape = [int(d) for d in q.padded_shape]
    if len(shape) != 3 or shape[-1] % 64 or shape[1] % 32:
        return False
    for t in (q, k, v):
        if (t.dtype != ttnn.bfloat16 or t.layout != ttnn.TILE_LAYOUT or [int(d) for d in t.padded_shape] != shape
                or t.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED):
            return False
    if int(cos.padded_shape[-1]) != 64 or int(cos.padded_shape[-2]) < shape[1]:
        return False
    return _l1_bytes(shape[-1] // 32) <= _L1_BUDGET


def _build(device, kt, lt, r, eps, tensors):
    g = device.compute_with_storage_grid_size()
    _, grid, cg1, cg2, work1, work2 = core_split.split_work_to_cores(g, 3 * r, 0)

    def cb(idx, tiles, dtype=ttnn.bfloat16):
        size = 2 * _TILE if dtype == ttnn.float32 else _TILE
        fmt = ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dtype, page_size=size)
        return ttnn.CBDescriptor(total_size=tiles * size, core_ranges=grid, format_descriptors=[fmt])

    f32 = ttnn.float32
    cbs = [cb(5, 2 * kt), cb(6, 2), cb(7, 2), cb(8, kt), cb(15, kt), cb(10, 1), cb(19, kt), cb(16, 4), cb(21, 2 * kt),
           cb(11, 1, f32), cb(12, 1, f32), cb(13, 1, f32), cb(14, 1, f32), cb(17, kt, f32), cb(18, kt, f32),
           cb(20, 2, f32)]
    inv = struct.unpack("<I", struct.pack("<f", 1.0 / (32 * kt)))[0] >> 16
    scaler = (inv << 16) | inv
    eps_bits = struct.unpack("<I", struct.pack("<f", float(eps)))[0]
    rd, cp, wr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    first = 0
    for group, per in ((cg1, work1), (cg2, work2)):
        for cr in group.ranges():
            for cx in range(cr.start.x, cr.end.x + 1):
                for cy in range(cr.start.y, cr.end.y + 1):
                    rd[cx][cy] = cp[cx][cy] = wr[cx][cy] = [first, per]
                    first += per
    assert first == 3 * r, (first, r)
    ins, outs = tensors[:7], tensors[7:]
    rct = [kt, lt, r, scaler, eps_bits]
    for t in ins:
        rct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    wct = [kt, lt, r]
    for t in outs:
        wct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    src = lambda n: str(KERNEL_DIR / n)  # noqa: E731
    path = ttnn.KernelDescriptor.SourceType.FILE_PATH
    reader = ttnn.KernelDescriptor(kernel_source=src("reader.cpp"), source_type=path, core_ranges=grid,
                                   compile_time_args=rct, runtime_args=rd, common_runtime_args=[0] * 7,
                                   config=ttnn.ReaderConfigDescriptor())
    writer = ttnn.KernelDescriptor(kernel_source=src("writer.cpp"), source_type=path, core_ranges=grid,
                                   compile_time_args=wct, runtime_args=wr, common_runtime_args=[0] * 3,
                                   config=ttnn.WriterConfigDescriptor())
    compute = ttnn.KernelDescriptor(kernel_source=src("compute.cpp"), source_type=path, core_ranges=grid,
                                    compile_time_args=[kt, r], runtime_args=cp,
                                    config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4,
                                                                        fp32_dest_acc_en=True))
    return [reader, writer, compute], cbs


def qk_rope_heads(q, k, v, q_gamma, k_gamma, cos, sin, eps: float = 1e-5):
    """(rope(LN(q)), rope(LN(k)), v) per head, [B, H, L, 64] bf16 DRAM each, from [B, L, D] q, k, v; or None where
    `eligible` declines. gamma bf16 TILE [1, D] as `torch_to_tt` leaves a norm parameter; cos / sin the
    [1, 1, L, 64] NeoX tables of `esmc.rope_tables`. Does not free its inputs."""
    if not eligible(q, k, v, cos):
        STATS[1] += 1
        return None
    device = q.device()
    b, length, d = (int(x) for x in q.padded_shape)
    kt, lt, h = d // 32, length // 32, d // 64
    outs = [ttnn.allocate_tensor_on_device(ttnn.Shape([b, h, length, 64]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device,
                                           ttnn.DRAM_MEMORY_CONFIG) for _ in range(3)]
    tensors = [q, k, v, q_gamma, k_gamma, cos, sin] + outs
    key = (device.id(), b, length, d, float(eps), tuple(t.memory_config().buffer_type for t in tensors[:7]))
    if key not in _CACHE:
        _CACHE[key] = _build(device, kt, lt, b * lt, eps, tensors)
    kernels, cbs = _CACHE[key]
    kernels[0].common_runtime_args = [t.buffer_address() for t in tensors[:7]]
    kernels[1].common_runtime_args = [t.buffer_address() for t in outs]
    ttnn.generic_op(tensors, ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
    STATS[0] += 1
    return tuple(outs)
