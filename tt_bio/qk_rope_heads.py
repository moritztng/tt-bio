"""ESMC's q/k LayerNorm, NeoX RoPE and head split as one Tensix program.

The protein LMs (ESMC 300M / 600M / 6B, and the ESMC inside ESMFold2) run, between the qkv matmul and SDPA,
``layer_norm(q)``, ``layer_norm(k)``, a concat of q, k, v, ``nlp_create_qkv_heads`` and two ``rotary_embedding``
calls: six ops and about 6x the activation's bytes through DRAM. On a Wormhole chip at the ESMC 300M page batch
(7 x 544 tokens) that chain was ~35 ms of a 134 ms forward. Every one of these models has head_dim 64, two tiles,
so rotate_half is a swap of a head's two tiles with a sign and the head split is only where a tile is written:
this program reads q, k and v once and writes the per-head [B, H, L, 64] tensors once (kernels/qk_rope_heads).

Units are row tiles of q, k and v (3 * B * L / 32 of them); the q / k units and the v units are each spread over the
whole grid. The LayerNorm is
kernels/common/ln_prologue.hpp's two-pass fp32 statistics without the beta, its output kept in fp32 for the RoPE;
the cos products go through an fp32 CB and the sin products are added in dest. Where both gammas do not fit L1
beside a row's fp32 statistics (the 6B, D = 2560), each q / k row re-reads its gamma (QKRH_RELOAD).
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


def _vc(kt: int) -> int:
    """v tiles a reader / writer hand-off: the largest even divisor of KT up to 10 (whole heads)."""
    return max(c for c in range(2, 11, 2) if kt % c == 0)


def _plan(kt: int):
    """(reload, x buffers): both gammas resident and x double-buffered where that fits, else less."""
    for reload, xb in ((False, 2), (False, 1), (True, 1)):
        if _l1_bytes(kt, reload, xb) <= _L1_BUDGET:
            return reload, xb
    return None


def _l1_bytes(kt: int, reload: bool = False, xb: int = 2) -> int:
    bf16 = (xb * kt + 2 + 2 + (1 if reload else 2) * kt + 1 + 4 + 2 * _vc(kt)) * _TILE   # x, cos, sin, gammas, scaler, out, v
    fp32 = (4 + 2 * kt + 2) * 2 * _TILE                                                  # eps, mean, var, rstd, xc, tmp, p
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
    return _plan(shape[-1] // 32) is not None


def _build(device, kt, lt, r, eps, tensors):
    g = device.compute_with_storage_grid_size()
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))])

    def cb(idx, tiles, dtype=ttnn.bfloat16):
        size = 2 * _TILE if dtype == ttnn.float32 else _TILE
        fmt = ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dtype, page_size=size)
        return ttnn.CBDescriptor(total_size=tiles * size, core_ranges=grid, format_descriptors=[fmt])

    f32 = ttnn.float32
    reload, xb = _plan(kt)
    vc = _vc(kt)
    cbs = [cb(5, xb * kt), cb(6, 2), cb(7, 2), cb(8, kt), cb(10, 1), cb(16, 4), cb(21, 2 * vc),
           cb(11, 1, f32), cb(12, 1, f32), cb(13, 1, f32), cb(14, 1, f32), cb(17, kt, f32), cb(18, kt, f32),
           cb(20, 2, f32)] + ([] if reload else [cb(15, kt)])
    defines = [("QKRH_RELOAD", "1")] if reload else []
    inv = struct.unpack("<I", struct.pack("<f", 1.0 / (32 * kt)))[0] >> 16
    scaler = (inv << 16) | inv
    eps_bits = struct.unpack("<I", struct.pack("<f", float(eps)))[0]
    # q / k units carry the LayerNorm and RoPE, v units are a copy: split each over the whole grid on its own, q / k
    # remainder to the front and v remainder to the back, so no core holds more LayerNorms than it must.
    rd, cp, wr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    cores = [(x, y) for x in range(g.x) for y in range(g.y)]
    fqk, fv = 0, 2 * r
    for (cx, cy), nqk, nv in zip(cores, core_split.even_split(2 * r, cores), core_split.even_split(r, cores)[::-1]):
        rd[cx][cy] = cp[cx][cy] = wr[cx][cy] = [fqk, nqk, fv, nv]
        fqk, fv = fqk + nqk, fv + nv
    assert (fqk, fv) == (2 * r, 3 * r), (fqk, fv, r)
    ins, outs = tensors[:7], tensors[7:]
    rct = [kt, lt, r, scaler, eps_bits, vc]
    for t in ins:
        rct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    wct = [kt, lt, r, vc]
    for t in outs:
        wct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    src = lambda n: str(KERNEL_DIR / n)  # noqa: E731
    path = ttnn.KernelDescriptor.SourceType.FILE_PATH
    reader = ttnn.KernelDescriptor(kernel_source=src("reader.cpp"), source_type=path, core_ranges=grid,
                                   compile_time_args=rct, runtime_args=rd, common_runtime_args=[0] * 7, defines=defines,
                                   config=ttnn.ReaderConfigDescriptor())
    writer = ttnn.KernelDescriptor(kernel_source=src("writer.cpp"), source_type=path, core_ranges=grid,
                                   compile_time_args=wct, runtime_args=wr, common_runtime_args=[0] * 3,
                                   config=ttnn.WriterConfigDescriptor())
    compute = ttnn.KernelDescriptor(kernel_source=src("compute.cpp"), source_type=path, core_ranges=grid,
                                    compile_time_args=[kt, r], runtime_args=cp, defines=defines,
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
