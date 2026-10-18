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

SaProt (ESM-2) has no q / k LayerNorm and one packed qkv: `rope_heads_packed` runs the same program on it with the
norm compiled out (QKRH_NO_LN) and the reader striding over the packed row.
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
#: compute fidelity of the norm and RoPE products (fp32 dest either way); read at build time, part of the cache key.
FIDELITY = ttnn.MathFidelity.HiFi4
#: tiles per dest acquire in the norm stages, 1 or 4 (2 heads per RoPE group at 4 where the head count is even; 0
#: reads as 1); read at build time, part of the key. 4 is bit-identical to 1; Galaxy chip at 1000 MHz, ms a call on
#: the page shapes: 300M 0.340 -> 0.332, 600M 0.398 -> 0.385, 6B 0.245 -> 0.233.
BLOCK = 4
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
    bf16 = (xb * kt + 2 + 2 + (1 if reload else 2) * kt + 1 + 8 + 2 * _vc(kt)) * _TILE   # x, cos, sin, gammas, scaler, out, v
    fp32 = (4 + 2 * kt + 4) * 2 * _TILE                                                  # eps, mean, var, rstd, xc, tmp, p
    return bf16 + fp32


def eligible(q, k, v, cos, packed: bool = False) -> bool:
    """bf16 TILE interleaved [B, L, D] q, k, v of one shape (or one [B, L, 3D] qkv, passed three times), D / 64 heads, a
    cos table covering L, CBs in budget."""
    shape = [int(d) for d in q.padded_shape]
    if len(shape) != 3 or shape[-1] % (192 if packed else 64) or shape[1] % 32:
        return False
    for t in (q, k, v):
        if (t.dtype != ttnn.bfloat16 or t.layout != ttnn.TILE_LAYOUT or [int(d) for d in t.padded_shape] != shape
                or t.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED):
            return False
    if int(cos.padded_shape[-1]) != 64 or int(cos.padded_shape[-2]) < shape[1]:
        return False
    return _plan(shape[-1] // (96 if packed else 32)) is not None


def _build(device, kt, lt, r, eps, tensors, packed=False, ln=True):
    g = device.compute_with_storage_grid_size()
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))])

    def cb(idx, tiles, dtype=ttnn.bfloat16):
        size = 2 * _TILE if dtype == ttnn.float32 else _TILE
        fmt = ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dtype, page_size=size)
        return ttnn.CBDescriptor(total_size=tiles * size, core_ranges=grid, format_descriptors=[fmt])

    f32 = ttnn.float32
    reload, xb = _plan(kt)
    reload = reload and ln
    vc = _vc(kt)
    cbs = [cb(5, xb * kt), cb(6, 2), cb(7, 2), cb(8, kt), cb(10, 1), cb(16, 8), cb(21, 2 * vc),
           cb(11, 1, f32), cb(12, 1, f32), cb(13, 1, f32), cb(14, 1, f32), cb(17, kt, f32), cb(18, kt, f32),
           cb(20, 4, f32)] + ([] if reload else [cb(15, kt)])
    defines = ([("QKRH_RELOAD", "1")] if reload else []) + ([] if ln else [("QKRH_NO_LN", "1")])
    hg = 2 if BLOCK >= 4 and kt // 2 % 2 == 0 else 1   # whole groups only: c_20 / c_16 never wrap mid-reservation
    defines += [("QKRH_BLK", str(max(1, BLOCK))), ("QKRH_HG", str(hg))]
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
    rct = [kt, lt, r, scaler, eps_bits, vc, 3 * kt if packed else kt, kt if packed else 0]
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
                                    config=ttnn.ComputeConfigDescriptor(math_fidelity=FIDELITY,
                                                                        fp32_dest_acc_en=True))
    return [reader, writer, compute], cbs


def qk_rope_heads(q, k, v, q_gamma, k_gamma, cos, sin, eps: float = 1e-5):
    """(rope(LN(q)), rope(LN(k)), v) per head, [B, H, L, 64] bf16 DRAM each, from [B, L, D] q, k, v; or None where
    `eligible` declines. gamma bf16 TILE [1, D] as `torch_to_tt` leaves a norm parameter; cos / sin the
    [1, 1, L, 64] NeoX tables of `esmc.rope_tables`. Does not free its inputs."""
    return _run(q, k, v, q_gamma, k_gamma, cos, sin, eps, packed=False)


def rope_heads_packed(qkv, cos, sin):
    """(rope(q), rope(k), v) per head from a packed [B, L, 3D] qkv (SaProt: no q / k LayerNorm), as `qk_rope_heads`
    in place of nlp_create_qkv_heads + two rotary_embedding calls; or None where declined. Does not free qkv."""
    return _run(qkv, qkv, qkv, None, None, cos, sin, 0.0, packed=True)


def _run(q, k, v, q_gamma, k_gamma, cos, sin, eps, packed):
    if not eligible(q, k, v, cos, packed):
        STATS[1] += 1
        return None
    ln = q_gamma is not None
    device = q.device()
    b, length, d = (int(x) for x in q.padded_shape)
    d = d // 3 if packed else d
    kt, lt, h = d // 32, length // 32, d // 64
    outs = [ttnn.allocate_tensor_on_device(ttnn.Shape([b, h, length, 64]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device,
                                           ttnn.DRAM_MEMORY_CONFIG) for _ in range(3)]
    # without a LayerNorm the gamma slots are never read; cos stands in so every accessor has a tensor
    tensors = [q, k, v, q_gamma if ln else cos, k_gamma if ln else cos, cos, sin] + outs
    key = (device.id(), b, length, d, float(eps), packed, ln, str(FIDELITY), BLOCK, tuple(t.memory_config().buffer_type for t in tensors[:7]))
    if key not in _CACHE:
        _CACHE[key] = _build(device, kt, lt, b * lt, eps, tensors, packed, ln)
    kernels, cbs = _CACHE[key]
    kernels[0].common_runtime_args = [t.buffer_address() for t in tensors[:7]]
    kernels[1].common_runtime_args = [t.buffer_address() for t in outs]
    ttnn.generic_op(tensors, ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
    STATS[0] += 1
    return tuple(outs)
