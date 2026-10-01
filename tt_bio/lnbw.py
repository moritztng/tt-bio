"""The layer-norm backward as one kernel: read x, g and gamma once, write dx once.

`autograd._layer_norm_bw` composes it from ~20 ttnn calls, about fourteen of them full passes over
the activation (two-pass statistics, then the three-term dx), because `moreh_layer_norm_backward`
is wrong on Blackhole. A BindCraft 2 round runs 384 pair-sized ones at 288 tokens. This computes
the same expression, with the same two-pass statistics, one tile-row (32 rows x K) at a time in
L1: float32 intermediates, float32 DEST for the row reductions.

x-only. The weights' gradients stay on the composed path, which is the only one that computes
them; BindCraft 2 differentiates the sequence, never the weights.
"""
from __future__ import annotations

import collections
import math
import pathlib
import struct

import ttnn

from tt_bio import core_split
from tt_bio.envflags import env_flag

KERNEL_DIR = pathlib.Path(__file__).parent / "kernels" / "lnbw"

#: The lever. Off unless armed; `bindcraft2.fast_round()` arms it for the round it was graded on.
FUSED = env_flag("TT_BIO_LNBW_FUSED", False)

#: Served and declined calls by reason, cumulative; sample at a round boundary.
REACH: collections.Counter = collections.Counter()

_CACHE: dict = {}
_TILE = 32
_BYTES = {ttnn.bfloat16: 2048, ttnn.float32: 4096}
CB_X, CB_G, CB_GAMMA, CB_SCALER, CB_EPS = 0, 1, 2, 3, 4
CB_MEAN, CB_XC, CB_TMP, CB_VAR, CB_RSTD, CB_NORM, CB_DN, CB_A, CB_B, CB_T, CB_OUT = \
    5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16


def _decline(reason: str) -> bool:
    REACH["declined: " + reason] += 1
    return False


def eligible(x, g, gamma) -> bool:
    """What the kernel serves: bf16 x and g of one shape, TILE, interleaved, a channel
    width that is a power-of-two multiple of 32 (1/K exact in the bf16 reduce scaler), and gamma
    absent or one row of K."""
    if not FUSED:
        return False
    # Graded on Blackhole only (qb1 p150a, qb2 p300c). Wormhole keeps the composed path until the
    # kernel's float64 grade and device test have run on a Wormhole chip.
    from tt_bio import tenstorrent
    if tenstorrent.is_wormhole():
        return _decline("arch")
    # A float32 cotangent (a fan-in accumulator) is declined: the composed path computes that
    # case in exact float32 (1.5e-4 rel L2 against float64 at 64x64x128), and this kernel's FPU
    # stages read float32 CBs at TF32 (1.3e-3), which would spend accuracy the gradient has now.
    if x.dtype != ttnn.bfloat16 or g.dtype != ttnn.bfloat16:
        return _decline("dtype")
    if x.layout != ttnn.TILE_LAYOUT or g.layout != ttnn.TILE_LAYOUT:
        return _decline("layout")
    xs, gs = [int(d) for d in x.shape], [int(d) for d in g.shape]
    if xs != gs:
        return _decline("shape")
    K = xs[-1]
    if K % _TILE or K & (K - 1) or len(xs) < 2 or xs[-2] % _TILE:
        return _decline("width")
    if any(t.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED for t in (x, g)):
        return _decline("sharded")
    if gamma is not None:
        gsh = [int(d) for d in gamma.shape]
        if gamma.dtype != ttnn.bfloat16 or gamma.layout != ttnn.TILE_LAYOUT or gsh[-1] != K or \
                any(d != 1 for d in gsh[:-1]):
            return _decline("gamma")
    return True


def _cb(idx, depth, dtype, grid):
    fmt = ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dtype, page_size=_BYTES[dtype])
    return ttnn.CBDescriptor(total_size=depth * _BYTES[dtype], core_ranges=grid,
                             format_descriptors=[fmt])


def _build(x, g, gamma, out, eps, device):
    K = int(x.shape[-1])
    Wt = K // _TILE
    rows = math.prod(int(d) for d in x.padded_shape) // (K * _TILE)
    grid_size = device.compute_with_storage_grid_size()
    num_cores, grid, cg1, cg2, work1, work2 = core_split.split_work_to_cores(grid_size, rows, 0)
    do_gamma = int(gamma is not None)
    f32, bf16 = ttnn.float32, ttnn.bfloat16
    cbs = [_cb(CB_X, 2 * Wt, x.dtype, grid), _cb(CB_G, 2 * Wt, g.dtype, grid),
           _cb(CB_SCALER, 1, bf16, grid), _cb(CB_EPS, 1, f32, grid),
           _cb(CB_OUT, 2 * Wt, out.dtype, grid)]
    if do_gamma:
        cbs.append(_cb(CB_GAMMA, Wt, bf16, grid))
    cbs += [_cb(i, 1, f32, grid) for i in (CB_MEAN, CB_VAR, CB_RSTD, CB_A, CB_B)]
    cbs += [_cb(i, Wt, f32, grid) for i in (CB_XC, CB_TMP, CB_NORM, CB_DN, CB_T)]

    inv = struct.unpack("<I", struct.pack("<f", 1.0 / K))[0] >> 16
    scaler = (inv << 16) | inv
    eps_bits = struct.unpack("<I", struct.pack("<f", float(eps)))[0]

    reader_ct = [Wt, do_gamma, scaler, eps_bits]
    for t in (x, g, gamma if gamma is not None else x):
        reader_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    writer_ct = [Wt] + list(ttnn.TensorAccessorArgs(out).get_compile_time_args())

    rr, wr, cr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    first = 0
    for group, per in ((cg1, work1), (cg2, work2)):
        for rng in group.ranges():
            for cx in range(rng.start.x, rng.end.x + 1):
                for cy in range(rng.start.y, rng.end.y + 1):
                    rr[cx][cy] = [first, per]
                    wr[cx][cy] = [first, per]
                    cr[cx][cy] = [per]
                    first += per
    assert first == rows, (first, rows)

    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)
    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "reader_lnbw.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=grid,
        compile_time_args=reader_ct, runtime_args=rr, common_runtime_args=[0, 0, 0],
        config=ttnn.ReaderConfigDescriptor())
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "writer_lnbw.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=grid,
        compile_time_args=writer_ct, runtime_args=wr, common_runtime_args=[0],
        config=ttnn.WriterConfigDescriptor())
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "compute_lnbw.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=grid,
        compile_time_args=[0, Wt, do_gamma], runtime_args=cr, config=cfg)
    return {"kernels": [reader, writer, compute], "cbs": cbs}


def layer_norm_bw(x, g, gamma, eps):
    """dx of `layer_norm(x) * gamma (+ beta)` for cotangent g, in the dtype
    `autograd._layer_norm_bw` returns (float32 if g is, else bfloat16). Nothing is deallocated."""
    device = x.device()
    out_dtype = ttnn.float32 if g.dtype == ttnn.float32 else ttnn.bfloat16
    out = ttnn.allocate_tensor_on_device(x.shape, out_dtype, ttnn.TILE_LAYOUT, device,
                                         ttnn.DRAM_MEMORY_CONFIG)
    key = (device.id(), tuple(int(d) for d in x.padded_shape), x.dtype, g.dtype,
           gamma is not None, float(eps), x.memory_config().buffer_type,
           g.memory_config().buffer_type,
           None if gamma is None else gamma.memory_config().buffer_type)
    entry = _CACHE.get(key)
    if entry is None:
        entry = _CACHE[key] = _build(x, g, gamma, out, eps, device)
    reader, writer, compute = entry["kernels"]
    reader.common_runtime_args = [x.buffer_address(), g.buffer_address(),
                                  gamma.buffer_address() if gamma is not None else 0]
    writer.common_runtime_args = [out.buffer_address()]
    pd = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=entry["cbs"])
    REACH["served"] += 1
    ins = [x, g] + ([gamma] if gamma is not None else [])
    return ttnn.generic_op(ins + [out], pd)
