"""The pair layer norm as one generic_op: read x once, write the normed pair once, in bf16 or bfp8.

``ttnn.layer_norm`` on the [S, S, c_z] pair runs at 0.77 of its DRAM roof in normal mode and 0.60 in fast
mode on Wormhole (state/spd/CENSUS.md, staging10), and it returns its input's format, so a consumer that
reads bfp8 cannot be handed one without a typecast. This kernel normalises one tile-row (32 pairs x c_z) at
a time in L1 with two-pass statistics in a float32 DEST (the forward half of kernels/lnbw) and packs the
affine result once, in whatever format the output tensor has.

Not wired into a model yet: ``perf/spd_pair/ops.py --groups pln`` measures it against ``ttnn.layer_norm``
and float64 first.
"""
from __future__ import annotations

import collections
import math
import pathlib
import struct

import torch
import ttnn

from tt_bio import core_split

KERNEL_DIR = pathlib.Path(__file__).parent / "kernels" / "pair_ln"
REACH: collections.Counter = collections.Counter()
_CACHE: dict = {}
_AFFINE: dict = {}
_TILE = 32
_BYTES = {ttnn.bfloat16: 2048, ttnn.float32: 4096, ttnn.bfloat8_b: 1088}
CB_X, CB_GAMMA, CB_BETA, CB_SCALER, CB_EPS, CB_MEAN, CB_XC, CB_SQ, CB_VAR, CB_RSTD, CB_OUT = \
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 16


def eligible(x) -> bool:
    """bf16 TILE interleaved x whose channel width is a power-of-two multiple of 32 (1/K exact in
    the bf16 reduce scaler)."""
    if x.dtype != ttnn.bfloat16 or x.layout != ttnn.TILE_LAYOUT:
        REACH["declined: dtype/layout"] += 1
        return False
    K = int(x.shape[-1])
    if K % _TILE or K & (K - 1) or int(x.padded_shape[-2]) % _TILE:
        REACH["declined: width"] += 1
        return False
    if x.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        REACH["declined: sharded"] += 1
        return False
    return True


def affine(gamma, beta, device):
    """gamma and beta as float32 [32, K] tile tensors, every row the same: what the kernel's
    DEST-reuse multiply and add read. Built once per (gamma, beta) pair of device buffers."""
    key = (device.id(), gamma.buffer_address(), beta.buffer_address())
    hit = _AFFINE.get(key)
    if hit is None:
        rows = [ttnn.to_torch(t).float().reshape(-1) for t in (gamma, beta)]
        hit = _AFFINE[key] = tuple(
            ttnn.from_torch(r.reshape(1, -1).expand(_TILE, -1).contiguous(), dtype=ttnn.float32,
                            layout=ttnn.TILE_LAYOUT, device=device)
            for r in rows)
    return hit


def _cb(idx, depth, dtype, grid):
    fmt = ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dtype, page_size=_BYTES[dtype])
    return ttnn.CBDescriptor(total_size=depth * _BYTES[dtype], core_ranges=grid, format_descriptors=[fmt])


def _build(x, g32, b32, out, eps, device, ckc):
    K = int(x.shape[-1])
    Wt = K // _TILE
    rows = math.prod(int(d) for d in x.padded_shape) // (K * _TILE)
    num_cores, grid, cg1, cg2, work1, work2 = core_split.split_work_to_cores(
        device.compute_with_storage_grid_size(), rows, 0)
    f32, bf16 = ttnn.float32, ttnn.bfloat16
    db = 4 if Wt % 4 == 0 else (2 if Wt % 2 == 0 else 1)
    cbs = [_cb(CB_X, 2 * Wt, x.dtype, grid), _cb(CB_GAMMA, Wt, f32, grid), _cb(CB_BETA, Wt, f32, grid),
           _cb(CB_SCALER, 1, bf16, grid), _cb(CB_EPS, 1, f32, grid), _cb(CB_OUT, 2 * Wt, out.dtype, grid),
           _cb(CB_XC, Wt, f32, grid), _cb(CB_SQ, Wt, f32, grid)]
    cbs += [_cb(i, 1, f32, grid) for i in (CB_MEAN, CB_VAR, CB_RSTD)]
    inv = struct.unpack("<I", struct.pack("<f", 1.0 / K))[0] >> 16
    scaler = (inv << 16) | inv
    eps_bits = struct.unpack("<I", struct.pack("<f", float(eps)))[0]
    reader_ct = [Wt, scaler, eps_bits]
    for t in (x, g32, b32):
        reader_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    writer_ct = [Wt] + list(ttnn.TensorAccessorArgs(out).get_compile_time_args())
    rr, cr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    first = 0
    for group, per in ((cg1, work1), (cg2, work2)):
        for rng in group.ranges():
            for cx in range(rng.start.x, rng.end.x + 1):
                for cy in range(rng.start.y, rng.end.y + 1):
                    rr[cx][cy] = [first, per]
                    cr[cx][cy] = [per]
                    first += per
    assert first == rows, (first, rows)
    src = ttnn.KernelDescriptor.SourceType.FILE_PATH
    reader = ttnn.KernelDescriptor(kernel_source=str(KERNEL_DIR / "reader.cpp"), source_type=src, core_ranges=grid,
                                   compile_time_args=reader_ct, runtime_args=rr, common_runtime_args=[0, 0, 0],
                                   config=ttnn.ReaderConfigDescriptor())
    writer = ttnn.KernelDescriptor(kernel_source=str(KERNEL_DIR / "writer.cpp"), source_type=src, core_ranges=grid,
                                   compile_time_args=writer_ct, runtime_args=rr, common_runtime_args=[0],
                                   config=ttnn.WriterConfigDescriptor())
    compute = ttnn.KernelDescriptor(kernel_source=str(KERNEL_DIR / "compute.cpp"), source_type=src, core_ranges=grid,
                                    compile_time_args=[Wt, db], runtime_args=cr,
                                    config=ttnn.ComputeConfigDescriptor(math_fidelity=ckc[0], math_approx_mode=False,
                                                                        fp32_dest_acc_en=True))
    return {"kernels": [reader, writer, compute], "cbs": cbs}


def layer_norm(x, gamma, beta, eps=1e-5, dtype=ttnn.bfloat16, fidelity=ttnn.MathFidelity.HiFi4):
    """``ttnn.layer_norm(x, weight=gamma, bias=beta)`` written as `dtype` to DRAM. Call where ``eligible(x)``."""
    device = x.device()
    g32, b32 = affine(gamma, beta, device)
    out = ttnn.allocate_tensor_on_device(x.shape, dtype, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    key = (device.id(), tuple(int(d) for d in x.padded_shape), dtype, float(eps), str(fidelity),
           x.memory_config().buffer_type)
    entry = _CACHE.get(key)
    if entry is None:
        entry = _CACHE[key] = _build(x, g32, b32, out, eps, device, (fidelity,))
    reader, writer, _ = entry["kernels"]
    reader.common_runtime_args = [x.buffer_address(), g32.buffer_address(), b32.buffer_address()]
    writer.common_runtime_args = [out.buffer_address()]
    pd = ttnn.ProgramDescriptor(kernels=entry["kernels"], semaphores=[], cbs=entry["cbs"])
    REACH["served"] += 1
    return ttnn.generic_op([x, g32, b32, out], pd)
