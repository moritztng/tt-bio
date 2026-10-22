"""A pair layer norm as one program: `layer_norm(x) * gamma + beta` over the last axis of a bf16 tile
tensor in DRAM, every byte read once and written once.

`ttnn.layer_norm` on the trunk's `[1, N, N, C]` pair runs at ~1.7x the bytes it moves on Wormhole at
C = 256 and ~2.5x at C = 128. Each core here takes whole row blocks (32 rows x the C/32 channel tiles,
contiguous pages), keeps the centred rows and their squares in float32 L1 and applies the scale,
gamma and beta on DEST, so a tile is unpacked four times and packed three (kernels/pair_ln).

Diagnostic `abl` 1 copies x to y through the same reader, compute and writer: the program's movement
floor.
"""
from __future__ import annotations

import math
import struct

import torch
import ttnn

from tt_bio import core_split

_TILE = 32
_CACHE: dict = {}
_ROWS: dict = {}       # (id of weight, id of bias, device id) -> (gamma, beta) as full tiles
STATS = [0, 0]         # served, declined


def _kernel_dir():
    import pathlib
    return pathlib.Path(__file__).parent / "kernels" / "pair_ln"


def eligible(x) -> bool:
    """bf16, tile layout, interleaved DRAM, C a multiple of 128 up to 512, whole row blocks."""
    shp = [int(d) for d in x.padded_shape]
    ok = (x.dtype == ttnn.bfloat16 and x.layout == ttnn.TILE_LAYOUT
          and x.memory_config() == ttnn.DRAM_MEMORY_CONFIG
          and shp[-1] == int(x.shape[-1]) and shp[-1] % (4 * _TILE) == 0 and shp[-1] <= 512
          and shp[-2] % _TILE == 0)
    STATS[not ok] += 1
    return ok


def weight_tiles(weight, bias, device):
    """gamma and beta as `[32, C]` bf16 tiles with every row the weights. Cached per weight tensor."""
    key = (id(weight), id(bias), device.id())
    hit = _ROWS.get(key)
    if hit is None:
        rows = []
        for w in (weight, bias):
            v = (ttnn.to_torch(w) if isinstance(w, ttnn.Tensor) else w).reshape(-1).to(torch.bfloat16)
            rows.append(ttnn.from_torch(v.expand(_TILE, -1).contiguous(), dtype=ttnn.bfloat16,
                                        layout=ttnn.TILE_LAYOUT, device=device,
                                        memory_config=ttnn.DRAM_MEMORY_CONFIG))
        hit = _ROWS[key] = (rows, weight, bias)    # the refs keep the ids from being reused
    return hit[0]


def _cb(idx, depth, dtype, grid):
    page = {ttnn.bfloat16: 2048, ttnn.float32: 4096}[dtype]
    fmt = ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dtype, page_size=page)
    return ttnn.CBDescriptor(total_size=depth * page, core_ranges=grid, format_descriptors=[fmt])


def _build(x, out, gamma, beta, eps, fidelity, abl):
    device = x.device()
    C = int(x.padded_shape[-1])
    Ct = C // _TILE
    units = math.prod(int(d) for d in x.padded_shape) // (C * _TILE)
    _, grid, cg1, cg2, work1, work2 = core_split.split_work_to_cores(
        device.compute_with_storage_grid_size(), units, 0)
    f32, bf16 = ttnn.float32, ttnn.bfloat16
    cbs = [_cb(0, 2 * Ct, bf16, grid), _cb(2, Ct, bf16, grid), _cb(3, Ct, bf16, grid),
           _cb(4, 1, bf16, grid), _cb(16, 2 * Ct, bf16, grid)]
    cbs += [_cb(i, 1, f32, grid) for i in (6, 8)] + [_cb(i, Ct, f32, grid) for i in (9, 10)]

    inv = struct.unpack("<I", struct.pack("<f", 1.0 / C))[0] >> 16
    acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
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
    assert first == units, (first, units)

    src, kd = ttnn.KernelDescriptor.SourceType.FILE_PATH, _kernel_dir()
    reader = ttnn.KernelDescriptor(
        kernel_source=str(kd / "reader.cpp"), source_type=src, core_ranges=grid,
        compile_time_args=[Ct, (inv << 16) | inv] + acc(x) + acc(gamma) + acc(beta), runtime_args=rr,
        common_runtime_args=[0, 0, 0], config=ttnn.ReaderConfigDescriptor())
    writer = ttnn.KernelDescriptor(
        kernel_source=str(kd / "writer.cpp"), source_type=src, core_ranges=grid,
        compile_time_args=[Ct] + acc(out), runtime_args=wr, common_runtime_args=[0],
        config=ttnn.WriterConfigDescriptor())
    compute = ttnn.KernelDescriptor(
        kernel_source=str(kd / "compute.cpp"), source_type=src, core_ranges=grid,
        compile_time_args=[Ct, struct.unpack("<I", struct.pack("<f", float(eps)))[0], abl],
        runtime_args=cr,
        config=ttnn.ComputeConfigDescriptor(math_fidelity=fidelity, fp32_dest_acc_en=True,
                                            math_approx_mode=False))
    return {"kernels": [reader, writer, compute], "cbs": cbs}


def pair_ln(x, weight, bias, eps=1e-5, fidelity=ttnn.MathFidelity.HiFi4, abl=0):
    """`ttnn.layer_norm(x, weight=weight, bias=bias, epsilon=eps)` for an `eligible` x, in a new DRAM
    tensor. `weight` and `bias` are the norm's [C] parameters (device or host)."""
    device = x.device()
    gamma, beta = weight_tiles(weight, bias, device)
    out = ttnn.allocate_tensor_on_device(x.shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, device,
                                         ttnn.DRAM_MEMORY_CONFIG)
    key = (device.id(), str(x.padded_shape), float(eps), str(fidelity), abl)
    entry = _CACHE.get(key)
    if entry is None:
        entry = _CACHE[key] = _build(x, out, gamma, beta, eps, fidelity, abl)
    reader, writer, compute = entry["kernels"]
    reader.common_runtime_args = [x.buffer_address(), gamma.buffer_address(), beta.buffer_address()]
    writer.common_runtime_args = [out.buffer_address()]
    pd = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=entry["cbs"])
    ttnn.generic_op([x, gamma, beta, out], pd)
    return out
