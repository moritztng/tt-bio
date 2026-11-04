"""Layer norm over the last axis of an fp32 TILE tensor with fp32 statistics, as one Tensix program.

`tenstorrent.fp32_layer_norm` needs each row's mean and variance to fp32 precision (a single track's tokens share a
vector ~1e3x their differences) and Wormhole's FPU reductions do not hold it. Composed from ttnn ops that took ~61
programs a norm, two `_fp32_row_sum` folds of slices and fp32 adds; at the perf gate's 20 tokens that dispatch was
~100 ms a Protenix-family fold. Here every step runs on the SFPU in an fp32 DEST with the inputs unpacked straight to
DEST: the column tiles folded by SFPU adds, the 32 columns summed by `sfpu_reduce` (REDUCE_ROW), the statistics
broadcast back over the columns by the reader. Same two-pass formula and order of the final scale as the composed
form, so it matches it to fp32 rounding, not bit for bit.

One core takes a run of row tiles; x is interleaved (DRAM or L1), its last axis whole tiles.
"""

from __future__ import annotations

import math
import struct
from pathlib import Path

import ttnn

from . import mm_generic as MG

KERNEL_DIR = Path(__file__).resolve().parent / "kernels" / "fp32_ln"
TILE = 32
CB_X, CB_G, CB_B, CB_XC, CB_S1, CB_M, CB_S2, CB_R, CB_OUT = 0, 1, 2, 3, 4, 5, 6, 7, 16
_CACHE: dict = {}
_ROWS: dict = {}
#: calls served
STATS = [0]


def _bits(v):
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


def _cb(idx, grid, tiles, dtype):
    fmt = ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dtype, page_size=MG.tile_bytes(dtype))
    return ttnn.CBDescriptor(total_size=tiles * MG.tile_bytes(dtype), core_ranges=grid, format_descriptors=[fmt])


def _rows(v, k, dev):
    """`v` (k values) as a fp32 (32, k) TILE tensor whose every row is `v`, made once per vector."""
    key = id(v)
    hit = _ROWS.get(key)
    if hit is None or hit[0] is not v:
        v32 = ttnn.reshape(ttnn.typecast(v, ttnn.float32) if v.dtype != ttnn.float32 else v, (1, k))
        zeros = ttnn.zeros((TILE, k), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
        hit = _ROWS[key] = (v, ttnn.add(zeros, v32))
    return hit[1]


def _build(x, out, g, b, kt, mt, dtype, eps):
    dev = x.device()
    gs = dev.compute_with_storage_grid_size()
    ncores = min(mt, gs.x * gs.y)
    base, extra = divmod(mt, ncores)
    cores = [ttnn.CoreCoord(c % gs.x, c // gs.x) for c in range(ncores)]
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
    rd, cp, wr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    r0 = 0
    for i, c in enumerate(cores):
        nr = base + (i < extra)
        rd[c.x][c.y] = [r0, nr]
        wr[c.x][c.y] = [r0, nr]
        cp[c.x][c.y] = [nr]
        r0 += nr
    acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())  # noqa: E731
    src = ttnn.KernelDescriptor.SourceType.FILE_PATH
    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "reader.cpp"), source_type=src, core_ranges=grid,
        compile_time_args=[kt] + acc(x) + acc(g) + acc(b), runtime_args=rd, common_runtime_args=[0] * 3,
        config=ttnn.ReaderConfigDescriptor())
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "writer.cpp"), source_type=src, core_ranges=grid,
        compile_time_args=[kt] + acc(out), runtime_args=wr, common_runtime_args=[0],
        config=ttnn.WriterConfigDescriptor())
    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                       fp32_dest_acc_en=True)
    modes = [ttnn.UnpackToDestMode.Default] * 32
    for i in (CB_X, CB_G, CB_B, CB_XC, CB_M, CB_R):
        modes[i] = ttnn.UnpackToDestMode.UnpackToDestFp32
    cfg.unpack_to_dest_mode = modes
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "compute.cpp"), source_type=src, core_ranges=grid,
        compile_time_args=[kt, _bits(1.0 / (kt * TILE)), _bits(eps), int(dtype == ttnn.bfloat16)],
        runtime_args=cp, config=cfg)
    f32 = ttnn.float32
    cbs = [_cb(i, grid, kt, f32) for i in (CB_X, CB_G, CB_B, CB_XC)]
    cbs += [_cb(i, grid, 1, f32) for i in (CB_S1, CB_M, CB_S2, CB_R)]
    cbs.append(_cb(CB_OUT, grid, 2, dtype))
    return {"kernels": [reader, writer, compute], "cbs": cbs}


def layer_norm(x, weight, bias, epsilon, dtype, memory_config=None):
    """LN(x) * weight + bias over the last axis, written at `dtype` (bfloat16 or float32). `x` fp32 TILE
    interleaved with whole tiles on its last axis; `weight` / `bias` that axis's vectors."""
    pad = [int(d) for d in x.padded_shape]
    k = pad[-1]
    if x.dtype != ttnn.float32 or x.layout != ttnn.TILE_LAYOUT or x.is_sharded() or int(x.shape[-1]) != k:
        raise ValueError(f"fp32_ln: needs an interleaved fp32 TILE tensor with whole last-axis tiles, got "
                         f"{x.dtype} {x.layout} {list(x.shape)} / {pad}")
    dev = x.device()
    mc = memory_config or ttnn.DRAM_MEMORY_CONFIG
    kt = k // TILE
    mt = math.prod(pad) // k // TILE
    g, b = _rows(weight, k, dev), _rows(bias, k, dev)
    out = ttnn.allocate_tensor_on_device(x.shape, dtype, ttnn.TILE_LAYOUT, dev, mc)
    key = (dev.id(), mt, kt, dtype, float(epsilon), x.memory_config().buffer_type, mc.buffer_type)
    entry = _CACHE.get(key)
    if entry is None:
        entry = _CACHE[key] = _build(x, out, g, b, kt, mt, dtype, epsilon)
    reader, writer, _ = entry["kernels"]
    reader.common_runtime_args = [x.buffer_address(), g.buffer_address(), b.buffer_address()]
    writer.common_runtime_args = [out.buffer_address()]
    ttnn.generic_op([x, g, b, out], ttnn.ProgramDescriptor(kernels=entry["kernels"], semaphores=[],
                                                           cbs=entry["cbs"]))
    STATS[0] += 1
    return out
