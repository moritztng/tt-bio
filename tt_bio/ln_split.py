"""LayerNorm of a few wide rows with each row tile split over G cores.

`ttnn.layer_norm` gives each 32-row tile row to one core. The ESMC-6B's [544, 2560] activations are 17 tile rows,
so 17 of Wormhole's 72 cores do the whole norm, 80 tiles each: 0.092 ms a call against ~0.03 ms of DRAM traffic,
161 calls a forward. Here the G cores of a row each read, norm and write KT / G column tiles, and trade one fp32
partial tile twice over the NOC (the row sum, then the row sum of (x - mean)^2) to get the row statistics. Same
two-pass statistics and 1/K scaler as ttnn.layer_norm, so it matches it to rounding, not bit for bit.

Serves one row tile a core (rows x G <= the grid); declines (None) on anything else, the caller then runs
ttnn.layer_norm.
"""

from __future__ import annotations

import math
import struct
from pathlib import Path

import ttnn

from . import mm_generic as MG

KERNEL_DIR = Path(__file__).resolve().parent / "kernels" / "ln_split"
TILE = 32
_CACHE: dict = {}
#: (served, declined)
STATS = [0, 0]
#: data format of x - mean and the stage scratch (c_16 / c_17)
IM_DTYPE = ttnn.float32


def plan(mt, kt, ncores):
    """Column groups a row tile for `mt` row tiles of `kt` column tiles on `ncores` cores: the largest divisor of
    `kt` with mt * G <= ncores, or None. Pays only for G >= 4 on rows of >= 64 tiles (lns4, .110, device ms
    stock -> split: [544, 2560] G 4 0.110 -> 0.085, [256, 2560] G 8 0.106 -> 0.060, [1024, 2560] G 2 0.116 ->
    0.151, [544, 1152] G 4 0.055 -> 0.050): each tile op on the fp32 intermediates costs ~3x ttnn's."""
    if kt < 64:
        return None
    best = None
    for g in range(4, kt + 1):
        if kt % g == 0 and mt * g <= ncores:
            best = g
    return best


def _cb(idx, core_grid, tiles, dtype):
    fmt = ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dtype, page_size=MG.tile_bytes(dtype))
    return ttnn.CBDescriptor(total_size=tiles * MG.tile_bytes(dtype), core_ranges=core_grid,
                             format_descriptors=[fmt])


def _vec_ok(v, k):
    return v is None or (v.dtype == ttnn.bfloat16 and v.layout == ttnn.TILE_LAYOUT and int(v.shape[-1]) == k
                         and math.prod(int(d) for d in v.shape) == k and not v.is_sharded())


def _build(x, out, w, b, g, eps):
    dev = x.device()
    pad = [int(d) for d in x.padded_shape]
    k, kt = pad[-1], pad[-1] // TILE
    mt = int(x.volume()) // k // TILE
    kc = kt // g
    gx = dev.compute_with_storage_grid_size().x
    ncores = mt * g
    cores = [ttnn.CoreCoord(c % gx, c // gx) for c in range(ncores)]
    ranges = [ttnn.CoreRange(ttnn.CoreCoord(0, y), ttnn.CoreCoord(gx - 1, y)) for y in range(ncores // gx)]
    if ncores % gx:
        y = ncores // gx
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, y), ttnn.CoreCoord(ncores % gx - 1, y)))
    grid = ttnn.CoreRangeSet(ranges)
    noc = [dev.worker_core_from_logical_core(c) for c in cores]
    rd, wr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for c, lc in enumerate(cores):
        r, j = divmod(c, g)
        rd[lc.x][lc.y] = [r, j * kc]
        peers = sum(([noc[r * g + i].x, noc[r * g + i].y] for i in range(g)), [])
        wr[lc.x][lc.y] = [r, j * kc, j] + peers
    inv = struct.unpack("<I", struct.pack("<f", 1.0 / int(x.shape[-1])))[0] >> 16
    eps_bits = struct.unpack("<I", struct.pack("<f", float(eps)))[0]
    acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())  # noqa: E731
    src = ttnn.KernelDescriptor.SourceType.FILE_PATH
    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "reader.cpp"), source_type=src, core_ranges=grid,
        compile_time_args=[kt, kc, (inv << 16) | inv, eps_bits, int(w is not None), int(b is not None)]
        + acc(x) + acc(w if w is not None else x) + acc(b if b is not None else x),
        runtime_args=rd, common_runtime_args=[0] * 3, config=ttnn.ReaderConfigDescriptor())
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "writer.cpp"), source_type=src, core_ranges=grid,
        compile_time_args=[g, kt, kc] + acc(out), runtime_args=wr, common_runtime_args=[0],
        config=ttnn.WriterConfigDescriptor())
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "compute.cpp"), source_type=src, core_ranges=grid,
        compile_time_args=[kc, g, int(w is not None), int(b is not None)],
        config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                            fp32_dest_acc_en=True))
    bf, f32 = ttnn.bfloat16, ttnn.float32
    cbs = [_cb(2, grid, kc, bf), _cb(7, grid, kc, bf), _cb(10, grid, 1, bf)]
    cbs += [_cb(i, grid, kc, bf) for i, v in ((8, w), (9, b)) if v is not None]
    cbs += [_cb(i, grid, t, f32) for i, t in ((11, 1), (12, 1), (13, g), (14, 1), (15, 1), (18, g), (19, 1), (20, 1))]
    cbs += [_cb(i, grid, kc, IM_DTYPE) for i in (16, 17)]
    sems = [ttnn.SemaphoreDescriptor(id=i, core_ranges=grid, initial_value=0) for i in (0, 1)]
    return {"kernels": [reader, writer, compute], "cbs": cbs, "sems": sems}


def layer_norm(x, weight=None, bias=None, epsilon=1e-5, memory_config=None):
    """`ttnn.layer_norm(x, weight=, bias=, epsilon=)` over the last axis, or None (nothing allocated) when the
    split does not apply: bf16 TILE, interleaved DRAM in, interleaved DRAM or L1 out, whole tiles, rows x G within
    the grid."""
    pad = [int(d) for d in x.padded_shape]
    k = pad[-1]
    mc = memory_config or ttnn.DRAM_MEMORY_CONFIG
    if (x.dtype != ttnn.bfloat16 or x.layout != ttnn.TILE_LAYOUT or x.memory_config() != ttnn.DRAM_MEMORY_CONFIG
            or mc not in (ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG)
            or list(x.shape) != pad or k % TILE or not (_vec_ok(weight, k) and _vec_ok(bias, k))):
        STATS[1] += 1
        return None
    dev = x.device()
    gs = dev.compute_with_storage_grid_size()
    mt = int(x.volume()) // k // TILE
    g = plan(mt, k // TILE, gs.x * gs.y)
    if g is None:
        STATS[1] += 1
        return None
    out = ttnn.allocate_tensor_on_device(x.shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, mc)
    key = (dev.id(), tuple(pad), weight is not None, bias is not None, float(epsilon), g, IM_DTYPE,
           mc == ttnn.L1_MEMORY_CONFIG)
    entry = _CACHE.get(key)
    if entry is None:
        entry = _CACHE[key] = _build(x, out, weight, bias, g, epsilon)
    reader, writer, _ = entry["kernels"]
    reader.common_runtime_args = [x.buffer_address(), (weight if weight is not None else x).buffer_address(),
                                  (bias if bias is not None else x).buffer_address()]
    writer.common_runtime_args = [out.buffer_address()]
    ins = [x] + [v for v in (weight, bias) if v is not None]
    ttnn.generic_op(ins + [out], ttnn.ProgramDescriptor(kernels=entry["kernels"], semaphores=entry["sems"],
                                                        cbs=entry["cbs"]))
    STATS[0] += 1
    return out
