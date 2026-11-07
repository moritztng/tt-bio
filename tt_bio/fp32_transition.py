"""Wormhole's fp32 single-track Transition (`tenstorrent._fp32_single_transition`) as one Tensix program.

The composed form is ~21 ttnn programs a call (fp32 LN, two centrings, three centred matmuls of two matmuls and an add
each, silu, multiply); at OpenDDE's 512 tokens it takes 0.97 ms against 0.17 for the bf16 Transition, nearly all of it
per-program cost, ~760 calls a fold. Here G cores share each 32-token row tile, core j taking hidden tiles
j*NH/G .. of fc1 / fc2 and the same K run of fc3, and does every step of its share: the layer norm on the SFPU as
`fp32_ln` does, the matmuls in an fp32 dest, silu and the gate multiply on the SFPU in fp32. Each core then sends
the fp32 fc3 partial of output tile m to the core that owns m, which sums the G partials in a fixed order and
rounds once (one core per row tile, G = 1, keeps all K and owns every output tile). Every core also norms token 0's
row, which the reader copies over a whole tile, so the centring row, its matmul and token 0's hidden come out
broadcast over the rows without any core waiting on another (kernels/fp32_transition/compute.cpp). Same algebra and
bf16 rounding points as the composed form; only the fp32 summation order differs.
"""

from __future__ import annotations

import math
import struct
from pathlib import Path

import ttnn

from . import mm_generic as MG
from .fp32_ln import _rows

KERNEL_DIR = Path(__file__).resolve().parent / "kernels" / "fp32_transition"
TILE = 32
_CACHE: dict = {}
#: calls served
STATS = [0]


def _bits(v):
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


def _cb(idx, grid, tiles, dtype):
    fmt = ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dtype, page_size=MG.tile_bytes(dtype))
    return ttnn.CBDescriptor(total_size=tiles * MG.tile_bytes(dtype), core_ranges=grid, format_descriptors=[fmt])


def plan(mt, kt, nh, ncores):
    """Cores per row tile: the largest G dividing both KT and NH with mt * G cores on the grid, else 1 (one core
    then takes several row tiles)."""
    return max(g for g in range(1, math.gcd(kt, nh) + 1) if kt % g == 0 and nh % g == 0 and mt * g <= max(ncores, mt))


def _build(x, out, g, b, w1, w2, w3, kt, nh, mt, dtype, eps, ckc):
    dev = x.device()
    gs = dev.compute_with_storage_grid_size()
    grp = plan(mt, kt, nh, gs.x * gs.y)
    nhg = nh // grp
    kc3 = max(d for d in range(1, 9) if nhg % d == 0)      # fc3 K tiles a w3 chunk (double-buffered)
    ncores = mt * grp if grp > 1 else min(mt, gs.x * gs.y)
    base, extra = divmod(mt, ncores) if grp == 1 else (1, 0)
    cores = [ttnn.CoreCoord(c % gs.x, c // gs.x) for c in range(ncores)]
    noc = [dev.worker_core_from_logical_core(c) for c in cores]
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
    rd, cp, wr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    r0 = 0
    for i, c in enumerate(cores):
        if grp > 1:
            r, j = divmod(i, grp)
            rd[c.x][c.y] = [r, 1, j]
            wr[c.x][c.y] = [r, 1, j] + sum(([noc[r * grp + o].x, noc[r * grp + o].y] for o in range(grp)), [])
            cp[c.x][c.y] = [1]
        else:
            nr = base + (i < extra)
            rd[c.x][c.y] = [r0, nr, 0]
            wr[c.x][c.y] = [r0, nr, 0, noc[i].x, noc[i].y]
            cp[c.x][c.y] = [nr]
            r0 += nr
    acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())  # noqa: E731
    src = ttnn.KernelDescriptor.SourceType.FILE_PATH
    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "reader.cpp"), source_type=src, core_ranges=grid,
        compile_time_args=[kt, nh, nhg, kc3] + acc(x) + acc(g) + acc(b) + acc(w1) + acc(w2) + acc(w3),
        runtime_args=rd, common_runtime_args=[0] * 6, config=ttnn.ReaderConfigDescriptor())
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "writer.cpp"), source_type=src, core_ranges=grid,
        compile_time_args=[kt, grp] + acc(out), runtime_args=wr, common_runtime_args=[0],
        config=ttnn.WriterConfigDescriptor())
    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ckc.math_fidelity, math_approx_mode=False, fp32_dest_acc_en=True)
    modes = [ttnn.UnpackToDestMode.Default] * 64  # indexed by CB id; Blackhole has 64
    for i in (0, 1, 2, 3, 5, 7, 8, 17, 19):      # x, gamma, beta, x - mean, mean, rstd, token 0's x, c0, fc3 partials
        modes[i] = ttnn.UnpackToDestMode.UnpackToDestFp32
    cfg.unpack_to_dest_mode = modes
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "compute.cpp"), source_type=src, core_ranges=grid,
        compile_time_args=[kt, nhg, _bits(1.0 / (kt * TILE)), _bits(eps), int(dtype == ttnn.bfloat16), kc3, grp],
        runtime_args=cp, config=cfg)
    f32, bf = ttnn.float32, ttnn.bfloat16
    cbs = [_cb(i, grid, kt, f32) for i in (0, 1, 2, 3, 8, 17, 19)]
    cbs += [_cb(i, grid, 1, f32) for i in (4, 5, 6, 7)]
    cbs += [_cb(9, grid, kt, bf), _cb(10, grid, kt, bf), _cb(18, grid, 2, f32)]
    cbs += [_cb(11, grid, 2 * kt, w1.dtype), _cb(12, grid, 2 * kt, w2.dtype), _cb(13, grid, 2 * kc3, w3.dtype)]
    cbs += [_cb(14, grid, nhg, bf), _cb(15, grid, nhg, bf), _cb(16, grid, 2, dtype)]
    sems = [ttnn.SemaphoreDescriptor(id=0, core_ranges=grid, initial_value=0)]
    return {"kernels": [reader, writer, compute], "cbs": cbs, "sems": sems}


def transition(x, norm_weight, norm_bias, w1, w2, w3, ckc, dtype, memory_config=None, eps=1e-5):
    """`fc3(silu(fc1(LN(x))) * fc2(LN(x)))` with fp32 LN, fp32 intermediates and token-0-centred matmul inputs, at
    `dtype`. `x` [.., N, K] TILE interleaved, whole tiles on K; w1 / w2 [K, H], w3 [H, K]."""
    pad = [int(d) for d in x.padded_shape]
    k = pad[-1]
    if x.layout != ttnn.TILE_LAYOUT or x.is_sharded() or int(x.shape[-1]) != k:
        raise ValueError(f"fp32_transition: needs an interleaved TILE tensor with whole last-axis tiles, got "
                         f"{x.dtype} {x.layout} {list(x.shape)} / {pad}")
    dev = x.device()
    mc = memory_config or ttnn.DRAM_MEMORY_CONFIG
    x32 = ttnn.typecast(x, ttnn.float32) if x.dtype != ttnn.float32 else x
    kt, nh = k // TILE, int(w1.padded_shape[-1]) // TILE
    mt = math.prod(pad) // k // TILE
    g, b = _rows(norm_weight, k, dev), _rows(norm_bias, k, dev)
    out = ttnn.allocate_tensor_on_device(x.shape, dtype, ttnn.TILE_LAYOUT, dev, mc)
    key = (dev.id(), mt, kt, nh, dtype, float(eps), ckc.math_fidelity, w1.dtype, w3.dtype,
           x32.memory_config().buffer_type, mc.buffer_type)
    entry = _CACHE.get(key)
    if entry is None:
        entry = _CACHE[key] = _build(x32, out, g, b, w1, w2, w3, kt, nh, mt, dtype, eps, ckc)
    reader, writer, _ = entry["kernels"]
    reader.common_runtime_args = [x32.buffer_address(), g.buffer_address(), b.buffer_address(),
                                  w1.buffer_address(), w2.buffer_address(), w3.buffer_address()]
    writer.common_runtime_args = [out.buffer_address()]
    ttnn.generic_op([x32, g, b, w1, w2, w3, out],
                    ttnn.ProgramDescriptor(kernels=entry["kernels"], semaphores=entry["sems"], cbs=entry["cbs"]))
    if x32 is not x:
        ttnn.deallocate(x32)
    STATS[0] += 1
    return out
