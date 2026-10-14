"""The DiT's adaLN, `shift + layer_norm(x) * scale` with per-token scale and shift, as one kernel.

ttnn has no layer_norm with a per-token affine, so OpenFold3's token DiT runs `ttnn.layer_norm`
and then `ttnn.addcmul`: two launches and the normed activation written and read once between
them, twice per block and step (9600 calls a 512-token fold). On a Wormhole Galaxy chip the pair
costs 27 + 20 us a call (state/spd/of3-whcen-analysis.md), 0.45 s a fold. Here a core norms its
row tiles in L1 (kernels/common/ln_prologue.hpp's statistics, two-pass in fp32 CBs, as
ttnn.layer_norm) and applies scale and shift before the single write. Lever `dit_adaln`.

Not bit-exact with the two calls: the normed tile stays fp32 into the multiply instead of being
packed to the activation dtype first. Declines (None) on anything but interleaved TILE operands of
one shape whose last dim is whole tiles.
"""

from __future__ import annotations

import struct
from pathlib import Path

import ttnn

from . import mm_generic as MG
from . import ops
from .triatt_ln import split_rows

KERNEL_DIR = Path(__file__).resolve().parent / "kernels" / "adaln"
TILE = 32
_CACHE: dict = {}
#: (served, declined)
STATS = [0, 0]
REJECTS: dict = {}


def _check(x, sc, sb):
    if ops.taping():
        return "taping"
    shp = [int(d) for d in x.shape]
    for t in (x, sc, sb):
        if t.layout != ttnn.TILE_LAYOUT or t.dtype not in (ttnn.bfloat16, ttnn.float32):
            return "dtype"
        if t.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
            return "memcfg"
        if [int(d) for d in t.shape] != shp or [int(d) for d in t.padded_shape] != shp:
            return "shape"
    if len(shp) < 2 or shp[-1] % TILE or shp[-2] % TILE or shp[-1] > 32 * TILE:
        return "shape"
    return None


def _cb(idx, core_grid, tiles, dtype):
    fmt = ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dtype, page_size=MG.tile_bytes(dtype))
    return ttnn.CBDescriptor(total_size=tiles * MG.tile_bytes(dtype), core_ranges=core_grid,
                             format_descriptors=[fmt])


def _build(x, sc, sb, out, grid, ckc, eps):
    kt = int(x.shape[-1]) // TILE
    m = 1
    for d in list(x.shape)[:-1]:
        m *= int(d)
    m //= TILE
    gx, gy = grid
    n_cores = min(m, gx * gy)
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(c % gx, c // gx), ttnn.CoreCoord(c % gx, c // gx))
                                   for c in range(n_cores)])
    rd, wr, cp = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for c, (r0, n) in enumerate(split_rows(m, n_cores)):
        cx, cy = c % gx, c // gx
        rd[cx][cy] = [r0, n]
        wr[cx][cy] = [r0, n]
        cp[cx][cy] = [n]
    inv = struct.unpack("<I", struct.pack("<f", 1.0 / int(x.shape[-1])))[0] >> 16
    eps_bits = struct.unpack("<I", struct.pack("<f", float(eps)))[0]
    acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())  # noqa: E731
    src = ttnn.KernelDescriptor.SourceType.FILE_PATH
    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "reader.cpp"), source_type=src, core_ranges=core_grid,
        compile_time_args=[kt, (inv << 16) | inv, eps_bits] + acc(x) + acc(sc) + acc(sb),
        runtime_args=rd, common_runtime_args=[0] * 3, config=ttnn.ReaderConfigDescriptor())
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "writer.cpp"), source_type=src, core_ranges=core_grid,
        compile_time_args=[kt] + acc(out), runtime_args=wr, common_runtime_args=[0],
        config=ttnn.WriterConfigDescriptor())
    fid, approx, _, full = ckc
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "compute.cpp"), source_type=src, core_ranges=core_grid,
        compile_time_args=[kt], runtime_args=cp,
        config=ttnn.ComputeConfigDescriptor(math_fidelity=fid, math_approx_mode=approx,
                                            fp32_dest_acc_en=True, dst_full_sync_en=full))
    f32 = ttnn.float32
    cbs = [_cb(2, core_grid, kt, out.dtype), _cb(3, core_grid, kt, sc.dtype), _cb(4, core_grid, kt, sb.dtype),
           _cb(7, core_grid, kt, x.dtype), _cb(10, core_grid, 1, ttnn.bfloat16)]
    cbs += [_cb(i, core_grid, t, f32) for i, t in ((11, 1), (12, 1), (13, 1), (14, 1), (16, kt), (17, kt))]
    return {"kernels": [reader, writer, compute], "cbs": cbs}


def adaln(x, scale, shift, ckc, grid, eps=1e-5):
    """`shift + layer_norm(x) * scale` (no LN affine) in one kernel, or None (nothing allocated).
    The result takes `shift`'s dtype, as `ttnn.addcmul(shift, ln, scale)` does. `ckc` is a
    (fidelity, approx, fp32_acc, full_sync) tuple (`mm_generic.ckc_args`)."""
    why = _check(x, scale, shift)
    if why:
        k = (why, tuple(int(d) for d in x.padded_shape))
        REJECTS[k] = REJECTS.get(k, 0) + 1
        STATS[1] += 1
        return None
    out = ttnn.allocate_tensor_on_device(x.shape, shift.dtype, ttnn.TILE_LAYOUT, x.device(),
                                         x.memory_config())
    key = (x.device().id(), tuple(int(d) for d in x.padded_shape), str(x.dtype), str(scale.dtype),
           str(shift.dtype), tuple(grid), tuple(str(a) for a in ckc), float(eps))
    entry = _CACHE.get(key)
    if entry is None:
        entry = _CACHE[key] = _build(x, scale, shift, out, grid, ckc, eps)
    reader, writer, _ = entry["kernels"]
    reader.common_runtime_args = [x.buffer_address(), scale.buffer_address(), shift.buffer_address()]
    writer.common_runtime_args = [out.buffer_address()]
    ttnn.generic_op([x, scale, shift, out], ttnn.ProgramDescriptor(kernels=entry["kernels"], semaphores=[],
                                                                   cbs=entry["cbs"]))
    STATS[0] += 1
    return out
