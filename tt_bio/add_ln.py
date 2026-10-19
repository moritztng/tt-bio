"""A pre-LN transformer's residual add and the next block's LayerNorm as one kernel: `(a + b, layer_norm(a + b))`.

SaProt's --fast blocks (and ESM-2's in general) run `h = ttnn.add(x, r)` and then `ttnn.layer_norm(h)` on every
sublayer, 66 pairs a 650M page batch. Each is a short op over [rows, 1280]: at 3808 rows a core holds one or two
row tiles, so the two launches and the second read of h cost more than the arithmetic (WH 650M census: 12.9 ms of
adds and 13.7 of layer_norms out of 156). Here a core reads a and b once, adds them in an fp32 DEST and packs the
bf16 sum both to its output and to the norm's input, then norms it in L1 (kernels/common/ln_prologue.hpp, the
two-pass fp32 statistics of trimul T1 and triatt_ln).

h is `ttnn.add`'s answer element for element (an FPU add into a 32-bit DEST packed to bf16 is exactly what ttnn.add
computes on two bf16 operands, kernels/rne_add/compute_rne_add.cpp), so the residual stream is unchanged. The norm
is not bit-exact with ttnn.layer_norm: different statistics, same arithmetic.

`add_ln` returns None (nothing allocated) for anything but two bf16 TILE interleaved operands of one [.., K] shape
with K a whole number of tiles and [K] gamma/beta, or when the circular buffers do not fit beside what L1 holds.
"""

from __future__ import annotations

import struct
from pathlib import Path

import ttnn

from . import mm_generic as MG

KERNEL_DIR = Path(__file__).resolve().parent / "kernels" / "add_ln"
TILE = 32
_CACHE: dict = {}
#: [served, declined]
STATS = [0, 0]
REJECTS: dict = {}


def _reject(why):
    REJECTS[why] = REJECTS.get(why, 0) + 1
    STATS[1] += 1
    return None


def _check(a, b, w, beta):
    for t in (a, b, w, beta):
        if t.dtype != ttnn.bfloat16 or t.layout != ttnn.TILE_LAYOUT:
            return "dtype"
        if t.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
            return "memcfg"
    if tuple(a.padded_shape) != tuple(b.padded_shape) or tuple(a.shape) != tuple(b.shape):
        return "shape"
    k = int(a.shape[-1])
    if k % TILE or int(a.padded_shape[-1]) != k:
        return "k"
    for v in (w, beta):   # [K] or [1, K]: volume() counts the tile padding, the logical shape does not
        if int(v.shape[-1]) != k or any(int(d) != 1 for d in list(v.shape)[:-1]):
            return "ln_params"
    return None


def _bf16_rne(x: float) -> int:
    u = struct.unpack("<I", struct.pack("<f", x))[0]
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) & 0xFFFF


def _cb(idx, core_grid, tiles, dtype=ttnn.bfloat16):
    fmt = ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dtype, page_size=MG.tile_bytes(dtype))
    return ttnn.CBDescriptor(total_size=tiles * MG.tile_bytes(dtype), core_ranges=core_grid, format_descriptors=[fmt])


def cb_bytes(kt):
    """L1 bytes per core of the circular buffers: a, b double-buffered, h, norm input, y, gamma, beta (bf16), the
    centred row and a scratch row (fp32), the scaler and four fp32 stat tiles."""
    return (2 * kt + 2 * kt + 5 * kt + 1) * MG.tile_bytes(ttnn.bfloat16) + (2 * kt + 4) * MG.tile_bytes(ttnn.float32)


def _build(a, b, w, beta, h, y, grid, eps):
    kt = int(a.shape[-1]) // TILE
    m = a.volume() // int(a.shape[-1]) // TILE
    gx, gy = grid
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))])
    rd, wr, cp = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    ncores, r0 = gx * gy, 0
    for c in range(ncores):
        n = m // ncores + (c < m % ncores)
        cx, cy = c % gx, c // gx
        rd[cx][cy] = [r0, n]
        wr[cx][cy] = [r0, n]
        cp[cx][cy] = [n]
        r0 += n
    inv = _bf16_rne(1.0 / int(a.shape[-1]))
    eps_bits = struct.unpack("<I", struct.pack("<f", float(eps)))[0]
    acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())  # noqa: E731
    src = ttnn.KernelDescriptor.SourceType.FILE_PATH
    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "reader.cpp"), source_type=src, core_ranges=core_grid,
        compile_time_args=[kt, (inv << 16) | inv, eps_bits] + acc(a) + acc(b) + acc(w) + acc(beta),
        runtime_args=rd, common_runtime_args=[0] * 4, config=ttnn.ReaderConfigDescriptor())
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "writer.cpp"), source_type=src, core_ranges=core_grid,
        compile_time_args=[kt] + acc(h) + acc(y),
        runtime_args=wr, common_runtime_args=[0] * 2, config=ttnn.WriterConfigDescriptor())
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "compute.cpp"), source_type=src, core_ranges=core_grid,
        compile_time_args=[kt], runtime_args=cp,
        config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                            fp32_dest_acc_en=True, dst_full_sync_en=False))
    cbs = [_cb(0, core_grid, 2 * kt), _cb(1, core_grid, 2 * kt), _cb(2, core_grid, kt), _cb(3, core_grid, kt),
           _cb(7, core_grid, kt), _cb(8, core_grid, kt), _cb(9, core_grid, kt), _cb(10, core_grid, 1)]
    cbs += [_cb(i, core_grid, t, ttnn.float32) for i, t in ((11, 1), (12, 1), (13, 1), (14, 1), (16, kt), (17, kt))]
    return {"kernels": [reader, writer, compute], "cbs": cbs}


def add_ln(a, b, w, beta, eps=1e-5, h_memory_config=None, y_memory_config=None):
    """`(a + b, layer_norm(a + b) * w + beta)` in one program, or None (nothing allocated)."""
    why = _check(a, b, w, beta)
    if why:
        return _reject(why)
    dev = a.device()
    g = dev.compute_with_storage_grid_size()
    grid = (g.x, g.y)
    kt = int(a.shape[-1]) // TILE
    h = ttnn.allocate_tensor_on_device(a.shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, dev,
                                       h_memory_config or ttnn.DRAM_MEMORY_CONFIG)
    y = ttnn.allocate_tensor_on_device(a.shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, dev,
                                       y_memory_config or ttnn.DRAM_MEMORY_CONFIG)
    # The CBs sit below every live L1 buffer, so the largest free block is what they may take (triatt_ln._plan).
    lcf = ttnn.get_memory_view(dev, ttnn.BufferType.L1).largest_contiguous_bytes_free_per_bank
    lcf = min(lcf) if isinstance(lcf, (list, tuple)) else lcf
    if cb_bytes(kt) > lcf:
        ttnn.deallocate(h)
        ttnn.deallocate(y)
        return _reject(f"l1:kt{kt}")
    key = (dev.id(), tuple(int(d) for d in a.padded_shape), grid, float(eps),
           str(a.memory_config().buffer_type), str(b.memory_config().buffer_type),
           str(h.memory_config().buffer_type), str(y.memory_config().buffer_type))
    entry = _CACHE.get(key)
    if entry is None:
        entry = _CACHE[key] = _build(a, b, w, beta, h, y, grid, eps)
    reader, writer, _ = entry["kernels"]
    reader.common_runtime_args = [a.buffer_address(), b.buffer_address(), w.buffer_address(), beta.buffer_address()]
    writer.common_runtime_args = [h.buffer_address(), y.buffer_address()]
    ttnn.generic_op([a, b, w, beta, h, y], ttnn.ProgramDescriptor(kernels=entry["kernels"], semaphores=[],
                                                                   cbs=entry["cbs"]))
    STATS[0] += 1
    return h, y
