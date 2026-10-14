"""Triangle attention's input LayerNorm and its fused q/k/v/gate/bias projection as one kernel.

The pairformer's triangle attention norms the raw pair tensor with `ttnn.layer_norm` and then
reads the normed copy once in `triatt_qkv.qkvgb_heads`. At 512 tokens the normed copy is 67 MB
written and read again per call, 384 calls a fold: 0.153 s of p150a device time in the layer_norm
alone (cen68), at ~335 GB/s, so it is bytes, not math. With c_z = 128 a row tile carries the whole
channel axis in four tiles, so a core can norm a row tile in L1 and project it right there: the
LayerNorm prologue of trimul T1 (kernels/common/ln_prologue.hpp) followed by the qkvgb matmul
against the resident [K, 4c + 32] weight, written straight to the head-major q, k, v, gate and the
[B, S, 32] pair bias that `qkvgb_heads` produces.

Not bit-exact with the two calls it replaces: the statistics are ln_prologue's (two-pass, fp32
CBs) and the normed tile is packed to bf16 before the matmul, as T1 does. Lever `triatt_ln`.
Declines (None) on anything but bf16 TILE DRAM operands with 32-channel heads and K <= 8 tiles.
"""

from __future__ import annotations

import struct
from pathlib import Path

import ttnn

from . import mm_generic as MG
from . import ops

KERNEL_DIR = Path(__file__).resolve().parent / "kernels" / "triatt_ln"
TILE = 32
RB = 4          # row tiles per reader barrier
_CACHE: dict = {}
#: (served, declined)
STATS = [0, 0]
REJECTS: dict = {}


def _reject(why, x):
    k = (why, tuple(int(d) for d in x.padded_shape))
    REJECTS[k] = REJECTS.get(k, 0) + 1
    STATS[1] += 1
    return None


def split_rows(m, ncores):
    """Contiguous row-tile ranges [(first, count)] per core, the first m % ncores one longer."""
    out, r0 = [], 0
    for c in range(ncores):
        n = m // ncores + (c < m % ncores)
        out.append((r0, n))
        r0 += n
    return out


def out_page(r, h, H, MT):
    """writer.cpp: head-major page of row tile r, head tile h (kernels/triatt HEAD_MAJOR_MT)."""
    return ((r // MT) * H + h) * MT + r % MT


def _cb(idx, core_grid, tiles, dtype=ttnn.bfloat16):
    fmt = ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dtype, page_size=MG.tile_bytes(dtype))
    return ttnn.CBDescriptor(total_size=tiles * MG.tile_bytes(dtype), core_ranges=core_grid,
                             format_descriptors=[fmt])


def _build(x, w, ln_w, ln_b, outs, grid, ckc, eps, H, MT):
    kt, nw = int(x.shape[-1]) // TILE, int(w.shape[-1]) // TILE
    m = 1
    for d in x.padded_shape[:-1]:
        m *= int(d)
    m //= TILE
    gx, gy = grid
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))])
    rd, wr, cp = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for c, (r0, n) in enumerate(split_rows(m, gx * gy)):
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
        compile_time_args=[kt, nw, RB, (inv << 16) | inv, eps_bits] + acc(x) + acc(w) + acc(ln_w) + acc(ln_b),
        runtime_args=rd, common_runtime_args=[0] * 4, config=ttnn.ReaderConfigDescriptor())
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "writer.cpp"), source_type=src, core_ranges=core_grid,
        compile_time_args=[H, MT] + sum((acc(o) for o in outs), []),
        runtime_args=wr, common_runtime_args=[0] * 5, config=ttnn.WriterConfigDescriptor())
    fid, approx, _, full = ckc
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "compute.cpp"), source_type=src, core_ranges=core_grid,
        compile_time_args=[kt, nw, 0], runtime_args=cp,   # RNE off: pack truncates, as T1 ships
        config=ttnn.ComputeConfigDescriptor(math_fidelity=fid, math_approx_mode=approx,
                                            fp32_dest_acc_en=True, dst_full_sync_en=full))
    cbs = [_cb(0, core_grid, kt * nw), _cb(1, core_grid, 2 * kt), _cb(2, core_grid, 2 * nw),
           _cb(7, core_grid, 2 * RB * kt), _cb(8, core_grid, kt), _cb(9, core_grid, kt),
           _cb(10, core_grid, 1)]
    cbs += [_cb(i, core_grid, t, ttnn.float32) for i, t in ((11, 1), (12, 1), (13, 1), (14, 1), (16, kt), (17, kt))]
    return {"kernels": [reader, writer, compute], "cbs": cbs}


def ln_qkvgb_heads(x, ln_w, ln_b, w, ckc, n_heads, head_dim, bias_channels, grid, eps=1e-5):
    """`((q, k, v), gate, bias)` of `layer_norm(x)` in one kernel, or None (nothing allocated).

    `x` is the raw [B, S, K] pair, `w` the [K, 4c + 32] qkvgb weight of `qkvgb_heads`, `ckc` a
    (fidelity, approx, fp32_acc, full_sync) tuple as `trimul_tail` takes it."""
    from .tenstorrent import lever
    if not lever("triatt_ln"):
        return None
    if ops.taping():
        return _reject("taping", x)
    for t in (x, w, ln_w, ln_b):
        if t.dtype != ttnn.bfloat16 or t.layout != ttnn.TILE_LAYOUT:
            return _reject("dtype", x)
        if t.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
            return _reject("memcfg", x)
    if x.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
        # The ending variant's transposed pair is L1-resident: its ~0.6 MB a core beside these CBs
        # is unmeasured, so it keeps the two-call path until a device run says it fits.
        return _reject("memcfg", x)
    shp, pad = [int(d) for d in x.shape], [int(d) for d in x.padded_shape]
    K = shp[-1]
    c = n_heads * head_dim
    if len(shp) != 3 or shp != pad or K % TILE or K > 8 * TILE:
        return _reject("act_shape", x)
    if head_dim != TILE or list(w.shape) != [K, 4 * c + TILE] or not 0 < bias_channels <= TILE:
        return _reject("weight_shape", x)
    for v in (ln_w, ln_b):
        if int(v.shape[-1]) != K or any(int(d) != 1 for d in list(v.shape)[:-1]):
            return _reject("ln_params", x)
    H, MT = c // TILE, pad[-2] // TILE
    dev = x.device()
    outs = [ttnn.allocate_tensor_on_device(ttnn.Shape([shp[0], n_heads, shp[1], head_dim]), ttnn.bfloat16,
                                           ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG) for _ in range(4)]
    outs.append(ttnn.allocate_tensor_on_device(ttnn.Shape([shp[0], shp[1], bias_channels]), ttnn.bfloat16,
                                               ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG))
    key = (dev.id(), tuple(pad), tuple(int(d) for d in w.shape), bias_channels, tuple(grid),
           tuple(str(a) for a in ckc), float(eps))
    entry = _CACHE.get(key)
    if entry is None:
        entry = _CACHE[key] = _build(x, w, ln_w, ln_b, outs, grid, ckc, eps, H, MT)
    reader, writer, _ = entry["kernels"]
    reader.common_runtime_args = [x.buffer_address(), w.buffer_address(), ln_w.buffer_address(),
                                  ln_b.buffer_address()]
    writer.common_runtime_args = [o.buffer_address() for o in outs]
    pd = ttnn.ProgramDescriptor(kernels=entry["kernels"], semaphores=[], cbs=entry["cbs"])
    ttnn.generic_op([x, w, ln_w, ln_b] + outs, pd)
    STATS[0] += 1
    return tuple(outs[:3]), outs[3], outs[4]
