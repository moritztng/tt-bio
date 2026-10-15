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
`transposed=True` norms x^T for the ending variant without writing it: the reader gathers each row
tile of x^T from the 32 x tiles that hold it and shuffles rows in L1 (reader.cpp TR), so the
`_pair_transpose` before the norm, a full read and write of the pair, is gone. Same arithmetic on the
same values in the same order, so bit-exact with transpose-then-norm.
`ln_proj` is the same kernel for a narrow [K, n <= 64] weight with a plain [B, S, n] output
(lever `apb_ln`: the attention-pair-bias projection of the normed pair).
"""

from __future__ import annotations

import os
import struct
from pathlib import Path

import ttnn

from . import dest_guard
from . import mm_generic as MG
from . import ops

KERNEL_DIR = Path(__file__).resolve().parent / "kernels" / "triatt_ln"
TILE = 32
RB = 4          # row tiles per reader barrier
#: the ending variant norms z^T straight out of z (reader.cpp TR) instead of transposing first
TR = os.environ.get("TT_BIO_TRIATT_LN_TR", "1") == "1"
#: diagnostic only (spd-pairchain ptab2): at every ln_qkvgb_heads call, 1 reads the L1 allocator view
#: (what the TR fit test does), 2 synchronizes the device. TR=0 vs TR=1 moved the --fast digest while
#: every call took the same path, and the fit test's allocator read was the only difference.
DRAIN_PROBE = 0
_CACHE: dict = {}
#: (served, declined)
STATS = [0, 0]
REJECTS: dict = {}


def _reject(why, x):
    k = (why, tuple(int(d) for d in x.padded_shape))
    REJECTS[k] = REJECTS.get(k, 0) + 1
    STATS[1] += 1
    return None


def _check(x, w, ln_w, ln_b, ckc):
    """Why the operands cannot take the kernel, or None."""
    if ops.taping():
        return "taping"
    if dest_guard.exposed_args((ckc[0], ckc[1], True, ckc[3]), "tri"):
        return "dest_guard"   # all of K accumulates in fp32 DEST: the Wormhole cancellation fault
    for t in (x, w, ln_w, ln_b):
        if t.dtype != ttnn.bfloat16 or t.layout != ttnn.TILE_LAYOUT:
            return "dtype"
        if t.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
            return "memcfg"
    if x.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
        # The ending variant's transposed pair is L1-resident: its ~0.6 MB a core beside these CBs
        # is unmeasured, so it keeps the two-call path until a device run says it fits.
        return "memcfg"
    shp, pad = [int(d) for d in x.shape], [int(d) for d in x.padded_shape]
    K = shp[-1]
    if len(shp) != 3 or shp != pad or K % TILE or K > 8 * TILE:
        return "act_shape"
    for v in (ln_w, ln_b):
        if int(v.shape[-1]) != K or any(int(d) != 1 for d in list(v.shape)[:-1]):
            return "ln_params"
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


def _tr_buffers(kt, nw):
    """16-row x^T units the TR reader can hold in c_7 (2 = double-buffered, 1, or 0: does not fit)."""
    from .tenstorrent import _l1_bank_bytes
    fixed = kt * nw + 2 * kt + 2 * nw + 2 * kt + 1 + 16 + 2 * (4 + 2 * kt)   # fp32 CBs count twice
    for nb in (2, 1):
        if (fixed + nb * 16 * kt) * MG.tile_bytes(ttnn.bfloat16) <= 0.7 * _l1_bank_bytes():
            return nb
    return 0


def _build(x, w, ln_w, ln_b, outs, grid, ckc, eps, H, MT, tr):
    kt, nw = int(x.shape[-1]) // TILE, int(w.padded_shape[-1]) // TILE
    m = 1
    for d in list(x.padded_shape)[:-1]:
        m *= int(d)
    m //= TILE
    gx, gy = grid
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))])
    rd, wr, cp = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    # TR splits 16-row units (2 per x^T row-tile pair (bt, jt)), the rest splits row tiles
    for c, (r0, n) in enumerate(split_rows(m // 16 if tr else m, gx * gy)):
        cx, cy = c % gx, c // gx
        rows = 16 * n if tr else n
        rd[cx][cy] = [r0, n, int(bool(tr)), MT]
        wr[cx][cy] = [r0, rows, int(bool(tr)), MT]
        cp[cx][cy] = [rows]
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
        compile_time_args=[H, MT, nw] + sum((acc(o) for o in outs), []),
        runtime_args=wr, common_runtime_args=[0] * len(outs), config=ttnn.WriterConfigDescriptor())
    fid, approx, _, full = ckc
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "compute.cpp"), source_type=src, core_ranges=core_grid,
        compile_time_args=[kt, nw, 0], runtime_args=cp,   # RNE off: pack truncates, as T1 ships
        config=ttnn.ComputeConfigDescriptor(math_fidelity=fid, math_approx_mode=approx,
                                            fp32_dest_acc_en=True, dst_full_sync_en=full))
    cbs = [_cb(0, core_grid, kt * nw), _cb(1, core_grid, 2 * kt), _cb(2, core_grid, 2 * nw),
           _cb(7, core_grid, tr * 16 * kt if tr else 2 * RB * kt), _cb(8, core_grid, kt), _cb(9, core_grid, kt),
           _cb(10, core_grid, 1)]
    if tr:
        cbs.append(_cb(3, core_grid, 16))   # the 32 half tiles of one channel tile
    cbs += [_cb(i, core_grid, t, ttnn.float32) for i, t in ((11, 1), (12, 1), (13, 1), (14, 1), (16, kt), (17, kt))]
    return {"kernels": [reader, writer, compute], "cbs": cbs}


def ln_qkvgb_heads(x, ln_w, ln_b, w, ckc, n_heads, head_dim, bias_channels, grid, eps=1e-5, transposed=False):
    """`((q, k, v), gate, bias)` of `layer_norm(x)` (of `x^T` with `transposed`) in one kernel, or None
    (nothing allocated).

    `x` is the raw [B, S, K] pair (square and whole tiles with `transposed`), `w` the [K, 4c + 32] qkvgb weight of `qkvgb_heads`, `ckc` a
    (fidelity, approx, fp32_acc, full_sync) tuple as `trimul_tail` takes it."""
    from .tenstorrent import lever
    if DRAIN_PROBE:
        from .tenstorrent import _l1_bank_bytes
        _l1_bank_bytes() if DRAIN_PROBE == 1 else ttnn.synchronize_device(x.device())
    if not lever("triatt_ln"):
        return None
    why = _check(x, w, ln_w, ln_b, ckc)
    if why:
        return _reject(why, x)
    shp, pad = [int(d) for d in x.shape], [int(d) for d in x.padded_shape]
    K = shp[-1]
    c = n_heads * head_dim
    if head_dim != TILE or list(w.shape) != [K, 4 * c + TILE] or not 0 < bias_channels <= TILE:
        return _reject("weight_shape", x)
    H, MT = c // TILE, pad[-2] // TILE
    tr = 0
    if transposed:
        if not TR or shp[0] != shp[1] or shp[0] % TILE:
            return _reject("tr_shape", x)
        tr = _tr_buffers(K // TILE, int(w.padded_shape[-1]) // TILE)
        if not tr:
            from .tenstorrent import _l1_bank_bytes
            return _reject(f"tr_l1:kt{K // TILE}:nw{int(w.padded_shape[-1]) // TILE}:l1{_l1_bank_bytes()}", x)
    dev = x.device()
    outs = [ttnn.allocate_tensor_on_device(ttnn.Shape([shp[0], n_heads, shp[1], head_dim]), ttnn.bfloat16,
                                           ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG) for _ in range(4)]
    outs.append(ttnn.allocate_tensor_on_device(ttnn.Shape([shp[0], shp[1], bias_channels]), ttnn.bfloat16,
                                               ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG))
    return _launch(x, w, ln_w, ln_b, outs, grid, ckc, eps, H, MT, bias_channels, tr)


def _launch(x, w, ln_w, ln_b, outs, grid, ckc, eps, H, MT, tag, tr=0):
    key = (x.device().id(), tuple(int(d) for d in x.padded_shape), tuple(int(d) for d in w.padded_shape), H, tag,
           tuple(grid), tuple(str(a) for a in ckc), float(eps), tr)
    entry = _CACHE.get(key)
    if entry is None:
        entry = _CACHE[key] = _build(x, w, ln_w, ln_b, outs, grid, ckc, eps, H, MT, tr)
    reader, writer, _ = entry["kernels"]
    reader.common_runtime_args = [x.buffer_address(), w.buffer_address(), ln_w.buffer_address(),
                                  ln_b.buffer_address()]
    writer.common_runtime_args = [o.buffer_address() for o in outs]
    pd = ttnn.ProgramDescriptor(kernels=entry["kernels"], semaphores=[], cbs=entry["cbs"])
    ttnn.generic_op([x, w, ln_w, ln_b] + outs, pd)
    STATS[0] += 1
    if H:
        return tuple(outs[:3]), outs[3], outs[4]
    return outs[0]


def ln_proj(x, ln_w, ln_b, w, ckc, grid, eps=1e-5):
    """`layer_norm(x) @ w` in one kernel for a narrow `w` ([K, n], n <= 2 tiles), or None (nothing
    allocated). The pair-bias projections (lever `apb_ln`) read the whole normed pair to write one
    tile of width; here the normed tile never leaves L1. `x` [B, S, K] bf16 TILE DRAM, out [B, S, n]."""
    from .tenstorrent import lever
    if not lever("apb_ln"):
        return None
    why = _check(x, w, ln_w, ln_b, ckc)
    if why:
        return _reject(why, x)
    shp = [int(d) for d in x.shape]
    n = int(w.shape[-1])
    if int(w.shape[0]) != shp[-1] or w.padded_shape[-1] > 2 * TILE:
        return _reject("weight_shape", x)
    out = ttnn.allocate_tensor_on_device(ttnn.Shape([shp[0], shp[1], n]), ttnn.bfloat16, ttnn.TILE_LAYOUT,
                                         x.device(), ttnn.DRAM_MEMORY_CONFIG)
    return _launch(x, w, ln_w, ln_b, [out], grid, ckc, eps, 0, int(x.padded_shape[-2]) // TILE, "proj")
