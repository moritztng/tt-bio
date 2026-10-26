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
Lever `triatt_ln_b8` writes q, k and v bfp8 (gate and bias stay bf16, the gated out-projection reads a bf16
gate; the caller narrows the head-major bias after its permute and triatt_sdpa keeps a bf16 destination, so
the fused SDPA and out-projection both still serve): the kernel is bound by its DRAM writes, which cap near 120 GB/s on Wormhole, and this cuts them by 35 %.
Each tile is one core's full-K dot product packed once, so the bytes do not depend on the core grid.
`ln_proj` is the same kernel for a narrow [K, n <= 64] weight with a plain [B, S, n] output
(lever `apb_ln`: the attention-pair-bias projection of the normed pair).
Where the whole weight does not fit a core's L1 (OpenDDE: K 384, [12 x 49] tiles, 1.2 MB), the cores
split into G column groups: each holds 1/G of the weight's columns and norms the same rows, so x is read
G times instead of being normed into DRAM and read back. Each output tile is still one core's full-K
dot product, so the split changes no value. With triatt_ln_b8 two groups split at 2H instead (`b8_cols`),
so each group's bfp8 tiles form one run that ends on a DST block.
"""

from __future__ import annotations

import math
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
TR = True
#: bench-only stage skips (kernels/triatt_ln/compute.cpp ABL; perf/spd_m_protenix/tl_attr.py), 0 in production
ABL = 0
#: rotate each core's plain-read rows by its row-split index, so the cores of a group do not all start on the same
#: row residue and hit the same few DRAM banks in lockstep (head-major pages of one row tile are MT apart). Order
#: only: every row is read, computed and written exactly as before. OpenDDE WH normal fold, same chip, CIFs identical,
#: 1000 MHz: 512 -0.75 s, 640 -1.36 s (stgf); per call it moves only the HiFi2 bf16 write (512 9.67 -> 8.77 ms).
STAGGER = True
#: issue every second output tile on the reader's NOC (writer.cpp DUAL_NOC; both DM kernels on DM_DYNAMIC_NOC).
#: Same pages and bytes, so bit-exact. Bench flag while the WH write ceiling (~127 GB/s on one NOC) is measured.
DUAL_NOC = False
_CACHE: dict = {}
#: (served, declined, served with bfp8 q/k/v)
STATS = [0, 0, 0]
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
    if len(shp) != 3 or shp != pad or K % TILE or K > 16 * TILE:
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


def _cb_bytes(kt, nw, tr):
    """L1 bytes per core of `_build`'s circular buffers."""
    bf = kt * nw + 2 * kt + 2 * nw + (tr * 16 * kt + 16 if tr else 2 * RB * kt) + 2 * kt + 1
    return bf * MG.tile_bytes(ttnn.bfloat16) + (4 + 2 * kt) * MG.tile_bytes(ttnn.float32)


def _lcf(dev):
    lcf = ttnn.get_memory_view(dev, ttnn.BufferType.L1).largest_contiguous_bytes_free_per_bank
    return min(lcf) if isinstance(lcf, (list, tuple)) else lcf


def _plan(kt, nw, tr, dev, ncores):
    """(G, columns a group, TR units in c_7) for the fewest column groups whose CBs fit, or None.

    TR units are 16-row x^T units double-buffered (2) or not (1), 0 without TR. Against the L1
    actually free now: the CBs sit below every live L1 buffer, so the largest free block is what
    they may take. At 8 heads x 32 (w 33 tiles wide, Protenix-v2) and K 256 one unit needs 1.12 MB
    of a 1.40 MB Wormhole bank, which a fixed fraction of the bank refused."""
    lcf = _lcf(dev)
    for g in (1, 2, 4):
        if ncores % g:
            continue
        nwg = -(-nw // g)
        for nb in ((2, 1) if tr else (0,)):
            if _cb_bytes(kt, nwg, nb) <= lcf:
                return g, nwg, nb
    return None


def _dm(risc, noc):
    """The stock Reader/WriterConfigDescriptor (RISC-V 1 reads on NOC 0, RISC-V 0 writes on NOC 1), dynamic under DUAL_NOC."""
    if not DUAL_NOC:
        return ttnn.ReaderConfigDescriptor() if risc == ttnn.DataMovementProcessor.RISCV_1 else ttnn.WriterConfigDescriptor()
    return ttnn.DataMovementConfigDescriptor(processor=risc, noc=noc, noc_mode=ttnn.NOC_MODE.DM_DYNAMIC_NOC)


def _nq(nq):
    return [("NQ", "1")] if nq else []


def b8_cols(g, nw, H, gate=False):
    """Per column group (first tile, bfp8 tiles, end tile) for bfp8 q, k, v (and the gate), or None.

    One group: q|k|v(|g) are its first tiles. Two groups split at 2H instead of nw, so group 0 is q|k (and
    one v tile it computes bf16 but leaves to group 1) and group 1 is v(|g), then the bf16 rest. Each group's
    bfp8 run starts its row and ends on a DST block."""
    nb = (4 if gate else 3) * H
    if g == 1:
        return [(0, nb, 4 * H + 1)] if nb % 4 == 0 else None
    if g == 2 and H % 4 == 0 and nw == 2 * H + 1:
        return [(0, 2 * H, 2 * H), (2 * H, nb - 2 * H, 4 * H + 1)]
    return None


def _cb_tiles(cols, nw):
    """Tiles of c_2 (bfp8) and c_4 (bf16 rest) under `cols`: two rows of every group's chunk, since a ring
    whose size is not a multiple of a core's reservation would run that reservation past its end."""
    lcm = lambda v: math.lcm(*v) if v else 0   # noqa: E731
    return 2 * lcm([q for _, q, _ in cols]), 2 * lcm([nw - q for _, q, _ in cols if nw > q])


def _build(x, w, ln_w, ln_b, outs, grid, ckc, eps, H, MT, tr, g, nw, cols=None):
    kt, nwf = int(x.shape[-1]) // TILE, int(w.padded_shape[-1]) // TILE
    m = 1
    for d in list(x.padded_shape)[:-1]:
        m *= int(d)
    m //= TILE
    gx, gy = grid
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))])
    rd, wr, cp = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    # TR splits 16-row units (2 per x^T row-tile pair (bt, jt)), the rest splits row tiles. Core c is in
    # column group c % g, and every group splits the same rows.
    split = split_rows(m // 16 if tr else m, gx * gy // g)
    nq = max(q for _, q, _ in cols) if cols else 0
    c2, c4 = _cb_tiles(cols, nw) if cols else (2 * nw, 0)
    for c in range(gx * gy):
        r0, n = split[c // g]
        cx, cy = c % gx, c // gx
        rows = 16 * n if tr else n
        n0, q, end = cols[c % g] if cols else ((c % g) * nw, 0, nwf)
        s = (c // g) % n if STAGGER and not tr and n else 0
        rd[cx][cy] = [r0, n, int(bool(tr)), MT, n0, s]
        wr[cx][cy] = [r0, rows, int(bool(tr)), MT, n0, q, end, s]
        cp[cx][cy] = [rows, q]
    inv = struct.unpack("<I", struct.pack("<f", 1.0 / int(x.shape[-1])))[0] >> 16
    eps_bits = struct.unpack("<I", struct.pack("<f", float(eps)))[0]
    acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())  # noqa: E731
    src = ttnn.KernelDescriptor.SourceType.FILE_PATH
    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "reader.cpp"), source_type=src, core_ranges=core_grid,
        compile_time_args=[kt, nw, RB, (inv << 16) | inv, eps_bits, nwf] + acc(x) + acc(w) + acc(ln_w) + acc(ln_b),
        defines=[("ABL_R", "1")] if ABL & 8 else [],
        runtime_args=rd, common_runtime_args=[0] * 4, config=_dm(ttnn.DataMovementProcessor.RISCV_1, MG.NOC_FOR_DRAM_READ))
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "writer.cpp"), source_type=src, core_ranges=core_grid,
        compile_time_args=[H, MT, nw, nwf] + sum((acc(o) for o in outs), []),
        defines=([("ABL_W", "1")] if ABL & 4 else []) + ([("DUAL_NOC", "1")] if DUAL_NOC else []) + _nq(nq),
        runtime_args=wr, common_runtime_args=[0] * len(outs),
        config=_dm(ttnn.DataMovementProcessor.RISCV_0, MG.NOC_FOR_DRAM_WRITE))
    fid, approx, _, full = ckc
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "compute.cpp"), source_type=src, core_ranges=core_grid,
        compile_time_args=[kt, nw, 0, ABL], runtime_args=cp, defines=_nq(nq),   # RNE off: pack truncates, as T1 ships
        config=ttnn.ComputeConfigDescriptor(math_fidelity=fid, math_approx_mode=approx,
                                            fp32_dest_acc_en=True, dst_full_sync_en=full))
    cbs = [_cb(0, core_grid, kt * nw), _cb(1, core_grid, 2 * kt), _cb(2, core_grid, c2, outs[0].dtype),
           _cb(7, core_grid, tr * 16 * kt if tr else 2 * RB * kt), _cb(8, core_grid, kt), _cb(9, core_grid, kt),
           _cb(10, core_grid, 1)]
    if tr:
        cbs.append(_cb(3, core_grid, 16))   # the 32 half tiles of one channel tile
    if c4:
        cbs.append(_cb(4, core_grid, c4))   # the bf16 rest: bias, and the gate unless triatt_ln_b8g
    cbs += [_cb(i, core_grid, t, ttnn.float32) for i, t in ((11, 1), (12, 1), (13, 1), (14, 1), (16, kt), (17, kt))]
    return {"kernels": [reader, writer, compute], "cbs": cbs}


def ln_qkvgb_heads(x, ln_w, ln_b, w, ckc, n_heads, head_dim, bias_channels, grid, eps=1e-5, transposed=False):
    """`((q, k, v), gate, bias)` of `layer_norm(x)` (of `x^T` with `transposed`) in one kernel, or None
    (nothing allocated).

    `x` is the raw [B, S, K] pair (square and whole tiles with `transposed`), `w` the [K, 4c + 32]
    qkvgb weight of `qkvgb_heads`, `ckc` a (fidelity, approx, fp32_acc, full_sync) tuple as
    `trimul_tail` takes it."""
    from .tenstorrent import lever
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
    if transposed and (not TR or shp[0] != shp[1] or shp[0] % TILE):
        return _reject("tr_shape", x)
    plan = _plan(K // TILE, int(w.padded_shape[-1]) // TILE, transposed, x.device(), grid[0] * grid[1])
    if plan is not None and transposed and plan[0] > 1:
        # Each group would gather x^T in 1 KB pieces on its own: 13.99 ms against 13.62 for transpose,
        # norm and project (OpenDDE S512, Wormhole HiFi4, 1000 MHz). The caller transposes and comes
        # back without `transposed`, which beats both.
        return _reject("tr_groups", x)
    if plan is None:
        return _reject(f"l1:kt{K // TILE}:nw{int(w.padded_shape[-1]) // TILE}:tr{int(transposed)}", x)
    dev = x.device()
    # bfp8 q, k, v: each group's bfp8 run must start its row and end on a DST block (b8_cols)
    gate = lever("triatt_ln_b8g")   # the gate too: gated_out_proj reads a bfp8 g
    cols = b8_cols(plan[0], plan[1], H, gate) if lever("triatt_ln_b8") else None
    if cols and plan[0] > 1:
        # c_2 and c_4 against the bf16 c_2 that _plan priced
        c2, c4 = _cb_tiles(cols, plan[1])
        extra = (c2 - 2 * plan[1]) * MG.tile_bytes(ttnn.bfloat8_b) + c4 * MG.tile_bytes(ttnn.bfloat16)
        if _cb_bytes(K // TILE, plan[1], plan[2]) + extra > _lcf(x.device()):
            cols = None
    nq = (4 if gate else 3) * H if cols else 0
    STATS[2] += bool(nq)
    hd = lambda dt: ttnn.allocate_tensor_on_device(ttnn.Shape([shp[0], n_heads, shp[1], head_dim]), dt,  # noqa: E731
                                                   ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG)
    outs = [hd(ttnn.bfloat8_b if nq > i * H else ttnn.bfloat16) for i in range(4)]   # q, k, v, g
    outs.append(ttnn.allocate_tensor_on_device(ttnn.Shape([shp[0], shp[1], bias_channels]), ttnn.bfloat16,
                                               ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG))
    return _launch(x, w, ln_w, ln_b, outs, grid, ckc, eps, H, MT, bias_channels, plan, cols)


def _launch(x, w, ln_w, ln_b, outs, grid, ckc, eps, H, MT, tag, plan, cols=None):
    g, nw, tr = plan
    key = (x.device().id(), tuple(int(d) for d in x.padded_shape), tuple(int(d) for d in w.padded_shape), H, tag,
           tuple(grid), tuple(str(a) for a in ckc), float(eps), plan, ABL, STAGGER, DUAL_NOC, tuple(cols or ()))
    entry = _CACHE.get(key)
    if entry is None:
        entry = _CACHE[key] = _build(x, w, ln_w, ln_b, outs, grid, ckc, eps, H, MT, tr, g, nw, cols)
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
    plan = _plan(shp[-1] // TILE, int(w.padded_shape[-1]) // TILE, False, x.device(), grid[0] * grid[1])
    if plan is None:
        return _reject("l1", x)
    out = ttnn.allocate_tensor_on_device(ttnn.Shape([shp[0], shp[1], n]), ttnn.bfloat16, ttnn.TILE_LAYOUT,
                                         x.device(), ttnn.DRAM_MEMORY_CONFIG)
    return _launch(x, w, ln_w, ln_b, [out], grid, ckc, eps, 0, int(x.padded_shape[-2]) // TILE, "proj", plan)
