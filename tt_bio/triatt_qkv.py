"""Triangle attention with the head split never materialised: q, k, v, the gate and `out` all stay
in the layout the SDPA wants, so `nlp_create_qkv_heads` and `nlp_concat_heads` never run.

Today the fold runs `minimal_matmul` and then `nlp_create_qkv_heads`, and at the other end
`nlp_concat_heads` before the gate multiply. Together they move 1536 MiB per call at 93-96 % of the
copy roof purely to reorder tiles. Neither has to exist. `head_dim` is 32, exactly one tile, so
output tile *(i, n)* of the qkv matmul already **is** tile *(batch i/MT, head n%8, row i%MT)* of q,
k or v: no element moves inside a tile. Only the address the writer sends the tile to changes, and
symmetrically the address `out`'s reader fetches from.

So this drives the wheel's own `minimal_matmul` kernels through `ttnn.generic_op`
(:mod:`tt_bio.mm_generic`), with the two DM kernels taken from `tt_bio/kernels/triatt/` where three
guarded macros re-point the destination and source tile ids. Transaction count, transaction size and
every arithmetic operation are unchanged, so the results are **bit-exact** -- `torch.equal` against
the stock ops at 298, 320, 384, 512, 576 and 640 aa (`perf/triatt_fused/s1_gate.json`,
`s3_gate.json`), and at the fold the CIF sha256 and plDDT are identical arm to arm.

The gates below are deliberately narrow: 32-channel heads, a storage format
`mm_generic.fast_dtypes_ok` covers, interleaved DRAM both sides, and a shape the shipped
`_MM_BLOCK` entry already covers. Anything else falls through to the stock ops. The
bit-exactness above is the bf16 case, which is what ships; `TT_BIO_TRIATT_B8` narrows the
destination and is one rounding at the pack stage, scored against float64 in
`perf/bfp8_qkv/acc.json`.
"""

from __future__ import annotations

import os
from pathlib import Path

import ttnn

from . import mm_generic as G
from . import ops

KERNEL_DIR = Path(__file__).resolve().parent / "kernels" / "triatt"
TILE = 32

# (eligible calls served, calls that fell through to the two stock ops)
STATS = [0, 0]
# Why calls were refused, keyed by (reason, shape). A gate that never fires has to say why.
REJECTS: dict = {}

TRIATT_HEAD_MAJOR_QKV = True
_ENABLED = os.environ.get(
    "TT_BIO_TRIATT_HEAD_MAJOR_QKV", "1" if TRIATT_HEAD_MAJOR_QKV else "0") == "1"


def _reject(reason, shape):
    k = (reason, tuple(shape))
    REJECTS[k] = REJECTS.get(k, 0) + 1
    STATS[1] += 1
    return None


def _taping():
    """Is a tape open? Every entry point here drives `generic_op`, which has no backward, so each
    declines under one and the composed ops the tape can follow run instead (`ops.taping`)."""
    from . import ops
    return ops.taping()


def _common_ok(x, w, dtype):
    """The dtype, layout and memory-config conditions the transcription was verified under."""
    if not G.fast_dtypes_ok(x.dtype, w.dtype, dest=dtype):
        return False
    if x.layout != ttnn.TILE_LAYOUT or len(x.shape) != 3:
        return False
    xmc, wmc = x.memory_config(), w.memory_config()
    return (xmc.buffer_type == ttnn.BufferType.DRAM
            and xmc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
            and wmc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED)


@ops.fused_kernel("triatt_qkv_heads")
def qkv_heads(x, w, ckc, n_heads, head_dim, dtype, mm_config):
    """`nlp_create_qkv_heads(minimal_matmul(x, w))` as one op, or `None` to leave it alone.

    Returns `(q, k, v)`, each `[batch, n_heads, seq, head_dim]`, byte-identical to what the two
    stock ops produce. Under a tape it runs only through its entry in `taped_ttnn`
    (`triatt_qkv_heads`); without one the composed ops run instead.
    """
    if ops.declines_under_tape("triatt_qkv_heads"):
        return None

    if not _ENABLED:
        return None
    shape = [int(d) for d in x.shape]
    if head_dim != TILE or n_heads * head_dim * 3 != int(w.shape[-1]):
        return _reject("head_dim_or_width", shape)
    if not _common_ok(x, w, dtype):
        return _reject("dtype_or_memory", shape)
    # The descriptor is a transcription of the factory for the shipped block entry only.
    if mm_config is None:
        return _reject("no_mm_config", shape)
    from .tenstorrent import _mm_block_for, COMPUTE_GRID_MAIN
    blk = _mm_block_for(w)
    if blk is None:
        return _reject("no_block_entry", shape)

    pad = [int(d) for d in x.padded_shape]
    if pad[0] * pad[-2] <= int(w.shape[-1]):
        # transpose_core_grid is false there, a core-grid orientation this has never been run on
        return _reject("m_le_n", shape)

    dev = x.device()
    outs = [ttnn.allocate_tensor_on_device(
        ttnn.Shape([shape[0], n_heads, shape[1], head_dim]), dtype, ttnn.TILE_LAYOUT,
        dev, ttnn.DRAM_MEMORY_CONFIG) for _ in range(3)]
    G.generic_minimal_matmul(
        dev, x, w, outs, (blk, tuple(COMPUTE_GRID_MAIN)), G.ckc_args(ckc),
        {"HEAD_MAJOR_MT": pad[-2] // TILE}, KERNEL_DIR, guard="tri")
    STATS[0] += 1
    return tuple(outs)


# --- K1b: the tail stays head-major, so nlp_concat_heads never runs -------------------------------
#
# The SDPA leaves `o` as [batch, head, seq, 32]. Today the tail's first op undoes that so the gate
# multiply and the `out` projection can work on [batch, seq, 256]. Neither of them needs to: the
# multiply is elementwise, and `out`'s reader can take the same tile-id transform K1a gave the
# writer. So the gate projection writes head-major, the multiply runs where it is, and `out` reads
# head-major and writes an ordinary [batch, seq, 256] result.
#
# MEASURED on qb2 card 1 (perf/triatt_fused/s3_gate.json), torch.equal on the gate projection and on
# the final `out` at all six sizes:
#
#   n     nlp_concat_heads   gate proj        out proj                net ms/call
#   298   0.280 -> deleted   0.313 -> 0.322   0.318 -> 0.332  +4.5 %   +0.612
#   320   0.300 -> deleted   0.331 -> 0.337   0.337 -> 0.343  +1.7 %   +0.394
#   384   0.426 -> deleted   0.457 -> 0.480   0.466 -> 0.482  +3.5 %   +0.484
#   512   0.749 -> deleted   0.806 -> 0.801   0.803 -> 0.851  +5.9 %   +0.558
#   576   0.919 -> deleted   0.994 -> 0.975   1.007 -> 1.016  +0.9 %   +0.961
#   640   1.117 -> deleted   1.219 -> 1.182   1.233 -> 1.233  +0.0 %   +0.455
#
# The `out` read costs 0-6 % more head-major and the reason is NOT established. A DRAM bank conflict
# was predicted before this sweep and the sweep does not support it: this card has 8 banks, the
# reader walks 8 tile ids strided by `mt` per K block, and 512 is the only size whose `mt` is a
# multiple of 8 -- it is duly the worst at +5.9 %, but 640 (`mt` 20, two banks) reads +0.0 % and 298
# (`mt` 10, four banks) reads +4.5 %, so the predicted ordering is wrong. Against a 2.6 % A/A only
# 512 and 298 are outside noise at all. It is charged to the residual as unexplained.
#
# The gate declines whenever the `out` projection would have taken the L1-output leg, because that
# leg also removes the CONSUMER's operand read, which none of the numbers above can see. At 512 the
# allocator refuses it and this fires; at 298 it does not. Whether the 0.23 ms/call the head-major
# tail would save at 298 beats the L1 output is a fold question and is not answered here.

TRIATT_HEAD_MAJOR_TAIL = True
_TAIL_ENABLED = os.environ.get(
    "TT_BIO_TRIATT_HEAD_MAJOR_TAIL", "1" if TRIATT_HEAD_MAJOR_TAIL else "0") == "1"

# (tails served, tails declined)
TAIL_STATS = [0, 0]
TAIL_REJECTS: dict = {}

# Whether the head-major tail may take a call whose `out` projection would otherwise have used the
# L1-output leg. That leg also deletes the CONSUMER's operand read, which nothing measured off-fold
# can see, so it was held off by default until the fold answered it. The fold answered it: deleting
# nlp_concat_heads wins at every size measured, three folds per arm, alternating, byte-identical
# output (perf/triatt_fused/fold_ab_k1_full{,_r2}.json).
#
#   n     TriAtt body ms          ratio     A/A on the on arm
#   298   6080.7 ->  5034.2      1.2079x    2.67 ms
#   384  10450.6 ->  8834.6      1.1829x  468.88 ms
#   512  19719.8 -> 16716.5      1.1797x    4.54 ms
#
# 384 is the noisy one and is quoted as measured; the arms separate completely there anyway.
TRIATT_TAIL_OVER_L1 = True
_TAIL_OVER_L1 = os.environ.get(
    "TT_BIO_TRIATT_TAIL_OVER_L1", "1" if TRIATT_TAIL_OVER_L1 else "0") == "1"


def _tail_reject(reason, shape):
    k = (reason, tuple(shape))
    TAIL_REJECTS[k] = TAIL_REJECTS.get(k, 0) + 1
    TAIL_STATS[1] += 1
    return None


def gate_proj(x, w_g, w_o, ckc, n_heads, head_dim, dtype, mm_config):
    """The gate projection written head-major, or `None` to leave the whole tail alone.

    A 4-D return is the signal the rest of the tail reads: `attend` skips `nlp_concat_heads` and
    `gate_and_project` calls `out_proj`. `w_o` is only inspected, to ask whether the `out`
    projection it will feed would have taken the L1-output leg.
    """
    if _taping():
        return None   # no backward for `generic_op`; the composed ops run instead
    if not (_ENABLED and _TAIL_ENABLED):
        return None
    shape = [int(d) for d in x.shape]
    if head_dim != TILE or n_heads * head_dim != int(w_g.shape[-1]):
        return _tail_reject("head_dim_or_width", shape)
    if not _common_ok(x, w_g, dtype) or mm_config is None:
        return _tail_reject("dtype_or_memory_or_config", shape)
    if len(w_o.shape) != 2 or int(w_o.shape[-2]) // TILE != int(w_g.shape[-1]) // TILE:
        return _tail_reject("out_weight_shape", shape)

    from .tenstorrent import (_mm_block_for, COMPUTE_GRID_MAIN, _PAIR_PROJ_L1_OUT, _L1_OUT_REFUSED,
                              _PAIR_PROJ_MM, _MM_DEFAULT)
    if not _PAIR_PROJ_MM:
        # `out` would take the DRAM `ttnn.linear` leg, which this has never been compared against
        return _tail_reject("pair_proj_mm_off", shape)
    if _mm_block_for(w_g) is None or _mm_block_for(w_o) is None:
        return _tail_reject("no_block_entry", shape)
    # `_MM_DEFAULT` exists so K1a has a descriptor at a width the swept table does not cover
    # (opendde, c_z=384, kt=12). The TAIL is a different question there: `_pair_proj_minimal_matmul`
    # refuses kt != 8, so the stock `out` at that width is a `ttnn.linear` with `_pair_proj_config`,
    # and a head-major `out` swaps the op class. MEASURED at the fold, 512 aa,
    # perf/odde4x/ab_opendde_512.json: the tail moves the CIF digest and plDDT 0.754131 -> 0.750771
    # to buy 0.234 s, which is 1.15x the run's own 0.204 s A/A floor. Where the table has a SWEPT
    # entry the tail keeps serving and stays byte-identical (boltz2/openfold3 at kt = 4 and 2,
    # `on` vs `nonewmm` same CIF digest in perf/other512/ab_b2_rekey_512.json).
    if _mm_block_for(w_o) is _MM_DEFAULT:
        return _tail_reject("mm_default_entry_k1a_only", shape)
    # The L1-output leg of `out` also deletes the consumer's operand read; never trade it away.
    # `_L1_OUT_REFUSED` is the allocator's own verdict and is only populated after a real attempt,
    # so the first call at a new shape declines and the rest of the fold follows the verdict.
    if _PAIR_PROJ_L1_OUT and not _TAIL_OVER_L1:
        key = (tuple(x.padded_shape), tuple(w_o.shape), str(dtype))
        if key not in _L1_OUT_REFUSED:
            return _tail_reject("l1_out_leg_live", shape)

    pad = [int(d) for d in x.padded_shape]
    if pad[0] * pad[-2] <= int(w_g.shape[-1]):
        return _tail_reject("m_le_n", shape)

    dev = x.device()
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([shape[0], n_heads, shape[1], head_dim]), dtype, ttnn.TILE_LAYOUT,
        dev, ttnn.DRAM_MEMORY_CONFIG)
    G.generic_minimal_matmul(
        dev, x, w_g, out, (_mm_block_for(w_g), tuple(COMPUTE_GRID_MAIN)),
        G.ckc_args(ckc), {"HEAD_MAJOR_OUT_MT": pad[-2] // TILE}, KERNEL_DIR, guard="tri")
    TAIL_STATS[0] += 1
    return out


# --- K3: the gate rides the qkv projection, so the normed pair tensor is read once -------------
#
# `qkv_heads` and `gate_proj` are two matmuls over the same activation. At 512 aa that activation is
# 67.1 MB and each of them reads all of it, so the triangle attention reads its own normed pair
# tensor twice to fill four output buffers that differ only in which weight columns produced them.
# MEASURED by walking one block's operands keyed on the device ALLOCATION rather than the tensor id
# (`perf/b2z2_byte_floor/`): 134.2 MB of the block's 8.05 GB is this one re-read, and a third reader
# -- the 32-wide pair-bias projection -- makes the tensor's full redundancy 268.4 MB.
#
# Nothing about the kernel has to change. Its writer already splits the N axis into `N_chunks`
# buffers of `N_tiles_per_chunk` each: qkv is 12 N tiles as 3 x 4, the gate is 4 as 1 x 4. Four
# chunks of 4 is the same writer with one more destination. And it is bit-exact by the same argument
# `_MM_BLOCK` already makes for its neighbouring entries: boltz2's qkv key (4, 12) and gate key
# (4, 4) both carry K_block = 4 = the whole contraction, so the fused key (4, 16) folding K the same
# way computes every output element in the order it is computed today.
#
# Declines to exactly what the two separate calls would have declined to, including `gate_proj`'s
# own guards on the `out` projection -- a fused call that served where `gate_proj` would have
# refused would silently change the tail's op class.

TRIATT_FUSED_QKVG = True
_QKVG_ENABLED = os.environ.get(
    "TT_BIO_TRIATT_FUSED_QKVG", "1" if TRIATT_FUSED_QKVG else "0") == "1"

# (fused calls served, calls that fell back to the separate qkv + gate pair)
QKVG_STATS = [0, 0]
QKVG_REJECTS: dict = {}


def _qkvg_reject(reason, shape):
    k = (reason, tuple(shape))
    QKVG_REJECTS[k] = QKVG_REJECTS.get(k, 0) + 1
    QKVG_STATS[1] += 1
    return None


def qkvg_heads(x, w, w_o, ckc, n_heads, head_dim, dtype, mm_config):
    """`(q, k, v, gate)` from ONE pass over `x`, or `None` to leave the two calls alone.

    `w` is the qkv weight with the gate weight concatenated on its output axis, so the four
    destinations are four equal N chunks of one matmul. Byte-identical to
    `qkv_heads(x, w[:, :3c]) + gate_proj(x, w[:, 3c:])`.
    """
    if _taping():
        return None   # no backward for `generic_op`; the composed ops run instead
    if not (_ENABLED and _TAIL_ENABLED and _QKVG_ENABLED):
        return None
    shape = [int(d) for d in x.shape]
    c = n_heads * head_dim
    if head_dim != TILE or c * 4 != int(w.shape[-1]):
        return _qkvg_reject("head_dim_or_width", shape)
    if not _common_ok(x, w, dtype) or mm_config is None:
        return _qkvg_reject("dtype_or_memory_or_config", shape)
    if len(w_o.shape) != 2 or int(w_o.shape[-2]) // TILE != c // TILE:
        return _qkvg_reject("out_weight_shape", shape)

    from .tenstorrent import (_mm_block_for, COMPUTE_GRID_MAIN, _PAIR_PROJ_L1_OUT, _L1_OUT_REFUSED,
                              _PAIR_PROJ_MM, _MM_DEFAULT)
    blk = _mm_block_for(w)
    if blk is None or blk is _MM_DEFAULT:
        return _qkvg_reject("no_block_entry", shape)
    # Everything below is `gate_proj`'s guard set, because a fused call that served where the gate
    # refused would move the tail off `out_proj` and change what the block measures.
    if not _PAIR_PROJ_MM:
        return _qkvg_reject("pair_proj_mm_off", shape)
    if _mm_block_for(w_o) is None or _mm_block_for(w_o) is _MM_DEFAULT:
        return _qkvg_reject("out_block_entry", shape)
    if _PAIR_PROJ_L1_OUT and not _TAIL_OVER_L1:
        key = (tuple(x.padded_shape), tuple(w_o.shape), str(dtype))
        if key not in _L1_OUT_REFUSED:
            return _qkvg_reject("l1_out_leg_live", shape)

    pad = [int(d) for d in x.padded_shape]
    if pad[0] * pad[-2] <= c:
        return _qkvg_reject("m_le_n", shape)

    dev = x.device()
    outs = [ttnn.allocate_tensor_on_device(
        ttnn.Shape([shape[0], n_heads, shape[1], head_dim]), dtype, ttnn.TILE_LAYOUT,
        dev, ttnn.DRAM_MEMORY_CONFIG) for _ in range(4)]
    G.generic_minimal_matmul(
        dev, x, w, outs, (blk, tuple(COMPUTE_GRID_MAIN)), G.ckc_args(ckc),
        {"HEAD_MAJOR_MT": pad[-2] // TILE}, KERNEL_DIR, guard="tri")
    QKVG_STATS[0] += 1
    STATS[0] += 1
    TAIL_STATS[0] += 1
    return tuple(outs[:3]), outs[3]


def out_proj(gated, w, ckc, dtype, memory_config=None):
    """The `out` projection reading a head-major activation: `[B, H, S, 32] -> [B, S, H*32]`.

    `memory_config` names where the result lands. The kernel builds its output address generator
    from the tensor it is handed, so an L1 destination changes which banks a tile is written to
    and nothing else: same blocking, same accumulation order, same bytes.
    """
    if _taping():
        return None   # no backward for `generic_op`; the composed ops run instead
    from .tenstorrent import _mm_block_for, COMPUTE_GRID_MAIN
    B, H, S, D = (int(d) for d in gated.shape)
    pad = [int(d) for d in gated.padded_shape]
    dev = gated.device()
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([B, S, int(w.shape[-1])]), dtype, ttnn.TILE_LAYOUT, dev,
        memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG)
    G.generic_minimal_matmul(
        dev, gated, w, out, (_mm_block_for(w), tuple(COMPUTE_GRID_MAIN)),
        G.ckc_args(ckc), {"HEAD_MAJOR_IN0_MT": pad[-2] // TILE}, KERNEL_DIR,
        m_k=(pad[0] * pad[-2], H * D), guard="tri")
    return out


# --- the whole tail in one program: gate, `out` projection and residual -----------------------------
#
# Production runs the tail as three DRAM round trips: multiply_(o, g, SIGMOID on b) reads 2P and writes
# P, `out_proj` above reads P and writes P, and the layer's add_ reads 2P and writes P (P = one bf16
# pair tensor, 277 MB at 736 tokens). Measured on WH at 736: 4.99 + 3.38 + 3.40 = 11.77 ms per call.
# `gated_out_proj` reads o, g and z once and writes z once (4P), with Wo resident in L1: one output row
# tile per unit, the gate applied in DST, the whole K in one block, the residual added on the FPU.
# Numerics move at the bf16-ULP level: sigmoid(g) is never rounded to bf16 before the multiply.
#
# Variants at 736, WH 1000 MHz, z += tail with the residual, rel. RMS vs float64 (perf/spd_pair/ops.py, op3):
#   three ops today            11.95 ms   0.001723
#   fused, sigmoid_bf16, RNE    6.24 ms   0.001719
#   fused, poly, RNE            5.22 ms   0.001727
#   fused, sigmoid_bf16, no RNE 5.64 ms   0.001719  (output identical to RNE: the packer already rounds)
#   fused, HiFi4 no f32 (fast)  6.49 ms   0.001974
# So: poly sigmoid, no explicit RNE, and HiFi3 with fp32 DST in both modes (faster AND closer than fast's
# config, as for the pair layer norms).
TAIL_SIGPOLY = int(os.environ.get("TT_BIO_TRIATT_TAIL_SIGPOLY", "1"))
TAIL_RNE = int(os.environ.get("TT_BIO_TRIATT_TAIL_RNE", "0"))
# K one tile per DST pass, summed by the packer in fp32 (compute.cpp KB1): the Wormhole fp32-DST erratum
# lives in accumulation across a multi-tile K block (BOARD 2026-10-09 12:08Z, spd-swiglu). Counted vs
# float64 on WH, S 384/512/736 x 4 seeds, 974 M elements per arm (perf/spd_pair/tail_outliers.py): the
# whole-K block had 1 element off by 2.0 (S 736, seed 2), KB1 none (max 0.020), and KB1 is faster at
# 512 and 736 (2.85 vs 3.74 ms, 5.61 vs 6.45 ms).
TAIL_KB1 = int(os.environ.get("TT_BIO_TRIATT_TAIL_KB1", "1"))
TAIL_FORCE = os.environ.get("TT_BIO_TRIATT_TAIL")   # "1" / "0" overrides the `triatt_tail` lever
GOP_STATS = [0, 0]   # gated_out_proj served, declined (TAIL_STATS above is the head-major tail's)
_TAIL_DIR = Path(__file__).resolve().parent / "kernels" / "triatt_tail"
_TAIL_CACHE: dict = {}


def tail_on() -> bool:
    if TAIL_FORCE is not None:
        return TAIL_FORCE == "1"
    from .tenstorrent import lever
    return lever("triatt_tail")


def set_tail(v):
    """A/B switch for the harness: "1", "0" or None (follow the lever). Returns the previous value."""
    global TAIL_FORCE
    prev, TAIL_FORCE = TAIL_FORCE, v
    return prev


def _dram_tile_bf16(t):
    return (t.dtype == ttnn.bfloat16 and t.layout == ttnn.TILE_LAYOUT
            and t.memory_config() == ttnn.DRAM_MEMORY_CONFIG)


def _tail_ok(o, g, w, resid):
    po, pg = [int(d) for d in o.padded_shape], [int(d) for d in g.padded_shape]
    if not (len(po) == 4 and po == pg and po[-1] == TILE and _dram_tile_bf16(o) and _dram_tile_bf16(g)
            and _dram_tile_bf16(w) and len(w.shape) == 2 and int(w.shape[-2]) == po[1] * TILE
            and int(w.shape[-1]) % TILE == 0):
        return False
    if resid is not None:
        pz = [int(d) for d in resid.padded_shape]
        if not (_dram_tile_bf16(resid) and pz[-3:] == [po[0], po[2], int(w.shape[-1])]
                and all(d == 1 for d in pz[:-3])):
            return False
    from .tenstorrent import _l1_bank_bytes
    kt, nt = po[1], int(w.shape[-1]) // TILE
    tiles = kt * nt + 5 * kt + (8 if TAIL_KB1 else 7) * nt   # KB1's fp32 partials take two bf16 tiles' room
    return tiles * G.tile_bytes(ttnn.bfloat16) <= 0.6 * _l1_bank_bytes()


def _tail_cb(idx, core_grid, tiles, dt=ttnn.bfloat16):
    fmt = ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dt, page_size=G.tile_bytes(dt))
    return ttnn.CBDescriptor(total_size=tiles * G.tile_bytes(dt), core_ranges=core_grid,
                             format_descriptors=[fmt])


def _tail_dw(kt, nt):
    """Tiles per DST batch: four fit a 32-bit DST, and the batch must divide both K and N. A width that
    does not (the template stack's 64-channel pair has Kt = Nt = 2) packs past the end of the circular
    buffers and overwrites the residual's tiles."""
    return next(d for d in (4, 2, 1) if kt % d == 0 and nt % d == 0)


def _build_tail(o, g, w, out, z, ckc, grid, resid):
    B, kt, St = int(o.padded_shape[0]), int(o.padded_shape[1]), int(o.padded_shape[2]) // TILE
    nt = int(w.shape[-1]) // TILE
    nu = B * St
    gx, gy = grid
    ncores = gx * gy
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))])
    rd, wr, cp = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    u0 = 0
    for c in range(ncores):
        n = nu // ncores + (c < nu % ncores)
        x_, y_ = c % gx, c // gx
        rd[x_][y_] = [u0, n]
        wr[x_][y_] = [u0, n]
        cp[x_][y_] = [n]
        u0 += n
    acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    src = ttnn.KernelDescriptor.SourceType.FILE_PATH
    reader = ttnn.KernelDescriptor(
        kernel_source=str(_TAIL_DIR / "reader.cpp"), source_type=src, core_ranges=core_grid,
        compile_time_args=[kt, nt, St, int(resid)] + acc(o) + acc(g) + acc(w) + acc(z),
        runtime_args=rd, common_runtime_args=[0] * 4, config=ttnn.ReaderConfigDescriptor())
    writer = ttnn.KernelDescriptor(
        kernel_source=str(_TAIL_DIR / "writer.cpp"), source_type=src, core_ranges=core_grid,
        compile_time_args=[nt] + acc(out), runtime_args=wr, common_runtime_args=[0],
        config=ttnn.WriterConfigDescriptor())
    fid, approx, fp32, full = G.ckc_args(ckc)
    fid, fp32 = ttnn.MathFidelity.HiFi3, True
    compute = ttnn.KernelDescriptor(
        kernel_source=str(_TAIL_DIR / "compute.cpp"), source_type=src, core_ranges=core_grid,
        compile_time_args=[kt, nt, int(resid), TAIL_SIGPOLY, TAIL_RNE, _tail_dw(kt, nt), TAIL_KB1],
        runtime_args=cp,
        config=ttnn.ComputeConfigDescriptor(math_fidelity=fid, math_approx_mode=approx,
                                            fp32_dest_acc_en=fp32, dst_full_sync_en=full))
    cbs = [_tail_cb(0, core_grid, kt * nt), _tail_cb(1, core_grid, 2 * kt), _tail_cb(2, core_grid, 2 * kt),
           _tail_cb(4, core_grid, kt), _tail_cb(16, core_grid, 2 * nt)]
    if resid:
        cbs.append(_tail_cb(3, core_grid, 2 * nt))
    if TAIL_KB1:
        cbs.append(_tail_cb(6, core_grid, nt, ttnn.float32))
    elif resid:
        cbs.append(_tail_cb(5, core_grid, nt))
    return {"kernels": [reader, writer, compute], "cbs": cbs}


def gated_out_proj(o, g, w, ckc, resid=None):
    """`(o * sigmoid(g)) @ w` from head-major o, g `[B, H, S, 32]`, as `[B, S, N]`; with `resid`
    (the pair `[1?, B, S, N]` the update belongs to) the sum is written into `resid` in place and
    `resid` is returned. None where it declines; the caller then runs the three ops."""
    if _taping() or not tail_on() or not _tail_ok(o, g, w, resid):
        GOP_STATS[1] += 1
        return None
    from .tenstorrent import COMPUTE_GRID_MAIN
    B, S = int(o.shape[0]), int(o.shape[2])
    if resid is None:
        out = ttnn.allocate_tensor_on_device(ttnn.Shape([B, S, int(w.shape[-1])]), ttnn.bfloat16,
                                             ttnn.TILE_LAYOUT, o.device(), ttnn.DRAM_MEMORY_CONFIG)
    else:
        out = resid
    grid = tuple(COMPUTE_GRID_MAIN)
    key = (str(o.padded_shape), str(w.padded_shape), str(out.padded_shape), grid, G.ckc_args(ckc),
           resid is not None, TAIL_SIGPOLY, TAIL_RNE, TAIL_KB1)
    entry = _TAIL_CACHE.get(key)
    if entry is None:
        entry = _TAIL_CACHE[key] = _build_tail(o, g, w, out, out, ckc, grid, resid is not None)
    reader, writer, _ = entry["kernels"]
    reader.common_runtime_args = [o.buffer_address(), g.buffer_address(), w.buffer_address(),
                                  out.buffer_address()]
    writer.common_runtime_args = [out.buffer_address()]
    pd = ttnn.ProgramDescriptor(kernels=entry["kernels"], semaphores=[], cbs=entry["cbs"])
    ref = _tail_check_ref(o, g, w, resid)
    ttnn.generic_op([o, g, w, out], pd)
    if ref is not None:
        _tail_check_log(ref, out, o, g)
    GOP_STATS[0] += 1
    return out


# Diagnostic: TT_BIO_TRIATT_TAIL_CHECK=N compares the first N calls against float64 on the host and
# appends one JSON line per call to TT_BIO_TRIATT_TAIL_CHECK_LOG.
_CHECK_N = int(os.environ.get("TT_BIO_TRIATT_TAIL_CHECK", "0"))
_CHECK = [0]


def _tail_check_ref(o, g, w, resid):
    if _CHECK[0] >= _CHECK_N:
        return None
    _CHECK[0] += 1
    import torch
    O, Gt, W = (ttnn.to_torch(t).double() for t in (o, g, w))
    B, H, S, D = O.shape
    x = (O * torch.sigmoid(Gt)).permute(0, 2, 1, 3).reshape(B, S, H * D)
    u = x @ W
    if resid is not None:
        u = ttnn.to_torch(resid).double().reshape(u.shape) + u
    return u


def _tail_check_log(ref, out, o, g):
    import json, torch
    got = ttnn.to_torch(out).double().reshape(ref.shape)
    err = got - ref
    bad = int((~torch.isfinite(got)).sum())
    err = torch.nan_to_num(err)
    rel = float(err.norm() / ref.norm().clamp_min(1e-30))
    i = int(err.abs().argmax())
    G_ = ttnn.to_torch(g).float()
    O_ = ttnn.to_torch(o).float()
    rec = dict(call=_CHECK[0], shape=list(ref.shape), rel=rel, max_abs=float(err.abs().max()),
               at=list(map(int, torch.unravel_index(torch.tensor(i), ref.shape))),
               ref_at=float(ref.flatten()[i]), got_at=float(got.flatten()[i]), nonfinite=bad,
               g_min=float(G_.min()), g_max=float(G_.max()), o_absmax=float(O_.abs().max()),
               ref_absmax=float(ref.abs().max()))
    with open(os.environ.get("TT_BIO_TRIATT_TAIL_CHECK_LOG", "/tmp/triatt_tail_check.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")


# --- R1b: the pair-bias projection rides the qkv+gate pass, so `x_norm` is read ONCE -------------
#
# `qkvg_heads` above deleted one of the tri-attention's three reads of its own normed pair tensor.
# The third is the pair-BIAS projection, `c_z -> n_heads` -- one tile wide, and it reads all
# 67.1 MB of the same allocation at 512 aa. That is rank 1's remaining 134.2 MB over two
# attentions (`perf/b2z2_byte_floor/CENSUS.md`).
#
# It could not ride the same matmul until the split writer learned an unequal chunk: q, k, v and
# the gate are 4 N tiles each and the bias is 1, and the writer divided N into EQUAL chunks.
# `MM_SPLIT_LAST_TILES` gives the final chunk its own width (see
# `tt_bio/kernels/triatt/matmul_dataflow_common.hpp`), and the head-major tile-id transform
# reduces to the plain one at a width of a single tile, so the bias lands in an ordinary
# [batch, seq, 32] result while its four siblings stay head-major. No second define, no second
# kernel.
#
# Bit-exact by the same argument as `qkvg_heads`, plus one measured fact: the bias projection is a
# `ttnn.linear` today, not a `minimal_matmul`, so this changes its op class. At c_z=128 both block
# the whole 4-tile contraction at once and are `torch.equal` at max abs 0.0
# (`perf/b2z2_byte_round2/probe_opclass.py`, run before any of this was built).
#
# Declines to exactly what `qkvg_heads` declines to, plus a bias weight that is not one tile wide.

TRIATT_FUSED_QKVGB = True
_QKVGB_ENABLED = os.environ.get(
    "TT_BIO_TRIATT_FUSED_QKVGB", "1" if TRIATT_FUSED_QKVGB else "0") == "1"

# (fused calls served, calls that left the bias projection where it was)
QKVGB_STATS = [0, 0]
QKVGB_REJECTS: dict = {}


def _qkvgb_reject(reason, shape):
    k = (reason, tuple(shape))
    QKVGB_REJECTS[k] = QKVGB_REJECTS.get(k, 0) + 1
    QKVGB_STATS[1] += 1
    return None


def qkvgb_heads(x, w, w_o, ckc, n_heads, head_dim, dtype, mm_config, bias_channels):
    """`((q, k, v), gate, bias)` from ONE pass over `x`, or `None` to leave all three alone.

    `w` is the qkv weight with the gate weight and the (tile-padded) bias weight concatenated on
    its output axis, so the five destinations are five N chunks of one matmul -- four of four
    tiles and one of one. Byte-identical to the three calls it replaces.
    """
    if _taping():
        return None   # no backward for `generic_op`; the composed ops run instead
    if not (_ENABLED and _TAIL_ENABLED and _QKVG_ENABLED and _QKVGB_ENABLED):
        return None
    shape = [int(d) for d in x.shape]
    c = n_heads * head_dim
    if head_dim != TILE or c * 4 + TILE != int(w.shape[-1]):
        return _qkvgb_reject("head_dim_or_width", shape)
    if not 0 < bias_channels <= TILE:
        return _qkvgb_reject("bias_wider_than_a_tile", shape)
    if not _common_ok(x, w, dtype) or mm_config is None:
        return _qkvgb_reject("dtype_or_memory_or_config", shape)
    if len(w_o.shape) != 2 or int(w_o.shape[-2]) // TILE != c // TILE:
        return _qkvgb_reject("out_weight_shape", shape)

    from .tenstorrent import (_mm_block_for, COMPUTE_GRID_MAIN, _PAIR_PROJ_L1_OUT, _L1_OUT_REFUSED,
                              _PAIR_PROJ_MM, _MM_DEFAULT)
    blk = _mm_block_for(w)
    if blk is None or blk is _MM_DEFAULT:
        return _qkvgb_reject("no_block_entry", shape)
    # `gate_proj`'s guard set, because a fused call that served where the gate refused would move
    # the tail off `out_proj` and change what the block computes.
    if not _PAIR_PROJ_MM:
        return _qkvgb_reject("pair_proj_mm_off", shape)
    if _mm_block_for(w_o) is None or _mm_block_for(w_o) is _MM_DEFAULT:
        return _qkvgb_reject("out_block_entry", shape)
    if _PAIR_PROJ_L1_OUT and not _TAIL_OVER_L1:
        key = (tuple(x.padded_shape), tuple(w_o.shape), str(dtype))
        if key not in _L1_OUT_REFUSED:
            return _qkvgb_reject("l1_out_leg_live", shape)

    pad = [int(d) for d in x.padded_shape]
    if pad[0] <= TILE or pad[-2] <= TILE:
        # One tile on either axis is the single shape where this does not reproduce the three
        # calls it replaces to the byte: measured different at 32 residues and identical at 48,
        # 64, 96, 112 and every size on the 512-1536 ladder
        # (`perf/b2z2_trunk_ship/qkvgb_boundary.json`). A chain that fits in one tile is not a
        # performance case, so decline it and keep the fused path bit-exact wherever it serves.
        return _qkvgb_reject("single_tile_axis", shape)
    if pad[0] * pad[-2] <= int(w.shape[-1]):
        return _qkvgb_reject("m_le_n", shape)

    dev = x.device()
    outs = [ttnn.allocate_tensor_on_device(
        ttnn.Shape([shape[0], n_heads, shape[1], head_dim]), dtype, ttnn.TILE_LAYOUT,
        dev, ttnn.DRAM_MEMORY_CONFIG) for _ in range(4)]
    # One format across all five: `generic_minimal_matmul` sizes the output CB from `outs[0]`,
    # so a bias destination of a different width than its four siblings would be written
    # through the wrong page size.
    outs.append(ttnn.allocate_tensor_on_device(
        ttnn.Shape([shape[0], shape[1], bias_channels]), dtype, ttnn.TILE_LAYOUT,
        dev, ttnn.DRAM_MEMORY_CONFIG))
    G.generic_minimal_matmul(
        dev, x, w, outs, (blk, tuple(COMPUTE_GRID_MAIN)), G.ckc_args(ckc),
        {"HEAD_MAJOR_MT": pad[-2] // TILE}, KERNEL_DIR, n_widths=[c // TILE] * 4 + [1], guard="tri")
    QKVGB_STATS[0] += 1
    QKVG_STATS[0] += 1
    STATS[0] += 1
    TAIL_STATS[0] += 1
    return tuple(outs[:3]), outs[3], outs[4]
