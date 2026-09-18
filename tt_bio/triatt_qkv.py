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

The gates below are deliberately narrow: 32-channel heads, bf16, interleaved DRAM both sides, and a
shape the shipped `_MM_BLOCK` entry already covers. Anything else falls through to the stock ops.
"""

from __future__ import annotations

import os
from pathlib import Path

import ttnn

from . import mm_generic as G

KERNEL_DIR = Path(__file__).resolve().parent / "kernels" / "triatt"
TILE = 32

# (eligible calls served, calls that fell through to the two stock ops)
STATS = [0, 0]
# Why calls were refused, keyed by (reason, shape). A gate that never fires has to say why.
REJECTS: dict = {}

TRIATT_HEAD_MAJOR_QKV = True
_ENABLED = os.environ.get(
    "TT_BIO_TRIATT_HEAD_MAJOR_QKV", "1" if TRIATT_HEAD_MAJOR_QKV else "0") == "1"

# --- the same writer at the diffusion / trunk AttentionPairBias site ------------------------------
#
# `c12-tail-classes-screen` priced the diffusion side's own head split at 0.31376 s / 423.6 Mcycles
# per 512 aa fold, 7464 programs that compute nothing. Two of its four signatures are this one
# projection: the token transformer's 4800 calls at 0.20470 s and the trunk's 264 at 0.00581 s, both
# `AttentionPairBias` with `atom_level=False`. The atom-level pair (0.09016 s create + 0.01309 s
# concat) is a different mechanism -- its q comes from a second matmul and is row-padded from the
# 32-row window to 128 between the projection and the split -- and is not served here.
#
# Two things had to grow for it, both parameters rather than second code paths:
#   * head_dim is 64 there, TWO tiles, so the chunk's N tile index splits as (head, channel).
#     `HEAD_MAJOR_DT` carries that and defaults to 1, which is what the triangle attention compiles.
#   * the projection carries a bias where the triangle attention's does not. `mm_generic` gained the
#     wheel's own `FUSE_BIAS` branch, which adds the row in the fp32 accumulator before the pack --
#     the same place `ttnn.linear(bias=...)` adds it today.
#
# The tile re-point is bit-exact against `minimal_matmul + nlp_create_qkv_heads` by construction and
# is checked on the CPU at every converted shape (`perf/c12_diffusion_head/tile_map.py`). What is NOT
# bit-exact is the op class: this site runs `ttnn.linear(core_grid=CORE_GRID_MAIN)` today, and
# serving it means the projection becomes a `minimal_matmul` at the op's own default blocking. That
# is the same two-step opendde's tri-attention took (`_MM_DEFAULT` in tenstorrent.py), it needs an
# Angstrom reading against the seed-scatter floor, and it is why this ships OFF until the fold has
# measured it.
APB_HEAD_MAJOR_QKV = False
_APB_ENABLED = os.environ.get(
    "TT_BIO_APB_HEAD_MAJOR_QKV", "1" if APB_HEAD_MAJOR_QKV else "0") == "1"

# (served, declined) and the reject reasons at the AttentionPairBias site, kept apart from the
# triangle attention's so a dark gate at one site cannot be read off the other's counter.
APB_STATS = [0, 0]
APB_REJECTS: dict = {}

_SITES = {"triatt": (STATS, REJECTS), "apb": (APB_STATS, APB_REJECTS)}


def enabled(site):
    """Whether this site's gate is on at all, so a caller can skip building its arguments.

    `qkv_heads` checks it again; this exists so the off arm is byte-for-byte today's code path and
    not today's path plus a discarded `MinimalMatmulConfig` per call.
    """
    return _ENABLED if site == "triatt" else _APB_ENABLED


def _reject(reason, shape, site="triatt"):
    stats, rejects = _SITES[site]
    k = (reason, tuple(shape))
    rejects[k] = rejects.get(k, 0) + 1
    stats[1] += 1
    return None


def _m_rows(t):
    """The matmul M the factory infers from this activation: the product of every dim but the last.

    Written once because two call sites need it and one of them used to spell it `pad[0]*pad[-2]`,
    which is the same number only at rank 3 and is 140x too small for the rank-4 atom activation.
    """
    m = 1
    for d in t.padded_shape[:-1]:
        m *= int(d)
    return m


def _common_ok(x, w, dtype, rank=3):
    """The dtype, layout and memory-config conditions the transcription was verified under."""
    if dtype != ttnn.bfloat16 or x.dtype != ttnn.bfloat16 or w.dtype != ttnn.bfloat16:
        return False
    if x.layout != ttnn.TILE_LAYOUT or len(x.shape) != rank:
        return False
    xmc, wmc = x.memory_config(), w.memory_config()
    return (xmc.buffer_type == ttnn.BufferType.DRAM
            and xmc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
            and wmc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED)


def qkv_heads(x, w, ckc, n_heads, head_dim, dtype, mm_config, bias=None,
              allow_m_le_n=False, site="triatt", refuse=None):
    """`nlp_create_qkv_heads(minimal_matmul(x, w, bias))` as one op, or `None` to leave it alone.

    Returns `(q, k, v)`, each `[batch, n_heads, seq, head_dim]`, byte-identical to what the two
    stock ops produce. `head_dim` is in ELEMENTS and must be a whole number of tiles in the padded
    form the projection writes -- 32 for the triangle attention and the trunk, 64 for the diffusion
    token transformer. A head that is not a whole number of tiles cannot be served at all: elements
    would move inside a tile, which is a different op and not this one's destination change.

    `allow_m_le_n` opens the other core-grid orientation (`transpose_core_grid` false), which the
    AttentionPairBias shapes need and the triangle attention has never run.

    `refuse` is the caller's own precondition, recorded in this site's reject dict so a guard that
    lives at the call site still says why it fired.
    """
    if not enabled(site):
        return None
    stats, _ = _SITES[site]
    shape = [int(d) for d in x.shape]
    if refuse:
        return _reject(refuse, shape, site)
    if head_dim % TILE or n_heads * head_dim * 3 != int(w.shape[-1]):
        return _reject("head_dim_or_width", shape, site)
    if not _common_ok(x, w, dtype):
        return _reject("dtype_or_memory", shape, site)
    if bias is not None and (bias.dtype != ttnn.bfloat16
                             or bias.memory_config().buffer_type != ttnn.BufferType.DRAM
                             or int(bias.padded_shape[-1]) != int(w.shape[-1])):
        return _reject("bias_dtype_or_width", shape, site)
    # The descriptor is a transcription of the factory for the shipped block entry only.
    if mm_config is None:
        return _reject("no_mm_config", shape, site)
    from .tenstorrent import _mm_block_for, COMPUTE_GRID_MAIN
    blk = _mm_block_for(w)
    if blk is None:
        return _reject("no_block_entry", shape, site)

    pad = [int(d) for d in x.padded_shape]
    if _m_rows(x) <= int(w.shape[-1]) and not allow_m_le_n:
        # transpose_core_grid is false there, a core-grid orientation this has never been run on
        return _reject("m_le_n", shape, site)

    dev = x.device()
    outs = [ttnn.allocate_tensor_on_device(
        ttnn.Shape([shape[0], n_heads, shape[1], head_dim]), ttnn.bfloat16, ttnn.TILE_LAYOUT,
        dev, ttnn.DRAM_MEMORY_CONFIG) for _ in range(3)]
    defines = {"HEAD_MAJOR_MT": pad[-2] // TILE}
    if head_dim != TILE:
        defines["HEAD_MAJOR_DT"] = head_dim // TILE
    G.generic_minimal_matmul(
        dev, x, w, outs, (blk, tuple(COMPUTE_GRID_MAIN)), G.ckc_args(ckc),
        defines, KERNEL_DIR, bias=bias)
    stats[0] += 1
    return tuple(outs)


# --- the atom block's split, where head-major and plain coincide for q ----------------------------
#
# The remaining 0.09016 s / 121.7 Mcycles of the tail screen's lead, 1200 programs per 512 aa fold.
# The shipped chain is six ops for a split that computes nothing:
#
#     q  = linear(s, Wq, bq)                 [B, K, 32, 128]
#     kv = linear(s_kv, Wkv)                 [B, K, 128, 256]
#     pad q from 32 rows to ATOM_DIM=128     so nlp_create_qkv_heads sees matching row counts
#     reshape q, kv to [B*K, 1, 128, .]
#     nlp_create_qkv_heads(q, kv)            3 x [B*K, 4, 128, 32]
#     q = q[:, :, :ATOM_WINDOW, :]           throws the 96 padded rows straight back away
#
# The pad and the slice exist ONLY to satisfy the split's requirement that q and kv carry the same
# row count. A destination written per head does not have that requirement, so both go with it.
#
# The q half needs no tile transform at all, and that is worth stating because it is not obvious:
# the atom window is 32 rows, exactly ONE row tile, so `HEAD_MAJOR_MT` is 1 and the head-major
# expression collapses to `row * logical_d1 + tidx` -- the plain writer's own. `[B, K, 32, 128]` and
# `[B*K, 4, 32, 32]` are the SAME 560-tile buffer in the same order: tile (window, col) of the first
# holds window `w`, rows 0..31, channels 32*col..32*col+31, and tile (window, head) of the second
# holds window `w`, rows 0..31, head `col`'s 32 channels. Only the ttnn Shape differs. So q is
# head-major by choosing its destination's shape, with no define and no kernel involved.
# (`ttnn.reshape` cannot do this: it reads the last dim going 128 -> 32 as a re-tiling and moves the
# data, which is what makes L2's merge expensive. Writing the destination is what is free.)
#
# The kv half does need the transform, at `HEAD_MAJOR_MT = ATOM_DIM / 32 = 4`, and that is the shape
# `perf/c12_diffusion_head/tile_map.py` checks as `atom_kv_512aa`.
#
# Separate flag from the token/trunk site: same mechanism, but two projections instead of one, two
# op-class changes instead of one, and its own deleted pad and slice, so it earns its own arm.
APB_ATOM_HEAD_MAJOR_QKV = False
_ATOM_ENABLED = os.environ.get(
    "TT_BIO_APB_ATOM_HEAD_MAJOR_QKV", "1" if APB_ATOM_HEAD_MAJOR_QKV else "0") == "1"

ATOM_STATS = [0, 0]
ATOM_REJECTS: dict = {}


def atom_enabled():
    """Whether the atom-block gate is on, so the caller can skip building two configs per call."""
    return _ATOM_ENABLED


def _atom_reject(reason, shape):
    k = (reason, tuple(shape))
    ATOM_REJECTS[k] = ATOM_REJECTS.get(k, 0) + 1
    ATOM_STATS[1] += 1
    return None


def atom_qkv_heads(s, w_q, b_q, s_kv, w_kv, ckc, n_heads, head_dim, dtype, cfg_q, cfg_kv,
                   refuse=None):
    """The atom block's q and kv projections written head-major, or `None` to leave all six ops.

    Returns `(q, k, v)`, q as `[B, K * n_heads, W, head_dim]` and k, v as
    `[B, K * n_heads, ATOM_DIM, head_dim]` -- exactly the three tensors the shipped chain hands
    `_attention`, minus the pad and the slice.
    """
    if not _ATOM_ENABLED:
        return None
    shape = [int(d) for d in s.shape]
    if refuse:
        return _atom_reject(refuse, shape)
    if len(shape) != 4 or len(s_kv.shape) != 4:
        return _atom_reject("rank", shape)
    if head_dim % TILE or n_heads * head_dim != int(w_q.shape[-1]) \
            or n_heads * head_dim * 2 != int(w_kv.shape[-1]):
        return _atom_reject("head_dim_or_width", shape)
    if not (_common_ok(s, w_q, dtype, rank=4) and _common_ok(s_kv, w_kv, dtype, rank=4)):
        return _atom_reject("dtype_or_memory", shape)
    if b_q is not None and (b_q.dtype != ttnn.bfloat16
                            or b_q.memory_config().buffer_type != ttnn.BufferType.DRAM
                            or int(b_q.padded_shape[-1]) != int(w_q.shape[-1])):
        return _atom_reject("bias_dtype_or_width", shape)
    if cfg_q is None or cfg_kv is None:
        return _atom_reject("no_mm_config", shape)

    from .tenstorrent import _mm_block_for, COMPUTE_GRID_MAIN
    blk_q, blk_kv = _mm_block_for(w_q), _mm_block_for(w_kv)
    if blk_q is None or blk_kv is None:
        return _atom_reject("no_block_entry", shape)

    if shape[-2] % TILE or int(s_kv.shape[-2]) % TILE:
        # A ragged row count would make the destination's rows and the matmul's M disagree
        return _atom_reject("rows_not_whole_tiles", shape)
    if _m_rows(s) <= int(w_q.shape[-1]) or _m_rows(s_kv) <= int(w_kv.shape[-1]):
        # transpose_core_grid false, the orientation the token/trunk site runs and this one does not
        return _atom_reject("m_le_n", shape)

    dev = s.device()
    b, k, w = shape[0], shape[1], shape[-2]
    rows_kv = int(s_kv.shape[-2])
    dt = head_dim // TILE

    # q is ONE output, and at N_chunks == 1 the kernel takes `write_block_sync`, which reads
    # `MM_OUT_TILE_ID` -- a different macro from the split writer's `MM_SPLIT_TILE_ID`. So q's
    # define is HEAD_MAJOR_OUT_MT, the one `gate_proj` uses for its single output, and not
    # HEAD_MAJOR_MT, which that writer never looks at. Passing the wrong one is inert rather than
    # loud, and at the atom window it would even give the right answer by accident, because MT is 1
    # there and the head-major id IS the plain id.
    if dt != 1:
        # Only MM_SPLIT_TILE_ID was generalised to a multi-tile head; MM_OUT_TILE_ID carries no DT,
        # so a single-output head-major write is only expressible at one tile per head.
        return _atom_reject("multi_tile_head_on_single_output", shape)

    # kv is TWO outputs, so that call takes the split writer and HEAD_MAJOR_MT is the define it
    # reads. No HEAD_MAJOR_DT: the guard above already refused a multi-tile head here.
    def split_defines(rows):
        return {"HEAD_MAJOR_MT": rows // TILE}

    def alloc(rows):
        return ttnn.allocate_tensor_on_device(
            ttnn.Shape([b * k, n_heads, rows, head_dim]), ttnn.bfloat16, ttnn.TILE_LAYOUT,
            dev, ttnn.DRAM_MEMORY_CONFIG)

    q = alloc(w)
    G.generic_minimal_matmul(dev, s, w_q, [q], (blk_q, tuple(COMPUTE_GRID_MAIN)), G.ckc_args(ckc),
                             {"HEAD_MAJOR_OUT_MT": w // TILE}, KERNEL_DIR, bias=b_q)
    kv = [alloc(rows_kv), alloc(rows_kv)]
    G.generic_minimal_matmul(dev, s_kv, w_kv, kv, (blk_kv, tuple(COMPUTE_GRID_MAIN)),
                             G.ckc_args(ckc), split_defines(rows_kv), KERNEL_DIR)

    out = [ttnn.reshape(t, (b, k * n_heads, int(t.shape[-2]), head_dim)) for t in (q, *kv)]
    ATOM_STATS[0] += 1
    return tuple(out)


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
        ttnn.Shape([shape[0], n_heads, shape[1], head_dim]), ttnn.bfloat16, ttnn.TILE_LAYOUT,
        dev, ttnn.DRAM_MEMORY_CONFIG)
    G.generic_minimal_matmul(
        dev, x, w_g, out, (_mm_block_for(w_g), tuple(COMPUTE_GRID_MAIN)),
        G.ckc_args(ckc), {"HEAD_MAJOR_OUT_MT": pad[-2] // TILE}, KERNEL_DIR)
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
        ttnn.Shape([shape[0], n_heads, shape[1], head_dim]), ttnn.bfloat16, ttnn.TILE_LAYOUT,
        dev, ttnn.DRAM_MEMORY_CONFIG) for _ in range(4)]
    G.generic_minimal_matmul(
        dev, x, w, outs, (blk, tuple(COMPUTE_GRID_MAIN)), G.ckc_args(ckc),
        {"HEAD_MAJOR_MT": pad[-2] // TILE}, KERNEL_DIR)
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
    from .tenstorrent import _mm_block_for, COMPUTE_GRID_MAIN
    B, H, S, D = (int(d) for d in gated.shape)
    pad = [int(d) for d in gated.padded_shape]
    dev = gated.device()
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([B, S, int(w.shape[-1])]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev,
        memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG)
    G.generic_minimal_matmul(
        dev, gated, w, out, (_mm_block_for(w), tuple(COMPUTE_GRID_MAIN)),
        G.ckc_args(ckc), {"HEAD_MAJOR_IN0_MT": pad[-2] // TILE}, KERNEL_DIR,
        m_k=(pad[0] * pad[-2], H * D))
    return out


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
        ttnn.Shape([shape[0], n_heads, shape[1], head_dim]), ttnn.bfloat16, ttnn.TILE_LAYOUT,
        dev, ttnn.DRAM_MEMORY_CONFIG) for _ in range(4)]
    outs.append(ttnn.allocate_tensor_on_device(
        ttnn.Shape([shape[0], shape[1], bias_channels]), ttnn.bfloat16, ttnn.TILE_LAYOUT,
        dev, ttnn.DRAM_MEMORY_CONFIG))
    G.generic_minimal_matmul(
        dev, x, w, outs, (blk, tuple(COMPUTE_GRID_MAIN)), G.ckc_args(ckc),
        {"HEAD_MAJOR_MT": pad[-2] // TILE}, KERNEL_DIR, n_widths=[c // TILE] * 4 + [1])
    QKVGB_STATS[0] += 1
    QKVG_STATS[0] += 1
    STATS[0] += 1
    TAIL_STATS[0] += 1
    return tuple(outs[:3]), outs[3], outs[4]
