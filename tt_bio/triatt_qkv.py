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


def _reject(reason, shape):
    k = (reason, tuple(shape))
    REJECTS[k] = REJECTS.get(k, 0) + 1
    STATS[1] += 1
    return None


def _common_ok(x, w, dtype):
    """The dtype, layout and memory-config conditions the transcription was verified under."""
    if dtype != ttnn.bfloat16 or x.dtype != ttnn.bfloat16 or w.dtype != ttnn.bfloat16:
        return False
    if x.layout != ttnn.TILE_LAYOUT or len(x.shape) != 3:
        return False
    xmc, wmc = x.memory_config(), w.memory_config()
    return (xmc.buffer_type == ttnn.BufferType.DRAM
            and xmc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
            and wmc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED)


def qkv_heads(x, w, ckc, n_heads, head_dim, dtype, mm_config):
    """`nlp_create_qkv_heads(minimal_matmul(x, w))` as one op, or `None` to leave it alone.

    Returns `(q, k, v)`, each `[batch, n_heads, seq, head_dim]`, byte-identical to what the two
    stock ops produce.
    """
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
        ttnn.Shape([shape[0], n_heads, shape[1], head_dim]), ttnn.bfloat16, ttnn.TILE_LAYOUT,
        dev, ttnn.DRAM_MEMORY_CONFIG) for _ in range(3)]
    G.generic_minimal_matmul(
        dev, x, w, outs, (blk, tuple(COMPUTE_GRID_MAIN)), G.ckc_args(ckc),
        {"HEAD_MAJOR_MT": pad[-2] // TILE}, KERNEL_DIR)
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
                              _PAIR_PROJ_MM)
    if not _PAIR_PROJ_MM:
        # `out` would take the DRAM `ttnn.linear` leg, which this has never been compared against
        return _tail_reject("pair_proj_mm_off", shape)
    if _mm_block_for(w_g) is None or _mm_block_for(w_o) is None:
        return _tail_reject("no_block_entry", shape)
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


def out_proj(gated, w, ckc, dtype):
    """The `out` projection reading a head-major activation: `[B, H, S, 32] -> [B, S, H*32]`."""
    from .tenstorrent import _mm_block_for, COMPUTE_GRID_MAIN
    B, H, S, D = (int(d) for d in gated.shape)
    pad = [int(d) for d in gated.padded_shape]
    dev = gated.device()
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([B, S, int(w.shape[-1])]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev,
        ttnn.DRAM_MEMORY_CONFIG)
    G.generic_minimal_matmul(
        dev, gated, w, out, (_mm_block_for(w), tuple(COMPUTE_GRID_MAIN)),
        G.ckc_args(ckc), {"HEAD_MAJOR_IN0_MT": pad[-2] // TILE}, KERNEL_DIR,
        m_k=(pad[0] * pad[-2], H * D))
    return out
