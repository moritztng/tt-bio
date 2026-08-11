"""The triangle-attention qkv projection writing q, k and v straight into head-major layout.

Today the fold runs ``minimal_matmul`` and then ``nlp_create_qkv_heads``, and the second op moves
1152 MiB per call at 93 % of the copy roof purely to reorder tiles. It does not have to exist.
``head_dim`` is 32, exactly one tile, so output tile *(i, n)* of the qkv matmul already **is** tile
*(batch i/MT, head n%8, row i%MT)* of q, k or v: no element moves inside a tile. Only the address
the writer sends the tile to changes.

So this drives the wheel's own ``minimal_matmul`` kernels through ``ttnn.generic_op``
(:mod:`tt_bio.mm_generic`), with the two DM kernels taken from ``tt_bio/kernels/triatt/`` where one
macro re-points the split writer's destination index. Transaction count, transaction size and every
arithmetic operation are unchanged, so the result is **bit-exact** -- ``torch.equal`` against
``nlp_create_qkv_heads(minimal_matmul(...))`` at 298, 320, 384, 512, 576 and 640 aa
(``perf/triatt_fused/s1_gate.json``), where it is worth 1.92-2.03x on the pair and 0.75-3.27 ms/call.

The gate below is deliberately narrow: 32-channel heads, bf16, interleaved DRAM both sides, and a
shape the shipped ``_MM_BLOCK`` entry already covers. Anything else falls through to the two stock
ops.
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


def qkv_heads(x, w, ckc, n_heads, head_dim, dtype, mm_config):
    """``nlp_create_qkv_heads(minimal_matmul(x, w))`` as one op, or ``None`` to leave it alone.

    Returns ``(q, k, v)``, each ``[batch, n_heads, seq, head_dim]``, byte-identical to what the two
    stock ops produce.
    """
    if not _ENABLED:
        return None
    shape = [int(d) for d in x.shape]
    if head_dim != TILE or n_heads * head_dim * 3 != int(w.shape[-1]):
        return _reject("head_dim_or_width", shape)
    if dtype != ttnn.bfloat16 or x.dtype != ttnn.bfloat16 or w.dtype != ttnn.bfloat16:
        return _reject("dtype", shape)
    if x.layout != ttnn.TILE_LAYOUT or len(shape) != 3:
        return _reject("layout_or_rank", shape)
    xmc, wmc = x.memory_config(), w.memory_config()
    if (xmc.buffer_type != ttnn.BufferType.DRAM
            or xmc.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED
            or wmc.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED):
        return _reject("memory_config", shape)
    # The descriptor is a transcription of the factory for the shipped block entry only.
    if mm_config is None:
        return _reject("no_mm_config", shape)
    from .tenstorrent import _MM_BLOCK, COMPUTE_GRID_MAIN
    blk = _MM_BLOCK.get(int(w.shape[-1]) // TILE)
    if blk is None:
        return _reject("no_block_entry", shape)

    pad = [int(d) for d in x.padded_shape]
    m_tiles_per_batch = pad[-2] // TILE
    M = pad[0] * pad[-2]
    if M <= int(w.shape[-1]):
        # transpose_core_grid is false there, a core-grid orientation this has never been run on
        return _reject("m_le_n", shape)

    dev = x.device()
    outs = [ttnn.allocate_tensor_on_device(
        ttnn.Shape([shape[0], n_heads, shape[1], head_dim]), ttnn.bfloat16, ttnn.TILE_LAYOUT,
        dev, ttnn.DRAM_MEMORY_CONFIG) for _ in range(3)]
    G.generic_minimal_matmul(
        dev, x, w, outs, (blk, tuple(COMPUTE_GRID_MAIN)), G.ckc_args(ckc),
        {"HEAD_MAJOR_MT": m_tiles_per_batch}, KERNEL_DIR)
    STATS[0] += 1
    return tuple(outs)
