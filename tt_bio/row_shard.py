"""Row-wise (i-axis) sharding of a bucketed token axis across several devices.

A pair track is ``[1, N, N, C]``. Splitting it on the i axis gives each device a contiguous
slab of output rows; every device still executes the model's own arithmetic exactly once, so
this is a partition of work, never a reduction of it.

The one constraint that shapes the whole module is that **the token axis is bucketed to a
multiple of 32** (:data:`tt_bio.token_axis.TOKEN_BUCKET`), and the sharded ops require a
tile-aligned sub-range, so every slab boundary must also be a multiple of 32. When the axis is
an odd number of tiles the slabs cannot be equal and the split is unbalanced by one tile. That
costs load balance, and :func:`row_shard_ceiling` says exactly how much.

Padding the axis up to a multiple of ``32 * n_shards`` to make the split even is worse, not
better: the pair track is O(N^2), so at N=288 padding to 320 costs 1.235x of compute against
the 1.111x of imbalance it removes. The axis multiple stays 32 and the slabs stay ragged.

Nothing here knows a model name. It gates on axis length, alignment and shard count.
"""

from __future__ import annotations

from .token_axis import TOKEN_BUCKET

__all__ = ["row_shard_bounds", "row_shard_ceiling", "shard_rows", "gather_rows"]


def row_shard_bounds(
    n_rows: int, n_shards: int, align: int = TOKEN_BUCKET
) -> tuple[tuple[int, int], ...]:
    """Contiguous ``(start, stop)`` row slabs, every boundary a multiple of ``align``.

    ``n_rows`` must already be a multiple of ``align`` -- that is what bucketing the token axis
    guarantees. Tiles are handed out as evenly as they divide; the leading slabs take the
    remainder, so a 9-tile axis split two ways is 5 tiles and 4 tiles, not 4.5 and 4.5.

    Raises ``ValueError`` if the axis is too short to give every shard at least one tile, which
    is a real limit and not a rounding case: a 32-row track cannot be split at all.
    """
    if n_shards < 1:
        raise ValueError(f"n_shards must be >= 1, got {n_shards}")
    if align < 1 or n_rows % align:
        raise ValueError(
            f"{n_rows} rows is not a multiple of the {align}-row alignment; bucket the token "
            "axis before sharding it"
        )
    tiles = n_rows // align
    if tiles < n_shards:
        raise ValueError(
            f"{n_rows} rows is {tiles} tile(s) of {align}, too few to give {n_shards} shards a "
            "tile each; this axis is not shardable that many ways"
        )
    base, extra = divmod(tiles, n_shards)
    out, start = [], 0
    for k in range(n_shards):
        stop = start + (base + (1 if k < extra else 0)) * align
        out.append((start, stop))
        start = stop
    return tuple(out)


def row_shard_ceiling(n_rows: int, n_shards: int, align: int = TOKEN_BUCKET) -> float:
    """Best speedup an i-axis shard can reach on a perfectly parallel, gather-free op.

    ``n_rows / max_slab``: the shard runs as slow as its largest slab. 2.000x at N=512 two ways,
    1.800x at N=288, 1.000x at N=32. Communication is not in here -- this is the load-balance
    ceiling alone, and a real shard sits below it.
    """
    bounds = row_shard_bounds(n_rows, n_shards, align)
    return n_rows / max(r1 - r0 for r0, r1 in bounds)


def shard_rows(t, bounds, dim: int = 1):
    """Slice ``t`` into the slabs ``bounds`` names, along ``dim``. Works on torch and ttnn alike."""
    return tuple(t[(slice(None),) * dim + (slice(r0, r1),)] for r0, r1 in bounds)


def gather_rows(slabs, dim: int = 1):
    """Concatenate row slabs back into the full axis. The inverse of :func:`shard_rows`."""
    import torch

    return torch.cat(list(slabs), dim=dim)


# --- the Pairformer block, run as row slabs -----------------------------------------------------
#
# Each entry says what one op of the residual chain needs on the i axis:
#
#   FULL  the op reads rows it does not own -- a chip holding only its slab cannot compute its
#         own outputs, so the whole track has to be brought together before the op runs.
#   ROWS  the op's output row i depends only on input row i, so the op runs on the slab alone.
#
# The chain is z = z + op(z), five times, so an op that writes only its own rows leaves every
# other chip's copy stale. A FULL op therefore forces one gather of the *update* immediately
# before it, and consecutive FULL ops cannot share one. `block_split_bitexact.py` does not take
# these labels on trust: it demotes each FULL to ROWS in turn and requires the run to break.
#
# Why each op lands where it does, at tt_bio/tenstorrent.py:
#
#   trimul_start   its second operand is built from EVERY row (perm_b at 5773, matmul 5898):
#                  out[i,j] = sum_k a[i,k] b[j,k], and b[j] is row j of the same tensor.
#   trimul_end     a itself takes a COLUMN slab (a_axis = 2 at 5630), so a chip holding rows
#                  needs the other chip's rows on both sides.
#   triatt_start   q, k, v and the gate all come off the row slab. Only the triangle BIAS spans
#                  every row (built at 6553 from the whole normed tensor) -- and it is a
#                  4-channel projection, 2.10 MB against the track's 67.1 MB. This op is
#                  therefore row-local in everything but a tensor 32x smaller than the one it is
#                  charged for, and it is FULL here only because `TriangleAttention` builds its
#                  own bias and has no parameter to accept one. See `gathers_per_block`.
#   triatt_end     the qkv projection runs on the WHOLE tensor (6838) because k and v need every
#                  i; only q takes the slab. Two pair transposes bracket it.
#   transition_z   a SwiGLU per (i, j). Nothing crosses.
#
# The single track is a sixth consumer of z and it is ROW-LOCAL in z: AttentionPairBias reads
# bias[h,i,j] out of row i (7250), so a chip needs only its own rows. What it does need is all of
# s, which is [1, N, 384] = 0.39 MB, three orders of magnitude below a pair-track gather.
PAIR_CHAIN = (
    ("triangle_multiplication_start", "FULL"),
    ("triangle_multiplication_end", "FULL"),
    ("triangle_attention_start", "FULL"),
    ("triangle_attention_end", "FULL"),
    ("transition_z", "ROWS"),
)


def gathers_per_block(chain=PAIR_CHAIN, injectable_bias: bool = False) -> int:
    """Full-pair-track gathers one block costs in steady state.

    A block is entered with every chip fresh only on its own rows, so the leading FULL op pays a
    gather too, and that same gather is the previous block's closing one. With the chain as
    shipped that is FOUR. With ``injectable_bias``, `triatt_start` drops to a 2.10 MB gather of
    the triangle bias instead of 67.1 MB of z and the count is THREE -- which is the number
    `b2z2-dual-chip-fold` derived, reached by a different route than that derivation used.

    Below three is not available with today's primitives. Each of trimul_start, trimul_end and
    triatt_end needs every row at full channel width, and each is separated from the last by a
    residual update that every chip computes on its own half only. Removing one means either
    computing the other half twice, which is not a speedup, or turning the gather into a
    reduction -- a k-split of trimul_end, a flash-style key-split of triatt_end. Both move the
    same bytes AND regroup the accumulation, so both give up the bit-exactness that makes this
    shard free of an accuracy argument.
    """
    n = sum(1 for _, need in chain if need == "FULL")
    return n - 1 if injectable_bias else n


def pairformer_block_sharded(layer, s, z, bounds, *, mask=None, attn_mask_start=None,
                             attn_mask_end=None, extra_attn_bias=None, gather=None,
                             chain=PAIR_CHAIN):
    """One ``PairformerLayer``, with the pair track split into the row slabs ``bounds``.

    Same arithmetic as ``layer(s, z, ...)``, partitioned: every slab computes its own output
    rows and nothing else. ``gather`` rebuilds a full-height tensor out of per-slab pieces. On
    one device that is a concat and this function is a simulator for the split; on a mesh it is
    ``ttnn.all_gather(dim=1)`` and the same code is the shard. Nothing else changes, which is
    the point: the correctness of the split is settled without a second chip.

    The single track is run replicated, unsharded. It is a sixth consumer of z and it is row-local
    in z, but it has no query slab yet, so this function leaves it whole and honest.
    """
    import ttnn

    if gather is None:
        def gather(parts):
            return ttnn.concat(list(parts), dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    # One device standing in for several runs the slabs one after another and therefore holds all
    # of them at once, which a real shard never does -- each chip computes exactly one. Two 256-row
    # slabs of a 512 aa pair track are 33.5 MB each and an op that returns L1 leaves the next slab's
    # circular buffers nowhere to live. Staging each finished slab to DRAM is a memory move, not
    # arithmetic, so it costs the simulation nothing it is measuring.
    simulated = len(bounds) > 1

    def keep(t):
        if not simulated or t.memory_config() == ttnn.DRAM_MEMORY_CONFIG:
            return t
        moved = ttnn.to_memory_config(t, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(t)
        return moved

    args = {
        "triangle_multiplication_start": (mask,),
        "triangle_multiplication_end": (mask,),
        "triangle_attention_start": (attn_mask_start,),
        "triangle_attention_end": (attn_mask_end,),
        "transition_z": (),
    }
    for name, need in chain:
        op = getattr(layer, name)
        extra = args[name]
        parts = []
        for r0, r1 in bounds:
            if need == "FULL":
                parts.append(keep(op(z, *extra, row_slab=(r0, r1))))
            else:
                shp = [int(d) for d in z.shape]
                rows = ttnn.slice(z, [0, r0, 0, 0], [shp[0], r1, shp[2], shp[3]],
                                  memory_config=ttnn.DRAM_MEMORY_CONFIG)
                try:
                    parts.append(keep(op(rows, *extra)))
                finally:
                    ttnn.deallocate(rows)
        update = gather(parts)
        for p in parts:
            ttnn.deallocate(p)
        z = ttnn.add_(z, update)
        ttnn.deallocate(update)

    if layer.transform_s:
        s_norm = ttnn.layer_norm(
            s, weight=layer.pre_norm_s_weight, bias=layer.pre_norm_s_bias, epsilon=1e-5,
            compute_kernel_config=layer.compute_kernel_config,
        )
        s_update = layer.attention_pair_bias(
            s_norm, z,
            seq_mask=extra_attn_bias if extra_attn_bias is not None else attn_mask_start,
        )
        ttnn.deallocate(s_norm)
        s = ttnn.add_(s, s_update)
        ttnn.deallocate(s_update)
        s_update = layer.transition_s(s)
        s = ttnn.add_(s, s_update)
        ttnn.deallocate(s_update)
    return s, z
