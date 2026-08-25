"""Block-sparse atom attention for RFD3.

The atom site's 128-neighbour index is not row-sparse, it is BLOCK-sparse. The union of the
neighbour sets of Q neighbouring query rows is much narrower than the full key axis -- at
Q=1216 the median union over a 200-step schedule is 3296 of 6080 -- so the site is a batched
DENSE matmul over ``[nb, Q, U]``: stock ops, tile aligned, no per-row gather and no custom
kernel. Every earlier route at this site treated the index as an arbitrary per-row gather and
measured dead (a shared key block is not gathered attention; an honest per-row gather is 7.2x
slower than the dense chain; ``ttnn.gather`` is silently wrong above 1920 on the indexed axis).

Two constraints decide whether this is a win, and both were found by measurement:

* **Q must be a multiple of 32.** The blocked scores come out ``[H, nb, Q, U]`` and have to be
  seen as ``[1, H, nb*Q, U]`` for the bias kernel and the softmax. That reshape is free when Q
  is a whole number of tile rows and a real re-tile when it is not, worth 4-5 ms on a 6.5 ms
  chain. So the candidate block sizes are the divisors of the padded query axis that are also
  multiples of 32, not the divisors.
* **U is baked into the program, so it is bucketed.** The union is per step and its tail is
  wide: over a full schedule it reaches 6048 of 6080, i.e. steps with no block sparsity at all.
  A single compiled width would have to be that worst case, where the chain costs more than the
  dense one it replaces. Instead a step picks the narrowest bucket that fits it, and a step too
  wide for the widest bucket runs the shipped dense chain -- so the downside is capped at dense
  by construction rather than by tuning.

**Together those two make this arm target-specific, which is the reason it is off by default and
not merely unproven.** Q has to divide the tile-padded atom axis, so with Q=1216 (38 tiles) the arm
fires only when ``ceil(atoms/32) % 38 == 0``: 288 of the 11745 atom counts between 256 and 12000,
2.45%. Every other target takes the dense fallback on every step. `examples/rfd3_binder.json`, the
release gate's own RFD3 fixture, is one of them -- 1350 atoms, 1376 padded, and 1376 = 2**5 * 43
with 43 prime, so its only multiple-of-32 block sizes are 32 and 1376 and neither is usable. The
gate therefore passes with ``RFD3_BLOCK_SPARSE=1`` having scored the dense chain twice; measured,
`0 blocked, 1791 dense-fallback`. ``tests/test_rfd3_block_sparse.py`` pins both facts. U is
per-target for the same reason, since the union scales with the atom count.

Worth, on a target it does fit (R4, 6051 atoms, 6080 = 5 x 1216): **2.982 s/design**, median of 7
rounds over two interleaved fold A/Bs, range 2.218-3.988, against an A/A control floor of 0.124
(0.053-0.196). Every A/B round was flagged by the harness's load bar and the A/A rounds were not,
so read that as a bracket; the distributions do not overlap and the contamination biases the ON arm
the safe way, since only the ON arm does host work in :func:`plan`. Dense fall-back rate over a
full 200-step schedule: 360 of 1791 = 20.1%.

Not bit-exact, and the reason is only the softmax's reduction order. The scores gather exactly
(the QK dot is one tile deep, so its dot-product tree does not depend on the M/N tiling) and the
non-neighbour columns are exact post-exp zeros either way, because ``exp(-1e4 * head_dim**-0.5)``
underflows fp32. What changes is that the row sum reduces U terms instead of the full key axis,
so the 128 real terms are re-associated.

Off by default. ``RFD3_BLOCK_SPARSE=1`` or :func:`set_enabled`.
"""
import os

import torch
import ttnn

from .. import rfd3_bias, softmax_generic
from .tiles import TILE, align_tile, pad_axis

_ENABLED = os.environ.get("RFD3_BLOCK_SPARSE", "0") == "1"
#: Query rows per block. Must be a multiple of 32 and must divide the padded query axis, or the
#: step falls back to dense. 1216 is the measured optimum at 6051 atoms (6080 padded): it beats
#: 608 and 3040 once U is sized by the schedule rather than by one sampled step, because the
#: gathered row count is nb*U and so the gather grows with the block count.
_Q_BLOCK = int(os.environ.get("RFD3_BLOCK_SPARSE_Q", "1216"))
#: Compiled key widths, narrowest first. Chosen on one schedule and validated on a held-out one
#: (+3.787 in sample, +3.744 out of sample, 1.2 % overfit). These are specific to a 6080-row
#: query axis -- a different atom count needs its own set, so this is per-target tuning and not
#: a constant.
_BUCKETS = (3264, 3488, 4224)

#: ``[blocked calls, dense-fallback calls, shipped calls]``. A silent arm is what made an earlier
#: batching result wrong, so every branch counts itself: the on arm must show ``shipped == 0``
#: and the off arm ``blocked + fallback == 0``, at the same total.
STATS = [0, 0, 0]


def enabled() -> bool:
    return _ENABLED


def set_enabled(on):
    """Toggle from a screen without going through the environment. Returns the previous value."""
    global _ENABLED
    was = _ENABLED
    _ENABLED = bool(on)
    return was


def set_config(q_block=None, buckets=None):
    """Override the block size and the bucket set, for a sweep. Returns the previous pair."""
    global _Q_BLOCK, _BUCKETS
    was = (_Q_BLOCK, _BUCKETS)
    if q_block is not None:
        _Q_BLOCK = int(q_block)
    if buckets is not None:
        _BUCKETS = tuple(sorted(int(b) for b in buckets))
    return was


def config():
    return _Q_BLOCK, _BUCKETS


def stats_line():
    return "block-sparse atom attention: %d blocked, %d dense-fallback, %d shipped" % tuple(STATS)


if os.environ.get("RFD3_BLOCK_SPARSE_STATS", "0") == "1":
    # The release gate runs its fold in a subprocess, so the counters above are invisible to the
    # harness that set the flag: a run where RFD3_BLOCK_SPARSE never reached the child looks
    # exactly like a passing on-arm run. Print them at exit, the same idiom as
    # rfd3_bias's RFD3_BIAS_STATS, so the arm can be read off the fold's own log.
    import atexit

    atexit.register(lambda: print(stats_line(), flush=True))


def plan(indices, n_key, q_block=None, buckets=None):
    """Host side of one step's plan, or ``None`` when the step does not qualify.

    ``indices`` is the ``[B, L, K]`` neighbour index and ``n_key`` the padded key axis. Returns
    ``(nb, q_block, u_width, gather, pos)`` where ``gather`` is ``[nb, u_width]`` of key rows
    (unused slots read row 0, whose bias is -1e4 and whose weight is therefore exactly 0) and
    ``pos`` is ``[nb*q_block, K]`` giving every neighbour's column inside its own block.

    ``None`` means run the shipped dense chain: a batch this arm does not cover, a query axis
    the block size does not divide, or a union wider than the widest bucket.

    The union comes off a ``[nb, n_key]`` bool mask, not a per-block ``torch.unique``. The mask's
    row sums are the union widths, its nonzero columns are the union itself, and
    ``(cumsum - 1)`` gathered at a neighbour is that neighbour's block-local column. Byte-identical
    to the sort and 8.3x cheaper -- 2.0 ms/step against 16.8 -- which matters because this runs
    once per diffusion step and host cost here is additive. The sort form would have cost 3.35
    s/design against a device prize of 4.19.
    """
    q_block = _Q_BLOCK if q_block is None else int(q_block)
    buckets = _BUCKETS if buckets is None else tuple(sorted(buckets))
    if indices.shape[0] != 1:
        return None                       # one index per sample; this arm covers batch 1
    if q_block % TILE or not buckets:
        return None
    length, n_neigh = indices.shape[1], indices.shape[2]
    nb_rows = align_tile(length)
    if nb_rows % q_block:
        return None
    nb = nb_rows // q_block
    idx = indices[0].cpu().long()
    # Pad the query axis by repeating the last row. Those rows are sliced off the output, so what
    # they attend to is irrelevant -- but they must be valid indices, not whatever zeros would
    # make of a masked row.
    if nb_rows != length:
        idx = torch.cat([idx, idx[-1:].expand(nb_rows - length, n_neigh)], 0)
    blk = idx.reshape(nb, q_block * n_neigh)
    mask = torch.zeros(nb, n_key, dtype=torch.bool)
    mask.scatter_(1, blk, True)
    u_max = int(mask.sum(1).max())
    fit = [b for b in buckets if b >= u_max]
    if not fit:
        return None                       # no compiled width covers this step: dense fallback
    u_width = fit[0]
    rank = mask.cumsum(1) - 1
    pos = rank.gather(1, blk).reshape(nb * q_block, n_neigh)
    gather = torch.zeros(nb, u_width, dtype=torch.int64)
    nz = mask.nonzero()
    gather[nz[:, 0], rank[nz[:, 0], nz[:, 1]]] = nz[:, 1]
    return nb, q_block, u_width, gather, pos


def gather_index(gather, n_head, n_key, device):
    """Flat head-offset gather index for ``ttnn.embedding`` over head-major key rows.

    ``kk`` arrives ``[1, H, n_key, head_dim]``, which is already ``[H*n_key, head_dim]``
    contiguous, so offsetting block ``b``'s key rows by ``h * n_key`` lands the gathered rows
    directly in ``[H, nb, U, head_dim]`` with no permute on either side of the gather.
    """
    heads = (torch.arange(n_head, dtype=torch.int64) * n_key).view(n_head, 1, 1)
    flat = (gather.unsqueeze(0) + heads).reshape(1, -1).to(torch.int32).contiguous()
    return ttnn.from_torch(flat, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn.uint32)


def attention(qq, kk, vv, pair_bias, pos_rm, gather_dev, nb, q_block, u_width, scale, dt, ckc):
    """The blocked chain, replacing dense scores + bias + softmax + value matmul.

    Returns ``[1, H, length, head_dim]`` -- the same thing the dense chain hands
    ``_merge_heads`` -- so the caller's tail is untouched.
    """
    n_head, n_key, head_dim = kk.shape[1], kk.shape[2], kk.shape[3]
    length = qq.shape[2]
    nb_rows = nb * q_block

    def gath(x):
        rows = ttnn.reshape(ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT), (n_head * n_key, head_dim))
        g = ttnn.embedding(gather_dev, rows, layout=ttnn.TILE_LAYOUT,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(rows)
        return ttnn.reshape(g, (n_head, nb, u_width, head_dim))

    kg, vg = gath(kk), gath(vv)
    qb = ttnn.reshape(pad_axis(qq, nb_rows, 2, 0.0), (n_head, nb, q_block, head_dim))
    kgt = ttnn.permute(kg, (0, 1, 3, 2))
    ttnn.deallocate(kg)
    scores = ttnn.matmul(qb, kgt, compute_kernel_config=ckc)
    ttnn.deallocate(kgt)
    # One kernel writes `scores * scale + bias` in fp32 straight from the bf16 scores and the
    # compact neighbour bias, so the U-wide fp32 bias never exists in DRAM. It is the same kernel
    # the shipped sparse path uses; only the index differs, because `pos_rm` addresses the block's
    # own columns instead of the full key axis.
    scores = ttnn.reshape(scores, (1, n_head, nb_rows, u_width))
    scores = rfd3_bias.fused_scores_bias_fp32(scores, pair_bias, pos_rm, scale)
    weights = softmax_generic.softmax_bf16(scores, dt)
    ttnn.deallocate(scores)
    weights = ttnn.reshape(weights, (n_head, nb, q_block, u_width))
    out = ttnn.matmul(weights, vg, compute_kernel_config=ckc)
    ttnn.deallocate(weights)
    ttnn.deallocate(vg)
    out = ttnn.reshape(out, (1, n_head, nb_rows, head_dim))
    if nb_rows != length:
        out = ttnn.slice(out, [0, 0, 0, 0], [1, n_head, length, head_dim])
    return out
