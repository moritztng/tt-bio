"""Pair-track projections ``[.., M, K] x [K, N]`` on ``experimental.minimal_matmul``.

AF2's pair projections outside the trimul in-projection and qkv/g go through ``ops.linear`` with
a core grid, and their VJP's dX through ``ttnn.matmul(transpose_b=True)``; both run ttnn's
``bmm_large_block`` at 130-150 GB/s on Blackhole. At the BindCraft 2 round's 288-token grid
(M = 82944) minimal_matmul with a swept block config runs the same products 1.2-2.2x faster and
reads closer to float64 (rel L2 0.00169 against 0.00172; not bit-identical, the K blocks fold
differently): pair-transition fc2 810 -> 367 us, the OPM out-projection 1098 -> 686 us
(perf/bcp_evo/out/mm_pair_bench.json against the block profile perf/bcp_evo/out/prof_blk).

``matmul`` returns the product or None, and None means the caller runs its own call unchanged.
minimal_matmul takes no transpose, so a ``transpose_b`` call gets the weight transposed once and
cached. The cache holds the weight as well as its transpose, because a ttnn tensor takes no weak
reference and an id alone could hand a recycled id a stale transpose. So an entry keeps its weight
on card after the model that owned it is gone, and whoever drops a model calls `forget`.
"""

from __future__ import annotations

import ttnn

from .envflags import env_flag

#: The lever. Default OFF and release-gated; armed by `bindcraft2.fast_round()`.
PAIR_MM_FUSED = env_flag("TT_BIO_PAIR_MM", False)

#: Rows below this keep ttnn's plan: the MSA and single tracks were not swept.
MIN_ROWS = 128 * 128

#: (kt, nt) -> (M_block, K_block, N_block, subblock_h, subblock_w), the fastest config of
#: mm_pair_bench at [1, 288, 288, K]. Keys not here keep ttnn's plan.
_BLOCK = {
    (4, 4): (8, 4, 1, 4, 1),
    (4, 12): (4, 4, 2, 2, 2),
    (4, 16): (4, 4, 4, 1, 4),
    (4, 32): (8, 4, 4, 1, 4),
    (12, 4): (8, 4, 1, 4, 1),
    (16, 4): (8, 8, 1, 4, 1),
    (32, 4): (8, 8, 2, 2, 2),
}

#: (calls served, calls declined), cumulative; sample at a round boundary.
STATS = [0, 0]
#: Declines by (rows, K, N, transpose_b, why), for finding the next shape worth sweeping.
DECLINED: dict = {}
_WT: dict = {}


def _transposed(w):
    hit = _WT.get(id(w))
    if hit is None or hit[0] is not w:
        hit = _WT[id(w)] = (w, ttnn.transpose(w, -2, -1))
    return hit[1]


def forget() -> None:
    """Drop every cached transpose, and with it the last reference to an evicted weight.

    `bindcraft2.TrunkPool` calls this when it evicts a checkpoint. Without it every eviction
    left that checkpoint's pair weights and their transposes on card, and a `resident=1`
    campaign on the five multimer models reloads a trunk most rounds
    (github.com/moritztng/tt-bio/issues/18). The live trunk's entries go too and are rebuilt on
    their next dX, one transpose per weight.
    """
    _WT.clear()


def config(x, w, transpose_b: bool = False):
    """The block config for ``x @ op(w)``, or None if this call is not in the swept class."""
    if (x.dtype != ttnn.bfloat16 or w.dtype != ttnn.bfloat16 or len(w.shape) != 2
            or x.layout != ttnn.TILE_LAYOUT or w.layout != ttnn.TILE_LAYOUT):
        return None
    shape = [int(d) for d in x.shape]      # ttnn.Shape does not support slicing
    K, N = (int(w.shape[1]), int(w.shape[0])) if transpose_b else (int(w.shape[0]), int(w.shape[1]))
    if shape[-1] != K or K % 32 or N % 32:
        return None
    rows = 1
    for d in shape[:-1]:
        rows *= d
    blk = _BLOCK.get((K // 32, N // 32))
    if blk is None or rows < MIN_ROWS or rows % 32:
        return None
    Mb, Kb, Nb, sh, sw = blk
    if (rows // 32) % Mb or (K // 32) % Kb or (N // 32) % Nb:
        return None
    from .tenstorrent import COMPUTE_GRID_MAIN, _mm_core_coord
    return ttnn.MinimalMatmulConfig(
        M_block_size=Mb, K_block_size=Kb, N_block_size=Nb, subblock_h=sh, subblock_w=sw,
        compute_with_storage_grid_size=_mm_core_coord(*COMPUTE_GRID_MAIN))


def _decline(x, w, transpose_b, why):
    STATS[1] += 1
    try:
        rows = 1
        for d in [int(d) for d in x.shape][:-1]:
            rows *= d
        key = (rows, int(x.shape[-1]), int(w.shape[0 if transpose_b else -1]), transpose_b, why)
    except Exception:
        key = ("?", why)
    DECLINED[key] = DECLINED.get(key, 0) + 1


#: Activations minimal_matmul applies at pack, on the fp32 DST. ReLU only: it is a sign test, so
#: relu(round(v)) == round(relu(v)) and fusing it moves no bit against a separate relu.
_ACTIVATIONS = {"relu": ttnn.UnaryOpType.RELU}


def matmul(x, w, bias=None, compute_kernel_config=None, dtype=None, transpose_b: bool = False,
           memory_config=None, activation=None):
    """``act(x @ op(w) (+ bias))`` on minimal_matmul, DRAM out, or None to leave the call alone."""
    if not PAIR_MM_FUSED:
        return None
    if activation is not None and activation not in _ACTIVATIONS:
        _decline(x, w, transpose_b, "activation")
        return None
    if memory_config is not None and memory_config.buffer_type != ttnn.BufferType.DRAM:
        _decline(x, w, transpose_b, "l1_out")
        return None
    cfg = config(x, w, transpose_b)
    if cfg is None:
        _decline(x, w, transpose_b, "class")
        return None
    try:
        out = ttnn.experimental.minimal_matmul(
            input_tensor=x, weight_tensor=_transposed(w) if transpose_b else w,
            bias_tensor=bias, compute_kernel_config=compute_kernel_config,
            fused_activation=(ttnn.UnaryWithParam(_ACTIVATIONS[activation])
                              if activation else None),
            dtype=dtype, config=cfg)
    except Exception:
        _decline(x, w, transpose_b, "refused")
        return None
    STATS[0] += 1
    return out
