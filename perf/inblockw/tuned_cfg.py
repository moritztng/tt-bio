"""Reference implementation of the `in0_block_w` matmul config, validated on qb2 card 2.

`ttnn.linear(core_grid=...)` makes ttnn derive `in0_block_w = 1`: one K tile per inner block, so
the weight block is re-multicast and the fp32 accumulator re-primed once per K tile, with too
little compute in a block to cover either. Choosing the block width instead is worth 1.12x-1.93x
on the shapes tt-bio runs at 298 aa, with rmsd against an fp32 torch reference unchanged in the
first four decimals (and never worse).

`core_grid` and `program_config` are mutually exclusive: `matmul_device_operation.cpp:1509`
asserts `!(has_user_grid && has_program_config)`. A call site takes one or the other.

Validation: `perf/inblockw/validate.py`, 15 shapes, full 1D + 2D ladder, production compute kernel
config (HiFi4 / fp32_dest_acc_en / packer_l1_acc). The config this returns is within 1.5%-7.6% of
the fastest config on either ladder wherever it fires.
"""
from functools import lru_cache

import ttnn

_FP32_TILE = 4096          # fp32 accumulation tile: fp32_dest_acc_en + packer_l1_acc
_L1_SLACK = 192 * 1024     # semaphores, dispatch, and the rest of the block's live allocations
_BW_CAP = 16               # past 16 K tiles the ladder is flat and L1 gets tight
_MIN_TILE_MACS = 12288     # below this the op is under ~0.06 ms and the lever is unmeasurable


def _largest_divisor(n, cap):
    return max((d for d in range(min(cap, n), 0, -1) if n % d == 0), default=1)


def _subblock(h, w):
    """out_subblock_h * out_subblock_w <= 4: with fp32_dest_acc_en the DEST file holds 4 tiles."""
    sh = _largest_divisor(h, 4)
    return sh, _largest_divisor(w, max(1, 4 // sh))


@lru_cache(maxsize=None)
def _tuned_matmul_config(mt, kt, nt, elem_bytes, grid, l1):
    """Program config for an (mt x kt) @ (kt x nt) tile matmul, or None to keep ttnn's choice.

    Fires in the two regimes measured on card 2 and stays out of the way everywhere else:

    - `mt >= 4 * cores`: 1D M-split. Every core owns whole rows of M, `in1` is multicast.
      Measured 1.12x-1.93x at mt = 512 and mt = 3200, which is the whole pair track.
    - `mt <= 32`: 2D MxN split. A 1D M-split cannot fill the grid here, and `core_grid=` already
      picks 2D correctly; only the block width is wrong. Measured 1.03x-1.53x at mt = 4 and 10,
      the single track.
    - in between (mt 33..4*cores-1, the atom track): **None.** The two shapes measured there
      (mt = 56, mt = 140) were neutral or worse, and both sit under `_MIN_TILE_MACS` anyway,
      so nothing the 298 aa ledger attributes is given up by declining them.
    """
    gx, gy = grid
    cores = gx * gy
    if min(mt, kt, nt) < 1 or mt * kt * nt < _MIN_TILE_MACS:
        return None
    tile = 1024 * elem_bytes

    if mt >= 4 * cores:
        per_core_M = -(-mt // cores)
        for bw in [d for d in range(min(_BW_CAP, kt), 0, -1) if kt % d == 0]:
            for ob_w in [d for d in range(nt, 0, -1) if nt % d == 0]:
                for ob_h in [d for d in range(per_core_M, 0, -1) if per_core_M % d == 0]:
                    need = (ob_h * ob_w * (tile + _FP32_TILE) * 2
                            + (ob_h + ob_w) * bw * tile * 2 + _L1_SLACK)
                    if need > l1:
                        continue
                    sh, sw = _subblock(ob_h, ob_w)
                    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                        compute_with_storage_grid_size=(gx, gy),
                        in0_block_w=bw, out_subblock_h=sh, out_subblock_w=sw,
                        out_block_h=ob_h, out_block_w=ob_w,
                        per_core_M=per_core_M, per_core_N=nt,
                        fuse_batch=True, fused_activation=None, mcast_in0=False,
                    )
        return None

    if mt > 32:
        return None

    per_core_M = -(-mt // gy)
    per_core_N = -(-nt // gx)
    for bw in [d for d in range(min(_BW_CAP, kt), 0, -1) if kt % d == 0]:
        need = (per_core_M * per_core_N * (tile + _FP32_TILE) * 2
                + (per_core_M + per_core_N) * bw * tile * 2 + _L1_SLACK)
        if need > l1:
            continue
        sh, sw = _subblock(per_core_M, per_core_N)
        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(gx, gy),
            in0_block_w=bw, out_subblock_h=sh, out_subblock_w=sw,
            out_block_h=per_core_M, out_block_w=per_core_N,
            per_core_M=per_core_M, per_core_N=per_core_N,
            transpose_mcast=False, fused_activation=None, fuse_batch=False,
        )
    return None


def _tuned_config_for(x, w, elem_bytes=2):
    """Shape-derive the config from live ttnn tensors, or None.

    Uses PADDED shapes. A guard written against the logical length has silently disabled the
    optimisation it was written for twice in this codebase already (`_tri_att_qkv_l1_config` at
    298 aa, and W6's SDPA band guard on `q.shape[2] == 298` when the tiles are 320).
    """
    xp, wp = tuple(x.padded_shape), tuple(w.padded_shape)
    if len(wp) != 2 or len(xp) < 2:
        return None                       # per-batch weights: a real batched matmul, not this
    m = 1
    for d in xp[:-1]:
        m *= d
    if m % 32 or xp[-1] % 32 or wp[-1] % 32 or wp[0] != xp[-1]:
        return None
    from .tenstorrent import COMPUTE_GRID_MAIN   # live grid, set at device open
    return _tuned_matmul_config(m // 32, xp[-1] // 32, wp[-1] // 32, elem_bytes,
                                COMPUTE_GRID_MAIN,
                                int(ttnn.get_max_worker_l1_unreserved_size()))
