"""A core grid for the batched matmuls that ask ttnn for no plan at all.

`bcx-p10-mmlay` ranked a BindCraft 2 gradient round's 8968 matmul calls by shape and placement
rather than by op name, and the split was clean. 98.2 % of the family's seconds runs
DRAM/bf16/TILE/INTERLEAVED on every operand, none of it is compute-bound, and the shapes that
already carry a `program_config` are within a few per cent of the best plan available to them.
The slow ones are the calls that hand `ttnn.matmul` a batched operand and nothing else: ttnn's
default planner spreads a rank-3 batch badly, and supplying any core grid at all fixes it.

Measured on pc card 0 (p150a, AICLK 1350), each point the model's OWN captured call with one
thing changed, median of 25 warm reps, PCC 1.000000 against the baseline's answer
(`perf/bcx_p10_mmlay/out/micro_grid.json`):

    batch x M x K x N   issued by               no plan     8x8 grid   speedup
    288x32x768x256 tb   MSA column attention     2.6403 ms   0.1842 ms   14.34x
    288x32x256x256 tb   MSA column attention     0.8965      0.1008       8.89x
    128x288x288x288     triangle multiplication  0.8520      0.1880       4.53x
    288x128x32x288      outer product mean       0.4843      0.1591       3.04x

The grid is flat over its own range -- 8x8, 8x13, 10x8 and 10x13 are within 5 % of each other on
every one of those shapes -- so this picks one side length rather than a per-shape table, and
the table would be worth 0.4 % more.

It is deliberately narrow, because the same sweep measured the other direction. A core grid on a
BATCH-1 matmul is worth 1.00-1.04x where it helps and 0.74x where it hurts: 1x82944x512x128
loses 26 % at its best grid. Those calls keep ttnn's planner, and the gate below is the batch.

`ttnn.matmul(core_grid=...)` selects a different program factory, one that derives
`in0_block_w = 1`, so this is not a narrowing of a tuned config -- it is the plan a call that
had none. Anything that already carries a `program_config` or a `core_grid` is left alone.
"""

from __future__ import annotations

import collections
from functools import lru_cache

import ttnn

from .envflags import env_flag, env_int

#: Release-gated, default OFF. It moves a reassociation (bf16 max abs deviation 2.0e-3 on the
#: widest shape, PCC 1.000000) and has not been through a release gate.
MM_LAYOUT = env_flag("TT_BIO_MM_LAYOUT", False)

#: Side length of the grid handed to a plan-less batched matmul, clamped to the device's own
#: compute grid. 8 is the measured best over the six shape classes the round issues.
MM_LAYOUT_SIDE = env_int("TT_BIO_MM_LAYOUT_GRID", 8)

#: Reach, so an A/B can tell a lever that did nothing from one that did not pay. `served` is a
#: call that got a grid it did not have; every `declined:*` is a call this site saw and left
#: alone, named by the reason. A round with served == 0 measured nothing.
REACH: collections.Counter = collections.Counter()


@lru_cache(maxsize=None)
def _grid(gx: int, gy: int) -> ttnn.CoreGrid:
    return ttnn.CoreGrid(y=min(MM_LAYOUT_SIDE, gy), x=min(MM_LAYOUT_SIDE, gx))


def _device_grid() -> tuple:
    # Imported here, not at module scope: `tenstorrent` imports half the package, and it is
    # what reads the real grid off the device at open time, so by the time a matmul runs this
    # is the measured grid and not the import-time guess.
    from .tenstorrent import COMPUTE_GRID_MAIN
    return COMPUTE_GRID_MAIN


def _batch(t) -> int:
    n = 1
    for d in tuple(t.shape)[:-2]:
        n *= int(d)
    return n


def _dram_interleaved(t) -> bool:
    try:
        mc = t.memory_config()
    except Exception:                                                         # noqa: BLE001
        return False
    return (mc.buffer_type == ttnn.BufferType.DRAM
            and mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED)


#: B2 (r4t-bwd / r4t-perf): explicit 2D multicast program_config for long-K `transpose_a=True`
#: bf16->fp32 weight gradients (`[K, M]^T @ [K, N]`, `K >= 8192`, `32 < Mt*Nt <= 1024`). Stock
#: ttnn.matmul plans those shapes onto 12-66 cores (6.8-30.8 TF/s); sizing the 2D grid to the
#: device compute grid with L1-safe `in0_block_w` hits 42-62 TF/s (1.96-6.52x) at equal or tighter
#: float64 rel_L2 with zero transpose and zero extra DRAM allocation.
DW_2D = env_flag("TT_BIO_DW_2D", True)
DW_2D_REACH: collections.Counter = collections.Counter()


def _pick_subblock(pm: int, pn: int, max_dst: int = 4) -> tuple[int, int]:
    best = (1, 1)
    for sh in range(1, min(pm, max_dst) + 1):
        if pm % sh:
            continue
        for sw in range(1, min(pn, max_dst // sh) + 1):
            if pn % sw:
                continue
            if sh * sw > best[0] * best[1]:
                best = (sh, sw)
    return best


@lru_cache(maxsize=None)
def _dw_2d_config(Mt: int, Nt: int, Kt: int, gx_dev: int, gy_dev: int):
    """Device-adaptive 2D multicast config for `[K, M]^T @ [K, N]` bf16->fp32."""
    best_key, best_cfg = None, None
    for tr_m in (False, True):
        gx_max, gy_max = (gy_dev, gx_dev) if tr_m else (gx_dev, gy_dev)
        gy = min(Mt, gy_max)
        gx = min(Nt, gx_max)
        pm = -(-Mt // gy)
        pn = -(-Nt // gx)
        gy = -(-Mt // pm)
        gx = -(-Nt // pn)
        sh, sw = _pick_subblock(pm, pn, 4)
        bw = 0
        for cand in (32, 16, 8, 4, 2, 1):
            if Kt % cand:
                continue
            # double-buffered bf16 in0/in1 CBs + fp32 output CB, safely under 1.5 MB L1
            cb_bytes = 2 * (pm * cand + pn * cand) * 2048 + pm * pn * 4096
            if cb_bytes <= 950_000:
                bw = cand
                break
        if not bw:
            continue
        key = (gx * gy, bw, -int(tr_m))
        if best_key is None or key > best_key:
            best_key = key
            cg = (gy, gx) if tr_m else (gx, gy)
            best_cfg = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=cg, in0_block_w=bw,
                out_subblock_h=sh, out_subblock_w=sw, out_block_h=pm, out_block_w=pn,
                per_core_M=pm, per_core_N=pn, transpose_mcast=tr_m, fused_activation=None)
    return best_cfg


def dw_2d_eligible(a, b, *, dtype=None) -> bool:
    """True when `DW_2D` will serve `a^T @ b` (no transpose, no extra DRAM)."""
    if not DW_2D or (dtype is not None and dtype != ttnn.float32):
        return False
    try:
        if (len(a.shape) != 2 or len(b.shape) != 2
                or a.dtype != ttnn.bfloat16 or b.dtype != ttnn.bfloat16
                or a.layout != ttnn.TILE_LAYOUT or b.layout != ttnn.TILE_LAYOUT):
            return False
        K, M = int(a.shape[0]), int(a.shape[1])
        Kb, N = int(b.shape[0]), int(b.shape[1])
        if K != Kb or K % 32 or M % 32 or N % 32:
            return False
        Kt, Mt, Nt = K // 32, M // 32, N // 32
        if Kt < 256 or Mt < 4 or Nt < 4 or Mt * Nt <= 32 or Mt * Nt > 1024:
            return False
        return _dram_interleaved(a) and _dram_interleaved(b)
    except Exception:  # noqa: BLE001
        return False


def _dw_2d_plan(a, b, kw: dict) -> dict:
    if not DW_2D or not kw.get("transpose_a") or kw.get("transpose_b"):
        return kw
    if kw.get("program_config") is not None or kw.get("config") is not None or kw.get("core_grid") is not None:
        return kw
    if kw.get("dtype") != ttnn.float32:
        return kw
    try:
        if (len(a.shape) != 2 or len(b.shape) != 2
                or a.dtype != ttnn.bfloat16 or b.dtype != ttnn.bfloat16
                or a.layout != ttnn.TILE_LAYOUT or b.layout != ttnn.TILE_LAYOUT):
            return kw
        K, M = int(a.shape[0]), int(a.shape[1])
        Kb, N = int(b.shape[0]), int(b.shape[1])
        if K != Kb or K % 32 or M % 32 or N % 32:
            return kw
        Kt, Mt, Nt = K // 32, M // 32, N // 32
        if Kt < 256 or Mt < 4 or Nt < 4 or Mt * Nt <= 32 or Mt * Nt > 1024:
            return kw
        if not (_dram_interleaved(a) and _dram_interleaved(b)):
            DW_2D_REACH["declined: not DRAM interleaved"] += 1
            return kw
        gx, gy = _device_grid()
        cfg = _dw_2d_config(Mt, Nt, Kt, int(gx), int(gy))
    except Exception:  # noqa: BLE001
        DW_2D_REACH["declined: error"] += 1
        return kw
    if cfg is None:
        DW_2D_REACH["declined: no config"] += 1
        return kw
    DW_2D_REACH["served"] += 1
    return {**kw, "program_config": cfg}


def plan(a, b, kw: dict) -> dict:
    """`kw` for `ttnn.matmul(a, b, **kw)`, with a core grid added where this lever applies.

    Returns `kw` itself when it does not, so an off arm pays one boolean.
    """
    kw = _dw_2d_plan(a, b, kw)
    if not MM_LAYOUT:
        return kw
    if kw.get("program_config") is not None or kw.get("config") is not None:
        REACH["declined: has a program config"] += 1
        return kw
    if kw.get("core_grid") is not None:
        REACH["declined: has a core grid"] += 1
        return kw
    try:
        if len(a.shape) < 3 or _batch(a) <= 1:
            # ttnn's planner is already at or near the best grid on a batch-1 matmul, and on
            # four of the round's shapes every grid in the sweep was worse than no grid.
            REACH["declined: not batched"] += 1
            return kw
        if not (_dram_interleaved(a) and _dram_interleaved(b)):
            REACH["declined: not DRAM interleaved"] += 1
            return kw
        gx, gy = _device_grid()
    except Exception:                                                         # noqa: BLE001
        REACH["declined: could not read the operands"] += 1
        return kw
    REACH["served"] += 1
    return {**kw, "core_grid": _grid(int(gx), int(gy))}


def reach() -> dict:
    """A snapshot of `REACH`, for a round boundary or a run stamp."""
    return dict(REACH)
