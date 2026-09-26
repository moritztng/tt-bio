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


def plan(a, b, kw: dict) -> dict:
    """`kw` for `ttnn.matmul(a, b, **kw)`, with a core grid added where this lever applies.

    Returns `kw` itself when it does not, so an off arm pays one boolean.
    """
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


#: Release-gated, default OFF. A second gate, for the call site the one above cannot see.
#: `mm_layout.plan` is wired into `autograd._matmul`, `_v_linear`'s dX and `taped_ttnn._v_matmul`
#: -- three sites, all of them `ttnn.matmul`. The forward `ttnn.linear` the trunk issues through
#: `ops.linear` is not one of them, and `bcx-p10-mmroof`'s census found 0.0488 s of a cell there
#: running with no plan at all.
#:
#: The grid is NOT the one above. Measured on qb2 card 0 at AICLK 1350, each point the model's
#: own captured call with one thing changed, 25 warm reps
#: (`perf/bcx_p10_mmroof/out/sweep_E.json`):
#:
#:     288x288x128x512 linear   11x10  1.1004x    8x8  0.5937x
#:     288x288x512x128 linear   11x10  1.0403x    8x8  0.9028x
#:
#: So an 8x8 grid is a 6-41 % LOSS on exactly the calls this gate serves, and the two gates want
#: opposite sides of the grid. They are separate flags for that reason and not one flag widened.
MM_PLAN = env_flag("TT_BIO_MM_PLAN", False)


def plan_linear(a, b, kw: dict) -> dict:
    """`kw` for `ttnn.linear(a, b, **kw)`, with the WHOLE compute grid added where this applies.

    Same shape of gate as `plan`: a call that already carries a plan keeps it, an unbatched call
    keeps ttnn's planner, and an operand that is not DRAM-interleaved is left alone because the
    sweep never measured one.
    """
    if not MM_PLAN:
        return kw
    if kw.get("program_config") is not None or kw.get("config") is not None:
        REACH["linear declined: has a program config"] += 1
        return kw
    if kw.get("core_grid") is not None:
        REACH["linear declined: has a core grid"] += 1
        return kw
    try:
        if len(a.shape) < 3 or _batch(a) <= 1:
            # The batch-1 direction is measured and it is a loss: 1x82944x512x128 reads 0.7339x
            # at 11x10 and 1x82944x128x128 reads 0.6982x. Those keep ttnn's planner.
            REACH["linear declined: not batched"] += 1
            return kw
        if not (_dram_interleaved(a) and _dram_interleaved(b)):
            REACH["linear declined: not DRAM interleaved"] += 1
            return kw
        gx, gy = _device_grid()
    except Exception:                                                         # noqa: BLE001
        REACH["linear declined: could not read the operands"] += 1
        return kw
    REACH["linear served"] += 1
    return {**kw, "core_grid": ttnn.CoreGrid(y=int(gy), x=int(gx))}


def reach() -> dict:
    """A snapshot of `REACH`, for a round boundary or a run stamp."""
    return dict(REACH)
