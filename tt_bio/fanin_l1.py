"""L1 for the transient the backward's fan-in writes and immediately reads back.

`bcx-p10-l1fuse` censused a BindCraft 2 gradient round by producer -> consumer EDGE instead of by
op, and the largest single chain in it is not a matmul operand. It is `Tensor.add_grad`'s fp32
promotion: `ttnn.typecast(grad, float32)` writes a `1x288x288x128` float32 tensor and the
`ttnn.add` on the very next call reads it straight back. **576 instances a round, 24.46 GB
written then re-read, and every byte of it goes to DRAM and comes back.**

That transient is the cheapest resident set a chain can have. It has exactly one consumer, the
consumer is adjacent half the time, and nothing else ever reads it: it is not an `autograd.Tensor`
and no tape node holds it, so `Tensor.evict` -- which is what made `bcx-p10-tril1`'s forward
residency worth 1.0289x instead of its byte model -- never sees it. A cotangent is not taped.

So the accumulator and the sum stay where they are and only the transient moves. One
`1x288x288x128` float32 tensor is 40.50 MiB against qb2 card 0's 153.34 MiB of aggregate L1
(11x10 cores at 1,461,760 B a bank, read off the allocator by `perf/bcx_p10_l1fuse/budget.py`),
which is 26.4 % of the grid. Placing all three would be 79.2 % and is not a resident set to plan
for beside a live backward.

It is fail-closed twice over. The budget is priced on the tensor's own bytes against the grid's
own L1, never on a shape or a token axis, and every refusal is counted by name. And because L1 is
allocated out of the same banks the block's live activations are in, a placement that the
allocator refuses at runtime falls back to DRAM and is counted as `spilled`, so a round can never
die of this lever: the worst it can do is the DRAM path it replaced.

Release-gated, default OFF. Placement does not change arithmetic -- `bcx-p10-mmroof` measured
max absolute deviation 0 across nine classes moved between buffers -- so anything that moves the
answer here is this gate and not the residency.
"""

from __future__ import annotations

import collections

import ttnn

from .envflags import env_flag, env_float

#: Release-gated, default OFF.
FANIN_L1 = env_flag("TT_BIO_GRAD_FANIN_L1", False)

#: Share of the compute grid's aggregate L1 one transient may take. 0.35 admits the 26.4 % the
#: round's `1x288x288x128` float32 cotangent needs and refuses the 79.2 % the triangle-attention
#: bias's `288x1x288x384` float32 would need, which is the chain the budget has to say no to.
FANIN_L1_SHARE = env_float("TT_BIO_GRAD_FANIN_L1_SHARE", 0.35)

#: `served` is a transient that got L1; every `declined:*` is one this gate saw and left in DRAM,
#: named by the reason; `spilled` is one the allocator refused at runtime and that fell back.
#: A round with served == 0 measured nothing and has to say so beside its own number.
REACH: collections.Counter = collections.Counter()

_ELEM = {ttnn.float32: 4, ttnn.bfloat16: 2, ttnn.uint32: 4, ttnn.int32: 4, ttnn.uint16: 2}


def _grid_l1_bytes() -> int:
    # Imported here and not at module scope: `tenstorrent` imports half the package, and it is
    # what reads the real grid off the device at open time, so by the time a backward runs this
    # is the measured grid and the allocator's own per-bank number.
    from .tenstorrent import COMPUTE_GRID_MAIN, _l1_bank_bytes
    gx, gy = COMPUTE_GRID_MAIN
    return _l1_bank_bytes() * gx * gy


def fits(t, dtype) -> bool:
    """Would `t`'s padded shape at `dtype` fit the share of the grid's L1 this lever may take?

    Priced on the tensor's own bytes, the way `_trimul_l1_fits` is, and never on a sequence
    length: a budget keyed on a shape is a budget that is wrong on the next shape.
    """
    elem = _ELEM.get(dtype)
    if elem is None:
        return False
    n = 1
    for d in tuple(t.padded_shape):
        n *= int(d)
    return n * elem <= FANIN_L1_SHARE * _grid_l1_bytes()


def config(t, dtype) -> "ttnn.MemoryConfig | None":
    """`memory_config` for the promoted cotangent, or None to leave it in DRAM."""
    if not FANIN_L1:
        return None
    try:
        if t.layout != ttnn.TILE_LAYOUT:
            # A row-major transient is a different allocation shape and none of the round's
            # fan-in is row-major; admitting one would be a guess.
            REACH["declined: not TILE"] += 1
            return None
        mc = t.memory_config()
        if mc.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
            REACH["declined: not interleaved"] += 1
            return None
        if mc.buffer_type != ttnn.BufferType.DRAM:
            REACH["declined: already off DRAM"] += 1
            return None
        if not fits(t, dtype):
            REACH["declined: over the L1 budget"] += 1
            return None
    except Exception:                                                         # noqa: BLE001
        REACH["declined: could not read the operand"] += 1
        return None
    REACH["served"] += 1
    return ttnn.L1_MEMORY_CONFIG


def typecast(t, dtype):
    """`ttnn.typecast(t, dtype)` into L1 where this lever applies, and into DRAM where it does not.

    The allocator is the second gate and it is the one that cannot be predicted: L1 is the same
    memory the block's live activations are in, so a transient that fits the budget can still be
    refused at the moment it is allocated. That falls back to the path it replaced rather than
    ending the round.
    """
    mc = config(t, dtype)
    if mc is None:
        return ttnn.typecast(t, dtype)
    try:
        return ttnn.typecast(t, dtype, memory_config=mc)
    except Exception:                                                         # noqa: BLE001
        REACH["served"] -= 1
        REACH["spilled: the allocator refused L1"] += 1
        return ttnn.typecast(t, dtype)


def reach() -> dict:
    """A snapshot of `REACH`, for a round boundary or a run stamp."""
    return dict(REACH)
