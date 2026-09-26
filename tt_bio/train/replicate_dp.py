"""Data parallelism over a per-step SAMPLE axis, and the reduce at a tape cut.

`tt_bio.train.sharding` splits a global batch of dataset examples across chips. That axis
raises throughput and leaves the step's latency alone, because the per-chip batch is what it
holds fixed. This module is the other axis, and it is the one a diffusion trainer has:
**OpenFold3 differentiates 48 noised structures inside ONE step at batch 1**, and BindCraft 2
runs an equivalent per-step replicate set. Those replicates are independent given the trunk
output, so splitting them across chips divides the step's own latency without doing less of
the model's work -- every one of the 48 still runs, just not all on one card.

The design is one sentence: each rank forwards the shared prefix itself, owns a disjoint
slice of the replicates, and the prefix is differentiated once against the SUMMED cut
cotangent.

**The prefix is recomputed, not exchanged.** Every rank runs the identical trunk forward from
identical weights, so `si_trunk`/`zij_pad` agree without a message. Sending them instead would
cost 37.7 MB of `zij_pad` alone at crop 384 and save a forward the rank has to be able to do
anyway to hold its own tape.

**One reduce, at the optimizer.** `AdamW.step(replicas=...)` already sums gradients across
`tt_bio.train.launcher`'s ranks over /dev/shm and it is the only place the numbers have to
meet, so this module does not add a collective beside it. What it does add is
:func:`reduce_cut`, for the cotangent at the cut, which has to be summed BEFORE the prefix
backward rather than after it.

**Every rank runs the prefix backward, from the SAME summed cotangent, and that is what
keeps the prefix's weights off the wire entirely.** :func:`reduce_cut` sums the cotangent at
the cut before anything is differentiated, so each rank then computes the FULL prefix
gradient rather than a partial one -- identical on every rank, bit for bit, because the input
and the weights are identical. Identical gradients need no reduce: they are simply excluded
from it. The earlier design here ran the prefix backward on rank 0 alone out of a fear of
N-counting, which does not apply once the cotangent is summed first; it cost the same wall
clock (the redundant backwards are parallel) and put 660.7 MB of prefix gradient on the wire
every step to tell the other ranks what they could have computed themselves.

**The order is load-bearing and it is the one thing a second implementer would get wrong.**
Reduce the replicate-side gradients BEFORE the prefix backward, not at the optimizer. At that
moment they hold only this rank's own replicates; afterwards they also hold the primed
replicate, which every rank runs, and reducing then would count it ``world`` times. See
:func:`reduce_grads`'s ``names``.

**The honest cap on this axis is that the prefix does not divide at all.** How big it is,
measured rather than projected: at 48
replicates on OpenFold3 crop 384 the trunk is 89.6 % of a 1035.31 s step (`of3t-p10wall`,
taped cycle 233.808 s plus trunk backward 687.661 s, qb2 card 1, AICLK 1350 median DURING
n=834) against a replicate loop of 102.32 s. An earlier 20 s reading for the trunk backward
was taken at 4 replicates and is 34x low at 48. **Sharding a term that is 9.9 % of the step
cannot pay**, so this module waits on the trunk defect (`of3t-p10trunk`) before it is worth
a chip.
"""

from __future__ import annotations

from typing import List, Sequence

__all__ = ["shard", "replicate_noise", "reduce_cut", "reduce_grads"]


def shard(n: int, world: int, rank: int) -> List[int]:
    """This rank's indices of an ``n``-long per-step sample axis. Round-robin, not contiguous.

    Round-robin because the replicate index is the sigma index and a diffusion schedule is
    ordered by noise level: a contiguous split hands rank 0 the loudest structures and rank
    ``world-1`` the quietest, so the ranks' step times would differ by the schedule rather
    than by the hardware and the slowest one would set the barrier every step.

    ``n`` need not divide by ``world``: the remainder lands on the low ranks, one each, which
    is at most one replicate of imbalance. Refusing a 47-replicate axis on 4 chips would be
    refusing the shape OpenFold3 actually has, since one of its 48 is the primed replicate.
    """
    if world < 1:
        raise ValueError(f"world must be at least 1, got {world}")
    if not 0 <= rank < world:
        raise ValueError(f"rank {rank} is not in range(world={world})")
    return list(range(rank, n, world))


def replicate_noise(seed: int, index: int, shape, dtype="float32"):
    """Standard normal noise for the replicate at GLOBAL ``index``, independent of draw order.

    The obvious implementation -- one generator per step, drawn once per replicate inside the
    loop -- makes a replicate's noise a function of its POSITION IN THE DRAW ORDER rather than
    of its index. On one chip running the replicates in order the two coincide, which is why a
    chunked-against-unchunked test passes against it and why the defect survives a card-free
    suite.

    Under :func:`shard` they come apart. Rank ``r`` executes global indices ``r, r+world, ...``
    and draws from position 0, so every rank gets the SAME noise sequence paired with a
    DIFFERENT set of sigmas: the structures differentiated are no longer the ones a single
    chip differentiates, and several are duplicates of each other. Nothing crashes, the draw
    count and every tensor shape are unchanged, and the step time is exactly right -- which is
    the dangerous part, because the arm then publishes a correct speed for a computation that
    is not the one being measured.

    Keying the generator on ``(seed, index)`` makes the noise a pure function of the global
    index. Replicate ``i`` is the same structure whoever owns it, at any ``world`` and any
    chunk size, so an N-chip arm and its N=1 control are the same computation and a gradient
    comparison between them means something. It is timing-neutral -- same draw count, same
    shapes, same device ops -- so it does not move a banked baseline.
    """
    import numpy as np
    return np.random.default_rng([seed, index]).standard_normal(shape).astype(dtype, copy=False)


def reduce_cut(axis, leaves: Sequence, device) -> int:
    """Sum the cotangent on each cut leaf across ``axis``, in place. Returns bytes exchanged.

    ``leaves`` is the detached side of the cut -- the leaves whose ``.grad`` the replicate
    backwards accumulated into. Every rank must pass the SAME leaves in the SAME order; a
    leaf whose gradient is None on one rank and set on another is the shape where a silent
    rank-dependent sum comes from, so a None is sent as explicit zeros rather than skipped.

    The sum comes back on the device because that is where the prefix backward reads it,
    which is the opposite of the gradient reduce: a gradient's next reader is the fp32 master
    on the host, a cotangent's next reader is a ttnn kernel.
    """
    if axis is None or axis.width == 1:
        return 0
    import numpy as np
    import ttnn
    from .tensors import to_host, to_device
    from tt_bio import autograd as ag

    per_param = {}
    shapes = []
    for i, leaf in enumerate(leaves):
        raw = ag._unwrap(leaf)
        shp = tuple(int(d) for d in raw.shape)
        g = leaf.grad
        a = (np.ascontiguousarray(to_host(g), dtype=np.float32) if g is not None
             else np.zeros(shp, np.float32))
        shapes.append((shp, raw.dtype))
        per_param[f"{i:06d}"] = [a.reshape(shp)]
    summed = axis.reduce_all(per_param)
    moved = 0
    for i, leaf in enumerate(leaves):
        shp, dtype = shapes[i]
        a = np.asarray(summed[f"{i:06d}"], np.float32).reshape(shp)
        moved += a.nbytes
        leaf.grad = to_device(a, device, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    return moved


def reduce_grads(axis, params, names=None) -> int:
    """Sum a parameter set's gradients across ``axis``, in place, on the host. Returns bytes.

    ``AdamW.step(replicas=...)`` is the shipped route and it is the right one when every rank
    holds the same set of gradients. Here they do not: the ranks that skip the prefix backward
    have no gradient on the prefix's weights, and
    :meth:`~tt_bio.train.launcher.ProcessAxis.reduce_all` flattens in name order, so two ranks
    with different name sets would sum vectors of different lengths into each other.

    ``names`` is the set that has to cross, and leaving it at ``None`` sends the whole
    declared set. **Pass the replicate half.** A parameter the prefix backward alone writes
    gets the identical gradient on every rank (same summed cotangent, same weights), so
    sending it buys nothing: on OpenFold3 at crop 384 the prefix is 165,180,992 of 381,302,188
    declared elements, so naming the 216,121,196-element diffusion half takes **660.7 MB off
    the wire every step, 43.3 %** (counts from `perf/of3t_stepfloor/out/step_rekey_384.json`
    and `out/d164_probeoff_384.json`; the two sets are disjoint -- the 275 duplicate handles
    are internal to the diffusion walk, 896 - 275 = 621). Every rank must pass the SAME names
    in the same order, which is why they come from the static weight walk and not from which
    parameters happen to hold a gradient this step.

    So the message is over every named parameter, absent gradients sent as zeros, with one
    extra float per parameter carrying whether this rank had one at all. The presence flags
    ride the same message rather than a second barrier, and they are what lets a parameter
    that took no gradient anywhere go back to ``grad = None`` instead of to a zero array --
    Adam with a zero gradient still moves a weight through its momentum, so the two are
    different updates and only ``None`` matches what one chip does.
    """
    if axis is None or axis.width == 1:
        return 0
    import numpy as np
    from .tensors import to_host

    names = sorted(params if names is None else names)
    per_param = {}
    present = np.zeros(len(names), np.float32)
    for i, n in enumerate(names):
        t = params[n]
        g = getattr(t, "grad", None)
        if g is None:
            # The shape off the handle, never a read of the weight: this branch is the one
            # taken by every prefix parameter on every rank but one, and downloading them to
            # learn a shape would put the model's whole weight set on the wire each step.
            per_param[n] = [np.zeros(_shape(t), np.float32)]
        else:
            present[i] = 1.0
            per_param[n] = [np.ascontiguousarray(to_host(g), dtype=np.float32)]
    # '~' sorts after every name the model uses, so the flags land at the tail of the vector
    # on every rank without anyone agreeing on anything but ASCII.
    per_param["~present"] = [present]
    summed = axis.reduce_all(per_param)
    flags = np.asarray(summed["~present"]).ravel()
    moved = int(sum(np.asarray(summed[n]).nbytes for n in names))
    for i, n in enumerate(names):
        params[n].grad = np.asarray(summed[n], np.float32) if flags[i] > 0 else None
    return moved


def _shape(t) -> tuple:
    """A parameter's shape without touching its storage."""
    shp = getattr(t, "shape", None)
    if shp is None:
        from tt_bio import autograd as ag
        shp = ag._unwrap(t).shape
    return tuple(int(d) for d in shp)
