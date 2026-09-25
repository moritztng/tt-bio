#!/usr/bin/env python3
"""Float64 VJPs for the two routing verbs, pure torch, no ttnn.

`_identity_grad` and `_sliced` are 47.97 % of one taped backward at padded 384 and neither has
ever been substituted. They are routing, so what there is to get right is not an algebraic form
but the INDEX ARITHMETIC and the accumulation of several contributions into one parent. Both are
written here against the shipped closures' own arithmetic, and `fdroute.py` validates them
against float64 central finite differences with no device in the loop.

What the shipped closures do, read off `tt_bio/taped_ttnn.py` rather than assumed:

  `_identity_grad` (line 87).  `bw(g)` re-layouts g to the source layout and then
      `x.add_grad(ttnn.typecast(g, src_dtype) if cast and g.dtype != src_dtype else g)`.
      The VALUE map is the identity; `to_layout` is a reshuffle and exact; the ONLY arithmetic
      in the verb is that typecast, and it is reached only when `cast` is true, which only
      `typecast` registers. Exact VJP: dx = g, in x's shape.

  `_sliced` (line 674).  `bw(g)` tiles g once, then for every axis that was cut prepends and
      appends a zero block of the missing extent and concatenates. Exact VJP: a zero tensor of
      the parent's shape with g written into the `[starts, ends)` box. `chunk` taping n blocks
      of one parent makes n such nodes, each a separate `add_grad` contribution, so the
      overlapping-region case is n adjacent boxes summed into one accumulator -- which is the
      composed thing `fdroute.py` checks, not just one box at a time.
"""
from __future__ import annotations

import torch


def identity_forward(x):
    """The value map of clone / reallocate / to_layout / to_memory_config / typecast."""
    return x


def identity_vjp(g, shape):
    """dx = g. The cotangent arrives in the OUTPUT's shape, which is the input's shape for
    every verb registered here; a rank normalisation elsewhere on the tape can still hand back
    the same volume in a different rank, so same-volume is reshaped and anything else raises."""
    if list(g.shape) == list(shape):
        return [g]
    if g.numel() == int(torch.tensor(list(shape)).prod()):
        return [g.reshape(tuple(int(d) for d in shape))]
    raise ValueError("identity_vjp: cotangent %s is not the parent's volume %s"
                     % (list(g.shape), list(shape)))


def _box(shape, starts, ends):
    if not (len(shape) == len(starts) == len(ends)):
        raise ValueError("sliced: rank %d/%d/%d disagree" % (len(shape), len(starts), len(ends)))
    for ax, (s, e, n) in enumerate(zip(starts, ends, shape)):
        if not (0 <= s <= e <= n):
            raise ValueError("sliced: axis %d box [%d, %d) outside extent %d" % (ax, s, e, n))
    return tuple(slice(int(s), int(e)) for s, e in zip(starts, ends))


def sliced_forward(x, starts, ends):
    return x[_box([int(d) for d in x.shape], starts, ends)]


def sliced_vjp(g, shape, starts, ends):
    """dx = zeros(shape); dx[starts:ends] = g.

    Written as a scatter into a zero tensor rather than as pad-and-concatenate. Those are the
    same map, and writing it the other way would reproduce the shipped closure's own axis loop
    -- including any off-by-one in it -- which is exactly what this reference exists to test.
    """
    shape = [int(d) for d in shape]
    idx = _box(shape, starts, ends)
    want = tuple(int(e) - int(s) for s, e in zip(starts, ends))
    if tuple(int(d) for d in g.shape) != want:
        if g.numel() != int(torch.tensor(list(want)).prod()):
            raise ValueError("sliced_vjp: cotangent %s is not the box %s"
                             % (list(g.shape), list(want)))
        g = g.reshape(want)
    out = torch.zeros(shape, dtype=g.dtype)
    out[idx] = g
    return [out]


VJP = {"_identity_grad": identity_vjp, "_sliced": sliced_vjp}
