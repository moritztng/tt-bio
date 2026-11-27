"""The rollout on the card: the fixed-shape step and its pose update as ttnn ops (D-dfd-perf-6).

An op-for-op port of the torch program that already defines a step over fixed shapes,
:func:`tt_bio.diffdock.static_net.static_forward` followed by :func:`tt_bio.diffdock.loop.diffusion_step`.
Every block here is graded against that torch twin in float64 on the card
(`scripts/diffdock_port/device_loop_grade.py`) before the next one builds on it, and nothing here
is new arithmetic.

Layout: a per-edge or per-node scalar is a logical ``[n, 1]`` TILE tensor, so every elementwise
op is fp32 on the vector engine, and a coordinate triple travels as three of them. Wide edge
features (harmonics, distance expansions) are assembled from those columns; only the embedding
MLPs touch the matrix engine.
"""

from __future__ import annotations

import math

import torch

_EPS = 1e-12          # harmonics.unit's clamp, e3nn's F.normalize default


def upload(device, x: torch.Tensor):
    """A host tensor as an fp32 TILE tensor on *device*; a 1-d tensor becomes ``[n, 1]``."""
    import ttnn
    x = x.to(torch.float32)
    if x.dim() == 1:
        x = x[:, None]
    return ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)


def norm3(x, y, z):
    """``|(x, y, z)|`` per row."""
    import ttnn
    return ttnn.sqrt(ttnn.add(ttnn.add(ttnn.multiply(x, x), ttnn.multiply(y, y)),
                              ttnn.multiply(z, z)))


def harmonics(x, y, z, ls=(0, 1, 2)):
    """:func:`tt_bio.diffdock.harmonics.spherical_harmonics` of ``(x, y, z)``, normalised.

    Returns ``[n, sum(2l+1)]``, e3nn's component order. A zero vector stays zero past l=0, as
    ``unit`` makes it on the host.
    """
    import ttnn
    inv = ttnn.reciprocal(ttnn.clamp(norm3(x, y, z), min=_EPS))
    x, y, z = (ttnn.multiply(c, inv) for c in (x, y, z))
    s3, s5, s15 = math.sqrt(3), math.sqrt(5), math.sqrt(15)
    cols = []
    for l in ls:
        if l == 0:
            cols.append(ttnn.add(ttnn.multiply(x, 0.0), 1.0))
        elif l == 1:
            cols += [ttnn.multiply(c, s3) for c in (x, y, z)]
        elif l == 2:
            y2 = ttnn.multiply(y, y)
            x2z2 = ttnn.add(ttnn.multiply(x, x), ttnn.multiply(z, z))
            cols += [ttnn.multiply(ttnn.multiply(x, z), s15),
                     ttnn.multiply(ttnn.multiply(x, y), s15),
                     ttnn.multiply(ttnn.subtract(y2, ttnn.multiply(x2z2, 0.5)), s5),
                     ttnn.multiply(ttnn.multiply(y, z), s15),
                     ttnn.multiply(ttnn.subtract(ttnn.multiply(z, z), ttnn.multiply(x, x)),
                                   0.5 * s15)]
        else:
            raise ValueError(f"harmonics are implemented to l=2, asked for {l}")
    return ttnn.concat(cols, dim=1)


class Smearing:
    """:class:`tt_bio.diffdock.features.GaussianSmearing` on the card: ``exp(coeff (d - mu)^2)``."""

    def __init__(self, device, module):
        self.offset = upload(device, module.offset.reshape(1, -1))
        self.coeff = float(module.coeff)

    def __call__(self, dist):
        import ttnn
        d = ttnn.subtract(dist, self.offset)
        return ttnn.exp(ttnn.multiply(ttnn.multiply(d, d), self.coeff))


class MLP:
    """An ``nn.Sequential`` of Linear / ReLU / Identity on the card, fp32 weights."""

    def __init__(self, device, seq: torch.nn.Sequential):
        self.ops = []
        for m in seq:
            if isinstance(m, torch.nn.Linear):
                self.ops.append((upload(device, m.weight.detach().t().contiguous()),
                                 upload(device, m.bias.detach().reshape(1, -1))))
            elif isinstance(m, torch.nn.ReLU):
                self.ops.append("relu")
            elif not isinstance(m, (torch.nn.Identity, torch.nn.Dropout)):
                raise NotImplementedError(f"no card form for {type(m).__name__}")

    def __call__(self, x):
        import ttnn
        for op in self.ops:
            x = ttnn.relu(x) if op == "relu" else ttnn.linear(x, op[0], bias=op[1])
        return x


# ---------------------------------------------------------------------------
# the pose update: `tt_bio.diffdock.loop.diffusion_step` on the card
# ---------------------------------------------------------------------------

def axis_angle_matrix(ax, ay, az):
    """:func:`tt_bio.diffdock.sampling.axis_angle_to_matrix` on three ``[B, 1]`` columns.

    Returns R's nine entries as ``[B, 1]`` columns, row-major. Upstream's quaternion route and
    its 1e-6 small-angle series, which is already a ``where`` and so needs no branch here.
    """
    import ttnn
    angles = norm3(ax, ay, az)
    half = ttnn.multiply(angles, 0.5)
    # ttnn.subtract has no (scalar, tensor) overload: negate and add.
    series = ttnn.add(ttnn.neg(ttnn.multiply(ttnn.multiply(angles, angles), 1.0 / 48)), 0.5)
    ratio = ttnn.where(ttnn.lt(ttnn.abs(angles), 1e-6), series,
                       ttnn.multiply(ttnn.sin(half),
                                     ttnn.reciprocal(ttnn.clamp(angles, min=1e-30))))
    r = ttnn.cos(half)
    i, j, k = (ttnn.multiply(c, ratio) for c in (ax, ay, az))
    sq = ttnn.add(ttnn.add(ttnn.multiply(r, r), ttnn.multiply(i, i)),
                  ttnn.add(ttnn.multiply(j, j), ttnn.multiply(k, k)))
    two = ttnn.multiply(ttnn.reciprocal(sq), 2.0)
    mul = lambda a, b: ttnn.multiply(two, ttnn.multiply(a, b))
    diag = lambda a, b: ttnn.add(ttnn.neg(ttnn.add(mul(a, a), mul(b, b))), 1.0)
    return (diag(j, k), ttnn.subtract(mul(i, j), mul(k, r)), ttnn.add(mul(i, k), mul(j, r)),
            ttnn.add(mul(i, j), mul(k, r)), diag(i, k), ttnn.subtract(mul(j, k), mul(i, r)),
            ttnn.subtract(mul(i, k), mul(j, r)), ttnn.add(mul(j, k), mul(i, r)), diag(i, j))


def rotate(cols, m):
    """``x @ R.T`` for coordinate columns *cols* ``(x, y, z)`` and R's nine entries *m*.

    Columns are ``[n, 1]``; R's entries are whatever row layout the caller gathered them into,
    one row per atom when a pose's matrix applies to its atoms.
    """
    import ttnn
    x, y, z = cols
    row = lambda a, b, c: ttnn.add(ttnn.add(ttnn.multiply(x, a), ttnn.multiply(y, b)),
                                   ttnn.multiply(z, c))
    return row(m[0], m[1], m[2]), row(m[3], m[4], m[5]), row(m[6], m[7], m[8])


def guard(cols, big: float = 1e30):
    """:func:`tt_bio.diffdock.sampling.guard_scores`'s repair, over a head's ``[B, 1]`` columns.

    A non-finite entry becomes ``0.01 * mean|finite entries| * sign``, the head's columns sharing
    one eps as the host's whole-tensor reduction does.

    **Not built on `ttnn.where`.** Measured on a card (`scripts/diffdock_port/nan_probe.py`):
    `where` evaluates both lanes, so a NaN -- or an inf, through ``inf * 0`` -- in the lane not
    taken still reaches the output, and the whole repair came back NaN. Two ops do the job:
    `ttnn.clamp` maps NaN to its max and each infinity to the matching bound, and `ttnn.sign`
    already returns +1 for NaN and +-1 for +-inf, which is exactly the host's
    ``nan_to_num(nan=1, posinf=1, neginf=-1)``. After the clamp every value is finite, so the
    select is arithmetic (``m * x + (1 - m) * repair``) and multiplying by zero is safe.
    """
    import ttnn
    finite = [ttnn.typecast(ttnn.isfinite(c), ttnn.float32) for c in cols]
    safe = [ttnn.clamp(c, min=-big, max=big) for c in cols]
    signs = [ttnn.sign(c) for c in cols]
    total, count = None, None
    for c, m in zip(safe, finite):
        mag = ttnn.multiply(ttnn.abs(c), m)
        total = mag if total is None else ttnn.add(total, mag)
        count = m if count is None else ttnn.add(count, m)
    eps = ttnn.multiply(ttnn.multiply(ttnn.sum(total), 0.01),
                        ttnn.reciprocal(ttnn.clamp(ttnn.sum(count), min=1.0)))
    out = []
    for c, m, sg in zip(safe, finite, signs):
        repair = ttnn.multiply(ttnn.multiply(sg, eps), ttnn.add(ttnn.neg(m), 1.0))
        out.append(ttnn.add(ttnn.multiply(c, m), repair))
    return out


# ---------------------------------------------------------------------------
# one tower's edge features, built on the card from the positions it already holds
# ---------------------------------------------------------------------------

def gather(table, idx, n: int):
    """``table[idx]`` for a TILE ``[N, d]`` fp32 table, exactly.

    :func:`tt_bio.diffdock.device.gather_rows` splits an fp32 table into three bf16 slices so
    the copy loses nothing; positions are fp32 and a one-hot in bf16 would not be exact
    (`WH.md`), so every gather in the loop goes through it.
    """
    from .device import gather_rows
    return gather_rows(table, idx, n)


class EdgeGeometry:
    """Per-edge vector, distance, harmonics and radius mask for one fixed edge set.

    The edge index never moves (the static graphs carry the step's changes in their masks), so
    the two row-index tensors are uploaded once and only the positions change per step.
    """

    def __init__(self, device, src_idx, dst_idx, n_edges: int):
        import ttnn
        self.device, self.n = device, n_edges
        self.src = ttnn.from_torch(src_idx.reshape(1, -1).to(torch.int32), dtype=ttnn.uint32,
                                   layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        self.dst = ttnn.from_torch(dst_idx.reshape(1, -1).to(torch.int32), dtype=ttnn.uint32,
                                   layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    def vectors(self, src_pos, dst_pos=None):
        """``dst_pos[dst] - src_pos[src]`` as three ``[E, 1]`` columns.

        *src_pos* and *dst_pos* are ``[N, 3]`` TILE tensors (one table when the edge set is
        within a tower, two across the cross edges).
        """
        import ttnn
        a = gather(src_pos, self.src, self.n)
        b = gather(dst_pos if dst_pos is not None else src_pos, self.dst, self.n)
        d = ttnn.subtract(b, a)
        return [ttnn.slice(d, [0, k], [self.n, k + 1]) for k in range(3)]

    def within(self, dist, cutoff):
        """``(dist <= cutoff)`` as an fp32 ``[E, 1]`` mask, the host's ``<=``."""
        import ttnn
        return ttnn.typecast(ttnn.le(dist, cutoff), ttnn.float32)


# ---------------------------------------------------------------------------
# the step's masks: StaticGraphs.lig_mask / cross_mask / rec_edge_mask on the card
# ---------------------------------------------------------------------------

class StepMasks:
    """The three per-step edge masks over fixed edge sets, as fp32 ``[E, 1]`` columns.

    Each mask is "always on here, computed there, zero on the pad", and which row is which
    never changes over the rollout. So the two 0/1 selectors are uploaded once and the step
    is one comparison plus a multiply-add: no dynamic slicing, no host round trip.

        mask = always + computed * predicate

    The predicates are the host's own floating-point expressions, not their intent.
    :func:`tt_bio.diffdock.graph.cross_mask` divides the edge vector by the cutoff and
    compares the norm with 1, because ``build_cross_conv_graph`` does; comparing the norm
    with the cutoff instead moves edges across the boundary, so this divides too.
    """

    def __init__(self, device, sg, n_lig_edges_padded: int, n_cross_edges_padded: int):
        self.device = device
        n_bond = sg.lig_n_bond_edges
        n_pairs = sg.n_lig * (sg.n_lig - 1) * sg.n_poses
        lig_always = torch.zeros(n_lig_edges_padded)
        lig_always[:n_bond] = 1.0
        lig_calc = torch.zeros(n_lig_edges_padded)
        lig_calc[n_bond:n_bond + n_pairs] = 1.0
        cross_calc = torch.zeros(n_cross_edges_padded)
        cross_calc[:sg.n_lig * sg.n_rec * sg.n_poses] = 1.0
        self.lig_always = upload(device, lig_always)
        self.lig_calc = upload(device, lig_calc)
        self.cross_calc = upload(device, cross_calc)

    def lig(self, dist, lig_max_radius: float):
        """:meth:`StaticGraphs.lig_mask`: bond edges always on, the complete block by radius."""
        import ttnn
        inside = ttnn.typecast(ttnn.le(dist, lig_max_radius), ttnn.float32)
        return ttnn.add(self.lig_always, ttnn.multiply(self.lig_calc, inside))

    def cross(self, vx, vy, vz, cutoff):
        """:meth:`StaticGraphs.cross_mask`. *cutoff* is a float or an ``[E, 1]`` column."""
        import ttnn
        inv = 1.0 / cutoff if isinstance(cutoff, float) else ttnn.reciprocal(cutoff)
        scaled = [ttnn.multiply(c, inv) for c in (vx, vy, vz)]
        inside = ttnn.typecast(ttnn.le(norm3(*scaled), 1.0), ttnn.float32)
        return ttnn.multiply(self.cross_calc, inside)

    @staticmethod
    def rec_edges(keep, src, dst, n_edges: int):
        """:meth:`StaticGraphs.rec_edge_mask`: an edge survives the crop iff both ends do."""
        import ttnn
        return ttnn.multiply(gather(keep, src, n_edges), gather(keep, dst, n_edges))


def rec_keep(cross_near, n_poses: int, n_lig: int, n_rec: int, n_rec_padded: int):
    """:meth:`StaticGraphs.rec_keep` without a ``cdist`` and without a scatter.

    The host builds the receptor crop either as an ``[n_rec, n_lig]`` distance matrix per
    pose (`rec_keep`) or as an ``index_add`` over the cross edges (`rec_keep_from_cross`).
    Neither is needed on the card: the cross graph is the **complete** ligand x receptor
    pair set per pose, laid out source-major by :func:`graph.complete_pairs` and tiled by
    pose, so edge ``((p * n_lig) + l) * n_rec + r`` is exactly the pair ``(p, l, r)``. "Some
    ligand atom of this pose is within the radius of this residue" is then a reduction over
    the ligand axis of a ``[n_poses, n_lig, n_rec]`` view, not a scatter into a node axis.

    The reduction is a max of a 0/1 column, so it is exact in any dtype, and it agrees with
    `rec_keep_from_cross` row for row (checked in torch at (24, 48, 4), (11, 37, 3) and
    (40, 120, 10), equal everywhere).

    *cross_near* is the ``[E, 1]`` cross mask; its first ``n_poses * n_lig * n_rec`` rows are
    the real edges and the rest is pad. Returns an ``[n_rec_padded, 1]`` fp32 node mask with
    the pad rows zero, which is what :meth:`StepMasks.rec_edges` and the cross product want.
    """
    import ttnn
    real = n_poses * n_lig * n_rec
    flat = ttnn.to_layout(ttnn.slice(cross_near, [0, 0], [real, 1]), ttnn.ROW_MAJOR_LAYOUT)
    block = ttnn.to_layout(ttnn.reshape(flat, [n_poses, n_lig, n_rec]), ttnn.TILE_LAYOUT)
    kept = ttnn.max(block, dim=1)                      # [n_poses, 1, n_rec] or [n_poses, n_rec]
    rows = ttnn.reshape(ttnn.to_layout(kept, ttnn.ROW_MAJOR_LAYOUT), [n_poses * n_rec, 1])
    if n_rec_padded > n_poses * n_rec:
        pad = ttnn.zeros([n_rec_padded - n_poses * n_rec, 1], dtype=ttnn.float32,
                         layout=ttnn.ROW_MAJOR_LAYOUT, device=cross_near.device())
        rows = ttnn.concat([rows, pad], dim=0)
    return ttnn.to_layout(rows, ttnn.TILE_LAYOUT)
