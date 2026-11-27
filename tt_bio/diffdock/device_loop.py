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


_ZEROS: dict = {}


def zeros(device, shape, layout=None):
    """A cached fp32 zero tensor on *device*: padding is read, never written, so one per shape.

    Allocating it per step puts a host-to-device write inside every step, which a trace
    capture refuses and an untraced step pays for nothing.
    """
    import ttnn
    layout = ttnn.TILE_LAYOUT if layout is None else layout
    key = (id(device), tuple(shape), str(layout))
    if key not in _ZEROS:
        _ZEROS[key] = ttnn.zeros(list(shape), dtype=ttnn.float32, layout=layout, device=device)
    return _ZEROS[key]


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
    """An ``nn.Sequential`` of Linear / ReLU / Tanh / Identity on the card, fp32 weights.

    A ``Linear`` with ``bias=False`` is carried as such rather than as a zero bias row: the
    torsion head's final layer is biasless on both linears and its activation is ``Tanh``,
    not ``ReLU``, unlike every other MLP in the model.
    """

    _ACTS = {torch.nn.ReLU: "relu", torch.nn.Tanh: "tanh"}

    def __init__(self, device, seq: torch.nn.Sequential):
        self.ops = []
        for m in seq:
            if isinstance(m, torch.nn.Linear):
                bias = (None if m.bias is None else
                        upload(device, m.bias.detach().reshape(1, -1)))
                self.ops.append((upload(device, m.weight.detach().t().contiguous()), bias))
            elif type(m) in self._ACTS:
                self.ops.append(self._ACTS[type(m)])
            elif not isinstance(m, (torch.nn.Identity, torch.nn.Dropout)):
                raise NotImplementedError(f"no card form for {type(m).__name__}")

    def __call__(self, x):
        import ttnn
        for op in self.ops:
            if op == "relu":
                x = ttnn.relu(x)
            elif op == "tanh":
                x = ttnn.tanh(x)
            else:
                x = ttnn.linear(x, op[0], bias=op[1])
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
        pad = zeros(cross_near.device(), [n_rec_padded - n_poses * n_rec, 1],
                    ttnn.ROW_MAJOR_LAYOUT)
        rows = ttnn.concat([rows, pad], dim=0)
    return ttnn.to_layout(rows, ttnn.TILE_LAYOUT)


# ---------------------------------------------------------------------------
# the three towers' edge and node features: the first half of `static_forward`
# ---------------------------------------------------------------------------

class TowerFeatures:
    """Edge attributes, harmonics and node embeddings for the receptor, ligand and cross towers.

    The port of :func:`tt_bio.diffdock.static_net.static_forward` down to its first
    :func:`run_stack`. Two things make this cheaper on the card than the host expression
    reads, and both are exact rather than approximations:

    * **The sigma embedding is a broadcast, not a gather.** The host writes
      ``lig_sig[lei[0]]`` and ``lig_sig[cei[0]]``, but ``lig_sig`` is one row expanded over
      every ligand node, so every edge reads the same values. An ``[E, 1]`` column of ones
      times the ``[1, D]`` row is that block, and no index tensor is touched.
    * **The receptor's distance work is per complex, not per step.** The receptor does not
      move, so ``rvec``, its expansion, its edge embedding and its harmonics are the same
      on every one of the 19 steps. They are computed in :meth:`receptor_static` once and
      held; only the crop mask changes, and that is :class:`StepMasks`' job.

    The node embeddings are **not** here. ``lig_node_embedding`` and ``rec_node_embedding``
    are :class:`AtomEncoder`, not ``Sequential``: a sum of one embedding table per
    categorical atom feature, then one linear over that sum concatenated with the continuous
    block. That is a gather-and-sum followed by a GEMM, not an MLP, so it is its own block.
    Worth noting for it: the categorical half is static per complex, so the embedding sum can
    be computed once and only the linear re-run per step.

    The receptor's sigma term is the one per-step piece of the receptor tower, and the host
    adds it *after* the embedding stack (``rec_h[:, :ns] += rec_sig``), so it is not here.
    """

    def __init__(self, device, model, sg, n_lig_edges: int, n_cross_edges: int):
        self.device, self.sg = device, sg
        self.sh_ls = tuple(model.sh_ls)
        self.rec_smear = Smearing(device, model.rec_distance_expansion)
        self.lig_smear = Smearing(device, model.lig_distance_expansion)
        self.cross_smear = Smearing(device, model.cross_distance_expansion)
        self.rec_edge_mlp = MLP(device, model.rec_edge_embedding)
        self.lig_edge_mlp = MLP(device, model.lig_edge_embedding)
        self.cross_edge_mlp = MLP(device, model.cross_edge_embedding)
        self.ones_lig = upload(device, torch.ones(n_lig_edges))
        self.ones_cross = upload(device, torch.ones(n_cross_edges))

    @staticmethod
    def _broadcast(ones, row):
        """``[E, 1] x [1, D] -> [E, D]``: one row repeated down the edge axis."""
        import ttnn
        return ttnn.multiply(ones, row)

    def receptor_static(self, rvec_cols):
        """``(rec_eattr, rec_sh)`` for the receptor graph, which never moves.

        Call once per complex and hold the result: nothing in it depends on the step.
        """
        dist = norm3(*rvec_cols)
        return (self.rec_edge_mlp(self.rec_smear(dist)),
                harmonics(*rvec_cols, self.sh_ls))

    def ligand(self, lvec_cols, bond_block, sigma_row):
        """``(lig_eattr, lig_sh)``. *bond_block* is the padded ``[E, bond_dim]`` bond table.

        The host concatenates the bond attributes, the per-edge sigma embedding and the
        distance expansion in that order, and the embedding MLP's input width fixes it.
        """
        import ttnn
        dist = norm3(*lvec_cols)
        sig = self._broadcast(self.ones_lig, sigma_row)
        x = ttnn.concat([bond_block, sig, self.lig_smear(dist)], dim=1)
        return self.lig_edge_mlp(x), harmonics(*lvec_cols, self.sh_ls)

    def cross(self, cvec_cols, sigma_row):
        """``(cross_eattr, cross_sh, rev_sh)``: the cross edges and their reverse harmonics."""
        import ttnn
        dist = norm3(*cvec_cols)
        sig = self._broadcast(self.ones_cross, sigma_row)
        eattr = self.cross_edge_mlp(ttnn.concat([sig, self.cross_smear(dist)], dim=1))
        rev = [ttnn.multiply(c, -1.0) for c in cvec_cols]
        return eattr, harmonics(*cvec_cols, self.sh_ls), harmonics(*rev, self.sh_ls)


def pose_centers(lig_pos, n_poses: int, n_lig: int):
    """Each pose's ligand centre of mass, ``[n_poses, 3]``.

    The host writes ``lig_pos[:n_lig * P].reshape(P, n_lig, 3).mean(1)``. The pose axis is
    the leading one and every pose holds the same atom count, so this is the same kind of
    reduction as :func:`rec_keep`: a view and a mean, no scatter and no index tensor.
    """
    import ttnn
    real = n_poses * n_lig
    flat = ttnn.to_layout(ttnn.slice(lig_pos, [0, 0], [real, 3]), ttnn.ROW_MAJOR_LAYOUT)
    block = ttnn.to_layout(ttnn.reshape(flat, [n_poses, n_lig, 3]), ttnn.TILE_LAYOUT)
    return ttnn.reshape(ttnn.mean(block, dim=1), [n_poses, 3])


class CenterHead:
    """The centre head's geometry and its final scaling, around the existing tensor product.

    `static_forward`'s centre head is one conv from every ligand atom into its own pose's
    centre, then a norm-and-rescale of the two output vectors. The convolution itself is the
    stack machinery that `dfd-perf` already runs on the card, so it is not re-ported here;
    what is here is everything around it, which was all host torch.

    ``center[pose_of]`` is a gather with a fixed index, and the index never changes over the
    rollout, so it is uploaded once and goes through the exact fp32 three-way gather rather
    than a bf16 one-hot (`WH.md`).
    """

    def __init__(self, device, head, sg, sigma_embed_dim: int):
        self.device, self.sg = device, sg
        self.sh_ls = tuple(head.sh_ls)
        self.smear = Smearing(device, head.center_distance_expansion)
        self.edge_mlp = MLP(device, head.center_edge_embedding)
        self.tr_final = MLP(device, head.tr_final_layer)
        self.rot_final = MLP(device, head.rot_final_layer)
        self.odd_parity = bool(head.odd_parity)
        self.scale_by_sigma = bool(head.scale_by_sigma)
        pose_of = (torch.arange(sg.n_lig_padded).div(sg.n_lig, rounding_mode="floor")
                   .clamp(max=sg.n_poses - 1))
        import ttnn
        self.pose_of = ttnn.from_torch(pose_of.reshape(1, -1).to(torch.int32),
                                       dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
                                       device=device)
        self.ones_atoms = upload(device, torch.ones(sg.n_lig_padded))
        self.ones_poses = upload(device, torch.ones(sg.n_poses))

    def vectors(self, lig_pos, valid):
        """``head_vec = (lig_pos - center[pose_of]) * lig_valid``, as three columns."""
        import ttnn
        centers = pose_centers(lig_pos, self.sg.n_poses, self.sg.n_lig)
        per_atom = gather(centers, self.pose_of, self.sg.n_lig_padded)
        d = ttnn.multiply(ttnn.subtract(lig_pos, per_atom), valid)
        return [ttnn.slice(d, [0, k], [self.sg.n_lig_padded, k + 1]) for k in range(3)]

    def attributes(self, head_vec, lig_h, ns: int, sigma_row, src_idx):
        """``cattr``: the distance expansion and sigma through the MLP, then ``lig_h[src, :ns]``.

        *src_idx* is ``ce[1]`` when the head is ``fixed_center_conv`` and ``ce[0]`` otherwise,
        which the caller decides once, exactly as the host does.
        """
        import ttnn
        sig = ttnn.multiply(self.ones_atoms, sigma_row)
        x = ttnn.concat([self.smear(norm3(*head_vec)), sig], dim=1)
        return ttnn.concat([self.edge_mlp(x),
                            ttnn.slice(gather(lig_h, src_idx, self.sg.n_lig_padded),
                                       [0, 0], [self.sg.n_lig_padded, ns])], dim=1)

    def scale(self, out, sigma_row, tr_sigma: float, so3_norm: float, n_poses: int):
        """The host's norm-and-rescale of the conv output into ``(tr, rot)``.

        ``out`` is the ``[P, 12]`` (or ``[P, 6]`` when ``odd_parity``) convolution result.
        Returns two ``[P, 3]`` tensors. The division by the norm is the host's own: a zero
        vector would divide by zero there too, and the guard that handles it is
        :func:`guard`, which runs later on the sampler's side.
        """
        import ttnn
        col = lambda a, b: ttnn.slice(out, [0, a], [n_poses, b])
        if self.odd_parity:
            tr, rot = col(0, 3), col(3, 6)
        else:
            tr = ttnn.add(col(0, 3), col(6, 9))
            rot = ttnn.add(col(3, 6), col(9, 12))
        sig = ttnn.multiply(ttnn.slice(self.ones_poses, [0, 0], [n_poses, 1]), sigma_row)
        out_pair = []
        for vec, final in ((tr, self.tr_final), (rot, self.rot_final)):
            n = norm3(*[ttnn.slice(vec, [0, k], [n_poses, k + 1]) for k in range(3)])
            scale = final(ttnn.concat([n, sig], dim=1))
            out_pair.append(ttnn.multiply(ttnn.multiply(vec, ttnn.reciprocal(n)), scale))
        tr, rot = out_pair
        if self.scale_by_sigma:
            tr = ttnn.multiply(tr, 1.0 / tr_sigma)
            rot = ttnn.multiply(rot, so3_norm)
        return tr, rot


class NodeEncoder:
    """:class:`score_model.AtomEncoder` on the card, with its static half hoisted out.

    The encoder sums one embedding table per categorical atom feature and then runs one
    linear over ``cat([sum, continuous block])``. Atom types do not change over a rollout, so
    **the entire categorical sum is a per-complex constant**: it is evaluated once on the host
    in fp32 (an embedding lookup is a copy, so this is exact) and uploaded with the static
    continuous columns beside it. Only the sigma block changes per step, and that is one row
    broadcast down the node axis, so a step costs one concat and one GEMM -- no gathers on the
    card at all.

    This is hoisting a loop invariant, not skipping the model's work: every step still
    evaluates the same linear over the same values the host would have built.

    *node_x* is the host feature table **without** the sigma columns, categorical first, in
    the encoder's own column order. ``sigma_width=0`` means the features already carry
    everything the encoder takes, which is the receptor's case: `static_forward` passes
    ``rec_x`` straight in and adds the receptor's sigma term after the embedding stack
    instead, so the whole receptor node embedding is constant over the 19 steps.
    """

    def __init__(self, device, encoder, node_x: torch.Tensor, sigma_width: int = 0):
        n_cat = encoder.n_categorical
        expected = encoder.n_categorical + encoder.extra_dim - sigma_width
        if node_x.shape[1] != expected:
            raise ValueError(f"encoder takes {expected} static columns, got {node_x.shape[1]}")
        with torch.no_grad():
            total = None
            for i, emb in enumerate(encoder.atom_embedding_list):
                term = emb(node_x[:, i].long())
                total = term if total is None else total + term
            base = torch.cat([total, node_x[:, n_cat:]], dim=1)
        self.base = upload(device, base.float())
        self.sigma_width = sigma_width
        self.ones = upload(device, torch.ones(node_x.shape[0]))
        self.linear = MLP(device, torch.nn.Sequential(
            encoder.additional_features_embedder)) if encoder.extra_dim > 0 else None

    def __call__(self, sigma_row=None):
        """``[N, emb_dim]`` node features for this step."""
        import ttnn
        if self.linear is None:
            return self.base
        if self.sigma_width == 0:
            return self.linear(self.base)
        if sigma_row is None:
            raise ValueError("this encoder takes a sigma row per step")
        return self.linear(ttnn.concat([self.base, ttnn.multiply(self.ones, sigma_row)], dim=1))


class TorsionHead:
    """The torsion head around its two tensor products: bond frame, edge geometry, attributes.

    :class:`tt_bio.diffdock.heads.TorsionHead` builds, per step, the rotatable bonds' midpoints
    and axes, the bond-centre <- atom edge vectors, their radius mask and harmonics, the axis
    harmonics each edge pairs with in ``final_tp_tor``, and three scalar blocks per edge
    (embedded distance, the atom's first ``ns`` features, the bond's). All of that is here.
    ``final_tp_tor`` and ``tor_bond_conv`` are the two tensor products and are not.

    Both index sets (the bonds' two atoms, the complete bond x atom edge set) are fixed for the
    rollout, so they are uploaded once and every lookup is the exact fp32 gather. The released
    model does not set ``smooth_edges``, so the edge weight is 1 and is not built.
    """

    def __init__(self, device, head, sg, rot_bonds):
        import ttnn
        if head.smooth_edges:
            raise NotImplementedError("smooth_edges: the released model does not set it")
        idx = lambda t: ttnn.from_torch(t.reshape(1, -1).to(torch.int32), dtype=ttnn.uint32,
                                        layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        self.ns, self.sh_ls = head.ns, tuple(head.sh_ls)
        self.cutoff = float(head.lig_max_radius)
        self.nb = rot_bonds.shape[1]
        self.b0, self.b1 = idx(rot_bonds[0]), idx(rot_bonds[1])
        self.ne = sg.tor_edge_index.shape[1]
        self.edges = EdgeGeometry(device, sg.tor_edge_index[0], sg.tor_edge_index[1], self.ne)
        real = torch.zeros(self.ne)
        real[:sg.n_rot_bonds * sg.n_lig * sg.n_poses] = 1.0
        self.real = upload(device, real)
        self.smear = Smearing(device, head.lig_distance_expansion)
        self.edge_mlp = MLP(device, head.final_edge_embedding)
        self.final = MLP(device, head.tor_final_layer)
        self.scale_by_sigma = bool(head.scale_by_sigma)

    def bonds(self, lig_pos):
        """``(bond_pos, bond_vec)``, each ``[n_bonds, 3]``: midpoint and ``pos[b1] - pos[b0]``."""
        import ttnn
        a, b = gather(lig_pos, self.b0, self.nb), gather(lig_pos, self.b1, self.nb)
        return ttnn.multiply(ttnn.add(a, b), 0.5), ttnn.subtract(b, a)

    def __call__(self, lig_pos, lig_h):
        """``(edge_attr [E, 3 ns], edge_sh [E, sh], bond_sh [E, 5], mask [E, 1])`` for one step.

        ``mask`` is :meth:`StaticGraphs.tor_mask`: ``|v| <= lig_max_radius`` on the real edges,
        0 on the pad. ``bond_sh`` is each edge's own bond's ``2e`` harmonics, the second operand
        of ``final_tp_tor``.
        """
        import ttnn
        bond_pos, bond_vec = self.bonds(lig_pos)
        v = self.edges.vectors(bond_pos, lig_pos)
        dist = norm3(*v)
        mask = ttnn.multiply(self.edges.within(dist, self.cutoff), self.real)
        axis = [ttnn.slice(bond_vec, [0, k], [self.nb, k + 1]) for k in range(3)]
        bond_sh = gather(harmonics(*axis, ls=(2,)), self.edges.src, self.ne)
        ns = self.ns
        h = ttnn.slice(lig_h, [0, 0], [lig_h.shape[0], ns])
        bond_h = ttnn.add(gather(h, self.b0, self.nb), gather(h, self.b1, self.nb))
        attr = ttnn.concat([self.edge_mlp(self.smear(dist)),
                            gather(h, self.edges.dst, self.ne),
                            gather(bond_h, self.edges.src, self.ne)], dim=1)
        return attr, harmonics(*v, ls=self.sh_ls), bond_sh, mask

    def score(self, conv_out, tor_sigma: float):
        """``tor_final_layer(conv_out) * sqrt(torus_score_norm(tor_sigma))``, ``[n_bonds, 1]``.

        Every bond of a rollout shares one sigma per step, so the host's per-bond table lookup
        is one scalar, taken from the host's own table.
        """
        import ttnn
        tor = self.final(conv_out)
        if not self.scale_by_sigma:
            return tor
        from .schedules import torus_score_norm
        s = float(torus_score_norm(torch.tensor([tor_sigma], dtype=torch.float64))[0]) ** 0.5
        return ttnn.multiply(tor, s)


class FullTP:
    """:class:`tt_bio.diffdock.full_tensor_product.FullTensorProduct` on the card.

    The product has no weights, so it is one fixed bilinear map: ``out = (x2 (x) x1) @ M`` with
    ``M[(j, i), k] = d out_k / d(x1_i x2_j)``. ``M`` is read off the host module itself in
    float64 by feeding it every pair of basis vectors, so it is the host's own arithmetic and
    not a re-derivation of the Wigner and sort conventions. On the card the outer product is
    ``d2`` broadcast multiplies (fp32, vector engine, exact) and the contraction is one
    ``[E, d1 d2] @ [d1 d2, d_out]`` GEMM at HiFi4 with fp32 accumulation.
    """

    def __init__(self, device, module):
        import ttnn
        d1, d2 = module.in1_irreps.dim, module.in2_irreps.dim
        eye1 = torch.eye(d1, dtype=torch.float64)
        with torch.no_grad():
            rows = [module(eye1, torch.eye(d2, dtype=torch.float64)[j].expand(d1, d2))
                    for j in range(d2)]
        self.d1, self.d2 = d1, d2
        self.m = upload(device, torch.cat(rows, 0))          # row j * d1 + i
        self.ckc = ttnn.types.BlackholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
            fp32_dest_acc_en=True, packer_l1_acc=True)

    def __call__(self, x1, x2):
        """``[E, d1]`` and ``[E, d2]`` fp32 TILE tensors -> ``[E, d_out]``."""
        import ttnn
        n = x1.shape[0]
        outer = ttnn.concat([ttnn.multiply(x1, ttnn.slice(x2, [0, j], [n, j + 1]))
                             for j in range(self.d2)], dim=1)
        return ttnn.matmul(outer, self.m, compute_kernel_config=self.ckc)
# ---------------------------------------------------------------------------
# the persistent pose buffer: `loop.fixed_update` and Horn's Kabsch on the card
# ---------------------------------------------------------------------------

def _pose_reduce(col, n_poses: int, n_lig: int, width: int, how: str):
    """Reduce a per-atom ``[n, width]`` column over each pose's real atoms -> ``[n_poses, width]``.

    The same view :func:`pose_centers` and :func:`rec_keep` take: the pose axis is the leading
    one and every pose holds the same atom count, so this is a reshape and a reduction, with no
    index tensor and no scatter.
    """
    import ttnn
    real = n_poses * n_lig
    flat = ttnn.to_layout(ttnn.slice(col, [0, 0], [real, width]), ttnn.ROW_MAJOR_LAYOUT)
    block = ttnn.to_layout(ttnn.reshape(flat, [n_poses, n_lig, width]), ttnn.TILE_LAYOUT)
    out = ttnn.sum(block, dim=1) if how == "sum" else ttnn.mean(block, dim=1)
    return ttnn.reshape(out, [n_poses, width])


def unit3(cols):
    """:func:`tt_bio.diffdock.harmonics.unit` on three columns: divide by ``max(|v|, 1e-12)``."""
    import ttnn
    inv = ttnn.reciprocal(ttnn.clamp(norm3(*cols), min=_EPS))
    return [ttnn.multiply(c, inv) for c in cols]


def proper_rotation(k16, n_poses: int, squarings: int):
    """:func:`tt_bio.diffdock.kabsch.proper_rotation` on the card, Davenport's K as 16 columns.

    *k16* is K row-major: ``k16[4 a + b]`` is ``K[a, b]``, each a ``[n_poses, 1]`` column. The
    squarings run as a batched ``[P, 1, 4, 4]`` matmul, which is ten ops for the whole rollout
    step rather than the 640 column multiplies the unrolled form would dispatch.

    The dominant eigenvector is read off ``A^(2^squarings) = lambda v v^T`` by picking the column
    with the largest diagonal entry, as the host's ``argmax`` does. The pick here is the *first*
    index attaining the maximum (``eq_j`` times the running product of ``1 - eq_k``), so a tie
    resolves to one fixed column instead of a sum of two that could cancel, and the result never
    depends on a reduction order.
    """
    import ttnn
    shift = None                                   # Gershgorin: max_a sum_b |K[a, b]|
    for a in range(4):
        row = None
        for b in range(4):
            m = ttnn.abs(k16[4 * a + b])
            row = m if row is None else ttnn.add(row, m)
        shift = row if shift is None else ttnn.maximum(shift, row)
    cols = [ttnn.add(k16[4 * a + b], shift) if a == b else k16[4 * a + b]
            for a in range(4) for b in range(4)]
    mat = ttnn.reshape(ttnn.concat(cols, dim=1), [n_poses, 1, 4, 4])
    ckc = ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    tiny = 1.175494e-38                            # torch.finfo(float32).tiny, the host's clamp
    for _ in range(squarings):
        mat = ttnn.matmul(mat, mat, compute_kernel_config=ckc)
        scale = ttnn.max(ttnn.max(ttnn.abs(mat), dim=-1, keepdim=True), dim=-2, keepdim=True)
        mat = ttnn.multiply(mat, ttnn.reciprocal(ttnn.clamp(scale, min=tiny)))
    flat = ttnn.reshape(mat, [n_poses, 16])
    col = lambda a, b: ttnn.slice(flat, [0, 4 * a + b], [n_poses, 4 * a + b + 1])
    diag = [col(a, a) for a in range(4)]
    best = diag[0]
    for d in diag[1:]:
        best = ttnn.maximum(best, d)
    q, taken = [None] * 4, None
    for j in range(4):
        hit = ttnn.typecast(ttnn.eq(diag[j], best), ttnn.float32)
        pick = hit if taken is None else ttnn.multiply(hit, ttnn.add(ttnn.neg(taken), 1.0))
        taken = pick if taken is None else ttnn.add(taken, pick)
        for a in range(4):
            term = ttnn.multiply(col(a, j), pick)
            q[a] = term if q[a] is None else ttnn.add(q[a], term)
    sq = None
    for a in range(4):
        m = ttnn.multiply(q[a], q[a])
        sq = m if sq is None else ttnn.add(sq, m)
    two = ttnn.multiply(ttnn.reciprocal(ttnn.clamp(sq, min=tiny)), 2.0)
    r, i, j, k = q
    mul = lambda a, b: ttnn.multiply(two, ttnn.multiply(a, b))
    diag_entry = lambda a, b: ttnn.add(ttnn.neg(ttnn.add(mul(a, a), mul(b, b))), 1.0)
    return (diag_entry(j, k), ttnn.subtract(mul(i, j), mul(k, r)), ttnn.add(mul(i, k), mul(j, r)),
            ttnn.add(mul(i, j), mul(k, r)), diag_entry(i, k), ttnn.subtract(mul(j, k), mul(i, r)),
            ttnn.subtract(mul(i, k), mul(j, r)), ttnn.add(mul(j, k), mul(i, r)), diag_entry(i, j))


class PoseUpdate:
    """:func:`tt_bio.diffdock.loop.fixed_update` over the card's own pose buffer.

    The buffer is the ``[n_lig_padded, 3]`` fp32 tensor every other block already reads: pose
    major, pad rows at the end. A step rotates and translates each pose about its own centre,
    applies the rotatable-bond torsions one unrolled bond at a time, and re-superposes the
    flexible pose onto the rigid one with Horn's quaternion Kabsch, exactly as the torch twin
    does. Nothing here reads a value back to the host.

    Every index is fixed for the rollout and uploaded once: the pose of each atom, the two atoms
    of each bond (``StaticGraphs`` already tiles ``rot_bonds`` over poses, pose-major, so bond
    ``i`` of pose ``p`` is row ``p * n_rot_bonds + i``), and each bond's rotating-atom mask,
    tiled over poses into a ``[n_lig_padded, 1]`` 0/1 column. The mask chooses arithmetically
    rather than through ``ttnn.where``: every lane here is finite, and `where` evaluates both
    (`guard`).

    A pad atom belongs to the clamped last pose and a pad bond has both ends on the pad atom, so
    its axis is zero, :func:`unit3` sends it to zero and the rotation is the identity -- the same
    way the torch twin keeps the shapes static.
    """

    def __init__(self, device, sg, rot_bonds, mask_rotate, squarings: int | None = None):
        import ttnn
        from .kabsch import SQUARINGS
        from .sampling import _as_mask
        mask_rotate = _as_mask(mask_rotate)       # the graph builder hands it over as numpy
        self.device, self.sg = device, sg
        self.n, self.P, self.n_lig = sg.n_lig_padded, sg.n_poses, sg.n_lig
        self.nb = sg.n_rot_bonds
        self.squarings = SQUARINGS if squarings is None else squarings
        idx = lambda t: ttnn.from_torch(t.reshape(1, -1).to(torch.int32), dtype=ttnn.uint32,
                                        layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        pose_of = (torch.arange(self.n).div(self.n_lig, rounding_mode="floor")
                   .clamp(max=self.P - 1))
        self.pose_of = idx(pose_of)
        self.u, self.v, self.rot_mask = [], [], []
        for i in range(self.nb):
            rows = torch.arange(self.P) * self.nb + i
            self.u.append(idx(rot_bonds[0][rows]))
            self.v.append(idx(rot_bonds[1][rows]))
            col = torch.zeros(self.n)
            m = mask_rotate[i].to(torch.float32)
            col[:self.P * self.n_lig] = m.repeat(self.P)
            self.rot_mask.append(upload(device, col))

    def _per_atom(self, per_pose):
        """A ``[n_poses, d]`` tensor spread over the atoms, through the exact fp32 gather."""
        return gather(per_pose, self.pose_of, self.n)

    def _move(self, pos, m, pivot):
        """``(pos - pivot) @ R.T + pivot`` with R's nine entries *m* already per atom."""
        import ttnn
        d = ttnn.subtract(pos, pivot)
        cols = rotate([ttnn.slice(d, [0, k], [self.n, k + 1]) for k in range(3)], m)
        return ttnn.add(ttnn.concat(cols, dim=1), pivot)

    def rigid(self, pos, tr, rot):
        """The rigid half: rotate each pose about its centre by *rot*, then translate by *tr*."""
        import ttnn
        centre = self._per_atom(pose_centers(pos, self.P, self.n_lig))
        rcols = [ttnn.slice(rot, [0, k], [self.P, k + 1]) for k in range(3)]
        m = self._per_atom(ttnn.concat(list(axis_angle_matrix(*rcols)), dim=1))
        m = [ttnn.slice(m, [0, k], [self.n, k + 1]) for k in range(9)]
        return ttnn.add(self._move(pos, m, centre), self._per_atom(tr))

    def torsions(self, pos, tor):
        """The unrolled torsion updates. *tor* is ``[n_poses * n_rot_bonds, 1]``, pose-major."""
        import ttnn
        if not self.nb:
            return pos
        rows = ttnn.reshape(ttnn.slice(tor, [0, 0], [self.P * self.nb, 1]), [self.P, self.nb])
        for i in range(self.nb):
            a, b = gather(pos, self.u[i], self.P), gather(pos, self.v[i], self.P)
            d = ttnn.subtract(a, b)
            axis = unit3([ttnn.slice(d, [0, k], [self.P, k + 1]) for k in range(3)])
            angle = ttnn.slice(rows, [0, i], [self.P, i + 1])
            m = self._per_atom(ttnn.concat(
                list(axis_angle_matrix(*[ttnn.multiply(c, angle) for c in axis])), dim=1))
            m = [ttnn.slice(m, [0, k], [self.n, k + 1]) for k in range(9)]
            moved = self._move(pos, m, self._per_atom(b))
            keep = ttnn.add(ttnn.neg(self.rot_mask[i]), 1.0)
            pos = ttnn.add(ttnn.multiply(moved, self.rot_mask[i]), ttnn.multiply(pos, keep))
        return pos

    def align(self, flexible, rigid):
        """Horn's Kabsch, `b ~ a @ R.T + t.T`: superpose *flexible* back onto *rigid* per pose.

        The correlation matrix is nine per-pose sums of products of centred columns, taken with
        the same pose-axis reduction as the centres, so no ``[3, 3]`` batched transpose is built.
        """
        import ttnn
        ca, cb = pose_centers(flexible, self.P, self.n_lig), pose_centers(rigid, self.P, self.n_lig)
        da = ttnn.subtract(flexible, self._per_atom(ca))
        db = ttnn.subtract(rigid, self._per_atom(cb))
        acol = [ttnn.slice(da, [0, k], [self.n, k + 1]) for k in range(3)]
        bcol = [ttnn.slice(db, [0, k], [self.n, k + 1]) for k in range(3)]
        h = [_pose_reduce(ttnn.multiply(acol[a], bcol[b]), self.P, self.n_lig, 1, "sum")
             for a in range(3) for b in range(3)]
        xx, xy, xz, yx, yy, yz, zx, zy, zz = h
        add, sub = ttnn.add, ttnn.subtract
        k16 = [add(add(xx, yy), zz), sub(yz, zy), sub(zx, xz), sub(xy, yx),
               sub(yz, zy), sub(sub(xx, yy), zz), add(xy, yx), add(zx, xz),
               sub(zx, xz), add(xy, yx), sub(sub(yy, xx), zz), add(yz, zy),
               sub(xy, yx), add(zx, xz), add(yz, zy), sub(sub(zz, xx), yy)]
        r = proper_rotation(k16, self.P, self.squarings)
        per_atom = [ttnn.slice(self._per_atom(ttnn.concat(list(r), dim=1)), [0, k], [self.n, k + 1])
                    for k in range(9)]
        rotated = ttnn.concat(rotate(acol, per_atom), dim=1)      # (a - ca) @ R.T
        return ttnn.add(rotated, self._per_atom(cb))              # ... + cb, which is R(a-ca)+cb

    def __call__(self, pos, tr, rot, tor):
        """One pose update: rigid, torsions, re-superposition. Returns the new ``[n, 3]`` buffer."""
        rigid = self.rigid(pos, tr, rot)
        if not self.nb or tor is None:
            return rigid
        return self.align(self.torsions(rigid, tor), rigid)
# ---------------------------------------------------------------------------
# a stack of convolutions over edge features that never leave the card
# ---------------------------------------------------------------------------

def assemble_rows(device, parts, blocks, width: int):
    """Per-block ``[hi - lo, width]`` device columns -> one tensor in the padded edge layout.

    *parts* is one tensor per entry of *blocks*, in order, each holding that block's real rows;
    the dead rows between blocks are zeros, which is what their mask row means anyway. Owning
    the padding here is what lets a caller build its edge features per edge set and stay out of
    where the 32-row boundaries fall.
    """
    import ttnn
    if len(parts) != len(blocks):
        raise ValueError(f"{len(parts)} parts for {len(blocks)} blocks")
    out = []
    for part, (lo, hi, extra) in zip(parts, blocks):
        out.append(part)
        if extra:
            out.append(zeros(device, [extra, width]))
    return ttnn.concat(out, dim=0) if len(out) > 1 else out[0]


def tile_width(x):
    """Zero-pad a card tensor's columns to a multiple of 32, as :func:`device.to_device` does.

    The convolution's split first layer (:func:`device.fc1_blocks`) has ``pad32`` rows per
    block, for the node halves as well as the edge block, so every tensor entering a stack is
    widened first. The host form gets this free: ``device.to_device`` column-pads on upload, and
    the resident path, whose tensors never leave the card, has to do it itself.
    """
    from .device import pad32
    return pad_width(x.device(), x, pad32(x.shape[1]))


class ResidentStack:
    """:meth:`tt_bio.diffdock.device.DeviceConvModule.run_stack` with its inputs already on card.

    ``run_stack`` keeps the *node* features resident and uploads the edge features each call:
    it takes host tensors, pads them into the stack's edge layout and crosses them to the card.
    In the traced loop the edge features are produced on the card by :class:`TowerFeatures`, so
    there is nothing to upload and nothing to read back -- the stack has to take device tensors
    and return one.

    The edge layout is :func:`tt_bio.diffdock.device.stack_layout`, the same function
    ``run_stack`` uses, so the two forms cannot drift: a block of edges between two cut/split
    boundaries is padded out to a multiple of 32 with dead ``pad -> pad`` rows. The edge index
    is a rollout invariant, so every layer's gather index and scatter matrix is uploaded once
    here, at capture, rather than per step.

    :meth:`assemble` is how a caller gets its per-block columns into that layout: it owns the
    padding, so no caller has to know where the 32-row boundaries fall.
    """

    def __init__(self, mods, edge_index, plan, n_nodes: int):
        from .device import stack_layout
        from .graph import pad_count
        if len(mods) != len(plan):
            raise ValueError(f"{len(mods)} modules, {len(plan)} plan rows")
        self.mods, self.plan = list(mods), list(plan)
        self.n_nodes, self.n_p = n_nodes, pad_count(n_nodes)
        self.ei, self.pos, self.blocks = stack_layout(edge_index, self.n_p, self.plan)
        self.n_rows = self.ei.shape[1]
        self.graphs, self.groups = [], []
        for mod, (cut, splits) in zip(self.mods, self.plan):
            c = self.pos[cut]
            edges = [0, *(self.pos[x] for x in splits or ()), c]
            if len(edges) - 1 != len(mod.dev.tps):
                raise ValueError(f"{mod.name}: {len(edges) - 1} edge groups for "
                                 f"{len(mod.dev.tps)} FC blocks")
            self.graphs.append(mod._graph(self.ei[:, :c], self.n_p, self.n_p))
            self.groups.append((c, list(zip(edges, edges[1:]))))

    def assemble(self, device, parts, width: int):
        """Per-block device columns into this stack's padded layout; see :func:`assemble_rows`."""
        return assemble_rows(device, parts, self.blocks, width)

    def __call__(self, x, edge_attr, edge_sh, edge_mask, ns: int):
        """*x* ``[n_p, d]`` and the three ``[n_rows, *]`` edge tensors, all on the card.

        Returns the stack's node features on the card. No upload, no read-back, one program.
        """
        import ttnn
        rows = lambda t, a, b: ttnn.slice(t, [a, 0], [b, t.shape[1]])
        x, edge_attr, edge_sh = tile_width(x), tile_width(edge_attr), tile_width(edge_sh)
        for mod, graph, (cut, edges) in zip(self.mods, self.graphs, self.groups):
            x = mod.dev(x, graph, [rows(edge_attr, a, b) for a, b in edges],
                        rows(edge_sh, 0, cut), rows(edge_mask, 0, cut), ns=ns)
        return x


class ResidentConv:
    """One convolution whose destination node set is not its source, over card-side features.

    The two head convolutions are shaped this way: the centre head scatters every ligand atom
    into its own pose (``out_nodes = n_poses``) and the torsion head scatters bond-centre edges
    into bonds (``out_nodes = n_bonds``). :class:`ResidentStack` cannot carry them -- it assumes
    out nodes = in nodes, because that is what a stack does -- so they get this, which is the
    same contract with the two node counts separated and no read-back.
    """

    def __init__(self, mod, edge_index, n_out: int, n_in: int, groups=None):
        from .device import stack_layout
        from .graph import pad_count
        self.mod = mod
        self.n_out, self.n_out_p, self.n_in_p = n_out, pad_count(n_out), pad_count(n_in)
        plan = [(edge_index.shape[1], groups)]
        self.ei, self.pos, self.blocks = stack_layout(edge_index, self.n_out_p, plan,
                                                      self.n_in_p)
        self.n_rows = self.ei.shape[1]
        edges = [0, *(self.pos[x] for x in groups or ()), self.pos[edge_index.shape[1]]]
        if len(edges) - 1 != len(mod.dev.tps):
            raise ValueError(f"{mod.name}: {len(edges) - 1} edge groups for "
                             f"{len(mod.dev.tps)} FC blocks")
        self.groups = list(zip(edges, edges[1:]))
        self.graph = mod._graph(self.ei, self.n_out_p, self.n_in_p)

    def assemble(self, device, parts, width: int):
        """Per-block device columns into this layout; see :func:`assemble_rows`."""
        return assemble_rows(device, parts, self.blocks, width)

    def __call__(self, x, edge_attr, edge_sh, edge_mask, ns: int):
        """*x* ``[n_in_p, d]`` and the ``[n_rows, *]`` edge tensors on the card -> ``[n_out_p, d']``."""
        import ttnn
        rows = lambda t, a, b: ttnn.slice(t, [a, 0], [b, t.shape[1]])
        x, edge_attr, edge_sh = tile_width(x), tile_width(edge_attr), tile_width(edge_sh)
        return self.mod.dev(x, self.graph, [rows(edge_attr, a, b) for a, b in self.groups],
                            edge_sh, edge_mask, ns=ns)


# ---------------------------------------------------------------------------
# the whole step: `static_net.static_forward` with nothing on the host
# ---------------------------------------------------------------------------

def pad_rows(device, x, rows: int, width: int):
    """Zero-pad a ``[n, width]`` device tensor up to *rows*; the node axis a stack expects."""
    import ttnn
    n = x.shape[0]
    if n == rows:
        return x
    if n > rows:
        raise ValueError(f"{n} rows into {rows}")
    return ttnn.concat([x, zeros(device, [rows - n, width])], dim=0)


def pad_width(device, x, width: int):
    """Zero-pad a device tensor's column axis out to *width*: `static_net.pad_width`."""
    import ttnn
    w = x.shape[1]
    if w == width:
        return x
    return ttnn.concat([x, zeros(device, [x.shape[0], width - w])], dim=1)


class DeviceStep:
    """:func:`tt_bio.diffdock.static_net.static_forward` with no host tensor in the middle.

    The step the shipped device network runs today is host torch around 19 device
    convolutions: the harmonics, the distance expansions, the node embeddings, the masks and
    both heads' geometry are computed on the CPU and the node features cross the bus twice per
    convolution stack. This runs all of it on the card, from a pose buffer the card already
    holds to the three scores, and is the body a trace capture records
    (``device_resident_loop``).

    It is a composition, not new arithmetic: every piece is one of this module's blocks, each
    graded against its float64 torch twin on a card, and the order is `static_forward`'s own.
    The two things that are *not* re-derived here are the edge layout and the plan -- those
    come from :func:`tt_bio.diffdock.device.stack_layout` and
    :func:`tt_bio.diffdock.static_net.main_stack_plan`, which the host form uses too.

    What stays on the host: the per-step scalars (the sigmas and the sigma embedding row), which
    a traced rollout uploads once as a ``[19, *]`` table and slices by the step index at capture
    (:func:`tt_bio.diffdock.loop.step_inputs`), not per step.
    """

    def __init__(self, device, model, c, convs):
        import ttnn
        from .static_net import main_stack_plan
        sg = c.sg
        self.device, self.model, self.sg, self.c = device, model, sg, c
        self.ns, self.cfg = model.ns, model.config
        self.P, self.off = sg.n_poses, sg.rec_offset
        rei = c.rec_edge_index
        nl, nc, nr = (sg.lig_edge_index.shape[1], sg.cross_edge_index.shape[1], rei.shape[1])
        self.nl, self.nc = nl, nc

        self.masks = StepMasks(device, sg, nl, nc)
        self.towers = TowerFeatures(device, model, sg, nl, nc)
        self.lig_geo = EdgeGeometry(device, sg.lig_edge_index[0], sg.lig_edge_index[1], nl)
        self.cross_geo = EdgeGeometry(device, sg.cross_edge_index[0], sg.cross_edge_index[1], nc)
        self.rec_geo = EdgeGeometry(device, rei[0], rei[1], nr)

        # the receptor never moves: its vectors, edge embedding and harmonics are per complex
        self.rec_pos = upload(device, c.rec_pos.float())
        self.rec_eattr, self.rec_sh = self.towers.receptor_static(
            self.rec_geo.vectors(self.rec_pos))
        self.rec_sig_mlp = MLP(device, model.rec_sigma_embedding)
        self.rec_enc = NodeEncoder(device, model.rec_node_embedding, c.rec_x)
        self.lig_enc = NodeEncoder(device, model.lig_node_embedding, c.lig_x,
                                   sigma_width=self.cfg.sigma_embed_dim)

        bond = torch.zeros(nl, c.bond_attr.shape[1])
        bond[:sg.lig_n_bond_edges] = c.bond_attr.float()
        self.bond_block = upload(device, bond)
        # every constant column a step broadcasts over, uploaded once: a host-to-device copy
        # inside the body is work a captured trace would repeat for nothing
        self.ones_rec_edges = upload(device, torch.ones(nr))
        self.lig_valid = upload(device, c.lig_valid.float())
        self.keep_all = upload(device, torch.cat([torch.ones(sg.n_rec_total),
                                                  torch.zeros(sg.n_rec_padded - sg.n_rec_total)]))

        flat = lambda n: [(n, None)]
        self.rec_stack = ResidentStack(
            [convs[f"rec_emb_layers.{i}"] for i in range(len(model.rec_emb_layers))],
            rei, flat(nr) * len(model.rec_emb_layers), sg.n_rec_padded)
        # the stack carries one padding node past the receptor's own rows, so the column the
        # sigma term broadcasts over is the stack's node count, not `sg.n_rec_padded`
        self.ones_rec_nodes = upload(device, torch.ones(self.rec_stack.n_p))

        self.lig_stack = ResidentStack(
            [convs[f"lig_emb_layers.{i}"] for i in range(len(model.lig_emb_layers))],
            sg.lig_edge_index, flat(nl) * len(model.lig_emb_layers), sg.n_lig_padded)
        ei, plan, (s1, s2, s3) = main_stack_plan(model, sg, rei)
        self.main = ResidentStack([convs[f"conv_layers.{i}"] for i in range(len(model.conv_layers))],
                                  ei, plan, sg.n_lig_padded + sg.n_rec_padded)

        head = model.center_head
        self.center = CenterHead(device, head, sg, self.cfg.sigma_embed_dim)
        pose_of = (torch.arange(sg.n_lig_padded).div(sg.n_lig, rounding_mode="floor")
                   .clamp(max=self.P - 1))
        ce = torch.stack([pose_of, torch.arange(sg.n_lig_padded)])
        self.center_conv = ResidentConv(convs["center_head.final_conv"], ce, self.P,
                                        sg.n_lig_padded)
        self.center_src = ttnn.from_torch(
            (ce[1] if head.fixed_center_conv else ce[0]).reshape(1, -1).to(torch.int32),
            dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

        self.torsion = None
        if model.torsion_head is not None and sg.n_rot_bonds:
            th = model.torsion_head
            self.torsion = TorsionHead(device, th, sg, c.rot_bonds)
            self.tor_tp = FullTP(device, th.final_tp_tor)
            self.tor_conv = ResidentConv(convs["torsion_head.tor_bond_conv"],
                                         sg.tor_edge_index, c.rot_bonds.shape[1],
                                         sg.n_lig_padded)

    # -- the three towers ---------------------------------------------------------------

    def _receptor(self, sigma_row, keep):
        """``(rec_h, rec_eattr, rec_mask)``: the crop, the stack, then the sigma term."""
        import ttnn
        mask = StepMasks.rec_edges(keep, self.rec_geo.src, self.rec_geo.dst, self.rec_geo.n)
        base = self.rec_enc()
        h = pad_rows(self.device, base, self.rec_stack.n_p, base.shape[1])
        ea = self.rec_stack.assemble(self.device, [self.rec_eattr], self.rec_eattr.shape[1])
        sh = self.rec_stack.assemble(self.device, [self.rec_sh], self.rec_sh.shape[1])
        m = self.rec_stack.assemble(self.device, [mask], 1)
        h = self.rec_stack(h, ea, sh, m, self.ns)
        # the stack's trailing padding node is its own bookkeeping, not a receptor node: the
        # main stack concatenates `sg.n_rec_padded` receptor rows after the ligand's, and one
        # extra row here overflows its node count
        h = ttnn.slice(h, [0, 0], [self.sg.n_rec_padded, h.shape[1]])
        # the host adds the sigma term after the stack: rec_h[:, :ns] += rec_sig, and the same
        # row onto every receptor edge attribute
        sig = self.rec_sig_mlp(sigma_row)
        head = ttnn.add(ttnn.slice(h, [0, 0], [h.shape[0], self.ns]),
                        ttnn.multiply(ttnn.slice(self.ones_rec_nodes, [0, 0], [h.shape[0], 1]), sig))
        h = ttnn.concat([head, ttnn.slice(h, [0, self.ns], [h.shape[0], h.shape[1]])], dim=1)
        ea_edges = ttnn.add(self.rec_eattr, ttnn.multiply(self.ones_rec_edges, sig))
        return h, ea_edges, mask

    def _ligand(self, lig_pos, sigma_row):
        """``(lig_h, lig_eattr, lig_sh, lig_mask)`` for this step's positions."""
        vec = self.lig_geo.vectors(lig_pos)
        dist = norm3(*vec)
        mask = self.masks.lig(dist, self.cfg.lig_max_radius)
        eattr, sh = self.towers.ligand(vec, self.bond_block, sigma_row)
        base = self.lig_enc(sigma_row)
        h = pad_rows(self.device, base, self.lig_stack.n_p, base.shape[1])
        ea = self.lig_stack.assemble(self.device, [eattr], eattr.shape[1])
        esh = self.lig_stack.assemble(self.device, [sh], sh.shape[1])
        m = self.lig_stack.assemble(self.device, [mask], 1)
        return self.lig_stack(h, ea, esh, m, self.ns), eattr, sh, mask

    # -- the step -----------------------------------------------------------------------

    def __call__(self, lig_pos, sigma_row, tr_sigma: float, rot_sigma: float,
                 tor_sigma: float, crop_cutoff: float | None = None):
        """``(tr [P, 3], rot [P, 3], tor [n_bonds, 1])``, all on the card.

        *lig_pos* is the card's ``[n_lig_padded, 3]`` pose buffer and *sigma_row* this step's
        ``[1, sigma_embed_dim]`` timestep embedding.
        """
        import ttnn
        from .schedules import so3_score_norm
        sg, dev = self.sg, self.device
        cvec = self.cross_geo.vectors(lig_pos, self.rec_pos)
        cutoff = (3 * tr_sigma + 20) if self.cfg.dynamic_max_cross else self.cfg.cross_max_distance

        if crop_cutoff is None:
            keep = self.keep_all
        else:
            keep = rec_keep(self.masks.cross(*cvec, float(crop_cutoff)), self.P, sg.n_lig,
                            sg.n_rec, sg.n_rec_padded)
        rec_h, rec_eattr, rec_mask = self._receptor(sigma_row, keep)
        lig_h, lig_eattr, lig_sh, lig_mask = self._ligand(lig_pos, sigma_row)

        cross_eattr, cross_sh, rev_sh = self.towers.cross(cvec, sigma_row)
        cross_m = ttnn.multiply(self.masks.cross(*cvec, float(cutoff)),
                                gather(keep, self.cross_geo.dst, self.nc))

        node = ttnn.concat([pad_width(dev, lig_h, rec_h.shape[1]), rec_h], dim=0)
        node = pad_rows(dev, node, self.main.n_p, rec_h.shape[1])
        ea = self.main.assemble(dev, [lig_eattr, cross_eattr, rec_eattr, cross_eattr],
                                lig_eattr.shape[1])
        esh = self.main.assemble(dev, [lig_sh, cross_sh, self.rec_sh, rev_sh], lig_sh.shape[1])
        em = self.main.assemble(dev, [lig_mask, cross_m, rec_mask, cross_m], 1)
        node = self.main(node, ea, esh, em, self.ns)
        lig_h = ttnn.slice(node, [0, 0], [sg.n_lig_padded, node.shape[1]])

        head = self.model.center_head
        valid = self.lig_valid
        head_vec = self.center.vectors(lig_pos, valid)
        cattr = self.center.attributes(head_vec, lig_h, self.ns, sigma_row, self.center_src)
        hsh = harmonics(*head_vec, ls=tuple(head.sh_ls))
        # both heads hand the convolution a COMPLETE edge input: `CenterHead.attributes` has
        # already concatenated `lig_h[src, :ns]` and the torsion head its node and bond blocks,
        # exactly as `static_net` builds them for `final_conv`. So no `ns` here: the split first
        # layer is for the embedding stacks, whose edge input is the edge block alone.
        out = self.center_conv(lig_h,
                               self.center_conv.assemble(dev, [cattr], cattr.shape[1]),
                               self.center_conv.assemble(dev, [hsh], hsh.shape[1]),
                               self.center_conv.assemble(dev, [valid], 1), None)
        so3 = float(so3_score_norm(torch.tensor([rot_sigma], dtype=torch.float64))[0])
        tr, rot = self.center.scale(out, sigma_row, float(tr_sigma), so3, self.P)

        if self.torsion is None:
            return tr, rot, None
        attr, sh, bond_sh, tmask = self.torsion(lig_pos, lig_h)
        tsh = self.tor_tp(sh, bond_sh)
        conv = self.tor_conv(lig_h, self.tor_conv.assemble(dev, [attr], attr.shape[1]),
                             self.tor_conv.assemble(dev, [tsh], tsh.shape[1]),
                             self.tor_conv.assemble(dev, [tmask], 1), None)
        return tr, rot, self.torsion.score(conv, float(tor_sigma))


# ---------------------------------------------------------------------------
# the rollout: every step on the card, one persistent pose buffer
# ---------------------------------------------------------------------------

class DeviceRollout:
    """`loop.run_loop` over :class:`DeviceStep` and :class:`PoseUpdate`, the body a trace records.

    Everything that varies by step is fixed before the first one: the scalars come from
    :func:`tt_bio.diffdock.loop.step_inputs` and :func:`~tt_bio.diffdock.loop.coefficient_table`,
    the noise from :func:`~tt_bio.diffdock.loop.draw_noise` in the sampler's own RNG order, and
    the per-step sigma embedding rows and noise draws are uploaded here, once. A step is then
    ``guard -> perturb -> move`` on the card with no host value in it, and its result is copied
    into the one pose buffer :meth:`read_poses` reads.

    Step 0 reads the start poses from their own buffer rather than from the pose buffer, so
    the body is the same program however often it ran before: the eager warm-up pass a capture
    needs leaves the pose buffer one step on, and the captured step 0 starts from the start
    poses regardless.
    """

    def __init__(self, device, model, c, convs, start, mask_rotate, cfg, bounds, noise,
                 crop_beyond: float | None):
        import ttnn
        from .conv import pad_nodes
        from .features import timestep_embedding
        from .loop import coefficient_table, draw_noise, step_inputs
        self.device, self.sg = device, c.sg
        self.P, self.n_lig = start.shape[0], start.shape[1]
        self.nb = c.sg.n_rot_bonds
        self.step = DeviceStep(device, model, c, convs)
        self.update = PoseUpdate(device, c.sg, c.rot_bonds, mask_rotate)
        self.inputs = step_inputs(cfg, bounds, crop_beyond)
        table = coefficient_table(cfg, bounds, dtype=torch.float64)
        self.score_c, self.noise_c = table.score.tolist(), table.noise.tolist()
        tr, rot, tor = draw_noise(noise, table, self.P, self.nb if self.step.torsion else 0)
        mc = model.config
        self.sigma_rows = [upload(device, timestep_embedding(
            torch.tensor([s.t_tr]), mc.sigma_embed_dim, mc.embedding_scale).float())
            for s in self.inputs]
        self.z = [(upload(device, tr[t]), upload(device, rot[t]),
                   None if tor is None else upload(device, tor[t].reshape(-1)))
                  for t in range(table.steps)]
        lp = pad_nodes(start.reshape(self.P * self.n_lig, 3).float(), c.sg.n_lig_padded)
        self.start = upload(device, lp)
        self.pos = upload(device, lp)

    def _perturb(self, score, rows: int, width: int, t: int, head: int, z):
        """``score_coeff * guard(score) + noise_coeff * z`` over the real ``[rows, width]`` block."""
        import ttnn
        cols = guard([ttnn.slice(score, [0, k], [rows, k + 1]) for k in range(width)])
        out = ttnn.multiply(ttnn.concat(cols, dim=1) if width > 1 else cols[0],
                            self.score_c[t][head])
        if self.noise_c[t][head]:
            out = ttnn.add(out, ttnn.multiply(z, self.noise_c[t][head]))
        return out

    def body(self, t: int):
        """Step *t*: score, guard, perturb and move, written into the pose buffer."""
        import ttnn
        s = self.inputs[t]
        pos = self.start if t == 0 else self.pos
        tr, rot, tor = self.step(pos, self.sigma_rows[t], s.tr_sigma, s.rot_sigma, s.tor_sigma,
                                 crop_cutoff=s.crop_cutoff)
        ztr, zrot, ztor = self.z[t]
        tr = self._perturb(tr, self.P, 3, t, 0, ztr)
        rot = self._perturb(rot, self.P, 3, t, 1, zrot)
        if tor is not None:
            tor = self._perturb(tor, self.P * self.nb, 1, t, 2, ztor)
        ttnn.copy(self.update(pos, tr, rot, tor), self.pos)
        return self.pos

    def read_poses(self):
        """The pose buffer as ``[n_poses, n_lig, 3]`` host fp32, the one read of the rollout."""
        import ttnn
        flat = ttnn.to_torch(self.pos).float()[:self.P * self.n_lig, :3]
        return flat.reshape(self.P, self.n_lig, 3)
