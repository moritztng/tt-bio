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
        pad = ttnn.zeros([n_rec_padded - n_poses * n_rec, 1], dtype=ttnn.float32,
                         layout=ttnn.ROW_MAJOR_LAYOUT, device=cross_near.device())
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
        sig = ttnn.multiply(upload(self.device, torch.ones(n_poses)), sigma_row)
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
