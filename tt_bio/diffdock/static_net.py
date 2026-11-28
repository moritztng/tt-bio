"""The score network over fixed-shape graphs: every pose of a step in one forward, one shape for
the whole rollout.

:meth:`ScoreModel.forward` is upstream's function for one pose. It builds its radius, cross and
torsion graphs with ``nonzero``, so the edge count moves every step and nothing can be captured.
This module runs the *same submodules* over :class:`~tt_bio.diffdock.graph.StaticGraphs`: complete
edge sets built once from counts, with the per-step graphs as 0/1 edge masks. ``tp_scatter``
divides the mean by the mask sum, so a masked edge is gone exactly as an edge ``nonzero`` never
produced is gone, and the receptor crop is a node mask (D-dfd-orch-11).

Same arithmetic, different edge sets: the result matches the per-pose forward to float32
rounding, except for an edge exactly on a cutoff sphere (``<=`` here, ``<`` there), which a real
complex does not produce. Pad edges point at the pad node with mask 0, and a zero edge vector has
zero harmonics, so no NaN can ride a masked edge into the sum.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from tt_bio.diffdock.conv import pad_nodes
from tt_bio.diffdock.features import timestep_embedding
from tt_bio.diffdock.graph import StaticGraphs, pad_edge_index, tile_disjoint
from tt_bio.diffdock.harmonics import reversed_harmonics, spherical_harmonics
from tt_bio.diffdock.schedules import so3_score_norm


def pad_width(x: Tensor, width: int) -> Tensor:
    """Zero columns up to *width* (the ligand rows meet the wider receptor rows)."""
    return torch.nn.functional.pad(x, (0, width - x.shape[1]))


@dataclass
class StaticComplex:
    """Everything about one complex that the loop does not move, laid out for *n_poses*."""

    sg: StaticGraphs
    lig_x: Tensor            # [n_lig_padded, F] atom features, tiled over poses
    bond_attr: Tensor        # [lig_n_bond_edges, B] bond features, tiled
    rec_x: Tensor            # [n_rec_padded, F'] residue features (+ ESM), tiled
    rec_pos: Tensor          # [n_rec_padded, 3]
    rec_edge_index: Tensor   # [2, E_rec] receptor-local, tiled, padded
    rot_bonds: Tensor        # [2, n_bond_padded] atom indices of rotatable bonds, tiled
    lig_valid: Tensor        # [n_lig_padded] 1 on real atoms
    _receptor_base: tuple | None = field(default=None, repr=False)

    def receptor_base(self, model):
        """``(node embedding, edge embedding before sigma, harmonics)`` of the receptor graph.

        The receptor does not move, so these are the same for every pose and every step: they
        are computed on the first step and held. The crop is a mask and the sigma term is added
        afterwards, so neither is in here.
        """
        if self._receptor_base is None or self._receptor_base[0] is not model:
            rei = self.rec_edge_index
            rvec = self.rec_pos[rei[1]] - self.rec_pos[rei[0]]
            self._receptor_base = (model, (
                model.rec_node_embedding(self.rec_x),
                model.rec_edge_embedding(model.rec_distance_expansion(rvec.norm(dim=-1))),
                spherical_harmonics(rvec, model.sh_ls)))
        return self._receptor_base[1]

    @classmethod
    def build(cls, lig: dict, rec_x: Tensor, rec_pos: Tensor, rec_edges: Tensor,
              n_poses: int) -> "StaticComplex":
        n_lig, n_rec = lig["x"].shape[0], rec_pos.shape[0]
        rot = lig["edge_index"][:, lig["edge_mask"].bool()]
        sg = StaticGraphs.build(n_lig, n_rec, lig["edge_index"], rot.shape[1], n_poses)

        def tile(t, n):
            return pad_nodes(t.repeat(n_poses, *([1] * (t.dim() - 1))), n)

        rot_bonds = tile_disjoint(rot, n_poses, n_lig, n_lig)
        if sg.n_bond_padded:
            rot_bonds = pad_edge_index(rot_bonds, sg.n_bond_padded, sg.lig_pad_node)
        valid = torch.zeros(sg.n_lig_padded)
        valid[:sg.n_lig_total] = 1
        return cls(sg=sg, lig_x=tile(lig["x"], sg.n_lig_padded),
                   bond_attr=lig["edge_attr"].repeat(n_poses, 1),
                   rec_x=tile(rec_x, sg.n_rec_padded), rec_pos=tile(rec_pos, sg.n_rec_padded),
                   rec_edge_index=sg.rec_edge_index(rec_edges), rot_bonds=rot_bonds,
                   lig_valid=valid)


def _conv(lyr, node_attr, edge_index, edge_attr, edge_sh, mask, ns, groups=None, **kw):
    src, dst = edge_index
    e = torch.cat([edge_attr, node_attr[src, :ns], node_attr[dst, :ns]], -1)
    if groups is not None:
        e = list(torch.tensor_split(e, groups, dim=0))
    return lyr(node_attr, edge_index, e, edge_sh, edge_mask=mask, **kw)


def run_stack(layers, node_attr, edge_index, edge_attr, edge_sh, mask, ns, plan=None):
    """Convolutions in sequence over one edge set; *plan* is ``[(cut, splits)]`` per layer (the
    layer sees edges ``[:cut]``, FC groups split at *splits*), default all edges, one group.

    Card layers that are not grading run as one device-resident stack
    (:meth:`tt_bio.diffdock.device.DeviceConvModule.run_stack`), same arithmetic.
    """
    layers = list(layers)
    plan = plan or [(edge_index.shape[1], None)] * len(layers)
    if layers and all(getattr(lyr, "resident", False) for lyr in layers):
        return type(layers[0]).run_stack(layers, node_attr, edge_index, edge_attr, edge_sh,
                                         mask, ns, plan)
    for lyr, (cut, splits) in zip(layers, plan):
        node_attr = _conv(lyr, node_attr, edge_index[:, :cut], edge_attr[:cut], edge_sh[:cut],
                          mask[:cut], ns, splits)
    return node_attr


def main_stack_plan(model, sg, rec_edge_index):
    """The main convolution stack's edge index, per-layer plan and segment offsets.

    The stack runs over four blocks in this order: the ligand edges, the cross edges, the
    receptor edges shifted onto the receptor node block, and the cross edges reversed. The last
    layer only needs the rows that reach a ligand node, so it is cut at the end of the cross
    block; with ``differentiate_convolutions`` each layer splits its edges into FC groups at the
    block boundaries.

    Returns ``(ei, plan, (s1, s2, s3))``. Shared by :func:`static_forward`, which concatenates
    host edge features into it, and by the card-resident loop, which builds them on the card --
    one definition, so the two cannot disagree about which edges a layer sees.
    """
    rei, lei, cg = rec_edge_index, sg.lig_edge_index, sg.cross_edge_index_global()
    ei = torch.cat([lei, cg, rei + sg.rec_offset, cg.flip(0)], 1)
    s1 = lei.shape[1]
    s2, s3 = s1 + cg.shape[1], s1 + cg.shape[1] + rei.shape[1]
    plan = []
    for i in range(len(model.conv_layers)):
        last = i == len(model.conv_layers) - 1
        groups = ([s1] if last else [s1, s2, s3]) if model.config.differentiate_convolutions else None
        plan.append((s2 if last else ei.shape[1], groups))
    return ei, plan, (s1, s2, s3)


def static_forward(model, c: StaticComplex, lig_pos: Tensor, t_tr: float, tr_sigma, rot_sigma,
                   tor_sigma, crop_cutoff=None):
    """One step for every pose. *lig_pos* is ``[n_lig_padded, 3]`` (pose-major, pad rows 0).

    Returns ``(tr [P, 3], rot [P, 3], tor [P * n_rot])`` in :meth:`HostScoreNetwork.score_fn`'s
    order. *crop_cutoff* is upstream's per-step ``3 * tr_sigma + crop_beyond`` or None.
    """
    cfg, ns, sg = model.config, model.ns, c.sg
    P, dt = sg.n_poses, lig_pos.dtype
    tr_sigma, rot_sigma, tor_sigma = (torch.as_tensor(s, dtype=dt).reshape(1)
                                      for s in (tr_sigma, rot_sigma, tor_sigma))
    sigma_emb = timestep_embedding(torch.as_tensor([t_tr], dtype=dt), cfg.sigma_embed_dim,
                                   cfg.embedding_scale)
    lig_sig = sigma_emb.expand(sg.n_lig_padded, -1)

    # The receptor tower. Its graph never moves, but the crop is per pose, so the tower runs
    # per step as soon as a crop is on -- which is upstream's behaviour too (its cache is only
    # used on an uncropped step). The features going into it are held: the crop is a mask.
    rei = c.rec_edge_index
    keep = (sg.rec_keep(lig_pos, c.rec_pos, crop_cutoff) if crop_cutoff is not None else
            pad_nodes(torch.ones(sg.n_rec_total, 1, dtype=dt), sg.n_rec_padded)[:, 0])
    rec_mask = sg.rec_edge_mask(keep, rei)
    rec_h, rec_eattr, rec_sh = c.receptor_base(model)
    rec_h = run_stack(model.rec_emb_layers, rec_h, rei, rec_eattr, rec_sh, rec_mask, ns)
    rec_sig = model.rec_sigma_embedding(sigma_emb)
    rec_h = rec_h + 0
    rec_h[:, :ns] = rec_h[:, :ns] + rec_sig
    rec_eattr = rec_eattr + rec_sig

    # The ligand tower.
    lei = sg.lig_edge_index
    lig_mask = sg.lig_mask(lig_pos, cfg.lig_max_radius)
    lvec = lig_pos[lei[1]] - lig_pos[lei[0]]
    bond = torch.zeros(lei.shape[1], c.bond_attr.shape[1], dtype=dt)
    bond[:sg.lig_n_bond_edges] = c.bond_attr
    lig_eattr = model.lig_edge_embedding(torch.cat(
        [bond, lig_sig[lei[0]], model.lig_distance_expansion(lvec.norm(dim=-1))], 1))
    lig_sh = spherical_harmonics(lvec, model.sh_ls)
    lig_h = model.lig_node_embedding(torch.cat([c.lig_x, lig_sig], 1))
    lig_h = run_stack(model.lig_emb_layers, lig_h, lei, lig_eattr, lig_sh, lig_mask, ns)

    # Cross edges, then the main stack on cat([ligand rows, receptor rows]).
    cutoff = (tr_sigma * 3 + 20).expand(P) if cfg.dynamic_max_cross else cfg.cross_max_distance
    cei = sg.cross_edge_index
    cross_m = sg.cross_mask(lig_pos, c.rec_pos, cutoff) * keep[cei[1]]
    cvec = c.rec_pos[cei[1]] - lig_pos[cei[0]]
    cross_eattr = model.cross_edge_embedding(torch.cat(
        [lig_sig[cei[0]], model.cross_distance_expansion(cvec.norm(dim=-1))], 1))
    cross_sh = spherical_harmonics(cvec, model.sh_ls)
    rev_sh = reversed_harmonics(cross_sh, model.sh_ls)
    off = sg.rec_offset
    cg = sg.cross_edge_index_global()
    node = torch.cat([pad_width(lig_h, rec_h.shape[1]), rec_h], 0)
    ei, plan, _ = main_stack_plan(model, sg, rei)
    ea = torch.cat([lig_eattr, cross_eattr, rec_eattr, cross_eattr], 0)
    esh = torch.cat([lig_sh, cross_sh, rec_sh, rev_sh], 0)
    em = torch.cat([lig_mask, cross_m, rec_mask, cross_m], 0)
    node = run_stack(model.conv_layers, node, ei, ea, esh, em, ns, plan)
    lig_h = node[:off]

    # The centre head: every atom into its own pose's centre of mass, pad atoms masked out.
    head = model.center_head
    v = c.lig_valid.to(dt)
    pose_of = torch.arange(sg.n_lig_padded).div(sg.n_lig, rounding_mode="floor").clamp(max=P - 1)
    center = lig_pos[:sg.n_lig_total].reshape(P, sg.n_lig, 3).mean(1)
    head_vec = (lig_pos - center[pose_of]) * v[:, None]
    ce = torch.stack([pose_of, torch.arange(sg.n_lig_padded)])
    cattr = head.center_edge_embedding(torch.cat(
        [head.center_distance_expansion(head_vec.norm(dim=-1)), lig_sig], 1))
    src = ce[1] if head.fixed_center_conv else ce[0]
    cattr = torch.cat([cattr, lig_h[src, :ns]], -1)
    out = head.final_conv(lig_h, ce, cattr, spherical_harmonics(head_vec, head.sh_ls),
                          out_nodes=P, edge_mask=v)
    tr, rot = ((out[:, :3], out[:, 3:6]) if head.odd_parity else
               (out[:, :3] + out[:, 6:9], out[:, 3:6] + out[:, 9:]))
    g = sigma_emb.expand(P, -1)
    trn = torch.linalg.vector_norm(tr, dim=1, keepdim=True)
    tr = tr / trn * head.tr_final_layer(torch.cat([trn, g], 1))
    rotn = torch.linalg.vector_norm(rot, dim=1, keepdim=True)
    rot = rot / rotn * head.rot_final_layer(torch.cat([rotn, g], 1))
    if head.scale_by_sigma:
        tr = tr / tr_sigma
        rot = rot * so3_score_norm(rot_sigma).to(dt)

    if model.torsion_head is None or sg.n_rot_bonds == 0:
        return tr, rot, lig_pos.new_zeros(0)
    b = c.rot_bonds
    bond_pos = (lig_pos[b[0]] + lig_pos[b[1]]) / 2
    tm = sg.tor_mask(lig_pos, bond_pos, cfg.lig_max_radius)
    tor = model.torsion_head(lig_h, lig_pos, b, sg.tor_edge_index,
                             tor_sigma.expand(b.shape[1]), edge_mask=tm)
    return tr, rot, tor[:sg.n_rot_bonds_total]
