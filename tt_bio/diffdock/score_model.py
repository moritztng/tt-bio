"""The assembled DiffDock-L score network: a complex in, ``(tr, rot, tor)`` out.

This is upstream's ``models/cg_model.TensorProductScoreModel`` (``CGModel``) with every piece
taken from this package rather than from e3nn and torch_cluster: the irrep algebra, the
tensor products, the convolution and its scatter, the harmonics, the featurisers, the
noise-schedule tables and the three output heads are each graded against upstream's own
module with the released v1.1 checkpoint before they are wired together here.

Shape of the computation, in upstream's order:

1. **Receptor tower**, cached across diffusion steps: residue encoder -> ``num_prot_emb_layers``
   convolutions on the fixed receptor graph. The receptor does not move while the ligand is
   being denoised, so only the per-step sigma term is re-added to the cached features --
   ``rec[:, :ns] += rec_sigma_emb``. That cache is why the receptor tower is not in the
   per-step cost.
2. **Ligand tower**, per step: atom encoder -> ``num_prot_emb_layers`` convolutions on the
   ligand graph, which is *bonds concatenated with a radius graph*, in that order, with the
   bond features zero-padded on the radius edges. One edge set, one FC block, not two groups.
3. **Cross graph and the main stack**: ligand and receptor nodes are concatenated into one
   node set and the edges into one edge set in the fixed order
   ``[ligand, lig->rec, receptor, rec->lig]``, whose three boundaries ``s1, s2, s3`` are what
   the last convolution slices on -- it runs on ``[:s2]`` only, i.e. it updates the ligand
   from the ligand and the receptor but never the receptor itself.
4. **Heads**: :class:`~tt_bio.diffdock.heads.CenterHead` for tr/rot and
   :class:`~tt_bio.diffdock.heads.TorsionHead` for the torsions.

Three things that are easy to get wrong here and silent if wrong:

* **The reverse cross edges reuse the forward edges' features but the *reversed* harmonics**
  (``rev_edge_sh`` is the harmonics of ``-edge_vec``). Same ``edge_attr`` tensor, same FC
  block, different geometry -- not a second embedding.
* **``dynamic_max_cross``** recomputes the cross cutoff as ``3*tr_sigma + 20`` every step, so
  the cross edge count changes along the schedule. It is not a fixed graph.
* **The last convolution's edge slice** is ``[:s2]``, not the whole set, and its edge features
  are rebuilt from the *current* node features. Running it on everything loads cleanly and
  quietly changes every ligand node.

Upstream builds its neighbour lists with ``torch_cluster``, whose ``radius_graph`` caps each
node at 32 neighbours by default. This port takes every neighbour inside the cutoff, which is
the same edge set wherever the cap does not bind and a superset where it does; see
``state/dfd/TRACE.md`` for the one PoseBusters complex in the graded set where it binds.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from tt_bio.diffdock.config import DiffDockConfig
from tt_bio.diffdock.constants import lig_feature_dims, rec_residue_feature_dims
from tt_bio.diffdock.conv import TensorProductConvLayer
from tt_bio.diffdock.features import GaussianSmearing, timestep_embedding
from tt_bio.diffdock.harmonics import spherical_harmonics
from tt_bio.diffdock.heads import CenterHead, TorsionHead, _degrees, _mlp


class AtomEncoder(nn.Module):
    """Sum of one embedding table per categorical feature, then one linear over the sum
    concatenated with the continuous block (sigma embedding, ESM-2 block, scalar features).

    The sum comes first and the linear sees ``emb_dim + extra``, not the raw categories --
    upstream's ``models/layers.AtomEncoder``. Getting the concatenation order wrong loads
    cleanly because the linear's width is the same either way.
    """

    def __init__(self, emb_dim: int, feature_dims, sigma_embed_dim: int,
                 lm_embedding_dim: int = 0):
        super().__init__()
        self.n_categorical = len(feature_dims[0])
        self.extra_dim = feature_dims[1] + sigma_embed_dim + lm_embedding_dim
        self.atom_embedding_list = nn.ModuleList(
            [nn.Embedding(dim, emb_dim) for dim in feature_dims[0]])
        if self.extra_dim > 0:
            self.additional_features_embedder = nn.Linear(self.extra_dim + emb_dim, emb_dim)

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[1] != self.n_categorical + self.extra_dim:
            raise ValueError(
                f"encoder takes {self.n_categorical + self.extra_dim} columns, got {x.shape[1]}")
        out = 0
        for i, emb in enumerate(self.atom_embedding_list):
            out = out + emb(x[:, i].long())
        if self.extra_dim > 0:
            out = self.additional_features_embedder(
                torch.cat([out, x[:, self.n_categorical:]], dim=1))
        return out


def radius_edges(pos: Tensor, cutoff: float, *, loop: bool = False) -> Tensor:
    """Every ordered pair inside *cutoff*, as ``[2, E]`` of ``(src, dst)``.

    Single complex only -- the rollout batches poses of one complex, which share a graph, so
    there is no cross-graph pair to exclude.

    **Upstream caps a node at 32 neighbours here and this does not, deliberately.**
    ``torch_cluster.radius_graph`` keeps ``max_num_neighbors=32`` by default (upstream raises
    it to 10000 only for the cross graph, so the cap on this one and on the bond-conv graph is
    deliberate or forgotten, but it is what the weights were trained with). Matching it is not
    achievable by any clean reimplementation, measured on a 60-point cloud with every true
    degree at 57-59:

    * the kept 32 are **neither the first 32 by index nor the nearest 32** -- they fall out of
      the CPU kernel's cell-list traversal order (0 of 60 nodes kept an index-order prefix);
    * the selection is **not permutation-equivariant**: relabelling the same points changes 16
      of 1920 kept edges, so upstream's own score depends on ligand atom order once the cap
      binds.

    Taking every neighbour is therefore the defensible choice: it is deterministic, invariant
    to atom order, and a superset of whatever upstream kept. The divergence is inactive on
    every complex graded so far -- the densest golden ligand (``8AEU_M0L``, 41 heavy atoms)
    peaks at degree 29 against the cap of 32 -- but it WILL bind on a large compact ligand, and
    when it does the gap is upstream's artifact, not this function's error. Quantify it against
    the accuracy bar on such a ligand before changing anything here.
    """
    d = torch.cdist(pos, pos)
    mask = d < cutoff
    if not loop:
        mask.fill_diagonal_(False)
    dst, src = mask.nonzero(as_tuple=True)
    return torch.stack([src, dst])


def cross_edges(lig_pos: Tensor, rec_pos: Tensor, cutoff) -> Tensor:
    """``[2, E]`` of ``(ligand atom, receptor residue)`` inside *cutoff*, upstream's row order."""
    d = torch.cdist(lig_pos, rec_pos)
    c = cutoff if torch.is_tensor(cutoff) else torch.as_tensor(
        cutoff, dtype=d.dtype, device=d.device)
    src, dst = (d < c).nonzero(as_tuple=True)
    return torch.stack([src, dst])


class ScoreModel(nn.Module):
    """The released score checkpoint, assembled. Inference only."""

    def __init__(self, config: DiffDockConfig, lm_embedding_dim: int = 1280):
        super().__init__()
        if config.all_atoms or config.old_model:
            raise ValueError("ScoreModel is the residue-level score network (CGModel); "
                             "the all-atom confidence model is AAOldModel")
        self.config = config
        ns, nv = config.ns, config.nv
        self.ns, self.nv = ns, nv
        sh = str(config.sh_irreps)
        self.sh = sh
        self.sh_ls = _degrees(sh)

        self.lig_node_embedding = AtomEncoder(ns, lig_feature_dims, config.sigma_embed_dim)
        self.lig_edge_embedding = _mlp(
            config.in_lig_edge_features + config.sigma_embed_dim + config.distance_embed_dim,
            ns, ns, act_first=True)
        self.rec_node_embedding = AtomEncoder(ns, rec_residue_feature_dims, 0,
                                              lm_embedding_dim=lm_embedding_dim)
        self.rec_edge_embedding = _mlp(config.distance_embed_dim, ns, ns, act_first=True)
        self.rec_sigma_embedding = _mlp(config.sigma_embed_dim, ns, ns, act_first=True)
        self.cross_edge_embedding = _mlp(
            config.sigma_embed_dim + config.cross_distance_embed_dim, ns, ns, act_first=True)

        self.lig_distance_expansion = GaussianSmearing(0.0, config.lig_max_radius,
                                                       config.distance_embed_dim)
        self.rec_distance_expansion = GaussianSmearing(0.0, config.rec_max_radius,
                                                       config.distance_embed_dim)
        self.cross_distance_expansion = GaussianSmearing(0.0, config.cross_max_distance,
                                                         config.cross_distance_embed_dim)

        seq = config.irrep_seq
        last = len(seq) - 1
        def layer(i, n_edge_features, groups=1):
            return TensorProductConvLayer(
                seq[min(i, last)], sh, seq[min(i + 1, last)],
                n_edge_features=n_edge_features, hidden_features=n_edge_features,
                residual=True, batch_norm=config.batch_norm, edge_groups=groups,
                tp_weights_layers=config.tp_weights_layers)

        self.rec_emb_layers = nn.ModuleList(
            [layer(i, 3 * ns) for i in range(config.num_prot_emb_layers)])
        if not config.embed_also_ligand:
            raise ValueError("embed_also_ligand is set by the released checkpoint")
        self.lig_emb_layers = nn.ModuleList(
            [layer(i, 3 * ns) for i in range(config.num_prot_emb_layers)])
        # differentiate_convolutions: each conv layer gives the four edge categories their
        # own FC weight block, and the last layer has only two because it runs on [:s2].
        self.conv_layers = nn.ModuleList(
            [layer(config.num_prot_emb_layers + j, 3 * ns, config.conv_edge_groups(j))
             for j in range(config.num_conv_layers)])

        out_last = seq[min(config.num_prot_emb_layers + config.num_conv_layers, last)]
        head_kw = dict(in_irreps=out_last, sh_irreps=sh, ns=ns,
                       distance_embed_dim=config.distance_embed_dim,
                       odd_parity=config.odd_parity, scale_by_sigma=config.scale_by_sigma,
                       batch_norm=config.batch_norm)
        self.center_head = CenterHead(sigma_embed_dim=config.sigma_embed_dim,
                                      center_max_distance=config.center_max_distance,
                                      fixed_center_conv=True, **head_kw)
        self.torsion_head = None if config.no_torsion else TorsionHead(
            lig_max_radius=config.lig_max_radius,
            distance_expansion=self.lig_distance_expansion, **head_kw)

    # -- the two towers -----------------------------------------------------------------

    def receptor_tower(self, rec_x: Tensor, rec_pos: Tensor, rec_edge_index: Tensor):
        """The part that does not change along the diffusion: run it once per complex."""
        src, dst = rec_edge_index
        edge_vec = rec_pos[dst] - rec_pos[src]
        edge_attr = self.rec_edge_embedding(
            self.rec_distance_expansion(edge_vec.norm(dim=-1)))
        edge_sh = spherical_harmonics(edge_vec, self.sh_ls)
        node_attr = self.rec_node_embedding(rec_x)
        for lyr in self.rec_emb_layers:
            e = torch.cat([edge_attr, node_attr[src, :self.ns], node_attr[dst, :self.ns]], -1)
            node_attr = lyr(node_attr, rec_edge_index, e, edge_sh)
        return node_attr, edge_attr, edge_sh

    def ligand_tower(self, lig_x: Tensor, lig_pos: Tensor, bond_index: Tensor,
                     bond_attr: Tensor, node_sigma_emb: Tensor):
        rad = radius_edges(lig_pos, self.config.lig_max_radius)
        edge_index = torch.cat([bond_index, rad], dim=1)
        edge_attr = torch.cat(
            [bond_attr, bond_attr.new_zeros((rad.shape[1], bond_attr.shape[1]))], dim=0)
        src, dst = edge_index
        edge_vec = lig_pos[dst] - lig_pos[src]
        edge_attr = torch.cat([edge_attr, node_sigma_emb[src],
                               self.lig_distance_expansion(edge_vec.norm(dim=-1))], dim=1)
        edge_sh = spherical_harmonics(edge_vec, self.sh_ls)

        node_attr = self.lig_node_embedding(torch.cat([lig_x, node_sigma_emb], dim=1))
        edge_attr = self.lig_edge_embedding(edge_attr)
        for lyr in self.lig_emb_layers:
            e = torch.cat([edge_attr, node_attr[src, :self.ns], node_attr[dst, :self.ns]], -1)
            node_attr = lyr(node_attr, edge_index, e, edge_sh)
        return node_attr, edge_index, edge_attr, edge_sh

    # -- the whole network --------------------------------------------------------------

    def forward(self, lig_x: Tensor, lig_pos: Tensor, bond_index: Tensor, bond_attr: Tensor,
                edge_mask: Tensor, rec_x: Tensor, rec_pos: Tensor, rec_edge_index: Tensor,
                t_tr: float, tr_sigma: Tensor, rot_sigma: Tensor, tor_sigma: Tensor,
                receptor_cache=None):
        """One pose. *edge_mask* selects the rotatable bonds out of *bond_index*'s columns.

        Returns ``(tr [1,3], rot [1,3], tor [n_rot_bonds])``. *receptor_cache* is whatever a
        previous call returned in ``.receptor_cache``; passing it back skips the receptor
        tower, which is exactly what upstream's ``data['receptor'].rec_node_attr`` cache does.
        """
        cfg = self.config
        n_lig = lig_pos.shape[0]
        t = torch.as_tensor([t_tr], dtype=lig_pos.dtype, device=lig_pos.device)
        sigma_emb = timestep_embedding(t, cfg.sigma_embed_dim, cfg.embedding_scale)
        node_sigma_emb = sigma_emb.expand(n_lig, -1)

        if receptor_cache is None:
            receptor_cache = self.receptor_tower(rec_x, rec_pos, rec_edge_index)
        rec_node_attr, rec_edge_attr, rec_edge_sh = receptor_cache
        self.receptor_cache = receptor_cache

        rec_sigma_emb = self.rec_sigma_embedding(sigma_emb)
        rec_node_attr = rec_node_attr + 0
        rec_node_attr[:, :self.ns] = rec_node_attr[:, :self.ns] + rec_sigma_emb
        rec_edge_attr = rec_edge_attr + rec_sigma_emb

        lig_node_attr, lig_edge_index, lig_edge_attr, lig_edge_sh = self.ligand_tower(
            lig_x, lig_pos, bond_index, bond_attr, node_sigma_emb)

        cutoff = (tr_sigma * 3 + 20).reshape(1) if cfg.dynamic_max_cross \
            else cfg.cross_max_distance
        lr = cross_edges(lig_pos, rec_pos, cutoff)
        lr_vec = rec_pos[lr[1]] - lig_pos[lr[0]]
        lr_attr = self.cross_edge_embedding(torch.cat(
            [node_sigma_emb[lr[0]], self.cross_distance_expansion(lr_vec.norm(dim=-1))], dim=1))
        lr_sh = spherical_harmonics(lr_vec, self.sh_ls)
        rev_lr_sh = spherical_harmonics(-lr_vec, self.sh_ls)

        node_attr = torch.cat([lig_node_attr, rec_node_attr], dim=0)
        lr_shifted = torch.stack([lr[0], lr[1] + n_lig])
        edge_index = torch.cat([lig_edge_index, lr_shifted, rec_edge_index + n_lig,
                                torch.flip(lr_shifted, dims=[0])], dim=1)
        edge_attr = torch.cat([lig_edge_attr, lr_attr, rec_edge_attr, lr_attr], dim=0)
        edge_sh = torch.cat([lig_edge_sh, lr_sh, rec_edge_sh, rev_lr_sh], dim=0)
        s2 = lig_edge_index.shape[1] + lr_shifted.shape[1]

        s1 = lig_edge_index.shape[1]
        s3 = s2 + rec_edge_index.shape[1]
        for i, lyr in enumerate(self.conv_layers):
            last_layer = i == len(self.conv_layers) - 1
            cut = s2 if last_layer else edge_index.shape[1]
            idx = edge_index[:, :cut]
            e = torch.cat([edge_attr[:cut], node_attr[idx[0], :self.ns],
                           node_attr[idx[1], :self.ns]], -1)
            if cfg.differentiate_convolutions:
                bounds = [s1, s2] if last_layer else [s1, s2, s3]
                e = list(torch.tensor_split(e, bounds, dim=0))
                if last_layer:
                    e = e[:2]
            node_attr = lyr(node_attr, idx, e, edge_sh[:cut])
        lig_node_attr = node_attr[:n_lig]

        batch = torch.zeros(n_lig, dtype=torch.long, device=lig_pos.device)
        tr, rot = self.center_head(lig_node_attr, lig_pos, batch, 1, node_sigma_emb,
                                   sigma_emb, tr_sigma.reshape(1), rot_sigma.reshape(1))

        if self.torsion_head is None or int(edge_mask.sum()) == 0:
            return tr, rot, lig_pos.new_zeros(0)
        bonds = bond_index[:, edge_mask]
        bond_pos = (lig_pos[bonds[0]] + lig_pos[bonds[1]]) / 2
        d = torch.cdist(bond_pos, lig_pos)
        rows, cols = (d < cfg.lig_max_radius).nonzero(as_tuple=True)
        tor = self.torsion_head(lig_node_attr, lig_pos, bonds, torch.stack([rows, cols]),
                                tor_sigma.reshape(1).expand(bonds.shape[1]))
        return tr, rot, tor


#: ``checkpoint prefix -> ours``. Upstream keeps the head modules at the top level of
#: ``CGModel``; this port groups them into :class:`CenterHead` and :class:`TorsionHead`, which
#: is the only renaming between the released state dict and this module.
HEAD_PREFIXES = {
    "center_distance_expansion.": "center_head.center_distance_expansion.",
    "center_edge_embedding.": "center_head.center_edge_embedding.",
    "final_conv.": "center_head.final_conv.",
    "tr_final_layer.": "center_head.tr_final_layer.",
    "rot_final_layer.": "center_head.rot_final_layer.",
    "final_edge_embedding.": "torsion_head.final_edge_embedding.",
    "tor_bond_conv.": "torsion_head.tor_bond_conv.",
    "tor_final_layer.": "torsion_head.tor_final_layer.",
    "lig_distance_expansion.": "torsion_head.lig_distance_expansion.",
}

#: Keys the released checkpoint carries that are e3nn bookkeeping, not parameters: the
#: zero-length weight of an unweighted product, its output mask, and the Wigner 3j constants
#: e3nn compiles into the TorchScript submodule. This port keeps 3j at float64 in its own
#: table, so loading e3nn's fp32 copies would be a silent precision regression.
SKIP_SUBSTRINGS = ("tp.weight", "tp.output_mask", "_w3j", "final_tp_tor.")


def remap_checkpoint(state_dict: dict) -> dict:
    """The released ``CGModel`` state dict in this module's key names."""
    out = {}
    for k, v in state_dict.items():
        if any(s in k for s in SKIP_SUBSTRINGS):
            continue
        for src, dst in HEAD_PREFIXES.items():
            if k.startswith(src):
                k = dst + k[len(src):]
                break
        out[k] = v
    return out


def load_released(state_dict: dict, config: DiffDockConfig, *,
                  lm_embedding_dim: int = 1280) -> "ScoreModel":
    """Build the network and load the released weights into it, strictly.

    Strict is the point: a head this port grouped differently, a missing bias or an encoder
    whose categorical tables are in another order all show up here rather than as a number
    that is merely wrong.
    """
    model = ScoreModel(config, lm_embedding_dim=lm_embedding_dim)
    missing, unexpected = model.load_state_dict(remap_checkpoint(state_dict), strict=False)
    missing = [k for k in missing if "lig_distance_expansion" not in k]
    if missing or unexpected:
        raise ValueError(f"checkpoint does not fit: missing={missing[:8]} "
                         f"unexpected={unexpected[:8]}")
    return model.eval()


class HostScoreNetwork:
    """:class:`ScoreModel` behind the interface ``tt_bio.diffdock.sampling.sample`` expects.

    ``network.config`` and ``network.score_fn(graph)``, as documented in
    :mod:`tt_bio.diffdock.network`. The receptor graph, the ESM-2 embedding and the per-step
    crop live in the closure, which is why the device path can replace this class without the
    rollout changing.

    The receptor tower is cached **only when the checkpoint sets no ``crop_beyond``**. With a
    crop the kept residue set moves with the schedule *and* with the pose, so the tower's
    input is different every step -- upstream recomputes it there too, for the same reason.
    """

    def __init__(self, model: ScoreModel, lm_embeddings: Tensor | None = None):
        self.model = model
        self.config = model.config
        self.lm_embeddings = lm_embeddings

    def _receptor_x(self, receptor: dict) -> Tensor:
        x = receptor["x"]
        if self.lm_embeddings is None:
            return x
        if self.lm_embeddings.shape[0] != x.shape[0]:
            raise ValueError(f"{self.lm_embeddings.shape[0]} embedding rows for "
                             f"{x.shape[0]} residues")
        return torch.cat([x, self.lm_embeddings.to(x.dtype)], dim=1)

    def score_fn(self, graph):
        from tt_bio.diffdock.sampling import crop_beyond

        lig, rec = graph.ligand, graph.receptor
        full_rec_x = self._receptor_x(rec)
        cache = None

        def score(step):
            nonlocal cache
            tr_s = torch.as_tensor(step.tr_sigma, dtype=torch.float32)
            rot_s = torch.as_tensor(step.rot_sigma, dtype=torch.float32)
            tor_s = torch.as_tensor(step.tor_sigma, dtype=torch.float32)
            trs, rots, tors = [], [], []
            for pose in step.lig_pos:
                r, rx = rec, full_rec_x
                if step.crop_cutoff is not None:
                    keep = torch.any(
                        torch.cdist(rec["pos"], pose.float()) < step.crop_cutoff, dim=1)
                    r = crop_beyond(rec, pose, step.crop_cutoff)
                    rx = full_rec_x[keep]
                use = None if step.crop_cutoff is not None else cache
                tr, rot, tor = self.model(
                    lig["x"], pose.float(), lig["edge_index"], lig["edge_attr"],
                    lig["edge_mask"], rx, r["pos"], r["edge_index"],
                    step.t_tr, tr_s, rot_s, tor_s, receptor_cache=use)
                if step.crop_cutoff is None:
                    cache = self.model.receptor_cache
                trs.append(tr)
                rots.append(rot)
                tors.append(tor)
            return (torch.cat(trs), torch.cat(rots), torch.cat(tors))

        return score

    def confidence(self, graph, poses: Tensor):
        """No confidence model on this branch yet; ranking falls back to sample order."""
        return None
