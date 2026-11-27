"""The DiffDock-L confidence model: the network that ranks the sampled poses.

Upstream is ``models/old_aa_model.AAOldModel`` built with ``confidence_mode=True``. The
released v1.1 confidence checkpoint is ``best_model_epoch75.pt``, 4,778,902 parameters,
``ns=24, nv=6, 5 conv layers``, all-atom receptor. It runs **once per complex**, over the
final poses of a completed rollout, and emits ``[N, 2]``; upstream ranks on column 0 only
(verified here against ``state/dfd/goldens/*/poses.npz``, where ``confidence_raw[:, 0]``
equals the stored ``confidence`` and ``order`` equals ``argsort(-confidence)``).

Everything equivariant is reused from the score-model port: :class:`conv.TensorProductConvLayer`,
:mod:`tensor_product`, :mod:`irreps`, :mod:`harmonics`, :class:`features.GaussianSmearing` and
:func:`features.timestep_embedding`, plus the all-atom receptor graph that
``receptor.build_receptor_graph(all_atoms=True)`` already produces. What is genuinely new is
here and nowhere else: the old-style atom encoder, the 9-edge-group layer wiring, and the
pooled 2-way head.

Four things differ from the score network and each one is a silent wrong answer if assumed:

* **The encoder is upstream's ``OldAtomEncoder``, not its ``AtomEncoder``.** The checkpoint
  settles it: the keys are ``linear`` and ``lm_embedding_layer``, not
  ``additional_features_embedder``. The two do different arithmetic -- the old one *adds*
  ``linear(scalars)`` to the summed categorical embeddings, the new one *concatenates* and
  then projects -- so the score model's :class:`score_model.AtomEncoder` cannot be reused
  here even though both map the same widths.
* **There is no ``differentiate_convolutions``.** The confidence ``model_parameters.yml`` has
  no such key, and the 9 edge groups of a layer are **9 separate modules** in one flat
  ``conv_layers`` ModuleList of 45, not 9 weight blocks inside one layer as in the score
  network. ``config.conv_edge_groups`` describes the score model's packing and is deliberately
  not used here.
* **The residual is in the forward, not in the layer.** Every conv layer is built with
  ``residual=False``; the three updates that write a node type are summed onto the padded
  previous features by :meth:`ConfidenceModel.forward`.
* **The last layer runs only the 3 groups that write ligand nodes**, because nothing reads
  atom or receptor features afterwards. Running all 9 would not change the output, only the
  bill.

**Why one pose at a time is exact.** Upstream collates N copies of the graph into one
disjoint batch and finishes with ``scatter_mean`` per pose. This evaluates one pose per call
and takes a plain mean. That is not an approximation: the batched graph has no edge between
two poses, and in eval mode both normalisations here -- :class:`conv.IrrepBatchNorm` and the
head's ``nn.BatchNorm1d`` -- use stored running statistics, so no tensor in the network
couples two poses. Keeping the pose axis out of the graph also keeps every shape static,
which is what the device path needs.

**The zero-edge case is supported, not crashed** (D-dfd-orch-10). Upstream raises
``ValueError`` in ``tensor_layers.py:362`` when a conv layer sees no edges: ``0 // 100_000``
is 0, it takes the exact-division branch, and ``np.array_split(..., 0)`` refuses. It fires in
this model's ligand-receptor group on any pose whose ligand has drifted out of the pocket. A
layer over an empty edge set returns zeros here, with the mean's divisor guarded so the empty
mean is 0 rather than NaN. Upstream would have computed over zero edges too, had it not
crashed, so this is not doing less of the model's own work.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from tt_bio.diffdock.config import DiffDockConfig
from tt_bio.diffdock.constants import (
    lig_feature_dims,
    rec_atom_feature_dims,
    rec_residue_feature_dims,
)
from tt_bio.diffdock.conv import TensorProductConvLayer
from tt_bio.diffdock.features import GaussianSmearing, timestep_embedding
from tt_bio.diffdock.harmonics import spherical_harmonics
from tt_bio.diffdock.irreps import irrep_seq
from tt_bio.diffdock.score_model import cross_edges, radius_edges

#: Upstream's order inside one layer, ``conv_layers[9*l + k]`` for k = 0..8. The name says
#: which node type the group WRITES and which it reads: ``lr`` reads receptor, writes ligand.
EDGE_GROUPS = ("lig", "lr", "la", "atom", "al", "ar", "rec", "rl", "ra")

#: The three that write ligand nodes. The last layer runs only these (upstream's "last layer
#: optimisation"), because the head reads ligand features alone.
LIGAND_GROUPS = (0, 1, 2)

#: Per group k: ``(out set, source set, edge set, flipped)``. The edge input is
#: ``cat([attr, out[ei[0], :ns], source[ei[1], :ns]])``, the message gathers the source at
#: ``ei[1]`` and lands on the out set at ``ei[0]``. A flipped group runs a cross edge set
#: backwards and reuses its attributes and harmonics unchanged, as upstream does.
WIRING = (("lig", "lig", "lig", False), ("lig", "rec", "lr", False),
          ("lig", "atom", "la", False), ("atom", "atom", "atom", False),
          ("atom", "lig", "la", True), ("atom", "rec", "ar", False),
          ("rec", "rec", "rec", False), ("rec", "lig", "lr", True),
          ("rec", "atom", "ar", True))

#: The order each node set adds its updates in, upstream's (float addition does not commute
#: bit for bit).
SUMS = {"lig": (0, 2, 1), "atom": (3, 4, 5), "rec": (6, 8, 7)}


class ScalarAtomEncoder(nn.Module):
    """Upstream's ``models/layers.OldAtomEncoder``.

    Sum one embedding table per categorical column, **add** a linear map of the continuous
    block (here the sigma embedding), then -- if the node type carries a language-model
    embedding -- concatenate that and project back down.

    The add is the part that matters. ``score_model.AtomEncoder`` is upstream's newer class
    and concatenates instead; it has the same widths, so swapping them produces finite,
    plausible, wrong features. The checkpoint's key names (``linear``, ``lm_embedding_layer``
    against ``additional_features_embedder``) are what decides which class a checkpoint wants,
    and a strict load is what enforces it.
    """

    def __init__(self, emb_dim: int, feature_dims, sigma_embed_dim: int,
                 lm_embedding_dim: int = 0):
        super().__init__()
        self.n_categorical = len(feature_dims[0])
        self.n_scalar = feature_dims[1] + sigma_embed_dim
        self.lm_embedding_dim = lm_embedding_dim
        self.atom_embedding_list = nn.ModuleList(
            [nn.Embedding(dim, emb_dim) for dim in feature_dims[0]])
        if self.n_scalar > 0:
            self.linear = nn.Linear(self.n_scalar, emb_dim)
        if lm_embedding_dim:
            self.lm_embedding_layer = nn.Linear(lm_embedding_dim + emb_dim, emb_dim)

    def forward(self, x: Tensor) -> Tensor:
        want = self.n_categorical + self.n_scalar + self.lm_embedding_dim
        if x.shape[1] != want:
            raise ValueError(f"encoder takes {want} columns, got {x.shape[1]}")
        out: Tensor | int = 0
        for i, emb in enumerate(self.atom_embedding_list):
            out = out + emb(x[:, i].long())
        if self.n_scalar > 0:
            cols = x[:, self.n_categorical:self.n_categorical + self.n_scalar]
            out = out + self.linear(cols)
        if self.lm_embedding_dim:
            out = self.lm_embedding_layer(
                torch.cat([out, x[:, -self.lm_embedding_dim:]], dim=1))
        return out


def _edge_mlp(in_dim: int, ns: int) -> nn.Sequential:
    """``Linear -> ReLU -> Dropout -> Linear``, keeping upstream's module indices.

    Dropout is identity at inference but is kept as :class:`torch.nn.Identity` so the second
    linear stays at index 3 and the checkpoint's ``*.3.weight`` lands in it. Dropping it
    renumbers the block and the weights load into the wrong layer, or not at all.
    """
    return nn.Sequential(nn.Linear(in_dim, ns), nn.ReLU(), nn.Identity(), nn.Linear(ns, ns))


class ConfidenceModel(nn.Module):
    """``AAOldModel(confidence_mode=True)`` in plain torch, one pose per forward pass."""

    def __init__(self, config: DiffDockConfig, lm_embedding_dim: int = 1280):
        super().__init__()
        if not config.all_atoms:
            raise ValueError("the confidence checkpoint is the all-atom model")
        if not config.old_model:
            # The mirror of ScoreModel's own guard. Upstream ships two confidence
            # architectures at identical widths and picks between them with
            # `old_filtering_model`; its recipe sets that true and the released checkpoint is
            # the old one, so this class is `AAOldModel` and nothing else. Without this the
            # flag would be decorative: the wrong setting would load the old network anyway
            # and report a confidence that looked fine.
            raise ValueError(
                "ConfidenceModel is upstream's AAOldModel (old_filtering_model: true, the "
                "released checkpoint's architecture). The newer confidence network is not "
                "ported and has no released weights here.")
        self.config = config
        ns, nv = config.ns, config.nv
        self.ns, self.nv = ns, nv
        self.sh_ls = tuple(range(config.sh_lmax + 1))
        sed, ded, ced = (config.sigma_embed_dim, config.distance_embed_dim,
                         config.cross_distance_embed_dim)

        self.lig_node_embedding = ScalarAtomEncoder(ns, lig_feature_dims, sed)
        self.lig_edge_embedding = _edge_mlp(config.in_lig_edge_features + sed + ded, ns)
        self.rec_node_embedding = ScalarAtomEncoder(ns, rec_residue_feature_dims, sed,
                                                    lm_embedding_dim=lm_embedding_dim)
        self.rec_edge_embedding = _edge_mlp(sed + ded, ns)
        self.atom_node_embedding = ScalarAtomEncoder(ns, rec_atom_feature_dims, sed)
        self.atom_edge_embedding = _edge_mlp(sed + ded, ns)
        self.lr_edge_embedding = _edge_mlp(sed + ced, ns)
        self.ar_edge_embedding = _edge_mlp(sed + ded, ns)
        self.la_edge_embedding = _edge_mlp(sed + ced, ns)

        self.lig_distance_expansion = GaussianSmearing(0.0, config.lig_max_radius, ded)
        self.rec_distance_expansion = GaussianSmearing(0.0, config.rec_max_radius, ded)
        self.cross_distance_expansion = GaussianSmearing(0.0, config.cross_max_distance, ced)

        seq = irrep_seq(ns, nv, config.use_second_order_repr, config.reduce_pseudoscalars)
        last = len(seq) - 1
        layers = []
        for i in range(config.num_conv_layers):
            a, b = seq[min(i, last)], seq[min(i + 1, last)]
            for _ in EDGE_GROUPS:
                layers.append(TensorProductConvLayer(
                    a, str(config.sh_irreps), b, 3 * ns, residual=False,
                    batch_norm=config.batch_norm, tp_weights_layers=config.tp_weights_layers))
        self.conv_layers = nn.ModuleList(layers)

        # Indices are upstream's: the two BatchNorm1d and three Linear layers must stay at
        # 0, 1, 4, 5 and 8 for the checkpoint's keys to land.
        self.confidence_predictor = nn.Sequential(
            nn.Linear(2 * ns if config.num_conv_layers >= 3 else ns, ns),
            nn.BatchNorm1d(ns), nn.ReLU(), nn.Identity(),
            nn.Linear(ns, ns),
            nn.BatchNorm1d(ns), nn.ReLU(), nn.Identity(),
            nn.Linear(ns, config.num_confidence_outputs),
        )

    # -- graph construction -------------------------------------------------------------

    def _sigma_embedding(self, n: int, t: float, dtype, device) -> Tensor:
        """The per-node sigma embedding at diffusion time *t*.

        Upstream calls the confidence model with ``set_time(batch, 0, 0, 0, 0, ...)``
        (``utils/sampling.py:220``), so every node sees t = 0. It stays a parameter because
        the model itself does not assume it.
        """
        tt = torch.as_tensor([t], dtype=dtype, device=device)
        return timestep_embedding(tt, self.config.sigma_embed_dim,
                                  self.config.embedding_scale).expand(n, -1)

    def receptor_side(self, rec_x: Tensor, rec_pos: Tensor, rec_edge_index: Tensor,
                      atom_x: Tensor, atom_pos: Tensor, atom_edge_index: Tensor,
                      atom_rec_edge_index: Tensor, *, t: float = 0.0) -> dict:
        """Everything the receptor contributes before the convolutions: residue and atom node
        features and the ``rec``, ``atom`` and ``ar`` edge sets. None of it depends on the
        pose, so a pose set computes it once (:func:`score_poses`).

        Returns ``{"nodes": {set: [n, ns]}, "edges": {set: (edge_index, attr, sh)}, "pos":
        {set: [n, 3]}}``, the form :meth:`ligand_side` extends and :meth:`convolve` reads.
        """
        dtype, device = rec_pos.dtype, rec_pos.device
        rec_sigma = self._sigma_embedding(rec_pos.shape[0], t, dtype, device)
        atom_sigma = self._sigma_embedding(atom_pos.shape[0], t, dtype, device)

        # Receptor residues: a fixed k-NN graph from preprocessing.
        rec_vec = rec_pos[rec_edge_index[1]] - rec_pos[rec_edge_index[0]]
        rec_edge_attr = self.rec_edge_embedding(torch.cat(
            [rec_sigma[rec_edge_index[0]],
             self.rec_distance_expansion(rec_vec.norm(dim=-1))], dim=1))
        rec_node_attr = self.rec_node_embedding(torch.cat([rec_x, rec_sigma], dim=1))

        # Receptor atoms. Upstream expands these distances with lig_distance_expansion, not
        # rec_distance_expansion (old_aa_model.py:435); the cutoffs differ (5 A against 15 A)
        # so this is not interchangeable.
        atom_vec = atom_pos[atom_edge_index[1]] - atom_pos[atom_edge_index[0]]
        atom_edge_attr = self.atom_edge_embedding(torch.cat(
            [atom_sigma[atom_edge_index[0]],
             self.lig_distance_expansion(atom_vec.norm(dim=-1))], dim=1))
        atom_node_attr = self.atom_node_embedding(torch.cat([atom_x, atom_sigma], dim=1))

        ar_vec = rec_pos[atom_rec_edge_index[1]] - atom_pos[atom_rec_edge_index[0]]
        ar_edge_attr = self.ar_edge_embedding(torch.cat(
            [atom_sigma[atom_rec_edge_index[0]],
             self.rec_distance_expansion(ar_vec.norm(dim=-1))], dim=1))

        sh = lambda v: spherical_harmonics(v, self.sh_ls)
        return dict(
            nodes=dict(rec=rec_node_attr, atom=atom_node_attr),
            edges=dict(rec=(rec_edge_index, rec_edge_attr, sh(rec_vec)),
                       atom=(atom_edge_index, atom_edge_attr, sh(atom_vec)),
                       ar=(atom_rec_edge_index, ar_edge_attr, sh(ar_vec))),
            pos=dict(rec=rec_pos, atom=atom_pos))

    def ligand_side(self, receptor: dict, lig_x: Tensor, lig_pos: Tensor, bond_index: Tensor,
                    bond_attr: Tensor, *, t: float = 0.0, tr_sigma: float = 0.0) -> dict:
        """*receptor* (:meth:`receptor_side`) with one pose's ligand added: its node features
        and the ``lig``, ``lr`` and ``la`` edge sets. A new dict; *receptor* is not changed.

        *tr_sigma* sets the cross-edge cutoff ``3 * tr_sigma + 20``; it is 0 at the call site
        upstream uses.
        """
        cfg = self.config
        dtype, device = lig_pos.dtype, lig_pos.device
        lig_sigma = self._sigma_embedding(lig_pos.shape[0], t, dtype, device)
        rec_pos, atom_pos = receptor["pos"]["rec"], receptor["pos"]["atom"]

        # Ligand: the bond graph plus a 5 A radius graph, bonds first, radius edges with a
        # zero bond-type block.
        rad = radius_edges(lig_pos, cfg.lig_max_radius)
        lig_edge_index = torch.cat([bond_index, rad], dim=1)
        lig_bond_attr = torch.cat(
            [bond_attr, bond_attr.new_zeros((rad.shape[1], bond_attr.shape[1]))], dim=0)
        lig_vec = lig_pos[lig_edge_index[1]] - lig_pos[lig_edge_index[0]]
        lig_edge_attr = self.lig_edge_embedding(torch.cat(
            [lig_bond_attr, lig_sigma[lig_edge_index[0]],
             self.lig_distance_expansion(lig_vec.norm(dim=-1))], dim=1))
        lig_node_attr = self.lig_node_embedding(torch.cat([lig_x, lig_sigma], dim=1))

        # Cross graphs. ligand->receptor uses the annealed cutoff; ligand->atom uses 5 A but
        # is still expanded with cross_distance_expansion (old_aa_model.py:470), another
        # upstream asymmetry that looks like a typo and is part of the weights.
        cutoff = 3.0 * tr_sigma + 20.0 if cfg.dynamic_max_cross else cfg.cross_max_distance
        lr_edge_index = cross_edges(lig_pos, rec_pos, cutoff)
        lr_vec = rec_pos[lr_edge_index[1]] - lig_pos[lr_edge_index[0]]
        lr_edge_attr = self.lr_edge_embedding(torch.cat(
            [lig_sigma[lr_edge_index[0]],
             self.cross_distance_expansion(lr_vec.norm(dim=-1))], dim=1))

        la_edge_index = cross_edges(lig_pos, atom_pos, cfg.lig_max_radius)
        la_vec = atom_pos[la_edge_index[1]] - lig_pos[la_edge_index[0]]
        la_edge_attr = self.la_edge_embedding(torch.cat(
            [lig_sigma[la_edge_index[0]],
             self.cross_distance_expansion(la_vec.norm(dim=-1))], dim=1))

        sh = lambda v: spherical_harmonics(v, self.sh_ls)
        return dict(
            nodes=dict(receptor["nodes"], lig=lig_node_attr),
            edges=dict(receptor["edges"], lig=(lig_edge_index, lig_edge_attr, sh(lig_vec)),
                       lr=(lr_edge_index, lr_edge_attr, sh(lr_vec)),
                       la=(la_edge_index, la_edge_attr, sh(la_vec))),
            pos=dict(receptor["pos"], lig=lig_pos))

    def _conv_host(self, i: int, out: Tensor, src: Tensor, edges, flip: bool) -> Tensor:
        """Convolution *i* of :meth:`convolve` on host tensors (the default executor)."""
        edge_index, attr, sh = edges
        if flip:
            edge_index = torch.flip(edge_index, dims=[0])
        e = torch.cat([attr, out[:, :self.ns][edge_index[0]],
                       src[:, :self.ns][edge_index[1]]], dim=-1)
        return self.conv_layers[i](src, edge_index, e, sh, out_nodes=out.shape[0])

    @staticmethod
    def _combine_host(x: Tensor, updates: list) -> Tensor:
        acc = torch.nn.functional.pad(x, (0, updates[0].shape[-1] - x.shape[-1]))
        for u in updates:
            acc = acc + u
        return acc

    def convolve(self, nodes: dict, edges: dict, conv=None, combine=None):
        """The interaction layers over :data:`WIRING`; returns the ligand node features.

        *conv* ``(i, out, src, edges, flip) -> update`` runs convolution ``conv_layers[i]``
        and *combine* ``(x, updates) -> x'`` adds a node set's updates to its zero-padded
        features in :data:`SUMS` order. Both default to host torch; the card path
        (:class:`tt_bio.diffdock.device.DeviceConfidence`) passes its own and keeps every
        node table on the card, so the wiring exists once.
        """
        conv, combine = conv or self._conv_host, combine or self._combine_host
        n_layers = self.config.num_conv_layers
        for layer in range(n_layers):
            last = layer == n_layers - 1
            upd = {k: conv(9 * layer + k, nodes[out], nodes[src], edges[es], flip)
                   for k, (out, src, es, flip) in enumerate(WIRING)
                   if not last or k in LIGAND_GROUPS}
            nodes = {s: combine(x, [upd[k] for k in SUMS[s]]) if all(k in upd for k in SUMS[s])
                     else x for s, x in nodes.items()}
        return nodes["lig"]

    def head(self, lig_node_attr: Tensor) -> Tensor:
        """``[1, num_confidence_outputs]`` from the final ligand features.

        The head reads the true scalars and the pseudoscalars, the first and last ns columns,
        and pools over the pose's ligand atoms.
        """
        scalars = (torch.cat([lig_node_attr[:, :self.ns], lig_node_attr[:, -self.ns:]], dim=1)
                   if self.config.num_conv_layers >= 3 else lig_node_attr[:, :self.ns])
        return self.confidence_predictor(scalars.mean(dim=0, keepdim=True))

    def forward(self, lig_x: Tensor, lig_pos: Tensor, bond_index: Tensor, bond_attr: Tensor,
                rec_x: Tensor, rec_pos: Tensor, rec_edge_index: Tensor,
                atom_x: Tensor, atom_pos: Tensor, atom_edge_index: Tensor,
                atom_rec_edge_index: Tensor, *, t: float = 0.0,
                tr_sigma: float = 0.0) -> Tensor:
        """One pose in, ``[1, num_confidence_outputs]`` out.

        *atom_rec_edge_index* is ``[2, n_atom]`` of ``(atom, residue)``; it is fixed by the
        structure, which is why the receptor graph builder precomputes it.
        """
        rec = self.receptor_side(rec_x, rec_pos, rec_edge_index, atom_x, atom_pos,
                                 atom_edge_index, atom_rec_edge_index, t=t)
        g = self.ligand_side(rec, lig_x, lig_pos, bond_index, bond_attr, t=t,
                             tr_sigma=tr_sigma)
        return self.head(self.convolve(g["nodes"], g["edges"]))


#: e3nn keeps three artefacts in a TensorProduct's state dict that carry no learned
#: information: an empty ``weight`` (these tensor products take per-edge weights, so nothing
#: is stored), ``output_mask``, derived from the irreps, and the ``_w3j_*`` Wigner 3j tables
#: its TorchScript path caches, which are mathematical constants the port computes itself
#: (``w3j_tables.json``). Only layers whose input carries an l=1 track have the last ones,
#: which is why the first nine conv layers have none. ``score_model`` drops the same set.
SKIP_SUBSTRINGS = ("tp.weight", "tp.output_mask", "_w3j")


def remap_checkpoint(state_dict: dict) -> dict:
    """The released ``AAOldModel`` state dict in this module's key names.

    The only transformation is dropping :data:`SKIP_SUBSTRINGS`; every other key already
    matches, which is the point of mirroring upstream's module names.
    """
    return {k: v for k, v in state_dict.items()
            if not any(s in k for s in SKIP_SUBSTRINGS)}


def load_released(state_dict: dict, config: DiffDockConfig, *,
                  lm_embedding_dim: int = 1280) -> "ConfidenceModel":
    """Build the network and load the released confidence weights into it, strictly.

    Strict is the deliverable: the wrong atom encoder, a head grouped differently or a
    conv stack packed 5x9 instead of 45 all surface here rather than as a number that is
    merely wrong.
    """
    model = ConfidenceModel(config, lm_embedding_dim=lm_embedding_dim)
    missing, unexpected = model.load_state_dict(remap_checkpoint(state_dict), strict=False)
    if missing or unexpected:
        raise ValueError(f"confidence checkpoint does not fit: missing={missing[:8]} "
                         f"unexpected={unexpected[:8]}")
    return model.eval()


# ------------------------------------------------------- ranking poses on host torch

#: What upstream substitutes for a pose the network could not score
#: (``utils/sampling.py``: ``confidence = confidence.nan_to_num(nan=-1000)``). It is far
#: below any real logit, so such a pose sorts last rather than disappearing.
UNSCORABLE = -1000.0


def load_checkpoint(ckpt_path, params_dir=None, *, old_model: bool = True) -> "ConfidenceModel":
    """Load the released confidence network from a checkpoint file on disk.

    *params_dir* defaults to the checkpoint's own directory, which is upstream's layout
    (``best_model_epoch75.pt`` beside ``model_parameters.yml``) and what
    :func:`tt_bio.diffdock.checkpoints.resolve` returns.

    *old_model* is upstream's ``old_filtering_model``, which its default recipe sets true and
    which the released checkpoint needs. It is a parameter rather than a constant so that a
    caller asking for the other architecture is told it is missing instead of being handed
    this one.
    """
    from pathlib import Path

    from tt_bio.diffdock import config as dfd_config

    ckpt_path = Path(ckpt_path)
    directory = Path(params_dir) if params_dir is not None else ckpt_path.parent
    cfg = dfd_config.load(directory, old_model=old_model, name="confidence")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return load_released(state, cfg)


def score_poses(model: "ConfidenceModel", receptor: dict, ligand: dict,
                poses: Tensor, *, convolve=None) -> Tensor:
    """Confidence of every pose, as a ``[N]`` tensor. Higher is better.

    *receptor* is a graph built with ``all_atoms=True`` (``input.CONFIDENCE_GRAPH``) and
    must be in the same frame as *poses* -- both centred, or neither. *poses* is
    ``[N, n_lig, 3]``.

    The receptor side is embedded once (it does not depend on the pose), then one pass per
    pose. *convolve* ``(graph) -> ligand features`` replaces :meth:`ConfidenceModel.convolve`;
    the card path passes :class:`tt_bio.diffdock.device.DeviceConfidence`.

    Upstream collates the N poses into one disjoint graph, which is the same arithmetic: no edge joins two poses and both normalisations read running
    statistics in eval mode, so nothing couples them. Keeping the pose axis out of the graph
    is also what lets the device path hold every shape static.

    Only column 0 is returned. The head emits two class logits and upstream ranks on the
    first; column 1 is near but not equal to its negation, so averaging or flattening the
    two would change the order.
    """
    if "atom" not in receptor:
        raise ValueError(
            "the confidence model needs an all-atom receptor graph; build it with "
            "input.CONFIDENCE_GRAPH (all_atoms=True), not SCORE_GRAPH")
    atom = receptor["atom"]
    convolve = convolve or (lambda g: model.convolve(g["nodes"], g["edges"]))
    out = []
    with torch.no_grad():
        rec = model.receptor_side(
            receptor["x"].float(), receptor["pos"].float(), receptor["edge_index"].long(),
            atom["x"].float(), atom["pos"].float(), atom["edge_index"].long(),
            atom["atom_rec_edge_index"].long())
        for pose in poses:
            g = model.ligand_side(rec, ligand["x"].float(), pose.float(),
                                  ligand["edge_index"].long(), ligand["edge_attr"].float())
            out.append(model.head(convolve(g))[:, 0])
    return torch.cat(out).nan_to_num(nan=UNSCORABLE)


def rank_poses(poses: Tensor, scores: Tensor):
    """``(poses, scores)`` reordered best first. Ties keep their sampling order."""
    order = torch.argsort(-scores, stable=True)
    return poses[order], scores[order]
