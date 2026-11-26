"""The output heads of the DiffDock-L score network: translation, rotation and torsion.

Upstream: ``models/cg_model.CGModel.forward`` from ``build_center_conv_graph`` to
``scale_by_sigma``. One convolution from every ligand atom into its graph's centre of mass
gives two ``1o`` and two ``1e`` vectors per graph; the translation score is the first ``1o``
plus the first ``1e``, the rotation score the second of each. Each is then cut to a unit
direction and rescaled by a small MLP of ``[|v|, sigma embedding]``, and finally by the noise
level.

Things here that are easy to get wrong, all of which load cleanly and are visible only in the
numbers:

* **``fixed_center_conv``.** The released model sets it (``not_fixed_center_conv: false``),
  and the edge features take the *ligand atom's* scalars, ``node_attr[edge_index[1]]``. The
  constructor default upstream is the other branch, which takes ``node_attr[edge_index[0]]``
  -- and ``edge_index[0]`` there is a *graph* index, so it reads atom 0's, atom 1's ... scalars
  for graph 0, 1, ... The flag is kept, read from the config, and defaults to the released
  value, not to upstream's constructor default.
* **Adding ``1o`` and ``1e``.** ``tr = v[:, 0:3] + v[:, 6:9]`` sums a vector and a pseudovector,
  which is not parity-equivariant; that is upstream's choice (``odd_parity: false``) and the
  checkpoint was trained with it. Equivariance tests for these heads must therefore be
  rotation-only.
* **The norm-and-rescale.** ``v / |v| * mlp([|v|, emb])`` is rotation-equivariant only because
  the MLP sees the norm, which is invariant. A zero vector gives NaN here exactly as it does
  upstream; it does not arise from a real complex.
* **``scale_by_sigma``.** ``tr / tr_sigma`` but ``rot * so3_score_norm(rot_sigma)``: a divide
  and a multiply, by different things, and the second is a rounded table lookup
  (:mod:`tt_bio.diffdock.schedules`).

:class:`TorsionHead` is the third output and is documented on the class: it is the only head
whose convolution scatters to something other than atoms.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from tt_bio.diffdock.conv import TensorProductConvLayer
from tt_bio.diffdock.features import GaussianSmearing
from tt_bio.diffdock.full_tensor_product import FullTensorProduct
from tt_bio.diffdock.harmonics import spherical_harmonics
from tt_bio.diffdock.schedules import so3_score_norm, torus_score_norm


def _degrees(irreps: str) -> tuple[int, ...]:
    """``"1x0e + 1x1o + 1x2e"`` -> ``(0, 1, 2)``, the degrees :mod:`tt_bio.diffdock.harmonics`
    takes. Parity labels are ignored: for spherical harmonics they are fixed at ``(-1)^l``."""
    out = []
    for term in irreps.replace(" ", "").split("+"):
        mul, _, ir = term.rpartition("x")
        if mul not in ("", "1"):
            raise ValueError(f"harmonics come with multiplicity 1, got {term!r}")
        out.append(int(ir[:-1]))
    return tuple(out)


def _mlp(in_dim: int, hidden: int, out_dim: int, *, act_first: bool) -> nn.Sequential:
    """Two linears with upstream's module indices (0 and 3), dropout slot as Identity.

    The centre edge embedding is ``[Linear, ReLU, Dropout, Linear]``; the tr/rot final layers
    are ``[Linear, Dropout, ReLU, Linear]``. Both put the second linear at index 3, which is
    all the checkpoint keys see; at inference the two orders are the same function.
    """
    mid = [nn.ReLU(), nn.Identity()] if act_first else [nn.Identity(), nn.ReLU()]
    return nn.Sequential(nn.Linear(in_dim, hidden), *mid, nn.Linear(hidden, out_dim))


class CenterHead(nn.Module):
    """``(ligand node features, positions, batch, sigmas) -> (tr_score, rot_score)``."""

    def __init__(self, *, in_irreps: str, sh_irreps: str, ns: int, distance_embed_dim: int,
                 sigma_embed_dim: int, center_max_distance: float = 30.0,
                 fixed_center_conv: bool = True, odd_parity: bool = False,
                 scale_by_sigma: bool = True, batch_norm: bool = True):
        super().__init__()
        self.ns = ns
        self.sh_irreps = sh_irreps
        self.sh_ls = _degrees(sh_irreps)
        self.fixed_center_conv = fixed_center_conv
        self.odd_parity = odd_parity
        self.scale_by_sigma = scale_by_sigma
        self.center_distance_expansion = GaussianSmearing(0.0, center_max_distance,
                                                          distance_embed_dim)
        self.center_edge_embedding = _mlp(distance_embed_dim + sigma_embed_dim, ns, ns,
                                          act_first=True)
        self.final_conv = TensorProductConvLayer(
            in_irreps, sh_irreps, "1x1o + 1x1e" if odd_parity else "2x1o + 2x1e",
            n_edge_features=2 * ns, residual=False, batch_norm=batch_norm)
        self.tr_final_layer = _mlp(1 + sigma_embed_dim, ns, 1, act_first=False)
        self.rot_final_layer = _mlp(1 + sigma_embed_dim, ns, 1, act_first=False)

    def forward(self, node_attr: Tensor, pos: Tensor, batch: Tensor, num_graphs: int,
                node_sigma_emb: Tensor, graph_sigma_emb: Tensor, tr_sigma: Tensor,
                rot_sigma: Tensor) -> tuple[Tensor, Tensor]:
        n = pos.shape[0]
        edge_index = torch.stack([batch, torch.arange(n, device=pos.device)])
        center = torch.zeros((num_graphs, 3), dtype=pos.dtype, device=pos.device)
        center.index_add_(0, batch, pos)
        center = center / torch.bincount(batch, minlength=num_graphs).unsqueeze(1).to(pos.dtype)

        edge_vec = pos[edge_index[1]] - center[edge_index[0]]
        edge_attr = self.center_distance_expansion(edge_vec.norm(dim=-1))
        edge_attr = torch.cat([edge_attr, node_sigma_emb[edge_index[1]]], dim=1)
        edge_sh = spherical_harmonics(edge_vec, self.sh_ls)

        edge_attr = self.center_edge_embedding(edge_attr)
        src = edge_index[1] if self.fixed_center_conv else edge_index[0]
        edge_attr = torch.cat([edge_attr, node_attr[src, :self.ns]], dim=-1)
        out = self.final_conv(node_attr, edge_index, edge_attr, edge_sh, out_nodes=num_graphs)

        if self.odd_parity:
            tr, rot = out[:, :3], out[:, 3:6]
        else:
            tr, rot = out[:, :3] + out[:, 6:9], out[:, 3:6] + out[:, 9:]

        tr_norm = torch.linalg.vector_norm(tr, dim=1).unsqueeze(1)
        tr = tr / tr_norm * self.tr_final_layer(torch.cat([tr_norm, graph_sigma_emb], dim=1))
        rot_norm = torch.linalg.vector_norm(rot, dim=1).unsqueeze(1)
        rot = rot / rot_norm * self.rot_final_layer(torch.cat([rot_norm, graph_sigma_emb], dim=1))

        if self.scale_by_sigma:
            tr = tr / tr_sigma.unsqueeze(1)
            rot = rot * so3_score_norm(rot_sigma).unsqueeze(1).to(rot.device)
        return tr, rot


class TorsionHead(nn.Module):
    """``(ligand node features, positions, rotatable bonds) -> one score per bond``.

    Upstream: ``CGModel.forward`` from ``build_bond_conv_graph`` to ``tor_final_layer``. The
    graph here is unlike every other one in the model: its *destination* nodes are the
    midpoints of the rotatable bonds, not atoms, so the convolution scatters ``n_bonds``
    rows out of a node set of ``n_atoms``. Four details that load cleanly and differ only in
    the numbers:

    * **The harmonics are a tensor product, not a direction.** Each edge's ``0e+1o+2e``
      harmonics are combined with the ``2e`` harmonics of *its own bond's* axis
      (:class:`~tt_bio.diffdock.full_tensor_product.FullTensorProduct`), giving the 45-wide
      object the convolution actually sees. That is what makes the score a function of the
      angle between the neighbour and the rotation axis, which is what a torsion is.
    * **Three scalar blocks per edge**, in order: the embedded distance, the neighbouring
      *atom*'s first ``ns`` scalars, and the *bond*'s first ``ns`` scalars, where the bond's
      are the sum of its two endpoints'. Permuting them loads fine and is wrong.
    * **The final layer has no biases** (``bias=False`` on both linears) and the activation
      between them is ``Tanh``, not ``ReLU`` -- unlike every other MLP in the model.
    * **The sigma scaling is a square root** of a rounded table lookup, and the lookup is
      per *bond*, taken from the torsion sigma of the graph the bond belongs to.
    """

    def __init__(self, *, in_irreps: str, sh_irreps: str, ns: int, distance_embed_dim: int,
                 lig_max_radius: float = 5.0, odd_parity: bool = False,
                 scale_by_sigma: bool = True, batch_norm: bool = True,
                 smooth_edges: bool = False, distance_expansion: nn.Module | None = None):
        super().__init__()
        self.ns = ns
        self.sh_ls = _degrees(sh_irreps)
        self.lig_max_radius = lig_max_radius
        self.odd_parity = odd_parity
        self.scale_by_sigma = scale_by_sigma
        self.smooth_edges = smooth_edges
        # Upstream shares the ligand graph's smearing module with this head. It is stateless,
        # so sharing or rebuilding gives the same numbers; the caller can pass its own.
        self.lig_distance_expansion = distance_expansion or GaussianSmearing(
            0.0, lig_max_radius, distance_embed_dim)
        self.final_edge_embedding = _mlp(distance_embed_dim, ns, ns, act_first=True)
        self.final_tp_tor = FullTensorProduct(sh_irreps, "2e")
        self.tor_bond_conv = TensorProductConvLayer(
            in_irreps, self.final_tp_tor.out_irreps, f"{ns}x0o" if odd_parity
            else f"{ns}x0o + {ns}x0e", n_edge_features=3 * ns, residual=False,
            batch_norm=batch_norm)
        width = ns if odd_parity else 2 * ns
        self.tor_final_layer = nn.Sequential(
            nn.Linear(width, ns, bias=False), nn.Tanh(), nn.Identity(),
            nn.Linear(ns, 1, bias=False))

    def edge_weight(self, edge_vec: Tensor):
        """Upstream's ``get_edge_weight``: 1.0 unless ``smooth_edges``, which the released
        model does not set. Kept because it multiplies the message *and the mean divisor*."""
        if not self.smooth_edges:
            return 1.0
        normalised = torch.clip(edge_vec.norm(dim=-1) * torch.pi / self.lig_max_radius,
                                max=torch.pi)
        return 0.5 * (torch.cos(normalised) + 1.0).unsqueeze(-1)

    def forward(self, node_attr: Tensor, pos: Tensor, bonds: Tensor, edge_index: Tensor,
                tor_sigma: Tensor | None = None, edge_mask: Tensor | None = None,
                edge_chunk: int | None = None) -> Tensor:
        """*bonds* is ``[2, n_bonds]`` of atom indices; *edge_index* is ``[2, E]`` with row 0
        indexing *bonds* and row 1 indexing atoms. *tor_sigma* is one sigma per bond."""
        if bonds.shape[0] != 2 or edge_index.shape[0] != 2:
            raise ValueError("bonds and edge_index are [2, n]")
        n_bonds = bonds.shape[1]
        bond_pos = (pos[bonds[0]] + pos[bonds[1]]) / 2
        bond_vec = pos[bonds[1]] - pos[bonds[0]]
        bond_attr = node_attr[bonds[0]] + node_attr[bonds[1]]

        edge_vec = pos[edge_index[1]] - bond_pos[edge_index[0]]
        edge_attr = self.final_edge_embedding(
            self.lig_distance_expansion(edge_vec.norm(dim=-1)))
        edge_sh = spherical_harmonics(edge_vec, self.sh_ls)
        bonds_sh = spherical_harmonics(bond_vec, (2,))
        edge_sh = self.final_tp_tor(edge_sh, bonds_sh[edge_index[0]])

        edge_attr = torch.cat([edge_attr, node_attr[edge_index[1], :self.ns],
                               bond_attr[edge_index[0], :self.ns]], dim=-1)
        out = self.tor_bond_conv(node_attr, edge_index, edge_attr, edge_sh, out_nodes=n_bonds,
                                 reduce="mean", edge_weight=self.edge_weight(edge_vec),
                                 edge_mask=edge_mask, edge_chunk=edge_chunk)
        tor = self.tor_final_layer(out).squeeze(1)
        if self.scale_by_sigma:
            if tor_sigma is None:
                raise ValueError("scale_by_sigma needs one torsion sigma per bond")
            scale = torch.as_tensor(torus_score_norm(tor_sigma), dtype=tor.dtype,
                                    device=tor.device).sqrt()
            tor = tor * scale
        return tor
