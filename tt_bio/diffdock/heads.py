"""The translation and rotation score heads of the DiffDock-L score network.

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
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from tt_bio.diffdock.conv import TensorProductConvLayer
from tt_bio.diffdock.features import GaussianSmearing
from tt_bio.diffdock.harmonics import spherical_harmonics
from tt_bio.diffdock.schedules import so3_score_norm


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
