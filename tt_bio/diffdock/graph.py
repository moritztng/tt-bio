"""Static edge sets plus a distance mask, in place of upstream's per-step radius graphs.

Upstream rebuilds three graphs on the host at every one of the 19 diffusion steps, because
the ligand moves: ``radius_graph(lig.pos, 5)``, the ligand->receptor cross graph at the
sigma-dependent cutoff ``3*tr_sigma + 20``, and the torsion head's bond-centre graph. Each
call returns a different number of edges, so the step has a different shape every time, and a
different shape cannot be traced.

The port builds every one of those edge sets **once, complete**, and multiplies each edge's
message by a 0/1 mask recomputed from the moving coordinates. That is the same arithmetic:
the convolution sums messages over edges, so an edge whose message is multiplied by zero
contributes exactly what an absent edge contributes, provided the ``mean`` divisor counts the
mask and not the edges (:func:`tt_bio.diffdock.conv.tp_scatter` takes ``edge_mask`` for
exactly this). The shapes are then fixed for the whole run.

**What it costs.** `state/dfd/ARCH.md` 6.1 measured the cross graph over the real schedule:
complete for the first six of nineteen steps (the cutoff exceeds the protein's diameter),
96.6 % complete at step 8 and 69.5 % at the last step, so the complete set costs at most
1.44x the cross-edge arithmetic and nothing at all early on. The ligand radius graph is 586
of 1,406 possible edges on that complex, 2.4x on a set that is 0.3 % of the work.

**Where it is not identical, measured.** ``torch_cluster.radius`` caps neighbours per source
node -- 32 by default, which both ligand-side calls take -- and the kept subset is its grid
traversal order, not the nearest ones (the docstring calls it random). On the 428-ligand
PoseBusters benchmark set exactly one ligand reaches that cap (7F8T_FAD, an FAD cofactor
with 53 heavy atoms and 34 neighbours inside 5 A of one atom; 8AEU_M0L reaches it in the
bond-centre graph on the crystal pose). The port does not reproduce the cap: see
``D-dfd-trace-1``. Those complexes get the complete 5 A neighbourhood, which is the graph the
model intends; everywhere else the two edge sets are identical.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from tt_bio.diffdock.conv import bucket


def pad_count(n: int, multiple: int = 32) -> int:
    """Padded length of an *n*-row axis: a multiple of 32 with at least one row to spare.

    The spare row is the pad node. Padded edges have to point at a node no real edge touches
    (:func:`tt_bio.diffdock.conv.pad_edges` explains why), and an axis whose real length is
    already a multiple of 32 would otherwise have nowhere to put one.
    """
    return bucket(n + 1, multiple)


def complete_pairs(n_src: int, n_dst: int, *, exclude_diagonal: bool = False,
                   device=None) -> Tensor:
    """``[2, n_src*n_dst]`` (one fewer per row when *exclude_diagonal*), source-major.

    Row 0 is the node the message is scattered to, row 1 the node it is read from -- the
    same convention as ``tp_scatter``, and upstream's: its cross graph returns the ligand
    index first and the receptor index second.
    """
    src = torch.arange(n_src, device=device).repeat_interleave(n_dst)
    dst = torch.arange(n_dst, device=device).repeat(n_src)
    if exclude_diagonal:
        if n_src != n_dst:
            raise ValueError("exclude_diagonal needs a square pair set")
        keep = src != dst
        src, dst = src[keep], dst[keep]
    return torch.stack([src, dst])


def pad_edge_index(edge_index: Tensor, n_padded: int, pad_node) -> Tensor:
    """Append dead columns up to *n_padded*.

    *pad_node* is one index, or a pair when the two rows index different node blocks (the
    cross graph's rows are ligand and receptor nodes, each with its own pad row).
    """
    extra = n_padded - edge_index.shape[1]
    if extra < 0:
        raise ValueError(f"cannot pad {edge_index.shape[1]} edges down to {n_padded}")
    if extra == 0:
        return edge_index
    src_pad, dst_pad = (pad_node, pad_node) if isinstance(pad_node, int) else pad_node
    tail = torch.tensor([[src_pad] * extra, [dst_pad] * extra],
                        dtype=edge_index.dtype, device=edge_index.device)
    return torch.cat([edge_index, tail], dim=1)


def pad_mask(mask: Tensor, n_padded: int) -> Tensor:
    """Append zeros so a mask covers the padded edge axis."""
    return torch.nn.functional.pad(mask, (0, n_padded - mask.shape[0]))


def edge_vectors(src_pos: Tensor, dst_pos: Tensor, edge_index: Tensor) -> Tensor:
    """``dst_pos[edge_index[1]] - src_pos[edge_index[0]]``, upstream's sign convention."""
    return dst_pos[edge_index[1]] - src_pos[edge_index[0]]


def within(vec: Tensor, cutoff: float | Tensor) -> Tensor:
    """``[E] -> [E]`` 0/1 mask of ``|vec| <= cutoff``, in the input's dtype."""
    return (vec.norm(dim=-1) <= cutoff).to(vec.dtype)


def cross_mask(lig_pos: Tensor, rec_pos: Tensor, edge_index: Tensor,
               cutoff: float | Tensor) -> Tensor:
    """The cross-edge mask, computed the way upstream computes its cross graph.

    ``build_cross_conv_graph`` does not compare ``|rec - lig|`` with the cutoff: at a
    sigma-dependent cutoff it divides both coordinate sets by it and calls ``radius(..., 1)``.
    Dividing first and comparing with 1 is not the same floating-point expression as
    comparing with the cutoff, and an edge can land on the other side of the boundary. This
    reproduces the expression, not the intent.

    *cutoff* is one number, or one per edge (what :meth:`StaticGraphs.cross_mask` passes for
    a pose batch, where upstream gives every graph in the batch its own cutoff).
    """
    c = torch.as_tensor(cutoff, dtype=lig_pos.dtype, device=lig_pos.device)
    if c.numel() not in (1, edge_index.shape[1]):
        raise ValueError(f"cutoff has {c.numel()} entries for {edge_index.shape[1]} edges")
    c = c.reshape(-1, 1) if c.numel() > 1 else c
    vec = (rec_pos[edge_index[1]] - lig_pos[edge_index[0]]) / c
    return (vec.norm(dim=-1) <= 1.0).to(lig_pos.dtype)


def tile_disjoint(edge_index: Tensor, n_copies: int, src_stride: int,
                  dst_stride: int) -> Tensor:
    """*n_copies* of an edge set side by side, each shifted onto its own node block.

    The disjoint-union batching tt-atom uses (``tt_atom/disjoint.py``), with the one
    simplification DiffDock allows: every pose has the same node count, so the offsets are a
    stride rather than a cumulative sum, and the batched edge set is one broadcast of the
    single-pose one rather than a list to concatenate.
    """
    if n_copies == 1:
        return edge_index
    src = torch.arange(n_copies, device=edge_index.device) * src_stride
    dst = torch.arange(n_copies, device=edge_index.device) * dst_stride
    out = edge_index[None, :, :] + torch.stack([src, dst], dim=1)[:, :, None]
    return out.permute(1, 0, 2).reshape(2, -1)


@dataclass(frozen=True)
class StaticGraphs:
    """The fixed edge sets of one complex, padded, with the node axes they index.

    Built once per complex from counts alone -- no coordinates -- which is what makes the
    whole diffusion loop one shape. :meth:`masks` is the only per-step part, and it reads the
    moving coordinates.

    Node layout for the conv stack is upstream's: ligand nodes first, then receptor nodes,
    with each block padded to its own multiple of 32, so ``rec_offset`` is the padded ligand
    length and not ``n_lig``.
    """

    n_lig: int                    # per pose
    n_rec: int                    # per pose
    n_rot_bonds: int              # per pose
    n_poses: int
    lig_bond_edges: Tensor        # [2, n_bond_edges], the molecule's own bonds, always on
    lig_edge_index: Tensor        # [2, padded] bond edges then the complete non-self pairs
    lig_n_bond_edges: int         # over the whole batch
    cross_edge_index: Tensor      # [2, padded] ligand <- receptor, complete
    tor_edge_index: Tensor        # [2, padded] bond centre <- ligand atom, complete
    n_lig_padded: int             # the whole ligand block, every pose
    n_rec_padded: int
    n_bond_padded: int

    @property
    def n_lig_total(self) -> int:
        return self.n_lig * self.n_poses

    @property
    def n_rec_total(self) -> int:
        return self.n_rec * self.n_poses

    @property
    def n_rot_bonds_total(self) -> int:
        return self.n_rot_bonds * self.n_poses

    def cross_edge_index_global(self) -> Tensor:
        """The cross edge set on the conv stack's combined node axis.

        :attr:`cross_edge_index` keeps receptor indices local, because that is what the mask
        needs; the convolution runs on ``cat([ligand nodes, receptor nodes])``, where the
        receptor block starts at the *padded* ligand length.
        """
        out = self.cross_edge_index.clone()
        out[1] += self.rec_offset
        return out

    @property
    def rec_offset(self) -> int:
        return self.n_lig_padded

    @property
    def lig_pad_node(self) -> int:
        return self.n_lig_padded - 1

    @property
    def rec_pad_node(self) -> int:
        return self.n_lig_padded + self.n_rec_padded - 1

    @classmethod
    def build(cls, n_lig: int, n_rec: int, lig_bond_edges: Tensor, n_rot_bonds: int = 0,
              n_poses: int = 1, device=None) -> "StaticGraphs":
        """The complete edge sets for *n_poses* copies of one complex, padded.

        The poses are one disjoint-union graph, as upstream batches them: ten copies of the
        same ligand against ten copies of the same receptor, differing only in coordinates.
        The node axis is upstream's too -- every pose's ligand nodes, then every pose's
        receptor nodes -- because PyG batches a ``HeteroData`` by node type, and the score
        network reads ``node_attr[:n_ligand_nodes]`` at the end expecting exactly that.
        """
        device = device or lig_bond_edges.device
        n_lig_padded = pad_count(n_lig * n_poses)
        n_rec_padded = pad_count(n_rec * n_poses)
        n_bond_padded = pad_count(n_rot_bonds * n_poses) if n_rot_bonds else 0
        lig_pad = n_lig_padded - 1

        bonds = tile_disjoint(lig_bond_edges.to(device), n_poses, n_lig, n_lig)
        pairs = tile_disjoint(complete_pairs(n_lig, n_lig, exclude_diagonal=True,
                                             device=device), n_poses, n_lig, n_lig)
        lig_edges = torch.cat([bonds, pairs], dim=1)
        lig_edges = pad_edge_index(lig_edges, bucket(lig_edges.shape[1]), lig_pad)

        cross = tile_disjoint(complete_pairs(n_lig, n_rec, device=device),
                              n_poses, n_lig, n_rec)
        cross = pad_edge_index(cross, bucket(cross.shape[1]), (lig_pad, n_rec_padded - 1))

        if n_rot_bonds:
            tor = tile_disjoint(complete_pairs(n_rot_bonds, n_lig, device=device),
                                n_poses, n_rot_bonds, n_lig)
            tor = pad_edge_index(tor, bucket(tor.shape[1]), (n_bond_padded - 1, lig_pad))
        else:
            tor = torch.zeros((2, 0), dtype=torch.long, device=device)

        return cls(n_lig=n_lig, n_rec=n_rec, n_rot_bonds=n_rot_bonds, n_poses=n_poses,
                   lig_bond_edges=lig_bond_edges.to(device), lig_edge_index=lig_edges,
                   lig_n_bond_edges=bonds.shape[1], cross_edge_index=cross, tor_edge_index=tor,
                   n_lig_padded=n_lig_padded, n_rec_padded=n_rec_padded,
                   n_bond_padded=n_bond_padded)

    # -- per step ------------------------------------------------------------------
    def lig_mask(self, lig_pos: Tensor, lig_max_radius: float) -> Tensor:
        """Bond edges are always on; the complete block is masked at *lig_max_radius*.

        Upstream concatenates the bond edge list with ``radius_graph(pos, 5)`` and keeps
        both copies of a bonded pair that is also inside 5 A -- which every bonded pair is.
        The duplicate is not a bug to fix here: it doubles those pairs' weight in the sum and
        counts twice in the divisor, and the checkpoint was trained that way.
        """
        n_bond, n_padded = self.lig_n_bond_edges, self.lig_edge_index.shape[1]
        n_pairs = self.n_lig * (self.n_lig - 1) * self.n_poses
        radius_block = self.lig_edge_index[:, n_bond:n_bond + n_pairs]
        mask = torch.cat([
            torch.ones(n_bond, dtype=lig_pos.dtype, device=lig_pos.device),
            within(edge_vectors(lig_pos, lig_pos, radius_block), lig_max_radius),
        ])
        return pad_mask(mask, n_padded)

    def cross_mask(self, lig_pos: Tensor, rec_pos: Tensor, cutoff) -> Tensor:
        """*cutoff* is one number, or one per pose: upstream's cutoff is per graph."""
        per_pose = self.n_lig * self.n_rec
        real = per_pose * self.n_poses
        c = torch.as_tensor(cutoff, dtype=lig_pos.dtype, device=lig_pos.device)
        if c.numel() > 1:
            if c.numel() != self.n_poses:
                raise ValueError(f"{c.numel()} cutoffs for {self.n_poses} poses")
            c = c.reshape(-1).repeat_interleave(per_pose)
        mask = cross_mask(lig_pos, rec_pos, self.cross_edge_index[:, :real], c)
        return pad_mask(mask, self.cross_edge_index.shape[1])

    def tor_mask(self, lig_pos: Tensor, bond_pos: Tensor, lig_max_radius: float) -> Tensor:
        real = self.n_rot_bonds * self.n_lig * self.n_poses
        vec = edge_vectors(bond_pos, lig_pos, self.tor_edge_index[:, :real])
        return pad_mask(within(vec, lig_max_radius), self.tor_edge_index.shape[1])
