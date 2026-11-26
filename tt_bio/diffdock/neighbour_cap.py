"""Where upstream's 32-neighbour cap bites, and why this port does not reproduce it.

Upstream builds the ligand radius graph and the torsion head's bond-centre graph with
``torch_cluster``, which caps neighbours per source node at ``max_num_neighbors=32`` (it
raises the cap to 10000 only for the cross graph). This port takes every neighbour inside the
radius, so the two edge sets are identical until some atom has more than 32 neighbours inside
5 A, and differ after that.

**The cap cannot be reproduced, and that is a measurement, not a preference.** On a 60-point
cloud where every true degree is 59, ``scripts/diffdock_port/probe_cap_selection.py``
(evidence ``state/dfd/evidence/cap-selection.json``) finds the kept 32:

* are **not** the first 32 by index order: 0 of 60 nodes;
* are **not** the nearest 32: 0 of 60 nodes;
* are **not** stable under relabelling the same points: 7 of 60 nodes survive, 130 edges move;
* are **not** stable under *rotating* the same points: 0 of 60 nodes survive.

The last one is the decisive one. The kept subset falls out of the cell-list traversal order
of torch_cluster's CPU kernel, so it depends on the coordinate frame. Reproducing it would
make this port's output depend on the orientation of its input, which breaks the SE(3)
equivariance the model is built on and is a hard stop here. It also differs between
torch_cluster's CPU and CUDA builds, so there is no single "upstream" subset to match.

So the port keeps the complete neighbourhood (``D-dfd-trace-1``), which is the graph the model
intends, is permutation and rotation invariant, and is a superset of whatever upstream kept.
What this module adds is that the divergence is never silent: :func:`cap_report` says, for a
given set of coordinates, whether upstream would have truncated and by how much, so a grade
can quote those complexes separately instead of averaging them away.

Measured over all 428 PoseBusters ligands (``probe_radius_truncation.py``), exactly one
reaches the cap: 7F8T_FAD, an FAD cofactor whose densest atom has 34 heavy-atom neighbours
inside 5 A on the crystal pose and 32 on the RDKit start conformer.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import torch
from torch import Tensor

#: ``torch_cluster``'s default, which both ligand-side calls in upstream's ``cg_model.py`` take.
UPSTREAM_MAX_NEIGHBORS = 32


@dataclass(frozen=True)
class CapReport:
    """How far a coordinate set pushes past upstream's neighbour cap.

    *max_degree* counts neighbours inside the cutoff, excluding the self pair, the way
    upstream's ``radius_graph`` counts them. *truncated_nodes* is how many nodes upstream
    would have truncated, *excess_edges* how many edges in total this port keeps and upstream
    would have dropped.
    """

    max_degree: int
    cap: int
    truncated_nodes: int
    excess_edges: int

    @property
    def binds(self) -> bool:
        """True when upstream's graph would differ from this port's on these coordinates."""
        return self.truncated_nodes > 0

    def __str__(self) -> str:
        if not self.binds:
            return f"max degree {self.max_degree} is within upstream's cap of {self.cap}"
        return (f"{self.truncated_nodes} atom(s) exceed upstream's {self.cap}-neighbour cap "
                f"(max degree {self.max_degree}); this port keeps {self.excess_edges} edge(s) "
                f"upstream would have dropped")


def neighbour_degrees(pos: Tensor, cutoff: float) -> Tensor:
    """``[n]`` count of other points within *cutoff* of each point.

    The comparison is ``<`` on the direct distance, matching
    :func:`tt_bio.diffdock.score_model.radius_edges`, so this counts exactly the edges that
    function builds and not a near-miss of them.
    """
    if pos.ndim != 2 or pos.shape[-1] != 3:
        raise ValueError(f"expected [n, 3] coordinates, got {tuple(pos.shape)}")
    d = torch.cdist(pos, pos)
    inside = d < cutoff
    inside.fill_diagonal_(False)
    return inside.sum(dim=1)


def cross_degrees(src_pos: Tensor, dst_pos: Tensor, cutoff: float) -> Tensor:
    """``[n_src]`` count of *dst_pos* points within *cutoff* of each *src_pos* point.

    The bond-centre graph upstream builds with ``radius(lig_pos, bond_pos, 5)``, which caps
    per bond centre and has no self pair to exclude.
    """
    return (torch.cdist(src_pos, dst_pos) < cutoff).sum(dim=1)


def cap_report(degrees: Tensor, cap: int = UPSTREAM_MAX_NEIGHBORS) -> CapReport:
    """Summarise a degree vector against upstream's cap."""
    over = (degrees - cap).clamp(min=0)
    return CapReport(
        max_degree=int(degrees.max()) if degrees.numel() else 0,
        cap=cap,
        truncated_nodes=int((over > 0).sum()),
        excess_edges=int(over.sum()),
    )


def check_ligand_graph(lig_pos: Tensor, cutoff: float, *,
                       cap: int = UPSTREAM_MAX_NEIGHBORS,
                       label: str = "ligand", warn: bool = True) -> CapReport:
    """Report, and by default warn, when the ligand radius graph passes upstream's cap.

    Called on the coordinates a step actually runs on. The warning is the point: on such a
    complex this port and upstream are running different graphs, so upstream's published
    score is not the number to grade against without saying so.
    """
    report = cap_report(neighbour_degrees(lig_pos, cutoff), cap)
    if warn and report.binds:
        warnings.warn(
            f"{label}: {report}. tt-bio keeps the complete {cutoff} A neighbourhood "
            f"(D-dfd-trace-1); upstream DiffDock's torch_cluster graph would be truncated "
            f"here, so the two scores are not comparable on this complex.",
            RuntimeWarning, stacklevel=2)
    return report


def degrees_from_mask(edge_index: Tensor, mask: Tensor, n_nodes: int) -> Tensor:
    """``[n_nodes]`` degree of the masked static graph, from a per-step edge mask.

    The static-graph path (:class:`tt_bio.diffdock.graph.StaticGraphs`) never materialises a
    neighbour list: it carries a complete edge set and multiplies each edge by a 0/1 mask
    recomputed from the moving coordinates. The degree is then a segment sum of that mask over
    the destination row, which costs one scatter over a tensor the step already computes, so
    the guard composes with the mask instead of fighting it.

    Row 1 is the node the mask's degree is counted for, matching ``tp_scatter``'s convention
    that row 0 is scattered to and row 1 is read from; pass ``edge_index.flip(0)`` to count
    the other way.
    """
    if edge_index.shape[1] != mask.shape[0]:
        raise ValueError(f"{edge_index.shape[1]} edges but {mask.shape[0]} mask entries")
    out = torch.zeros(n_nodes, dtype=mask.dtype, device=mask.device)
    return out.index_add_(0, edge_index[1], mask)
