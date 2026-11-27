"""Upstream's 32-neighbour cap: where it bites, and why the port reports it instead of copying it.

The guard these tests pin exists because the cap is not reproducible. The properties that make
it not reproducible are measured in the reference environment by
``scripts/diffdock_port/probe_cap_selection.py`` and recorded in
``state/dfd/evidence/cap-selection.json``; the tests that need ``torch_cluster`` skip without
it, and the rest run anywhere.
"""
import json
import math
import warnings
from pathlib import Path

import pytest
import torch

from tt_bio.diffdock.neighbour_cap import (UPSTREAM_MAX_NEIGHBORS, cap_report, check_ligand_graph,
                                           cross_degrees, degrees_from_mask, neighbour_degrees)
from tt_bio.diffdock.score_model import radius_edges

CUTOFF = 5.0
FIXTURE = Path(__file__).parent / "fixtures/diffdock/7F8T_FAD_heavy_atoms.json"


@pytest.fixture(scope="module")
def fad():
    """7F8T_FAD's heavy atoms, the one PoseBusters ligand that passes the cap."""
    return json.loads(FIXTURE.read_text())


def dense_cloud(n=60, box=2.0, seed=0):
    """A cloud where every true degree is n-1, so the cap always binds."""
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, 3, generator=g) * box


# --------------------------------------------------------- the cap binds on a real ligand

def test_fad_crystal_pose_exceeds_the_cap(fad):
    """The production complex, at the production radius. Without this there is nothing to guard."""
    pos = torch.tensor(fad["crystal"]["pos"], dtype=torch.float32)
    report = cap_report(neighbour_degrees(pos, CUTOFF))
    assert report.max_degree == 34, report
    assert report.binds and report.truncated_nodes == 1, report
    assert report.excess_edges == 2, report


def test_fad_start_conformer_sits_exactly_on_the_cap(fad):
    """Degree 32 is not truncation: the cap keeps 32 and the atom has 32.

    This is why the divergence cannot be found by looking at the docking input alone. The
    ligand moves over the 19 diffusion steps, so an atom at the cap on the start conformer can
    cross it mid-rollout.
    """
    pos = torch.tensor(fad["start_conf"]["pos"], dtype=torch.float32)
    report = cap_report(neighbour_degrees(pos, CUTOFF))
    assert report.max_degree == UPSTREAM_MAX_NEIGHBORS
    assert not report.binds, report


def test_the_guard_warns_on_the_fad_crystal_pose_and_is_quiet_on_a_normal_ligand(fad):
    """Fails without the guard, passes with it: the whole point of the row."""
    over = torch.tensor(fad["crystal"]["pos"], dtype=torch.float32)
    with pytest.warns(RuntimeWarning, match="exceed upstream's 32-neighbour cap"):
        report = check_ligand_graph(over, CUTOFF, label="7F8T_FAD")
    assert report.binds

    under = torch.tensor(fad["start_conf"]["pos"], dtype=torch.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        quiet = check_ligand_graph(under, CUTOFF, label="7F8T_FAD start")
    assert not quiet.binds


def test_the_input_check_catches_a_ligand_sitting_exactly_on_the_cap(fad):
    """`on_input=True` must fire on 7F8T_FAD's start conformer, which is at 32, not over it.

    Without this the pre-flight check calls the one cap-binding complex in PoseBusters clear,
    because the truncation only appears once the pose has moved.
    """
    start = torch.tensor(fad["start_conf"]["pos"], dtype=torch.float32)
    report = cap_report(neighbour_degrees(start, CUTOFF))
    assert report.at_cap and not report.binds
    with pytest.warns(RuntimeWarning, match="sits exactly on upstream's 32-neighbour cap"):
        check_ligand_graph(start, CUTOFF, label="7F8T_FAD", on_input=True)


def test_a_normal_ligand_passes_the_input_check_quietly():
    """The check must not cry wolf on the 427 of 428 ligands that are nowhere near the cap."""
    g = torch.Generator().manual_seed(1)
    pos = torch.rand(30, 3, generator=g) * 14.0
    assert int(neighbour_degrees(pos, CUTOFF).max()) < UPSTREAM_MAX_NEIGHBORS
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        report = check_ligand_graph(pos, CUTOFF, on_input=True)
    assert not report.at_cap


def test_degree_count_matches_the_edge_set_the_port_actually_builds(fad):
    """The guard must count the port's own edges, not a near-miss of them."""
    pos = torch.tensor(fad["crystal"]["pos"], dtype=torch.float32)
    edges = radius_edges(pos, CUTOFF)
    counted = torch.bincount(edges[1], minlength=len(pos))
    assert torch.equal(counted, neighbour_degrees(pos, CUTOFF))


# ------------------------------------------------- the guard composes with the static mask

def test_degrees_from_mask_equals_the_dynamic_degree(fad):
    """`dfd-trace`'s static complete edge set plus a 0/1 mask gives the same degrees.

    This is the composition the trace path needs: a degree cap is a per-node sum over the
    mask, so the guard reads a tensor the step already computes and adds no `cdist`.
    """
    from tt_bio.diffdock.graph import StaticGraphs

    pos = torch.tensor(fad["crystal"]["pos"], dtype=torch.float32)
    n = len(pos)
    bonds = torch.zeros((2, 0), dtype=torch.long)
    graphs = StaticGraphs.build(n_lig=n, n_rec=1, lig_bond_edges=bonds)
    mask = graphs.lig_mask(pos, CUTOFF)
    # lig_mask uses `<=`; the degree guard and radius_edges use `<`. On this fixture no pair
    # sits exactly on 5 A, so the two agree and the composition is exact.
    degrees = degrees_from_mask(graphs.lig_edge_index, mask, graphs.n_lig_padded)
    assert torch.equal(degrees[:n].long(), neighbour_degrees(pos, CUTOFF))
    assert cap_report(degrees[:n].long()).max_degree == 34


def test_a_masked_static_graph_reports_the_cap_without_materialising_neighbours(fad):
    """The static path sees the same truncation the dynamic path does."""
    from tt_bio.diffdock.graph import StaticGraphs

    pos = torch.tensor(fad["crystal"]["pos"], dtype=torch.float32)
    n = len(pos)
    graphs = StaticGraphs.build(n_lig=n, n_rec=1, lig_bond_edges=torch.zeros((2, 0),
                                                                            dtype=torch.long))
    degrees = degrees_from_mask(graphs.lig_edge_index, graphs.lig_mask(pos, CUTOFF),
                               graphs.n_lig_padded)
    assert cap_report(degrees[:n].long()).binds


def test_pad_rows_do_not_invent_degree(fad):
    """Padded edges point at a pad node, so they must not raise any real node's degree."""
    from tt_bio.diffdock.graph import StaticGraphs

    pos = torch.tensor(fad["crystal"]["pos"], dtype=torch.float32)
    n = len(pos)
    graphs = StaticGraphs.build(n_lig=n, n_rec=1, lig_bond_edges=torch.zeros((2, 0),
                                                                            dtype=torch.long))
    degrees = degrees_from_mask(graphs.lig_edge_index, graphs.lig_mask(pos, CUTOFF),
                                graphs.n_lig_padded)
    assert degrees[n:].sum() == 0


# ------------------------------------------------- the bond-centre graph carries the same cap

def test_bond_centre_graph_degrees_count_per_centre():
    """Upstream caps `radius(lig_pos, bond_pos, 5)` per bond centre, with no self pair."""
    centres = torch.zeros(2, 3)
    centres[1, 0] = 50.0
    atoms = torch.rand(40, 3) * 2.0
    deg = cross_degrees(centres, atoms, CUTOFF)
    assert deg[0] == 40 and deg[1] == 0
    assert cap_report(deg).binds


# ------------------------------------- what the cap keeps, and why it is not reproducible

def test_recorded_cap_selection_evidence_says_the_cap_is_not_reproducible():
    """The measurement the no-cap decision rests on, pinned so a later change must face it."""
    path = Path("/home/moritz/.coworker/state/dfd/evidence/cap-selection.json")
    if not path.exists():
        pytest.skip("cap-selection.json not present")
    ev = json.loads(path.read_text())
    assert ev["kept_degree_max"] == UPSTREAM_MAX_NEIGHBORS
    assert ev["nodes_keeping_first_cap_by_index"] == 0
    assert ev["nodes_keeping_nearest_cap"] == 0
    assert ev["nodes_stable_under_rotation"] == 0
    assert ev["nodes_stable_under_relabelling"] < ev["n_nodes"]


def _torch_cluster():
    return pytest.importorskip("torch_cluster",
                               reason="reference env only; see probe_cap_selection.py")


def test_torch_cluster_cap_is_neither_index_order_nor_nearest():
    """Live against torch_cluster where it is installed. Same claim as the recorded evidence."""
    tc = _torch_cluster()
    pos = dense_cloud()
    ei = tc.radius_graph(pos, CUTOFF, max_num_neighbors=UPSTREAM_MAX_NEIGHBORS)
    kept = {}
    for src, dst in zip(ei[0].tolist(), ei[1].tolist()):
        kept.setdefault(dst, []).append(src)
    d = torch.cdist(pos.double(), pos.double())
    n = len(pos)
    by_index = by_distance = 0
    for i in range(n):
        others = [j for j in range(n) if j != i]
        if sorted(kept[i]) == others[:UPSTREAM_MAX_NEIGHBORS]:
            by_index += 1
        nearest = sorted(others, key=lambda j: d[i, j].item())[:UPSTREAM_MAX_NEIGHBORS]
        if sorted(kept[i]) == sorted(nearest):
            by_distance += 1
    assert by_index == 0, "cap kept an index-order prefix; the port's rationale needs rechecking"
    assert by_distance == 0, "cap kept the nearest neighbours; a nearest-k cap would match it"


def test_torch_cluster_cap_is_not_rotation_invariant():
    """The hard stop. A reproduced cap would make the port's output depend on the input frame."""
    tc = _torch_cluster()
    pos = dense_cloud()
    theta = 0.7
    rot = torch.tensor([[math.cos(theta), -math.sin(theta), 0.0],
                        [math.sin(theta), math.cos(theta), 0.0],
                        [0.0, 0.0, 1.0]])

    def kept_sets(p):
        ei = tc.radius_graph(p, CUTOFF, max_num_neighbors=UPSTREAM_MAX_NEIGHBORS)
        out = {}
        for src, dst in zip(ei[0].tolist(), ei[1].tolist()):
            out.setdefault(dst, []).append(src)
        return {k: sorted(v) for k, v in out.items()}

    base, turned = kept_sets(pos), kept_sets(pos @ rot.T)
    stable = sum(1 for i in base if base[i] == turned.get(i))
    assert stable < len(base), "cap survived a rotation; re-read D-dfd-trace-1 before relying on it"


def test_the_port_itself_is_rotation_invariant_where_upstream_is_not():
    """The property the port keeps by not capping: same edges in any frame."""
    pos = dense_cloud()
    theta = 0.7
    rot = torch.tensor([[math.cos(theta), -math.sin(theta), 0.0],
                        [math.sin(theta), math.cos(theta), 0.0],
                        [0.0, 0.0, 1.0]])
    base = radius_edges(pos, CUTOFF)
    turned = radius_edges(pos @ rot.T, CUTOFF)
    assert torch.equal(base, turned)
