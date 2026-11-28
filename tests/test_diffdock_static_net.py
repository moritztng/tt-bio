"""The fixed-shape forward against the per-pose one, card-free.

``static_net.static_forward`` runs every pose of a step in one forward over complete padded edge
sets with the step's graphs as masks; ``ScoreModel.forward`` runs one pose with ``nonzero``-built
graphs. Same submodules, same weights, so the two agree to float arithmetic -- which is the whole
claim the capture rests on. These run a small random model in float64, so a disagreement here is
a wiring error and not precision.
"""

import pytest

torch = pytest.importorskip("torch")

from tests.test_diffdock_score_model import _inputs, _model  # noqa: E402
from tt_bio.diffdock.conv import pad_nodes  # noqa: E402
from tt_bio.diffdock.static_net import StaticComplex, static_forward  # noqa: E402

SIGMAS = (4.0, 0.9, 0.7)
T_TR = 0.4


def _case(n_poses=2, n_lig=9, n_rec=14, dtype=torch.float64):
    m = _model().to(dtype)
    lig_x, lig_pos, bond_index, bond_attr, edge_mask, rec_x, rec_pos, rec_ei = _inputs(
        n_lig, n_rec)
    lig = dict(x=lig_x, edge_index=bond_index, edge_attr=bond_attr, edge_mask=edge_mask)
    g = torch.Generator().manual_seed(11)
    lig_pos, rec_x, rec_pos = lig_pos.to(dtype), rec_x.to(dtype), rec_pos.to(dtype)
    bond_attr = bond_attr.to(dtype)
    lig["edge_attr"] = bond_attr
    poses = torch.stack([lig_pos + torch.randn(n_lig, 3, generator=g, dtype=torch.float64).to(
        dtype) for _ in range(n_poses)])
    c = StaticComplex.build(lig, rec_x, rec_pos, rec_ei, n_poses)
    flat = pad_nodes(poses.reshape(n_poses * n_lig, 3), c.sg.n_lig_padded)
    return m, lig, rec_x, rec_pos, rec_ei, poses, c, flat


def _per_pose(m, lig, rec_x, rec_pos, rec_ei, poses, crop=None):
    from tt_bio.diffdock.sampling import crop_beyond
    tr, rot, tor = [], [], []
    for p in poses:
        r, rx = dict(x=rec_x, pos=rec_pos, edge_index=rec_ei), rec_x
        if crop is not None:
            keep = torch.any(torch.cdist(rec_pos, p) < crop, dim=1)
            r, rx = crop_beyond(r, p, crop), rec_x[keep]
        with torch.no_grad():
            a, b, t = m(lig["x"], p, lig["edge_index"], lig["edge_attr"], lig["edge_mask"],
                        rx, r["pos"], r["edge_index"], T_TR,
                        *(torch.as_tensor(s, dtype=p.dtype) for s in SIGMAS))
        tr.append(a)
        rot.append(b)
        tor.append(t)
    return torch.cat(tr), torch.cat(rot), torch.cat(tor)


def _close(got, ref, tol=1e-9):
    assert got.shape == ref.shape, f"{got.shape} != {ref.shape}"
    if ref.numel():
        rel = (got - ref).abs().max() / ref.abs().max()
        assert rel < tol, f"max rel err {rel:.3e}"


@pytest.mark.parametrize("n_poses", [1, 3])
def test_the_static_forward_matches_the_per_pose_one(n_poses):
    m, lig, rec_x, rec_pos, rec_ei, poses, c, flat = _case(n_poses)
    with torch.no_grad():
        got = static_forward(m, c, flat, T_TR, *SIGMAS)
    ref = _per_pose(m, lig, rec_x, rec_pos, rec_ei, poses)
    for g, r in zip(got, ref):
        _close(g, r)


def test_the_crop_is_a_node_mask_and_gives_the_cropped_answer():
    """D-dfd-orch-11: masking the residues upstream deletes is exact, not approximate.

    float32: upstream's ``crop_beyond`` casts the pose to float32 and compares it against the
    receptor as it stands, so the cropped path has no float64 form to grade against.
    """
    m, lig, rec_x, rec_pos, rec_ei, poses, c, flat = _case(2, dtype=torch.float32)
    crop = 8.0
    keep = c.sg.rec_keep(flat, c.rec_pos, crop)
    assert 0 < keep.sum() < c.sg.n_rec_total, "the crop has to actually drop residues"
    with torch.no_grad():
        got = static_forward(m, c, flat, T_TR, *SIGMAS, crop_cutoff=crop)
    ref = _per_pose(m, lig, rec_x, rec_pos, rec_ei, poses, crop=crop)
    for g, r in zip(got, ref):
        _close(g, r, tol=1e-5)


def test_the_shape_does_not_move_when_the_poses_do():
    """The point of the whole module: one capture covers every step of the rollout."""
    m, lig, rec_x, rec_pos, rec_ei, poses, c, flat = _case(2, dtype=torch.float32)
    shapes = []
    for scale in (1.0, 3.0, 12.0):
        with torch.no_grad():
            out = static_forward(m, c, flat * scale, T_TR, *SIGMAS, crop_cutoff=40.0)
        shapes.append(tuple(t.shape for t in out))
    assert len(set(shapes)) == 1, shapes


def test_the_row_grid_is_tile_aligned_geometric_and_exact_when_small():
    from tt_bio.diffdock.conv import bucket
    assert all(bucket(n) == -(-n // 32) * 32 for n in range(1, 1025))
    sizes = sorted({bucket(n) for n in range(1, 1_500_000, 5)})
    assert len(sizes) < 210, "the grid has to be small enough to compile once"
    for n in range(1025, 400_000, 37):
        b = bucket(n)
        assert b % 32 == 0 and n <= b <= n * 1.0625 + 32 and bucket(b) == b


def test_rows_padded_to_the_grid_contribute_nothing():
    """A complex whose edge sets sit between grid sizes: the cross set is padded by whole tiles
    of dead edges, and the float64 result is still the per-pose forward's."""
    m, lig, rec_x, rec_pos, rec_ei, poses, c, flat = _case(5, n_lig=11, n_rec=21)
    cross = 5 * 11 * 21
    assert c.sg.cross_edge_index.shape[1] - cross >= 32, "the case has to exercise the grid"
    with torch.no_grad():
        got = static_forward(m, c, flat, T_TR, *SIGMAS)
    for g, r in zip(got, _per_pose(m, lig, rec_x, rec_pos, rec_ei, poses)):
        _close(g, r)


def test_ligands_in_one_grid_cell_share_every_row_axis():
    from tt_bio.diffdock.graph import StaticGraphs
    bonds = torch.zeros(2, 60, dtype=torch.long)
    shape = lambda sg: (sg.n_lig_padded, sg.lig_edge_index.shape[1], sg.cross_edge_index.shape[1])
    a, b = (StaticGraphs.build(n, 330, bonds, 4, 10) for n in (30, 31))
    assert shape(a)[2] == shape(b)[2], "30 and 31 atoms on 330 residues share the cross axis"
