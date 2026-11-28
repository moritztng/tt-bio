"""The static graph is one pose tiled over the poses: `pose_blocks` finds the one pose."""
import pytest
import torch

from tt_bio.diffdock.device import pose_blocks, pose_layout
from tt_bio.diffdock.graph import StaticGraphs


def stack(n_lig=7, n_rec=13, poses=4):
    bonds = torch.tensor([[0, 1, 1, 2, 5, 6], [1, 0, 2, 1, 6, 5]])
    sg = StaticGraphs.build(n_lig, n_rec, bonds, n_poses=poses)
    g = torch.Generator().manual_seed(0)
    rec = torch.stack([torch.arange(n_rec).repeat_interleave(3),
                       torch.randint(0, n_rec, (3 * n_rec,), generator=g)])
    lei, cg = sg.lig_edge_index, sg.cross_edge_index_global()
    rei = sg.rec_edge_index(rec) + sg.rec_offset
    ei = torch.cat([lei, cg, rei, cg.flip(0)], 1)
    s1 = lei.shape[1]
    bounds = [0, s1, s1 + cg.shape[1], s1 + cg.shape[1] + rei.shape[1], ei.shape[1]]
    return sg, ei, bounds, [(0, n_lig), (sg.rec_offset, n_rec)]


def test_blocks_reproduce_the_dense_scatter_and_gather():
    sg, ei, bounds, sets = stack()
    P, N = sg.n_poses, sg.n_lig_padded + sg.n_rec_padded
    g = torch.Generator().manual_seed(1)
    x = torch.randn(N, 5, dtype=torch.float64, generator=g)
    msg = torch.randn(ei.shape[1], 5, dtype=torch.float64, generator=g)
    real = lambda n: sum((n >= s) & (n < s + k * P) for s, k in sets).bool()
    live = real(ei[0]) & real(ei[1])          # a dead row ends on a pad node
    dense = torch.zeros(N, 5, dtype=torch.float64).index_add_(0, ei[0, live], msg[live])
    got, seen = torch.zeros_like(dense), 0
    for lo, hi in zip(bounds, bounds[1:]):
        b = pose_blocks(ei, lo, hi, sets, P)
        assert b is not None
        e = b["src"].numel()
        seen += b["rows"].numel()
        (so, no), (si, ni) = sets[b["out_set"]], sets[b["in_set"]]
        S = torch.zeros(no, e, dtype=torch.float64)
        S[b["src"], torch.arange(e)] = 1
        got[so:so + no * P] += (S @ msg[b["rows"]]).reshape(no * P, 5)   # [n, e] @ [P, e, d]
        G = torch.zeros(e, ni, dtype=torch.float64)
        G[torch.arange(e), b["dst"]] = 1
        assert torch.equal(G @ x[si:si + ni * P].reshape(P, ni, 5), x[ei[1, b["rows"]]])
        assert torch.equal(ei[0, b["rows"]], so + b["src"] + no * torch.arange(P)[:, None])
    assert seen == int(live.sum())
    assert torch.allclose(got, dense, rtol=0, atol=1e-12)


@pytest.mark.parametrize("break_it", ["cross_pose", "one_pose_differs"])
def test_not_a_tiling_is_refused(break_it):
    sg, ei, bounds, sets = stack()
    ei = ei.clone()
    if break_it == "cross_pose":
        ei[1, 0] += sg.n_lig          # an edge that reads another pose
    else:
        ei[1, 0], ei[1, 1] = ei[1, 1].clone(), ei[1, 0].clone()
    assert pose_blocks(ei, bounds[0], bounds[1], sets, sg.n_poses) is None


def test_layout_gathers_and_scatters_like_the_dense_graph():
    """`pose_layout` rows, with the one-pose matrices a pose graph uploads, against index ops."""
    sg, ei, bounds, sets = stack()
    P, N, E = sg.n_poses, sg.n_lig_padded + sg.n_rec_padded, ei.shape[1]
    lay = pose_layout(ei, [(E, bounds[1:-1]), (bounds[2], bounds[1:2])], sets, P)
    assert [lay["pos"][b] for b in bounds] == [0] + [b["at"] + P * b["e"] for b in lay["blocks"]]
    assert all(b["at"] % 32 == 0 and b["e"] % 32 == 0 for b in lay["blocks"])
    g = torch.Generator().manual_seed(2)
    x = torch.randn(N, 5, dtype=torch.float64, generator=g)
    msg = torch.randn(E, 5, dtype=torch.float64, generator=g)
    take = lambda t, at: torch.cat([t, t.new_zeros(1, t.shape[1])])[at]
    xl, ml = take(x, lay["nodes"]), take(msg, lay["edges"])
    real = lay["edges"] >= 0
    assert torch.equal(lay["edges"][real].sort().values,
                       torch.cat([pose_blocks(ei, lo, hi, sets, P)["rows"].reshape(-1)
                                  for lo, hi in zip(bounds, bounds[1:])]).sort().values)
    out = torch.zeros_like(xl)
    for b in lay["blocks"]:
        k, e = b["src"].numel(), b["e"]
        (ao, no), (ai, ni) = ((lay["node_at"][s], lay["sets"][s]) for s in (b["out_set"], b["in_set"]))
        S, G = torch.zeros(no, e, dtype=torch.float64), torch.zeros(e, ni, dtype=torch.float64)
        S[b["src"], torch.arange(k)] = 1
        G[torch.arange(k), b["dst"]] = 1
        rows = slice(b["at"], b["at"] + P * e)
        live = lay["edges"][rows] >= 0
        for side, M, a, n in (("src", S.T, ao, no), ("dst", G, ai, ni)):
            got = (M @ xl[a:a + P * n].reshape(P, n, 5)).reshape(P * e, 5)
            want = x[ei[0 if side == "src" else 1, lay["edges"][rows][live]]]
            assert torch.equal(got[live], want) and not got[~live].any()
        out[ao:ao + P * no] += (S @ ml[rows].reshape(P, e, 5)).reshape(P * no, 5)
    live = lay["edges"][real]
    dense = torch.zeros(N, 5, dtype=torch.float64).index_add_(0, ei[0, live], msg[live])
    back = torch.zeros_like(dense)
    back[lay["nodes"][lay["nodes"] >= 0]] = out[lay["nodes"] >= 0]
    assert torch.allclose(back, dense, rtol=0, atol=1e-12)
    assert not out[lay["nodes"] < 0].any()


def test_layout_refuses_what_is_not_a_tiling():
    sg, ei, bounds, sets = stack()
    ei = ei.clone()
    ei[1, 0] += sg.n_lig
    assert pose_layout(ei, [(ei.shape[1], None)], sets, sg.n_poses) is None
