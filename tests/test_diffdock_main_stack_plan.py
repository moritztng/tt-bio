import types

import pytest

torch = pytest.importorskip("torch")

from tt_bio.diffdock import graph as G  # noqa: E402
from tt_bio.diffdock.static_net import main_stack_plan  # noqa: E402


def _model(n_layers=6, differentiate=True):
    return types.SimpleNamespace(conv_layers=list(range(n_layers)),
                                 config=types.SimpleNamespace(
                                     differentiate_convolutions=differentiate))


def _sg(n_poses=3):
    bonds = torch.stack([torch.arange(0, 6), torch.arange(1, 7)])
    return G.StaticGraphs.build(11, 37, bonds, n_rot_bonds=2, n_poses=n_poses)


def test_the_last_layer_sees_only_the_edges_that_reach_a_ligand_node():
    sg = _sg()
    rei = sg.rec_edge_index(torch.stack([torch.arange(0, 20), torch.arange(1, 21)]))
    ei, plan, (s1, s2, s3) = main_stack_plan(_model(), sg, rei)
    assert ei.shape[1] == s3 + (s2 - s1), "four blocks: lig, cross, rec, cross reversed"
    assert plan[-1][0] == s2, "the last layer is cut at the end of the cross block"
    assert all(cut == ei.shape[1] for cut, _ in plan[:-1])
    assert plan[-1][1] == [s1] and all(g == [s1, s2, s3] for _, g in plan[:-1])


def test_undifferentiated_convolutions_take_one_edge_group():
    sg = _sg()
    rei = sg.rec_edge_index(torch.stack([torch.arange(0, 20), torch.arange(1, 21)]))
    _, plan, _ = main_stack_plan(_model(differentiate=False), sg, rei)
    assert all(groups is None for _, groups in plan)


def test_the_reversed_cross_block_is_the_cross_block_flipped():
    sg = _sg()
    rei = sg.rec_edge_index(torch.stack([torch.arange(0, 20), torch.arange(1, 21)]))
    ei, _, (s1, s2, s3) = main_stack_plan(_model(), sg, rei)
    assert torch.equal(ei[:, s3:], ei[:, s1:s2].flip(0))
    assert torch.equal(ei[:, s2:s3], rei + sg.rec_offset)
