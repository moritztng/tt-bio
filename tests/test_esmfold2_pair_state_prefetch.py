"""The trunk's initial pair state drawn on a host thread is the reference's draw, and leaves the
global RNG where the reference leaves it (esmfold2_runtime._install_pair_state_prefetch)."""
import types

import torch
import torch.nn as nn

from tt_bio import esmfold2_runtime as R
from tt_bio._vendor.esmfold2_hf.modeling_esmfold2 import ESMFold2Model


class _Emb(nn.Module):
    def forward(self, **kw):
        return kw["aatype"].float()


def _model():
    m = types.SimpleNamespace(inputs_embedder=_Emb(), z_init_1=nn.Linear(4, 16, bias=False))
    m._init_pair_state = types.MethodType(ESMFold2Model._init_pair_state, m)
    return m


def _fold(m, L=40):
    torch.manual_seed(7)
    m.inputs_embedder(aatype=torch.ones(1, L, 33))
    return m._init_pair_state(torch.zeros(1, L, L, 16)), torch.rand(3)


def test_prefetched_state_is_the_reference_draw():
    m = _model()
    ref = _fold(m)
    R._install_pair_state_prefetch(m)
    for _ in range(2):
        got = _fold(m)
        assert torch.equal(ref[0], got[0]) and torch.equal(ref[1], got[1])


def test_shape_mismatch_and_switch_off_fall_back():
    m = _model()
    R._install_pair_state_prefetch(m)
    torch.manual_seed(7)
    m.inputs_embedder(aatype=torch.ones(1, 40, 33))
    z = m._init_pair_state(torch.zeros(1, 8, 8, 16))  # not the drawn shape: the reference call
    assert z.shape == (1, 8, 8, 16)
    R.PAIR_STATE_PREFETCH = False
    try:
        ref = _fold(_model())
        assert all(torch.equal(x, y) for x, y in zip(ref, _fold(m)))
    finally:
        R.PAIR_STATE_PREFETCH = True
