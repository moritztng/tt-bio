"""ESMFold2's initial pair state, drawn on a host thread from forward entry (`_PairStateDraw`).

With TRUNC_REJECT off the draw must be esm 3.4.1's inline `_init_pair_state` bit for bit. Either way it
must leave the global RNG where a draw on the spot leaves it, so every later draw in the fold is the
same, and a seed must give one state whether it was drawn on the thread or on the spot.
"""
import math

import pytest

torch = pytest.importorskip("torch")

import tt_bio.esmfold2_runtime as R  # noqa: E402
from tt_bio.esmfold2_runtime import _PairStateDraw  # noqa: E402

SHAPE = (1, 24, 24, 64)
STD = math.sqrt(2.0 / (5.0 * SHAPE[-1]))


def _inline(ref):
    """esm 3.4.1 `ESMFold2Model._init_pair_state`."""
    state = torch.empty_like(ref, dtype=torch.float32)
    torch.nn.init.trunc_normal_(state, mean=0.0, std=STD, a=-3 * STD, b=3 * STD)
    return state.to(dtype=ref.dtype)


def _on_the_spot(ref, seed, between=False):
    torch.manual_seed(seed)
    if between:
        torch.rand(1)
    return _PairStateDraw().take(ref), torch.rand(8)


def _threaded(ref, seed, between=False):
    torch.manual_seed(seed)
    d = _PairStateDraw()
    d.start(tuple(ref.shape))
    if between:
        torch.rand(1)
    return d.take(ref), torch.rand(8)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_off_it_is_the_inline_draw_and_leaves_the_rng_where_inline_does(monkeypatch, dtype):
    monkeypatch.setattr(R, "TRUNC_REJECT", False)
    ref = torch.zeros(SHAPE, dtype=dtype)
    torch.manual_seed(7)
    inline, after_inline = _inline(ref), torch.rand(8)
    for draw in (_threaded, _on_the_spot):
        state, after = draw(ref, 7)
        assert state.dtype == dtype
        assert torch.equal(state, inline) and torch.equal(after, after_inline)


@pytest.mark.parametrize("reject", [False, True])
@pytest.mark.parametrize("between", [False, True])
def test_a_seed_gives_one_state_on_the_thread_or_on_the_spot(monkeypatch, reject, between):
    monkeypatch.setattr(R, "TRUNC_REJECT", reject)
    ref = torch.zeros(SHAPE)
    (a, ra), (b, rb) = _threaded(ref, 3, between), _on_the_spot(ref, 3, between)
    assert torch.equal(a, b) and torch.equal(ra, rb)


def test_rejection_stays_inside_three_std(monkeypatch):
    monkeypatch.setattr(R, "TRUNC_REJECT", True)
    state, _ = _on_the_spot(torch.zeros((1, 64, 64, 128)), 11)
    assert state.abs().max() <= 3 * math.sqrt(2.0 / (5.0 * 128))


def test_a_shape_mismatch_draws_on_the_spot_and_a_take_consumes_the_draw():
    d = _PairStateDraw()
    d.start((1, 8, 8, 64))
    out = d.take(torch.zeros(SHAPE, dtype=torch.bfloat16))
    assert out.shape == SHAPE and out.dtype == torch.bfloat16
    assert d.nxt is None
