"""ESMFold2's initial pair state, drawn on a host thread from forward entry (`_PairStateDraw`).

The threaded draw must be esm 3.4.1's inline `_init_pair_state` bit for bit, and leave the global
RNG where the inline draw leaves it, so every later draw in the fold is unchanged too.
"""
import math

import pytest

torch = pytest.importorskip("torch")

from tt_bio.esmfold2_runtime import _PairStateDraw  # noqa: E402

SHAPE = (1, 24, 24, 64)


def _inline(ref):
    """esm 3.4.1 `ESMFold2Model._init_pair_state`."""
    std = math.sqrt(2.0 / (5.0 * ref.shape[-1]))
    state = torch.empty_like(ref, dtype=torch.float32)
    torch.nn.init.trunc_normal_(state, mean=0.0, std=std, a=-3 * std, b=3 * std)
    return state.to(dtype=ref.dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_threaded_draw_is_the_inline_draw_and_leaves_the_rng_where_inline_does(dtype):
    ref = torch.zeros(SHAPE, dtype=dtype)
    torch.manual_seed(7)
    inline, after_inline = _inline(ref), torch.rand(8)
    torch.manual_seed(7)
    d = _PairStateDraw()
    d.start(SHAPE)
    threaded = d.take(ref, _inline)
    assert threaded.dtype == dtype
    assert torch.equal(threaded, inline)
    assert torch.equal(torch.rand(8), after_inline)


def test_a_draw_in_between_falls_back_to_inline():
    ref = torch.zeros(SHAPE)
    torch.manual_seed(3)
    torch.rand(1)
    inline = _inline(ref)
    torch.manual_seed(3)
    d = _PairStateDraw()
    d.start(SHAPE)
    torch.rand(1)
    assert torch.equal(d.take(ref, _inline), inline)


def test_a_shape_mismatch_draws_inline_and_a_take_consumes_the_draw():
    d = _PairStateDraw()
    d.start((1, 8, 8, 64))
    out = d.take(torch.zeros(SHAPE, dtype=torch.bfloat16), _inline)
    assert out.shape == SHAPE and out.dtype == torch.bfloat16
    assert d.nxt is None
