"""ESMFold2's initial pair state, drawn on a host thread from forward entry (`_PairStateDraw`).

The draw must follow the fold's seed only: the same seed gives the same state whether the thread
finished early, late or was never started, and the distribution is esm 3.4.1's truncated normal.
"""
import math

import pytest

torch = pytest.importorskip("torch")

from tt_bio.esmfold2_runtime import _PairStateDraw  # noqa: E402

SHAPE = (1, 24, 24, 64)


def _ref(dtype=torch.float32):
    return torch.zeros(SHAPE, dtype=dtype)


def test_threaded_and_inline_draws_follow_the_seed():
    torch.manual_seed(7)
    d = _PairStateDraw()
    d.start(SHAPE)
    threaded = d.take(_ref())
    torch.manual_seed(7)
    inline = _PairStateDraw().take(_ref())
    assert torch.equal(threaded, inline)


def test_a_shape_mismatch_draws_inline_and_a_take_consumes_the_draw():
    d = _PairStateDraw()
    d.start((1, 8, 8, 64))
    out = d.take(_ref(torch.bfloat16))
    assert out.shape == SHAPE and out.dtype == torch.bfloat16
    assert d.nxt is None


def test_truncated_normal_with_upstream_std():
    torch.manual_seed(0)
    d = _PairStateDraw()
    d.start((1, 128, 128, 256))
    z = d.take(torch.zeros(1, 128, 128, 256))
    std = math.sqrt(2.0 / (5.0 * 256))
    assert float(z.abs().max()) <= 3 * std
    assert abs(float(z.std()) / std - 0.986) < 0.01  # std of a +-3 sigma truncated normal
    assert abs(float(z.mean())) < 1e-3 * std * 10
