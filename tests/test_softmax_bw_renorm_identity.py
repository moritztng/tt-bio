"""The identity that lets a fused softmax backward keep the renorm correction.

`ttnn.moreh_softmax_backward(y, g)` computes `y * (g - rowsum(g*y))`, which is the softmax
backward only when the rows of `y` sum to 1. Ours does not assume that: `SOFTMAX_BW_RENORM`
divides by `rowsum(y)` and has been default-on since 2026-09-21. With `S = rowsum(y)` and
`yn = y / S`,

    moreh(yn, g) = yn * (g - rowsum(g*yn)) = (y/S) * (g - rowsum(g*y)/S) = dx / S

so `dx = S * moreh(y/S, g)`. This checks that algebra on the host, over rows that deliberately do
NOT sum to 1 -- on rows that do, the identity is trivially true and the test would pass while
saying nothing.

What this does NOT check, and the split is the point: whether the wheel's kernel computes what its
documentation says on Blackhole. Its sibling `moreh_layer_norm_backward` does not
(dx 2.741e+06 relative L2 in bf16, upstream #12349). That question needs a device and belongs to
`perf/bcx_bwbytes/softmax_bw_probe.py`, which grades every arm against float64.
"""
import numpy as np
import pytest


def moreh(y, g):
    """`ttnn.moreh_softmax_backward`'s documented expression, on the host."""
    return y * (g - (g * y).sum(-1, keepdims=True))


def shipped(y, g):
    """`autograd.softmax_bw_dx`'s chain with SOFTMAX_BW_RENORM on."""
    inner = (g * y).sum(-1, keepdims=True) / y.sum(-1, keepdims=True)
    return y * (g - inner)


@pytest.mark.parametrize("rowsum", [0.9, 0.9895, 1.0, 1.02, 1.3])
@pytest.mark.parametrize("shape", [(4, 8), (3, 2, 16), (2, 4, 8, 8)])
def test_rescaled_moreh_equals_the_renormed_chain(shape, rowsum):
    rng = np.random.default_rng(len(shape) * 1000 + int(rowsum * 1e4))
    y = rng.random(shape) + 0.05
    y = y / y.sum(-1, keepdims=True) * rowsum          # rows sum to `rowsum`, not to 1
    g = rng.standard_normal(shape)

    s = y.sum(-1, keepdims=True)
    got = s * moreh(y / s, g)
    want = shipped(y, g)
    assert np.allclose(got, want, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("rowsum", [0.9, 1.3])
def test_unrescaled_moreh_differs_when_the_rows_do_not_sum_to_one(rowsum):
    """The control. Without the rescale the two disagree, so the test above is not vacuous."""
    rng = np.random.default_rng(7)
    y = rng.random((5, 16)) + 0.05
    y = y / y.sum(-1, keepdims=True) * rowsum
    g = rng.standard_normal((5, 16))
    assert not np.allclose(moreh(y, g), shipped(y, g), rtol=1e-6, atol=1e-9)


def test_they_agree_when_the_rows_do_sum_to_one():
    """And the other edge: at rowsum 1 the rescale is the identity, so nothing is being smuggled."""
    rng = np.random.default_rng(11)
    y = rng.random((5, 16)) + 0.05
    y = y / y.sum(-1, keepdims=True)
    g = rng.standard_normal((5, 16))
    assert np.allclose(moreh(y, g), shipped(y, g), rtol=1e-10, atol=1e-12)
