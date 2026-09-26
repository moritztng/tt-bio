"""`autograd._pairwise_sum0` sums a leading axis, for every row count including the odd ones.

Card-free by construction: the function is written out of `ttnn.add`, slicing and `deallocate`,
and numpy means the same thing by all three, so its control flow can be exercised on the host.
What that catches is the algebra of the odd level -- a leftover row carried on the side and
added once at the end -- which is where a halving tree goes wrong and where the loss is silent:
an off-by-one there drops one row out of a gradient the loop still happily descends.
"""
import pathlib
import types

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _host_pairwise_sum0():
    """The shipped source, run against numpy. Read from the file so it cannot drift from it."""
    src = (ROOT / "tt_bio" / "autograd.py").read_text()
    body = src[src.index("def _pairwise_sum0("):]
    body = body[:body.index("\ndef ", 1)]
    ns = {"ttnn": types.SimpleNamespace(add=np.add, deallocate=lambda t: None)}
    exec(compile(body, "autograd.py:_pairwise_sum0", "exec"), ns)
    return ns["_pairwise_sum0"]


@pytest.mark.parametrize("rows", list(range(1, 34)) + [63, 81, 128, 211, 224, 255, 256, 257, 300])
def test_pairwise_sum0_matches_a_plain_sum(rows):
    f = _host_pairwise_sum0()
    x = np.random.default_rng(rows).standard_normal((rows, 3, 4))
    got = f(x)
    want = x.sum(0, keepdims=True)
    assert got.shape == want.shape
    assert np.allclose(got, want, rtol=0, atol=1e-12)
