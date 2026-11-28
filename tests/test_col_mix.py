"""`col_mix`: what it serves, card-free. The arithmetic is graded on a card by `perf/bc4_msafuse/col_probe.py`
against the float64 function in `perf/bc4_msafuse/col_ref.py`, whose hand-written VJP is checked here."""
import importlib.util
import pathlib

import pytest
import torch
import ttnn

from tt_bio import col_mix as C


class _MC:
    memory_layout = ttnn.TensorMemoryLayout.INTERLEAVED


class _Op:
    def __init__(self, shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
        self.shape, self.dtype, self.layout = shape, dtype, layout

    def memory_config(self):
        return _MC


@pytest.mark.parametrize("qkv,head,bias,ok", [
    (_Op([2, 224, 768]), 32, None, True),
    (_Op([2, 200, 768]), 32, _Op([1, 200, 1]), True),
    (_Op([2, 224, 768]), 16, None, False),                    # a head is not one tile
    (_Op([3, 224, 768]), 32, None, False),                    # not two MSA rows
    (_Op([2, 224, 768], ttnn.float32), 32, None, False),
    (_Op([2, 224, 768]), 32, _Op([2, 224, 1]), False),        # a bias per MSA row
    (_Op([2, 224, 760]), 32, None, False),
])
def test_eligible(monkeypatch, qkv, head, bias, ok):
    monkeypatch.setattr(C, "COL_MIX", True)
    assert C.eligible(qkv, head, bias) is ok


def test_off_by_default_declines_everything():
    assert not C.COL_MIX and not C.eligible(_Op([2, 224, 768]), 32)


def test_reference_vjp_matches_autograd():
    path = pathlib.Path(__file__).resolve().parents[1] / "perf" / "bc4_msafuse" / "col_ref.py"
    spec = importlib.util.spec_from_file_location("col_ref", path)
    R = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(R)
    torch.manual_seed(1)
    n, c = 64, R.H * R.D
    r = lambda *s: torch.randn(*s, dtype=torch.float64)
    xn = r(2, n, c).requires_grad_()
    wq, wk, wv, wg, wo = (r(c, c) / c ** 0.5 for _ in range(5))
    dy = r(2, n, c)
    y, saved = R.forward(xn, wq, wk, wv, wg, r(c), wo, r(c), r(1, n, 1))
    (ref,) = torch.autograd.grad(y, xn, dy)
    got = R.vjp(dy, [t.detach() for t in saved], wq, wk, wv, wg, wo)
    assert float((got - ref).norm() / ref.norm()) < 1e-12
