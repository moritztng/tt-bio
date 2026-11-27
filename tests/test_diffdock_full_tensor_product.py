"""The torsion head's unweighted tensor product.

Upstream builds exactly one of these -- ``o3.FullTensorProduct(sh_irreps, "2e")`` -- and
feeds its 45-wide output to ``tor_bond_conv`` as that layer's harmonics. Two things about it
are easy to get silently wrong, and both change numbers without changing a single shape:

* **the output layout.** e3nn sorts the collected outputs, and ``Irrep`` is a ``tuple``, so
  the sort is ``(l, p)`` ascending -- *odd before even* at every l. Getting this backwards
  swaps the ``1o``/``1e`` and ``2o``/``2e`` blocks, which the next layer reads as weights.
* **the path normalisation.** ``uvuv`` connectivity reaches each output channel by exactly
  one path, so e3nn's ``element`` denominator is 1 and the coefficient is the bare
  ``component`` numerator ``sqrt(2l+1)`` -- not the fully-connected layer's rule.

Golden grade (reference env, e3nn 0.5.1): layout-identical for ``0e+1o+2e x 2e`` and max
relative error 6.86e-08 in fp32 -- fp32 rounding through one einsum. The float64 golden
below carries that comparison into this env, where the bar drops to ~1e-14.

Capturing it needed ``torch.set_default_dtype(torch.float64)`` *before* building the e3nn
module: ``.double()`` does not reach the Wigner constants, which live as plain attributes on
the TorchScript ``_compiled_main_left_right`` submodule rather than as buffers. A golden
taken from a ``.double()``-ed module is an fp32 answer in a float64 tensor, and grades this
port at 2.46e-08 -- fp32 eps -- while looking like a float64 comparison.
"""

import json
import math
import pathlib

import pytest

torch = pytest.importorskip("torch")

from tt_bio.diffdock.full_tensor_product import FullTensorProduct
from tt_bio.diffdock.harmonics import spherical_harmonics
from tt_bio.diffdock.irreps import Irreps

SH = "1x0e + 1x1o + 1x2e"
GOLDEN = pathlib.Path(__file__).parent / "data" / "diffdock_full_tp" / "sh_x_2e_float64.json"


def _ftp():
    return FullTensorProduct(SH, "2e").double()


def test_the_output_is_sorted_with_odd_before_even_at_every_l():
    out = str(_ftp().out_irreps)
    assert out.replace(" ", "") == "1x0e+1x1o+1x1e+1x2o+1x2e+1x2e+1x3o+1x3e+1x4e"
    assert _ftp().out_irreps.dim == 45


def test_it_matches_the_e3nn_golden_in_float64():
    d = json.loads(GOLDEN.read_text())
    x1 = torch.tensor(d["x1"], dtype=torch.float64)
    x2 = torch.tensor(d["x2"], dtype=torch.float64)
    want = torch.tensor(d["out"], dtype=torch.float64)
    got = _ftp()(x1, x2)
    rel = ((got - want).abs().max() / want.abs().max()).item()
    assert rel < 1e-14, f"off e3nn by {rel:.3e}"


def test_a_bare_term_parses_as_multiplicity_one():
    """``"2e"`` is e3nn's spelling of ``"1x2e"``; upstream writes the second input that way."""
    assert str(Irreps.parse("2e")) == str(Irreps.parse("1x2e"))


def test_the_scalar_output_is_rotation_invariant_and_the_vectors_rotate():
    """Both inputs here are harmonics of real vectors, so a rotation of those vectors is a
    rotation of the inputs. Then the ``0e`` column must not move at all, and each ``l=1``
    block must rotate by the same matrix -- the property the torsion score depends on."""
    g = torch.Generator().manual_seed(11)
    v1 = torch.randn(16, 3, generator=g, dtype=torch.float64)
    v2 = torch.randn(16, 3, generator=g, dtype=torch.float64)
    c, s_ = math.cos(0.6), math.sin(0.6)     # math, not torch: torch.tensor(0.6) is fp32
    r = torch.tensor([[1., 0., 0.], [0., c, -s_], [0., s_, c]], dtype=torch.float64)
    f = _ftp()
    out = f(spherical_harmonics(v1), spherical_harmonics(v2, ls=(2,)))
    rot = f(spherical_harmonics(v1 @ r.T), spherical_harmonics(v2 @ r.T, ls=(2,)))
    assert (rot[:, 0] - out[:, 0]).abs().max() < 1e-12
    for sl in (slice(1, 4), slice(4, 7)):          # 1o then 1e
        want = out[:, sl] @ r.T
        rel = (rot[:, sl] - want).abs().max() / want.abs().max()
        assert rel < 1e-12, f"l=1 block {sl} off by {rel:.3e}"


def test_a_wrong_width_is_caught_rather_than_broadcast():
    f = _ftp()
    with pytest.raises(ValueError, match="x2 width"):
        f(torch.zeros(2, 9, dtype=torch.float64), torch.zeros(2, 9, dtype=torch.float64))
    with pytest.raises(ValueError, match="x1 width"):
        f(torch.zeros(2, 5, dtype=torch.float64), torch.zeros(2, 5, dtype=torch.float64))
