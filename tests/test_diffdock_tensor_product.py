"""The tensor product, held to equivariance in float64 and to e3nn where e3nn is present.

Two independent grades, because they catch different mistakes:

* **Equivariance** (runs anywhere, no e3nn): rotate the inputs, rotate the output, they must
  agree. This catches a wrong Wigner 3j, a wrong path or a mis-sliced irrep, and it is the
  property the model's correctness actually rests on -- `dfd-equivariance` is a first-class
  accuracy gate for this port, not a nicety.
* **e3nn agreement** (skipped without e3nn): equivariance is blind to an overall per-path
  scale, and a wrong path normalisation is exactly such a scale. Only e3nn settles it, and
  only e3nn settles the 3j sign. `scripts/diffdock_port/tp_parity.py` runs the full grade in
  the reference environment; this test keeps a small version in the suite.

Measured in the reference environment (e3nn 0.5.0, torch 2.4.1+cpu) over all six distinct
signatures the released checkpoint builds: worst PCC 0.9999999999999879, worst max relative
error 3.72e-07.
"""

import math

import pytest

torch = pytest.importorskip("torch")

from tt_bio.diffdock.irreps import Irreps, spherical_harmonics_irreps
from tt_bio.diffdock.tensor_product import FullyConnectedTensorProduct, wigner_3j

SH = str(spherical_harmonics_irreps(2))
# The signatures the released DiffDock-L builds, from config.layer_plan().
SIGNATURES = [
    ("48x0e", SH, "48x0e + 10x1o"),
    ("48x0e + 10x1o", SH, "48x0e + 10x1o + 10x1e"),
    ("48x0e + 10x1o + 10x1e", SH, "48x0e + 10x1o + 10x1e + 10x0o"),
    ("48x0e + 10x1o + 10x1e + 10x0o", SH, "48x0e + 10x1o + 10x1e + 10x0o"),
    ("48x0e + 10x1o + 10x1e + 10x0o", SH, "2x1o + 2x1e"),
]
SMALL = [("4x0e + 2x1o + 2x1e + 2x0o", SH, "4x0e + 2x1o + 2x1e + 2x0o")]


def test_tables_are_normalised_and_have_the_selection_rules():
    for (l1, l2, l3) in [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0), (1, 1, 1), (1, 2, 1)]:
        w = wigner_3j(l1, l2, l3, dtype=torch.float64)
        assert w.shape == (2 * l1 + 1, 2 * l2 + 1, 2 * l3 + 1)
        assert torch.linalg.norm(w).item() == pytest.approx(1.0, abs=1e-12)
    # l x l -> 0 is the identity over sqrt(2l+1); this pins both scale and sign.
    assert torch.allclose(wigner_3j(1, 1, 0, dtype=torch.float64)[:, :, 0],
                          torch.eye(3, dtype=torch.float64) / math.sqrt(3), atol=1e-12)


def test_an_unlisted_signature_fails_loudly():
    with pytest.raises(KeyError, match="gen_w3j"):
        wigner_3j(2, 2, 2)


@pytest.mark.parametrize("sig", SIGNATURES)
def test_weight_numel_and_path_order(sig):
    tp = FullyConnectedTensorProduct(*sig)
    from tt_bio.diffdock.irreps import weight_numel
    assert tp.weight_numel == weight_numel(*sig)
    # in1 outermost, then the harmonics, then out: e3nn's order, and the order the FC
    # block's output is sliced in.
    assert list(tp.paths) == sorted(tp.paths)


def _rotation(alpha, beta, gamma, dtype=torch.float64):
    """A rotation, returned twice: as it acts on xyz coordinates and on an e3nn l=1 irrep.

    They are not the same matrix. e3nn orders the three components of an l=1 irrep as
    (y, z, x), so the irrep rotation is ``P R P^T`` for that permutation. Feeding a
    coordinate vector through the irrep matrix, or the other way round, leaves a residual
    that looks like a small numerical error rather than the basis mistake it is.
    """
    ca, sa = math.cos(alpha), math.sin(alpha)
    cb, sb = math.cos(beta), math.sin(beta)
    cg, sg = math.cos(gamma), math.sin(gamma)
    rz1 = torch.tensor([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]], dtype=dtype)
    ry = torch.tensor([[cb, 0.0, sb], [0.0, 1.0, 0.0], [-sb, 0.0, cb]], dtype=dtype)
    rz2 = torch.tensor([[cg, -sg, 0.0], [sg, cg, 0.0], [0.0, 0.0, 1.0]], dtype=dtype)
    r = rz1 @ ry @ rz2
    p = [1, 2, 0]                       # xyz -> e3nn's (y, z, x)
    return r, r[p][:, p]


def _real_sh_lmax2(v):
    """e3nn's ``o3.spherical_harmonics('1x0e+1x1o+1x2e', v, normalize=True, 'component')``.

    Written out so the equivariance test does not depend on e3nn. Component normalisation
    means each degree-l block has norm sqrt(2l+1); the l=2 block is the standard real
    harmonics in e3nn's (y, z, x) component order.
    """
    n = v / torch.linalg.norm(v, dim=-1, keepdim=True)
    y, z, x = n[..., 0], n[..., 1], n[..., 2]
    s3, s15, s5 = math.sqrt(3.0), math.sqrt(15.0), math.sqrt(5.0)
    l0 = torch.ones_like(x).unsqueeze(-1)
    l1 = s3 * torch.stack([y, z, x], dim=-1)
    l2 = torch.stack([
        s15 * x * y,
        s15 * y * z,
        (s5 / 2.0) * (3.0 * z * z - 1.0),
        s15 * x * z,
        (s15 / 2.0) * (x * x - y * y),
    ], dim=-1)
    return torch.cat([l0, l1, l2], dim=-1)


def _rotate_features(irreps: Irreps, x, d1):
    out = torch.empty_like(x)
    for term, sl in zip(irreps, irreps.slices()):
        block = x[..., sl].reshape(*x.shape[:-1], term.mul, 2 * term.l + 1)
        if term.l == 0:
            rot = block
        elif term.l == 1:
            rot = block @ d1.T
        else:
            raise AssertionError("feature irreps never exceed l=1 in DiffDock")
        out[..., sl] = rot.reshape(*x.shape[:-1], term.dim)
    return out


@pytest.mark.parametrize("sig", SMALL + SIGNATURES[-1:])
def test_equivariance_in_float64(sig):
    """Rotating the input rotates the output, to float64 precision.

    The harmonics are recomputed from the rotated direction vector rather than rotated as a
    tensor, which is the stronger statement: it is the path the model actually takes, where
    the harmonics come out of the edge geometry every diffusion step.
    """
    torch.manual_seed(7)
    tp = FullyConnectedTensorProduct(*sig).double()
    n = 64
    x1 = torch.randn(n, tp.in_irreps.dim, dtype=torch.float64)
    v = torch.randn(n, 3, dtype=torch.float64)
    w = torch.randn(n, tp.weight_numel, dtype=torch.float64)

    out = tp(x1, _real_sh_lmax2(v), w)

    # v is already in e3nn's (y, z, x) component order -- that is what _real_sh_lmax2
    # reads -- so it rotates with the irrep matrix, not the xyz one.
    _, d1 = _rotation(0.3, -1.1, 2.4)
    out_rot = tp(_rotate_features(tp.in_irreps, x1, d1), _real_sh_lmax2(v @ d1.T), w)
    want = _rotate_features(tp.out_irreps, out, d1)

    rel = (out_rot - want).abs().max() / want.abs().max()
    assert rel < 1e-12, f"equivariance broken at {rel:.3e}"


def test_parity_is_scaled_not_just_rotated():
    """A mirror flips the odd irreps and leaves the even ones alone.

    Parity is the half of O(3) equivariance a rotation test cannot see, and DiffDock's
    pseudoscalar track (``10x0o``) exists precisely to carry it.
    """
    torch.manual_seed(11)
    sig = ("4x0e + 2x1o + 2x1e + 2x0o", SH, "4x0e + 2x1o + 2x1e + 2x0o")
    tp = FullyConnectedTensorProduct(*sig).double()
    n = 32
    x1 = torch.randn(n, tp.in_irreps.dim, dtype=torch.float64)
    v = torch.randn(n, 3, dtype=torch.float64)
    w = torch.randn(n, tp.weight_numel, dtype=torch.float64)

    def flip(irreps, x):
        out = x.clone()
        for term, sl in zip(irreps, irreps.slices()):
            if term.p < 0:
                out[..., sl] = -out[..., sl]
        return out

    out = tp(x1, _real_sh_lmax2(v), w)
    out_mirror = tp(flip(tp.in_irreps, x1), _real_sh_lmax2(-v), w)
    want = flip(tp.out_irreps, out)
    rel = (out_mirror - want).abs().max() / want.abs().max()
    assert rel < 1e-12, f"parity equivariance broken at {rel:.3e}"




def _have_e3nn():
    try:
        import e3nn  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _have_e3nn(),
                    reason="e3nn is the upstream reference and is deliberately not a tt-bio "
                           "dependency; run scripts/diffdock_port/tp_parity.py in the "
                           "reference environment")
@pytest.mark.parametrize("sig", SIGNATURES)
def test_matches_e3nn(sig):
    from e3nn import o3

    torch.manual_seed(0)
    ours = FullyConnectedTensorProduct(*sig)
    ref = o3.FullyConnectedTensorProduct(*sig, shared_weights=False)
    assert ours.weight_numel == ref.weight_numel
    x1 = torch.randn(128, ours.in_irreps.dim)
    x2 = torch.randn(128, ours.sh_irreps.dim)
    w = torch.randn(128, ours.weight_numel)
    a, b = ours(x1, x2, w), ref(x1, x2, w)
    rel = ((a - b).abs().max() / b.abs().max()).item()
    pcc = torch.corrcoef(torch.stack([a.flatten().double(), b.flatten().double()]))[0, 1]
    assert rel < 1e-5, f"max relative error {rel:.3e}"
    assert pcc > 1 - 1e-12
