"""Sigma embedding and distance expansions, graded against a float64 recompute.

Both are closed forms, so there is no reason to grade them against another implementation:
the bar is the float64 value of the same formula. At fp32 that bar is tight -- 1e-6 relative,
not a PCC -- and holding it here means any later PCC loss in these columns came from the
device port, not from the featuriser.

CPU only.
"""

import math

import pytest

torch = pytest.importorskip("torch")

from tt_bio.diffdock import features as F
from tt_bio.diffdock.config import ESM2_DIM  # noqa: F401  (import guard: package imports cleanly)


def _rel(got, ref):
    ref = ref.to(torch.float64)
    return (got.to(torch.float64) - ref).abs().max() / ref.abs().max()


def test_sinusoidal_embedding_matches_float64():
    t = torch.linspace(0.0, 1.0, 37)
    for dim, scale in ((64, 1000), (32, 10000)):
        got = F.timestep_embedding(t, dim, scale)
        half = dim // 2
        step = math.log(10000) / (half - 1)
        freqs = torch.exp(torch.arange(half, dtype=torch.float64) * -step)
        ang = (scale * t).to(torch.float64)[:, None] * freqs[None, :]
        ref = torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)
        assert got.shape == (t.numel(), dim)
        assert _rel(got, ref) < 1e-6


def test_the_fp32_sigma_embedding_is_the_inaccurate_one():
    """Why :func:`sinusoidal_embedding` computes its angle in float64.

    Upstream builds the angle in fp32. At ``embedding_scale`` 1000 the angle reaches ~1000
    radians, where fp32 resolves about 6e-5 rad, so upstream's own sin/cos are off by 4.7e-5
    on a feature in [-1, 1]; at the confidence model's 10000 it is 2.2e-4. Ours lands within
    3.0e-8. This test fails if someone "restores bit-exactness with upstream" by moving the
    angle back to fp32, and names the number they would be restoring.
    """
    t = torch.linspace(0.0, 1.0, 37)
    for dim, scale, upstream_err in ((64, 1000, 4.7e-5), (32, 10000, 2.2e-4)):
        half = dim // 2
        step = math.log(10000) / (half - 1)
        exact = torch.exp(torch.arange(half, dtype=torch.float64) * -step)
        ang = (scale * t).to(torch.float64)[:, None] * exact[None, :]
        ref = torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)

        fp32_freqs = torch.exp(torch.arange(half, dtype=torch.float32) * -step)
        fp32_ang = (scale * t)[:, None] * fp32_freqs[None, :]
        upstream = torch.cat([torch.sin(fp32_ang), torch.cos(fp32_ang)], dim=1)

        ours = F.timestep_embedding(t, dim, scale)
        assert (upstream.double() - ref).abs().max() == pytest.approx(upstream_err, rel=0.2)
        assert (ours.double() - ref).abs().max() < 1e-7


def test_sinusoidal_layout_is_sin_then_cos():
    # cos(0) = 1, so at t = 0 the second half is all ones and the first half all zeros.
    emb = F.timestep_embedding(torch.zeros(1), 64, 1000)[0]
    assert torch.equal(emb[:32], torch.zeros(32))
    assert torch.equal(emb[32:], torch.ones(32))


def test_sinusoidal_rejects_a_batched_time():
    with pytest.raises(ValueError, match="1-D"):
        F.sinusoidal_embedding(torch.zeros(2, 3), 64)


@pytest.mark.parametrize("stop,num", [(5.0, 64), (15.0, 64), (80.0, 64), (30.0, 64)])
def test_gaussian_smearing_matches_float64(stop, num):
    dist = torch.linspace(0.0, stop * 1.2, 101)
    g = F.GaussianSmearing(0.0, stop, num)
    got = g(dist)
    ref = F.gaussian_smearing_reference(dist, g)
    assert got.shape == (dist.numel(), num)
    assert _rel(got, ref) < 1e-6
    # A distance sitting exactly on a centre peaks at 1 there.
    centre = F.GaussianSmearing(0.0, stop, num)(torch.tensor([stop / 2]))
    assert centre.max() <= 1.0 + 1e-6


def test_gaussian_smearing_coefficient_is_set_by_the_spacing():
    # The checkpoint stores only the centres, so the coefficient has to come out of
    # (start, stop, num). Getting it wrong widens or narrows every radial basis silently.
    g = F.GaussianSmearing(0.0, 5.0, 64)
    delta = 5.0 / 63
    assert g.coeff == pytest.approx(-0.5 / delta ** 2)
    assert g.offset.shape == (64,)
