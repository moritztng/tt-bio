"""`edm_device`'s step arithmetic against the host rollout it replaces, on the CPU.

`edm_device_step_reference` mirrors the device step op for op; the host path is
`OF3SampleDiffusion._augment` plus the EDM step in `_rollout_traced`. Same function, different
float order (reciprocal multiplies, mean as a mask-row product), so in float64 they agree to
rounding and in float32 to a few ulps of the coordinate scale. No device and no ttnn op runs.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("ttnn")

from tt_bio.openfold3_sample_diffusion import (  # noqa: E402
    OF3SampleDiffusion, edm_device_step_reference, edm_step_scalars)


def _rot(g, dtype):
    q, r = torch.linalg.qr(torch.randn(3, 3, generator=g, dtype=torch.float64))
    q = q * torch.sign(torch.diagonal(r))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q.to(dtype)


def _host_step(x, m, rot, trans, noise, t, c_tau, sd, step_scale, denoise):
    """The shipped host formulas (`_augment` + `_rollout_traced`'s EDM step), unpadded."""
    xn = OF3SampleDiffusion._augment(x[None], m, [[rot]], [[trans]], [[noise]], 0)
    mask = m[:, None]
    ru = denoise(xn[0] * mask / math.sqrt(t * t + sd ** 2))[None]
    sd2 = sd ** 2
    c_skip = sd2 / (sd2 + t * t)
    c_out = sd * t / math.sqrt(sd2 + t * t)
    xd = ((xn * mask) * c_skip + ru * c_out) * mask
    delta = (xn - xd) / t
    return (xn + step_scale * (c_tau - t) * delta)[0]


@pytest.mark.parametrize("dtype,tol", [(torch.float64, 1e-10), (torch.float32, 2e-5)])
@pytest.mark.parametrize("n_atom,NP", [(97, 128), (128, 128)])
def test_device_step_matches_host_step(dtype, tol, n_atom, NP):
    g = torch.Generator().manual_seed(0)
    sd, step_scale = 16.0, 1.5
    w = torch.randn(NP, 3, generator=g, dtype=torch.float64).to(dtype) * 0.1
    m = (torch.rand(n_atom, generator=g) > 0.1).to(dtype)
    x = torch.randn(n_atom, 3, generator=g, dtype=torch.float64).to(dtype) * 30.0
    for t, c_tau in [(160.0, 140.0), (12.0, 10.5), (0.05, 0.0)]:
        rot = _rot(g, dtype)
        trans = torch.randn(3, generator=g, dtype=torch.float64).to(dtype)
        noise = torch.randn(n_atom, 3, generator=g, dtype=torch.float64).to(dtype) * t

        def den(rl, n=n_atom):           # any smooth per-atom map stands in for the network
            return torch.tanh(rl) + rl * w[:n]

        host = _host_step(x, m, rot, trans, noise, t, c_tau, sd, step_scale, den)
        pad = lambda y: torch.nn.functional.pad(y, (0, 0, 0, NP - n_atom))
        mp = torch.nn.functional.pad(m, (0, NP - n_atom))
        dev = edm_device_step_reference(
            pad(x), mp, rot, trans, pad(noise), edm_step_scalars(t, c_tau, sd, step_scale),
            lambda rl: den(rl, NP))
        assert torch.all(dev[n_atom:] == 0), "pad rows must stay zero"
        scale = host.abs().max().item()
        err = (dev[:n_atom] - host).abs().max().item()
        assert err <= tol * max(scale, 1.0), (t, err, scale)
        x = host
