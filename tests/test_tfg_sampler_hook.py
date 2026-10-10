"""edm_sample's guidance hook: absent, every step is the Euler update it always was."""
import torch

from tt_bio.protenix import edm_sample


class _Denoiser:
    """Deterministic stand-in for the device denoiser: a contraction towards a fixed shape."""

    def __init__(self, n):
        self.target = torch.randn(1, n, 3, generator=torch.Generator().manual_seed(1)) * 10

    def denoise(self, x_noisy, t_hat, cond):
        w = 1.0 / (1.0 + float(t_hat[0]) / 16.0)
        return w * self.target.expand_as(x_noisy) + (1 - w) * x_noisy


class _Euler:
    """A guidance that applies the plain Euler update, written out independently."""

    def __init__(self):
        self.steps = []

    def step(self, x_noisy, x0, *, t_hat, sigma_t, eta, step, n_step):
        self.steps.append(step)
        return x_noisy + eta * (sigma_t - t_hat) * ((x_noisy - x0) / t_hat)


def _run(guidance, multiplicity):
    return edm_sample(_Denoiser(40), {}, 40, multiplicity=multiplicity, n_step=12, seed=3,
                      guidance=guidance)


def test_euler_guidance_is_bit_identical_to_no_guidance():
    for m in (1, 3):
        ref = _run(None, m)
        g = _Euler()
        assert torch.equal(_run(g, m), ref)
        assert g.steps == list(range(12))


def test_guidance_output_is_what_the_sampler_continues_from():
    class Shift(_Euler):
        def step(self, *a, **k):
            return super().step(*a, **k) + 1.0

    assert not torch.equal(_run(Shift(), 1), _run(None, 1))
