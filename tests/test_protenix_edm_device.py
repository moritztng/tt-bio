"""edm_sample's edm_device branch draws what the host loop draws and takes the same EDM step.

A fake diffusion module stands in for the card: its `rollout_device` replays the shared
`edm_device_step_reference` (the device step's op order) over `_draw`'s outputs. Both paths must
land on the same trajectory to fp32 rounding, which pins the RNG order and the scalars
(`edm_step_scalars(t_hat, sigma_t, sigma_data, eta)`) the device step reads.
"""
import torch

from tt_bio import protenix as P
from tt_bio import tenstorrent as T
from tt_bio.edm_device import edm_step_scalars
from tt_bio.openfold3_sample_diffusion import edm_device_step_reference

SD = 16.0


def _f(r):
    return 0.5 * torch.tanh(r) + 0.1 * r.flip(-2)


class _Fake:
    SIGMA_DATA = SD

    def denoise(self, x_noisy, t_hat, cond):
        t = float(t_hat)
        c_skip = SD ** 2 / (SD ** 2 + t * t)
        c_out = SD * t / (SD ** 2 + t * t) ** 0.5
        return c_skip * x_noisy + c_out * _f(x_noisy / (SD ** 2 + t * t) ** 0.5)

    denoise_traced = denoise

    def rollout_ok(self, cond, M):
        return True

    def rollout_device(self, x, cond, n_step, draw, progress_fn=None):
        M, N = x.shape[:2]
        x = x.double()
        for k in range(n_step):
            R, tr, eps, t_hat, sigma_t, eta = draw(k)
            sc = edm_step_scalars(t_hat, sigma_t, SD, float(eta))
            x = torch.stack([edm_device_step_reference(
                x[m], torch.ones(N, dtype=torch.float64), R[m].double().t(), tr[m, 0].double(),
                eps[m].double(), sc, _f) for m in range(M)])
        return x.float()


def _sample(lever_on, seed=7):
    with T.levers({"edm_device"} if lever_on else set()):
        return P.edm_sample(_Fake(), {}, 40, multiplicity=3, n_step=12, seed=seed, trace=True)


def test_device_branch_matches_host_loop():
    host, dev = _sample(False), _sample(True)
    assert host.shape == dev.shape == (3, 40, 3)
    assert torch.allclose(host, dev, rtol=1e-4, atol=1e-3), (host - dev).abs().max()


def test_host_loop_is_deterministic():
    assert torch.equal(_sample(False), _sample(False))
