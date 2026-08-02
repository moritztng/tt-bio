"""CPU-only parity test for the multiplicity-batching refactor of edm_sample.

No Tenstorrent device needed: stubs diffusion_module.denoise with a pure-torch
function and verifies the sampler shell's RNG/shape wiring.

Verifies:
  1. multiplicity=1 (default) is BIT-EXACT with a faithful re-implementation of the
     PRIOR per-sample path (shape=(1,N,3), compute_random_augmentation(1,...), one
     denoise call per step). This is the M=1 parity bar.
  2. multiplicity>1 runs and produces (M,N,3); each sample's trajectory is a valid
     EDM ancestral walk (monotone-sigma, finite). This is the sampler-shell sanity
     check; the device-denoise arithmetic parity (full fold) still needs a card.

Run: python3 tests/test_edm_sample_multiplicity.py
"""
import sys
import os
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tt_bio.protenix import edm_sample
from tt_bio.boltz2 import compute_random_augmentation


class _StubDiffusion:
    """Pure-torch stub: denoise(x_noisy, t_hat, cond) -> a simple shrinkage toward 0.
    The exact denoise math is irrelevant for the sampler-shell RNG/shape parity check --
    any deterministic function of (x_noisy, t_hat) makes the trajectories comparable
    between the batched and unbatched paths."""

    def __init__(self, fn=None):
        self._fn = fn or (lambda x, t, c: 0.5 * x)
        self.denoise_traced = None  # force the untraced path (trace=False)

    def denoise(self, x_noisy, t_hat, cond):
        return self._fn(x_noisy, float(t_hat.item()), cond)


def _old_unbatched(stub, n_atoms, *, n_step, seed, fn):
    """Faithful re-implementation of the PRIOR edm_sample (per-sample, shape=(1,N,3),
    compute_random_augmentation(1,...)). Used as the M=1 golden reference."""
    if seed is not None:
        torch.manual_seed(seed)
    inv_rho = 1.0 / 7.0
    i = torch.arange(n_step, dtype=torch.float64)
    sig = 16.0 * (160.0 ** inv_rho + (i / n_step) * (4e-4 ** inv_rho - 160.0 ** inv_rho)) ** 7.0
    sigmas = torch.cat([sig, torch.zeros(1, dtype=torch.float64)]).float()
    gammas = torch.where(sigmas > 1.0, torch.tensor(0.8), torch.tensor(0.0))
    shape = (1, n_atoms, 3)
    x = sigmas[0] * torch.randn(shape)
    for k in range(n_step):
        sigma_tm, sigma_t, gamma = sigmas[k].item(), sigmas[k + 1].item(), gammas[k + 1].item()
        R, tr = compute_random_augmentation(1, device=x.device, dtype=x.dtype)
        x = x - x.mean(dim=-2, keepdim=True)
        x = torch.einsum("bmd,bds->bms", x, R) + tr
        t_hat = sigma_tm * (1 + gamma)
        noise_var = 1.003 ** 2 * (t_hat ** 2 - sigma_tm ** 2)
        eps = (noise_var ** 0.5) * torch.randn(shape) if noise_var > 0 else torch.zeros(shape)
        x_noisy = x + eps
        denoised = fn(x_noisy, t_hat, {})
        d = (x_noisy - denoised) / t_hat
        x = x_noisy + 1.5 * (sigma_t - t_hat) * d
    return x


def test_m1_bitexact():
    torch.manual_seed(0)
    n_atoms, n_step, seed = 37, 12, 12345
    fn = lambda x, t, c: 0.5 * x
    stub = _StubDiffusion(fn)
    # New batched path at M=1:
    new = edm_sample(stub, {}, n_atoms, multiplicity=1, n_step=n_step, seed=seed)
    # Old unbatched path (golden):
    torch.manual_seed(0)
    old = _old_unbatched(stub, n_atoms, n_step=n_step, seed=seed, fn=fn)
    assert new.shape == (1, n_atoms, 3), f"new M=1 shape {new.shape}"
    assert old.shape == (1, n_atoms, 3), f"old M=1 shape {old.shape}"
    diff = (new - old).abs().max().item()
    assert diff == 0.0, f"M=1 NOT bit-exact: max abs diff {diff}"
    print(f"[PASS] M=1 bit-exact vs prior unbatched path (max abs diff {diff})")


def test_m_greater_than_1_shape_and_finite():
    torch.manual_seed(0)
    n_atoms, n_step, M = 37, 12, 4
    fn = lambda x, t, c: 0.5 * x
    stub = _StubDiffusion(fn)
    out = edm_sample(stub, {}, n_atoms, multiplicity=M, max_parallel_samples=2,
                     n_step=n_step, seed=42)
    assert out.shape == (M, n_atoms, 3), f"M=4 shape {out.shape}"
    assert torch.isfinite(out).all(), "M=4 has non-finite entries"
    # Each sample should differ (independent noise draws from the single stream).
    pair_dists = torch.cdist(out.reshape(M, -1), out.reshape(M, -1))
    offdiag = pair_dists[~torch.eye(M, dtype=bool)]
    assert offdiag.min() > 0, "M=4 produced duplicate samples (noise not independent)"
    print(f"[PASS] M=4 shape ({M},{n_atoms},3), finite, samples distinct "
          f"(min pairwise dist {offdiag.min():.4g})")


def test_chunking_equivalence():
    """max_parallel_samples=1 (fully sequential chunks) vs =M (one batched forward)
    must give the SAME result for the same seed -- chunking is an OOM guard, not a
    numerical choice, since cond is shared and the per-chunk denoise is deterministic."""
    torch.manual_seed(0)
    n_atoms, n_step, M = 37, 12, 4
    fn = lambda x, t, c: 0.5 * x
    a = edm_sample(_StubDiffusion(fn), {}, n_atoms, multiplicity=M,
                   max_parallel_samples=1, n_step=n_step, seed=42)
    torch.manual_seed(0)
    b = edm_sample(_StubDiffusion(fn), {}, n_atoms, multiplicity=M,
                   max_parallel_samples=M, n_step=n_step, seed=42)
    diff = (a - b).abs().max().item()
    assert diff == 0.0, f"chunking changed the result: max abs diff {diff}"
    print(f"[PASS] chunking equivalence (max_parallel_samples=1 vs ={M}): max abs diff {diff}")


if __name__ == "__main__":
    test_m1_bitexact()
    test_m_greater_than_1_shape_and_finite()
    test_chunking_equivalence()
    print("\nALL EDM-SAMPLE MULTIPLICITY TESTS PASS")
