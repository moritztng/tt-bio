"""The speculative atom index (tt_bio.rfd3.specidx, RFD3_SPEC_INDEX) returns the k-d tree's rows exactly, and the
sampler's early noise draw leaves the noise stream unchanged. Host only."""
import numpy as np
import pytest
import torch

S = pytest.importorskip("tt_bio.rfd3.specidx")
from scipy.spatial import cKDTree  # noqa: E402

from tt_bio.rfd3.sampler import RFD3Sampler  # noqa: E402


def _chain(L, sigma, seed):
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.normal(0, 1.5, (L, 3)), 0) + rng.normal(0, sigma, (L, 3))
    return x.astype(np.float32).astype(np.float64), rng


@pytest.mark.parametrize("sigma,err", [(0.1, 0.02), (5.0, 0.2), (60.0, 1.0), (60.0, 30.0)])
def test_rescore_equals_kdtree(sigma, err):
    L, k = 1500, 64
    x, rng = _chain(L, sigma, 7)
    xs = x + rng.normal(0, err, (L, 3))
    nn, n_bad = S.rescore(x, xs, *S.candidates(xs, 96, 2), k, 2)
    _, ref = cKDTree(x).query(x, k=k, workers=2)
    assert np.array_equal(nn, ref)
    if err >= 30:
        assert n_bad > 0  # a bad prediction is caught, not trusted


def test_rescore_ties_go_to_the_tree():
    x, _ = _chain(400, 0.0, 1)
    x = np.round(x)  # many equal distances
    nn, n_bad = S.rescore(x, x, *S.candidates(x, 48, 1), 32, 1)
    _, ref = cKDTree(x).query(x, k=32, workers=1)
    assert n_bad > 0 and np.array_equal(np.sort(nn, 1), np.sort(ref, 1))


def test_model_rows_equal_with_and_without_spec():
    M = pytest.importorskip("tt_bio.rfd3.model")
    L, K, NT = 900, 64, 90
    x, rng = _chain(L, 8.0, 3)
    tok = np.repeat(np.arange(NT), L // NT)
    mask = torch.from_numpy(np.abs(tok[:, None] - tok[None, :]) <= 1)
    seq = torch.where(mask, torch.arange(L).expand(L, L), L).topk(K, dim=1, largest=False, sorted=True).values
    xt = torch.from_numpy(x).float().unsqueeze(0)
    xs = x + rng.normal(0, 0.3, (L, 3))
    a = M._neighbours_kdtree(mask, seq, xt, K, 2)
    b = M._neighbours_kdtree(mask, seq, xt, K, 2, spec=(xs, *S.candidates(xs, 96, 2)))
    assert torch.equal(a, b)


class _Recorder:
    def __init__(self, takes):
        self.TAKES_SPEC_NEXT, self.calls = takes, []

    def __call__(self, X_noisy_L, t, f, n_recycle=None, spec_next=None, **_):
        den = 0.9 * X_noisy_L
        self.calls.append((X_noisy_L.clone(), den, spec_next))
        return {"X_L": den, "sequence_restype_I": None}


def test_sampler_noise_stream_and_prediction():
    L = 50
    coord = torch.randn(1, L, 3)
    fixed = torch.zeros(L, dtype=torch.bool)
    runs = []
    for takes in (False, True):
        m = _Recorder(takes)
        g = torch.Generator().manual_seed(5)
        RFD3Sampler(num_timesteps=12).sample(m, 1, L, coord, {}, {}, fixed, generator=g)
        runs.append(m.calls)
    assert all(torch.equal(a[0], b[0]) for a, b in zip(*runs))
    calls = runs[1]
    assert calls[-1][2] is None and all(c[2] is None for c in runs[0])
    for (xn, den, (coef, eps)), (xn1, _, _) in zip(calls, calls[1:]):
        # recycle 0 == the step's output here, so the prediction is the next X_noisy itself
        torch.testing.assert_close(xn + coef * (xn - den) + eps, xn1)
