"""tt_bio.rfd3.staticknn answers every row of a design trajectory exactly as cKDTree(x).query(x, k) does, order
included, while a static target is answered from its once-per-design cache. Host only."""
import numpy as np
import pytest

S = pytest.importorskip("tt_bio.rfd3.staticknn")
from scipy.spatial import cKDTree  # noqa: E402


def _trajectory(n_static=900, n_move=300, steps=8, quantise=False, seed=0):
    rng = np.random.default_rng(seed)
    target = np.cumsum(rng.normal(0, 1.5, (n_static, 3)), 0)
    if quantise:
        target = np.round(target)  # PDB-like grid: many equal distances
    binder = target[rng.integers(0, n_static, n_move)] + rng.normal(0, 6, (n_move, 3))
    for sigma in np.geomspace(200, 0.05, steps):  # far cloud -> docked on the target
        yield np.concatenate([target, binder + rng.normal(0, sigma, (n_move, 3))]).astype(np.float32).astype(np.float64)


@pytest.mark.parametrize("quantise", [False, True])
def test_trajectory_rows_equal_kdtree(quantise):
    sk, k = S.StaticKnn(), 64
    for x in _trajectory(quantise=quantise):
        _, ref = cKDTree(x).query(x, k=k, workers=2)
        assert np.array_equal(sk.query(x, k, 2), ref)
    assert sk.ids is not None and len(sk.ids) == 900
    if not quantise:
        assert sk.stats[1] < sk.stats[0]  # the cache answered rows (on the integer grid every row has a tie)


def test_cache_rebuilds_when_the_target_changes():
    sk, k = S.StaticKnn(), 32
    xs = list(_trajectory(steps=4, seed=1)) + list(_trajectory(steps=4, seed=2))
    for x in xs:
        _, ref = cKDTree(x).query(x, k=k, workers=1)
        assert np.array_equal(sk.query(x, k, 1), ref)


def test_nothing_static_is_a_plain_query():
    rng = np.random.default_rng(3)
    sk = S.StaticKnn()
    for _ in range(3):
        x = rng.normal(0, 10, (500, 3))
        _, ref = cKDTree(x).query(x, k=16, workers=1)
        assert np.array_equal(sk.query(x, 16, 1), ref)
    assert sk.ids is None


def test_skip_movers_answers_every_static_row():
    """With skip_movers the moving atoms' rows come back -1 (the card answers them) and every other row is the tree's."""
    sk, k = S.StaticKnn(), 64
    for i, x in enumerate(_trajectory(steps=6, seed=4)):
        _, ref = cKDTree(x).query(x, k=k, workers=1)
        mid = sk.movers(x)
        if mid is None:
            assert np.array_equal(sk.query(x, k, 1), ref)
            continue
        assert np.array_equal(mid, np.arange(900, 1200))
        got = sk.query(x, k, 1, skip_movers=True)
        assert (got[mid] == -1).all()
        assert np.array_equal(np.delete(got, mid, 0), np.delete(ref, mid, 0))
    assert i == 5 and sk.ids is not None
