"""Reference conformers from the helper process are the ones generated in this process, the helper restarts after it
dies, and a helper that cannot start falls back to in-process forks with the same molecules."""
import numpy as np
import pytest

rdkit_utils = pytest.importorskip("tt_bio._vendor.atomworks.ml.transforms.rdkit_utils")

JOBS = [("ALA", 2), ("TYR", 3), ("HEM", 1)]


def _coords(mols):
    return [np.stack([c.GetPositions() for c in m.GetConformers()]) for m in mols]


def _batch(seed):
    np.random.seed(seed)
    return rdkit_utils.ccd_codes_to_rdkit_with_conformers(JOBS)


def _same(a, b):
    assert len(a) == len(b)
    for x, y in zip(_coords(a), _coords(b)):
        np.testing.assert_array_equal(x, y)


@pytest.fixture(autouse=True)
def _fresh_helper():
    yield
    for p in rdkit_utils._HELPER:
        p.kill()
    rdkit_utils._HELPER.clear()


def test_helper_matches_in_process():
    np.random.seed(7)
    seeds = [rdkit_utils._get_random_seed() for _ in JOBS]
    here, _ = rdkit_utils._conformers_here(JOBS, seeds, (3.0, 0.15), {})
    helper = _batch(7)
    assert rdkit_utils._HELPER and rdkit_utils._HELPER[0].poll() is None
    _same(here, helper)


def test_helper_restarts_after_it_dies():
    first = _batch(1)
    rdkit_utils._HELPER[0].kill()
    rdkit_utils._HELPER[0].wait()
    _same(first, _batch(1))
    assert rdkit_utils._HELPER[0].poll() is None


def test_falls_back_when_helper_cannot_start(monkeypatch):
    ref = _batch(3)
    rdkit_utils._HELPER[0].kill()
    rdkit_utils._HELPER.clear()
    monkeypatch.setattr(rdkit_utils.sys, "executable", "/nonexistent/python")
    _same(ref, _batch(3))
    assert not rdkit_utils._HELPER


def test_pool_matches_fresh_forks_and_reuses_its_workers():
    from tt_bio._vendor.atomworks.ml.utils.timer import SubprocessPool

    np.random.seed(11)
    seeds = [rdkit_utils._get_random_seed() for _ in JOBS]
    forked, _ = rdkit_utils._conformers_here(JOBS, seeds, (3.0, 0.15), {})
    pool = SubprocessPool()
    first, _ = rdkit_utils._conformers_here(JOBS, seeds, (3.0, 0.15), {}, pool)
    workers = {p.pid for p, _ in pool._idle}
    again, _ = rdkit_utils._conformers_here(JOBS, seeds, (3.0, 0.15), {}, pool)
    _same(forked, first)
    _same(forked, again)
    assert {p.pid for p, _ in pool._idle} == workers


def _sleep(s):
    import time

    time.sleep(s)
    return s


def _die():
    import os

    os._exit(3)


def _raise():
    raise ValueError("boom")


def test_pool_times_out_raises_and_survives_a_dead_worker():
    from tt_bio._vendor.atomworks.ml.utils.timer import ChildProcessError, SubprocessPool

    pool = SubprocessPool()
    out = pool.run([(0.3, _sleep, (5,), {}), (2.0, _die, (), {}), (2.0, _raise, (), {}), (2.0, _sleep, (0.01,), {})])
    assert isinstance(out[0][1], TimeoutError) and not out[0][0]
    assert isinstance(out[1][1], ChildProcessError) and not out[1][0]
    assert isinstance(out[2][1], ValueError) and not out[2][0]
    assert out[3] == (True, 0.01)
    assert pool.run([(2.0, _sleep, (0.01,), {})] * 3) == [(True, 0.01)] * 3
