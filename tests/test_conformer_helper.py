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
