"""tt_bio.confidence_export: contact probabilities and the one confidence npz every model writes.

The contact rule is graded against upstream OpenDDE's compute_contact_prob, recorded in float64
into tests/fixtures/contact_probs_opendde_ref.npz by perf/fdx_confidence/make_contact_fixture.py.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from tt_bio import confidence_export as ce

REF = np.load(Path(__file__).parent / "fixtures/contact_probs_opendde_ref.npz")


@pytest.mark.parametrize("model", ["opendde", "boltz2"])
@pytest.mark.parametrize("thr", [8.0, 12.0])
def test_contact_probs_match_upstream_opendde(model, thr):
    p, _ = ce.contact_probs(REF[f"{model}_logits"], ce.bin_upper_edges(*ce.DISTOGRAM_GRID[model]),
                            thr)
    ref = REF[f"{model}_ref_{thr:g}"]
    off = ~np.eye(len(p), dtype=bool)
    # float32 storage is the only rounding: upstream's own float64 values, to 1e-7.
    assert np.abs(p[off] - ref[off]).max() < 1e-7


def test_cutoff_is_the_bin_edge_actually_used():
    want = {"opendde": 8.0, "boltz2": 7.8065, "openfold3": 7.9375, "af2ig": 7.9375,
            "rf3": 7.7143, "esmfold2": 7.8065}
    for model, edge in want.items():
        _, got = ce.contact_probs(np.zeros((2, 2, ce.DISTOGRAM_GRID[model][2])),
                                  ce.bin_upper_edges(*ce.DISTOGRAM_GRID[model]))
        assert got == pytest.approx(edge, abs=1e-4), model


def test_symmetric_unit_diagonal_and_a_probability():
    rng = np.random.default_rng(1)
    z = rng.normal(size=(9, 9, 64)) * 4                 # deliberately NOT symmetric
    p, _ = ce.contact_probs(z, ce.bin_upper_edges(2.0, 22.0, 64))
    assert p.dtype == np.float32 and np.array_equal(p, p.T)
    assert np.all(np.diag(p) == 1.0) and p.min() >= 0.0 and p.max() <= 1.0


def test_rejects_a_grid_that_does_not_fit_and_a_cutoff_below_it():
    with pytest.raises(ValueError, match="bins"):
        ce.contact_probs(np.zeros((2, 2, 63)), ce.bin_upper_edges(2.0, 22.0, 64))
    with pytest.raises(ValueError, match="below"):
        ce.contact_probs(np.zeros((2, 2, 64)), ce.bin_upper_edges(2.0, 22.0, 64), cutoff=1.5)


def test_write_crops_names_and_keeps_the_pae_key(tmp_path):
    rng = np.random.default_rng(2)
    pae, pde = rng.random((6, 6)) * 30, rng.random((6, 6)) * 30
    real = np.array([1, 1, 0, 1, 1, 0], bool)
    side = ce.write(tmp_path, "t", "boltz2", real=real, pae=pae, pde=pde,
                    distogram=rng.normal(size=(6, 6, 64)), absent={"x": "why"})
    z = np.load(tmp_path / "t_pae.npz")
    assert sorted(z.files) == ["contact_cutoff_A", "contact_probs", "pae", "pde"]
    assert np.array_equal(z["pae"], pae.astype(np.float32)[np.ix_(real, real)])
    assert z["contact_probs"].shape == (4, 4) and float(z["contact_cutoff_A"]) == pytest.approx(7.8065)
    assert json.loads((tmp_path / "t_pae.json").read_text()) == side
    assert side["n_tokens"] == 4 and side["absent"] == {"x": "why"}
    assert side["arrays"]["pae"] == {"shape": [4, 4], "dtype": "float32", "units": "angstrom"}


def test_write_refuses_matrices_on_different_axes(tmp_path):
    with pytest.raises(ValueError, match="token axis"):
        ce.write(tmp_path, "t", "boltz2", pae=np.zeros((4, 4)), pde=np.zeros((5, 5)))
