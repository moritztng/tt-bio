"""The refactor-verification harness compares what it claims to (perf/rfd4/verify_of3_refactor.py).

The script needs a card, so what is testable here is the comparison itself: it has to notice a
confidence number that moved, and it must not call two different folds identical because the keys
it forgot to compare happened to match.
"""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "verify_of3", ROOT / "perf" / "rfd4" / "verify_of3_refactor.py")
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)

ROW = {"plddt": 0.912345, "ptm": 0.81, "iptm": 0.44, "confidence_score": 0.77,
       "n_tokens": 256, "n_atoms": 2048, "msa_depth": 1, "n_residues": 256, "n_chains": 2}


def test_an_identical_row_is_no_drift():
    assert v.compare_rows(ROW, dict(ROW)) == []


def test_a_moved_confidence_number_is_caught():
    after = {**ROW, "plddt": 0.912400}
    assert v.compare_rows(after, ROW, tol=1e-6)
    assert "plddt" in v.compare_rows(after, ROW, tol=1e-6)[0]


def test_a_lost_feature_shows_up_as_a_shape_change():
    assert v.compare_rows(ROW, {**ROW, "n_tokens": 255})


def test_the_compared_keys_cover_confidence_and_shape():
    """Comparing runtime alone would call a broken featurization identical."""
    for key in ("plddt", "ptm", "iptm", "n_tokens", "n_atoms", "msa_depth"):
        assert key in v.COMPARED
    assert "runtime_s" not in v.COMPARED      # a timing is not a correctness signal


def test_a_missing_key_on_one_side_is_drift_not_a_pass():
    assert v.compare_rows(ROW, {k: val for k, val in ROW.items() if k != "iptm"})
