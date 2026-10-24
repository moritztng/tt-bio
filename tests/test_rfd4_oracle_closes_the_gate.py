"""End to end on the host: an OpenFold3-named refold cache reaches the quality grade.

The three metrics holding `state/rfd4/QUALITY.json` at ORACLE_MISSING -- `refolded_backbone_rmsd
.rmsd`, `oracle.overall_plddt` and `unconditional.pass` -- only exist once a refold does. Every
link of that chain is covered on its own: `tests/test_rfd4_eval_metrics.py` pins the evaluate half
against upstream's own CSV, `tests/test_rfd4_refold_cache.py` the files the Tenstorrent oracle
writes. What nothing covered is the chain, and the chain is where the of3-vs-af3 naming could
quietly drop the oracle columns on the floor and leave the harness reporting ORACLE_MISSING with
a refold sitting right there.

The refold here is the committed fixture (`perf/rfd4/eval_goldens/refold/`, the design rotated and
jittered by `make_refold_fixture.py`), renamed to the `of3` names the Tenstorrent oracle writes.
It is a fixture, not a measurement: this test says the plumbing carries numbers, never what the
numbers are.
"""
import json
import pathlib
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "perf" / "rfd4"))

import quality  # noqa: E402

GOLDENS = pathlib.Path(__file__).resolve().parents[1] / "perf" / "rfd4" / "eval_goldens"
FIXTURE_ID = "clean_generated_0ff2e25d"
OWED = ("refolded_backbone_rmsd.rmsd", "oracle.overall_plddt", "unconditional.pass")


def _arm(root, oracle_name):
    """One arm: a design, and the refold cache beside it under `oracle_name`'s file names."""
    from tt_bio.rfd4.evaluate import evaluate, example_id

    inputs = root / "designs"
    inputs.mkdir(parents=True)
    shutil.copy(GOLDENS / "clean_generated.cif", inputs / "clean_generated.cif")
    eid = example_id(inputs / "clean_generated.cif", inputs)

    cache = root / "out" / "evaluation_samples" / eid
    cache.mkdir(parents=True)
    for kind in ("confidences.json", "refolded.cif"):
        shutil.copy(GOLDENS / "refold" / FIXTURE_ID / f"{FIXTURE_ID}_af3_{kind}",
                    cache / f"{eid}_{oracle_name}_{kind}")

    csv_path = evaluate(inputs, "design,unconditional_refolding", root / "out")
    return quality.load_arm([inputs, pathlib.Path(csv_path)])


def test_an_openfold3_refold_reaches_the_grade(tmp_path):
    tt = _arm(tmp_path / "tt", "of3")
    ref = _arm(tmp_path / "ref", "of3")

    graded = quality.grade_feature("unconditional", tt, ref, floor={}, n_min=1,
                                   permutations=200, allow_no_floor=True, allow_unbound=True)

    for column in OWED:
        entry = graded["metrics"][column]
        assert not entry.get("missing"), f"{column} is still missing with a refold cached"
        assert entry["n_tt"] > 0, f"{column} graded {entry['n_tt']} TT designs"
    assert graded["missing_oracle_metrics"] == []
    assert graded["verdict"] != "ORACLE_MISSING", graded["notes"]


def test_without_a_refold_the_same_design_reads_oracle_missing(tmp_path):
    """The control: the harness is reporting the refold, not the design file beside it.

    Scored with the `design` tag alone, which is the whole oracle-free metric set, so the CSV is
    the one a box with no oracle produces. (Asking for `unconditional_refolding` with nothing
    cached is refused outright by the evaluate port, which folds nothing itself.)
    """
    from tt_bio.rfd4.evaluate import evaluate

    inputs = tmp_path / "designs"
    inputs.mkdir()
    shutil.copy(GOLDENS / "clean_generated.cif", inputs / "clean_generated.cif")
    csv_path = evaluate(inputs, "design", tmp_path / "out")
    arm = quality.load_arm([inputs, pathlib.Path(csv_path)])

    graded = quality.grade_feature("unconditional", arm, arm, floor={}, n_min=1,
                                   permutations=200, allow_no_floor=True, allow_unbound=True)
    assert graded["verdict"] == "ORACLE_MISSING"
    assert sorted(graded["missing_oracle_metrics"]) == sorted(
        OWED + ("unconditional.all_metrics_present",))


def test_an_af3_named_cache_still_reaches_the_grade(tmp_path):
    """AF3 is upstream's default oracle; teaching the reader `of3` must not cost it `af3`."""
    arm = _arm(tmp_path / "af3", "af3")
    graded = quality.grade_feature("unconditional", arm, arm, floor={}, n_min=1,
                                   permutations=200, allow_no_floor=True, allow_unbound=True)
    assert graded["missing_oracle_metrics"] == []


def test_the_fixture_carries_the_columns_the_gate_names(tmp_path):
    """A guard on the fixture itself: if upstream's refold cache stopped carrying overall_plddt,
    the two tests above would pass on an empty column rather than fail."""
    values = json.loads((GOLDENS / "refold" / FIXTURE_ID / f"{FIXTURE_ID}_af3_confidences.json").read_text())
    assert "overall_plddt" in values
