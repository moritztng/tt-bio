"""`<results>/ERR/<job>.txt` for every system that could not be predicted, as Mol-1 and Protenix write it.

Host-only: the predict command runs for real with the dispatch layer stubbed (tests/test_predict_exit_code.py's
fixture), so no device, network or weights are involved.
"""
from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from tests.test_predict_exit_code import _refused, _target, host_predict  # noqa: F401 (fixture)
from tt_bio.main import _save_results, _write_error_reports, cli


def _predict(data, out, model="mol1"):
    return CliRunner().invoke(cli, ["predict", str(data), "--model", model, "--single_sequence",
                                    "--out_dir", str(out)])


def _results_dir(out):
    return next(out.glob("*/results.json")).parent


@pytest.mark.parametrize("model", ["mol1", "boltz2"])
def test_a_failed_fold_writes_its_report_and_exits_nonzero(host_predict, monkeypatch, tmp_path, model):
    import tt_bio.main as m

    def dispatch(payload, workers, *, total, results_path, **_):
        host_predict["total"] = total
        _save_results([{"id": "a", "status": "failed", "error": "ran out of device memory"},
                       {"id": "b", "status": "ok"}], results_path)
        return 1

    monkeypatch.setattr(m, "_dispatch_run", dispatch)
    data = tmp_path / "data"
    data.mkdir()
    _target(data / "a.yaml")
    _target(data / "b.yaml")
    extra = ["--accelerator", "cpu"] if model == "boltz2" else []
    result = CliRunner().invoke(cli, ["predict", str(data), "--model", model, "--single_sequence",
                                      "--out_dir", str(tmp_path / "out"), *extra])
    assert result.exit_code == 2, result.output
    err = _results_dir(tmp_path / "out") / "ERR"
    assert sorted(p.name for p in err.iterdir()) == ["a.txt"]
    assert (err / "a.txt").read_text() == "ran out of device memory\n"


def test_a_refused_input_gets_a_report_beside_the_folds(host_predict, tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    _target(data / "a.yaml")
    _refused(data / "b.yaml")
    result = _predict(data, tmp_path / "out")
    assert result.exit_code == 2, result.output
    err = _results_dir(tmp_path / "out") / "ERR"
    assert [p.name for p in err.iterdir()] == ["b.txt"]
    assert "pocket" in (err / "b.txt").read_text()


def test_an_all_refused_batch_still_reports_every_input(host_predict, tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    _refused(data / "a.yaml")
    _refused(data / "b.yaml")
    result = _predict(data, tmp_path / "out")
    assert result.exit_code == 1 and host_predict["total"] is None
    err = _results_dir(tmp_path / "out") / "ERR"
    assert sorted(p.name for p in err.iterdir()) == ["a.txt", "b.txt"]


def test_a_clean_run_leaves_no_err_dir(host_predict, tmp_path):
    result = _predict(_target(tmp_path / "t.yaml"), tmp_path / "out")
    assert result.exit_code == 0, result.output
    assert not list((tmp_path / "out").glob("*/ERR"))


def test_a_job_that_folds_on_rerun_loses_its_report(tmp_path):
    results = tmp_path / "results.json"
    results.write_text(json.dumps([{"id": "a", "status": "failed", "error": "x"}]))
    _write_error_reports(results)
    assert (tmp_path / "ERR" / "a.txt").exists()
    results.write_text(json.dumps([{"id": "a", "status": "ok"}]))
    _write_error_reports(results)
    assert not (tmp_path / "ERR").exists()
