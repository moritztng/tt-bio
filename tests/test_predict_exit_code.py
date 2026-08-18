"""Host-only contract tests for the `tt-bio predict` process exit status.

Batch callers (the release gate, CI, fleet scripts) gate on the exit status,
so a run that lost targets must not exit 0 — but it did: every target could
fail and the process still reported success. These tests drive the real click
command with the dispatch layer stubbed, so no device, network or checkpoint
download is involved.

Contract: 0 = every target folded, 1 = every target failed, 2 = partial.
"""
from __future__ import annotations

import pytest
from click.testing import CliRunner

from tt_bio.main import cli
from tt_bio.runtime import build_local_workers


def _target(path):
    path.write_text("""version: 1
sequences:
  - protein:
      id: A
      sequence: MKVL
""")
    return path


@pytest.fixture
def host_predict(monkeypatch):
    """Stub everything predict touches that needs a device, network or weights.

    Returns the shared state: set ["failed"] to the failed-job count the
    stubbed dispatch should report; ["total"] records how many jobs the run
    was given (None until dispatch ran, so a crash before dispatch fails the
    test instead of masquerading as the asserted exit code).
    """
    import tt_bio.main as m

    state = {"failed": 0, "total": None}

    monkeypatch.setattr(m, "download_all", lambda *a, **k: None)
    monkeypatch.setattr(
        m, "_local_workers",
        lambda *a, **k: build_local_workers("cpu", [object()], [0]))

    def fake_dispatch_run(run_payload, workers, *, total, results_path,
                          struct_dir, model, listen, debug, log):
        state["total"] = total
        return state["failed"]

    def fake_dispatch_controller(controller_url, run_payload, *, total,
                                 results_path, struct_dir, model, debug, log,
                                 run_id=None):
        state["total"] = total
        return state["failed"]

    monkeypatch.setattr(m, "_dispatch_run", fake_dispatch_run)
    monkeypatch.setattr(m, "_dispatch_to_controller", fake_dispatch_controller)
    return state


def test_every_target_failed_exits_1(host_predict, tmp_path):
    host_predict["failed"] = 1
    result = CliRunner().invoke(
        cli, ["predict", str(_target(tmp_path / "t.yaml")), "--model", "boltz2",
              "--accelerator", "cpu", "--single_sequence",
              "--out_dir", str(tmp_path / "out")])
    assert host_predict["total"] == 1
    assert result.exit_code == 1


def test_every_target_folded_exits_0(host_predict, tmp_path):
    result = CliRunner().invoke(
        cli, ["predict", str(_target(tmp_path / "t.yaml")), "--model", "boltz2",
              "--accelerator", "cpu", "--single_sequence",
              "--out_dir", str(tmp_path / "out")])
    assert host_predict["total"] == 1
    assert result.exit_code == 0


def test_partial_failure_exits_2(host_predict, tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    _target(data / "a.yaml")
    _target(data / "b.yaml")
    host_predict["failed"] = 1
    result = CliRunner().invoke(
        cli, ["predict", str(data), "--model", "boltz2",
              "--accelerator", "cpu", "--single_sequence",
              "--out_dir", str(tmp_path / "out")])
    assert host_predict["total"] == 2
    assert result.exit_code == 2


def test_ttnn_model_branch_all_failed_exits_1(host_predict, tmp_path):
    """The esmfold2/protenix-v2/openfold3/opendde branch dispatches separately."""
    host_predict["failed"] = 1
    result = CliRunner().invoke(
        cli, ["predict", str(_target(tmp_path / "t.yaml")), "--model", "esmfold2",
              "--fast", "--out_dir", str(tmp_path / "out")])
    assert host_predict["total"] == 1
    assert result.exit_code == 1


def test_controller_mode_all_failed_exits_1(host_predict, tmp_path):
    """--controller submits to a remote cluster; the exit contract is the same."""
    host_predict["failed"] = 1
    result = CliRunner().invoke(
        cli, ["predict", str(_target(tmp_path / "t.yaml")), "--model", "boltz2",
              "--single_sequence", "--controller", "http://127.0.0.1:1",
              "--out_dir", str(tmp_path / "out")])
    assert host_predict["total"] == 1
    assert result.exit_code == 1


def test_ttnn_model_branch_controller_all_failed_exits_1(host_predict, tmp_path):
    host_predict["failed"] = 1
    result = CliRunner().invoke(
        cli, ["predict", str(_target(tmp_path / "t.yaml")), "--model", "esmfold2",
              "--fast", "--controller", "http://127.0.0.1:1",
              "--out_dir", str(tmp_path / "out")])
    assert host_predict["total"] == 1
    assert result.exit_code == 1


def test_exit_code_mapping():
    from tt_bio.main import _exit_for_failed_jobs

    assert _exit_for_failed_jobs(0, 3) is None
    with pytest.raises(SystemExit) as total_fail:
        _exit_for_failed_jobs(3, 3)
    assert total_fail.value.code == 1
    with pytest.raises(SystemExit) as partial:
        _exit_for_failed_jobs(1, 3)
    assert partial.value.code == 2
