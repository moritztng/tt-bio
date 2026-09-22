"""A failed fold's log has to outlive the scratch dir that held it.

`run_card.sh` walks one card through several models and the recorder deletes its workdir
whole between them, so the log a failure names is gone by the time anyone reads the campaign
log. rf3 failed at its 256 warm-up on two different cards on 2026-09-20 with
`SpawnProcess-1 exit 1` -- whose message ends "any other code prints its own fatal above" --
and the fatal was unrecoverable both times.
"""
import importlib.util
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture()
def rg(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "release_gate_keeplog", REPO_ROOT / "scripts" / "release_gate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    return mod


def test_the_log_is_copied_out_and_the_message_says_where(rg, tmp_path):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    log = scratch / "rf3-256-warmup.log"
    log.write_text("the fatal nobody got to read\n")

    where = rg._keep_failed_fold_log(log, "rf3-256-warmup")

    kept = sorted((tmp_path / "perf" / "sizegate" / "failures").glob("rf3-256-warmup-*.log"))
    assert len(kept) == 1
    assert kept[0].read_text() == "the fatal nobody got to read\n"
    assert str(kept[0].relative_to(tmp_path)) in where

    # the whole point: it survives the scratch tree going away
    import shutil
    shutil.rmtree(scratch)
    assert kept[0].exists()


def test_a_missing_log_reports_that_and_does_not_raise(rg, tmp_path):
    """This runs inside a path that is already reporting a failure. It must not add its own."""
    where = rg._keep_failed_fold_log(tmp_path / "gone.log", "rf3-256-warmup")
    assert "could not be kept" in where
