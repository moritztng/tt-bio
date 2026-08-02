"""Regression test: the platform must find the engine's predict results dir
under its per-model name (``<model>_results_<stem>``), not a hardcoded
``boltz_results_*`` prefix. The engine renamed folders via
``predict_results_dir_name`` and every predict job was then misclassified
as "Every target failed" despite a clean run — this locks the contract."""

import json
import os

import pytest

from tt_bio.main import PREDICT_MODELS, predict_results_dir_name
from tt_bio.platform.jobs import Job, JobManager


def test_results_dir_per_model(tmp_path):
    mgr = JobManager(tmp_path / "ws", msa_db_path=None)
    for model in PREDICT_MODELS:
        job = Job(id=f"j-{model}", kind="predict", name="t", created_at=0.0,
                  model=model, total=1)
        out = mgr._out_dir(job.id)
        rd = out / predict_results_dir_name(model, "inputs")
        rd.mkdir(parents=True)
        (rd / "results.json").write_text(json.dumps([{"id": "target_1",
                                                      "status": "ok"}]))
        assert mgr._results_dir(job) == rd, model
        assert mgr._ok_count(job) == 1, model


def test_results_dir_missing(tmp_path):
    mgr = JobManager(tmp_path / "ws", msa_db_path=None)
    job = Job(id="j-x", kind="predict", name="t", created_at=0.0,
              model="boltz2", total=1)
    (mgr._out_dir(job.id)).mkdir(parents=True)
    assert mgr._results_dir(job) is None
    assert mgr._ok_count(job) == 0


def test_msa_cache_dir_writable(tmp_path):
    mgr = JobManager(tmp_path / "ws", msa_db_path=str(tmp_path / "db"))
    assert mgr.msa_cache_dir == str(tmp_path / "msa_cache")


def test_msa_cache_dir_unwritable_falls_back(tmp_path):
    # The /data NFS-down state: parent exists, is not writable. MSA jobs must
    # fall back to per-job caching (no --msa_dir) instead of crashing.
    if os.geteuid() == 0:
        pytest.skip("root ignores permission bits")
    ro = tmp_path / "ro"
    ro.mkdir()
    ro.chmod(0o555)
    try:
        mgr = JobManager(tmp_path / "ws", msa_db_path=str(ro / "colabfold_db"))
        assert mgr.msa_cache_dir is None
    finally:
        ro.chmod(0o755)
