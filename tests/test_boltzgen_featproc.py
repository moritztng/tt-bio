"""bg_featproc's featurize worker: no named semaphores, and a worker that dies raises instead of hanging the design."""
import multiprocessing.synchronize
import os

import pytest

from tt_bio.boltzgen.task.predict.predict import _FeatureProcess


class _DiesOnLoad:
    """Unpickling this in the worker exits the worker, as a crash while it opens the data module would."""

    def __reduce__(self):
        return os._exit, (3,)


class _NoDesigns:
    def predict_dataloader(self):
        return []


def test_dead_worker_raises_and_creates_no_named_semaphore(monkeypatch):
    # logind's RemoveIPC deletes a user's named semaphores when their last login session closes; the worker must not
    # depend on one surviving between its start and its first request.
    def no_semlock(*a, **k):
        raise AssertionError("bg_featproc created a named semaphore")

    monkeypatch.setattr(multiprocessing.synchronize.SemLock, "__init__", no_semlock)
    with _FeatureProcess(_DiesOnLoad()) as worker:
        with pytest.raises(RuntimeError, match="exited with code 3"):
            worker.submit().result()


@pytest.mark.parametrize("wormhole", [True, False])
@pytest.mark.parametrize("fast", [False, True])
def test_shipped_sets_reach_the_featurize_process(monkeypatch, wormhole, fast):
    # Predict picks the featurizer before the model exists. It used to ask the shared sets, which never name
    # bg_featproc, so every unharnessed design featurized on a thread although each BoltzGen set names the process.
    from tt_bio import tenstorrent as T
    from tt_bio.boltzgen.model.models import boltz as B
    from tt_bio.boltzgen.task.predict import predict as P

    monkeypatch.delenv("TT_BIO_LEVERS", raising=False)
    monkeypatch.setattr(T, "is_wormhole", lambda: wormhole)
    monkeypatch.setattr(T, "_FAST_MODE", fast)
    assert P.design_levers is B.design_levers
    assert "bg_featproc" in B.design_levers()


def test_next_stage_takes_the_warm_spare_unless_the_environment_moved(monkeypatch):
    # Each featurizer starts the next one, so the next stage does not wait on a fresh process's imports. A spare carries
    # the environment it was spawned under, so one from before an environment change must not be used.
    from tt_bio.boltzgen.task.predict import predict as P

    monkeypatch.setattr(P, "_SPARE", None)
    with _FeatureProcess(_NoDesigns()) as first:
        spare = P._SPARE[1]
        assert spare is not first._proc and spare.is_alive()
    monkeypatch.setenv("BOLTZGEN_PROGRESS_FILE", "/tmp/next-job-progress.jsonl")
    with _FeatureProcess(_NoDesigns()) as second:
        assert second._proc is spare
        moved = P._SPARE[1]
    monkeypatch.setenv("TT_BIO_FEATPROC_TEST", "1")
    with _FeatureProcess(_NoDesigns()) as third:
        assert third._proc is not moved
    moved.join(10)
    assert not moved.is_alive() and moved.exitcode == 0
