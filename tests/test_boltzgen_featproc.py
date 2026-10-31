"""bg_featproc's featurize worker: no named semaphores, and a worker that dies raises instead of hanging the design."""
import multiprocessing.synchronize
import os

import pytest

from tt_bio.boltzgen.task.predict.predict import _FeatureProcess


class _DiesOnLoad:
    """Unpickling this in the worker exits the worker, as a crash while it opens the data module would."""

    def __reduce__(self):
        return os._exit, (3,)


def test_dead_worker_raises_and_creates_no_named_semaphore(monkeypatch):
    # logind's RemoveIPC deletes a user's named semaphores when their last login session closes; the worker must not
    # depend on one surviving between its start and its first request.
    def no_semlock(*a, **k):
        raise AssertionError("bg_featproc created a named semaphore")

    monkeypatch.setattr(multiprocessing.synchronize.SemLock, "__init__", no_semlock)
    with _FeatureProcess(_DiesOnLoad()) as worker:
        with pytest.raises(RuntimeError, match="exited with code 3"):
            worker.submit().result()
