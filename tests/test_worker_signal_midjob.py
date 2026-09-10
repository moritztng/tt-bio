"""A signal that stops a worker mid-fold must be reported, not swallowed.

Root-caused from rf3 at 1536 tokens on qb1 card 0, 2026-09-09 23:20Z: the worker
went from `trunk 3/10` to gone in ~30 s, returned 0, wrote no results row and left
`structures/` empty, and the CLI said "The worker's own traceback above says why"
with nothing above it.

Exit 0 out of a spawned worker means `run_worker_loop` returned. Everything that
escapes the job loop goes through `except BaseException`, which writes a fatal to
the launcher's REAL stderr (`_report_fatal` keeps a dup of fd 2 past
`_silence_subprocess_output`) and re-raises, so it cannot be silent. That leaves
one path: `except KeyboardInterrupt: pass`, i.e. a SIGINT or SIGTERM. SIGINT is
exactly what `main._stop_worker_processes` sends to end a finished run, so
swallowing it between leases is correct and must keep working; swallowing it
mid-job loses a fold that ran for eight minutes and tells the user nothing.

Host-only: no device, no weights. The model is stubbed and the signal is raised
from inside `predict_one`, which is where the real one landed.
"""

from __future__ import annotations

import base64
import signal
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tt_bio import worker as W  # noqa: E402
from tt_bio.distributed import ControllerClient, ControllerServer  # noqa: E402


def _controller(tmpdir: Path):
    server = ControllerServer("127.0.0.1", 0, tmpdir / "controller.sqlite3")
    server.serve_in_background()
    return server, ControllerClient(f"http://127.0.0.1:{server.port}")


def _make_run(client: ControllerClient, tmpdir: Path) -> str:
    return client.create_run({
        "data": str(tmpdir / "in.yaml"),
        "out_dir": str(tmpdir / "out"),
        "result_dir": str(tmpdir / "res"),
        "config": {"model": "esmfold2", "kind": "predict"},
        "jobs": [{"id": "cdk2x2_1536", "name": "cdk2x2_1536.yaml",
                  "input_b64": base64.b64encode(b"sequences: []\n").decode()}],
    })["run_id"]


def _worker_info():
    return {"worker_id": "w0", "device_id": 0, "host": "test", "accelerator": "cpu",
            "label": "test:cpu0", "visible_devices": "", "logical_device_id": 0}


class _StubModel:
    progress_fn = None


def _stub_load(self, cfg):
    self.model = _StubModel()
    self.model_id = cfg.get("model")
    self.config_hash = W._hash_run_config(cfg)


def _drive_loop(monkeypatch, url, *, on_predict):
    """Run the real run_worker_loop in this (main) thread, so the signal handler it
    installs is the one that fires, and restore the suite's handlers afterwards."""
    monkeypatch.delenv("TT_BIO_PARENT_PID", raising=False)
    monkeypatch.setattr(W._WorkerState, "load_model", _stub_load)
    monkeypatch.setattr(W._WorkerState, "predict_one",
                        lambda self, path, cfg: on_predict())
    # Neither belongs in a host test: one downloads weights, one is a heavy import.
    monkeypatch.setattr(W, "_ensure_local_artifacts", lambda cfg: None)
    monkeypatch.setattr("tt_bio.esmfold2.set_progress", lambda pfn: None, raising=False)

    saved = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        return W.run_worker_loop(url, _worker_info(), debug=True, idle_poll=0.05)
    finally:
        for s, handler in saved.items():
            signal.signal(s, handler)


def test_a_signal_mid_fold_fails_the_job_and_exits_nonzero(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        server, client = _controller(tmpdir)
        url = f"http://127.0.0.1:{server.port}"
        try:
            run_id = _make_run(client, tmpdir)

            def fold_then_get_signalled():
                # Where the real signal landed: inside the fold, before any output.
                signal.raise_signal(signal.SIGTERM)
                raise AssertionError("the handler did not raise")

            with pytest.raises(SystemExit) as exit_info:
                _drive_loop(monkeypatch, url, on_predict=fold_then_get_signalled)

            # 1. Not exit 0. 143 = 128 + SIGTERM, the shell convention, and distinct
            #    from every code already spoken for (75 contended, 70 orphaned,
            #    137/-9 host OOM kill).
            assert exit_info.value.code == 128 + int(signal.SIGTERM), (
                f"a killed fold exited {exit_info.value.code}; the defect was exit 0")

            # 2. The job is completed as failed, so the run turns terminal instead of
            #    leaving the dispatcher polling a run no worker will ever finish.
            rows = client.results(run_id)
            assert rows, "the killed job was never completed; the run cannot end"
            row = rows[0]
            assert row.get("status") == "failed", f"job status is {row.get('status')!r}"

            # 3. The error names the signal, the job and the loss. Reading `SIGTERM`
            #    is the point: it separates "someone stopped me" from every allocator
            #    and model failure, which is what the rf3 record could not do.
            err = row.get("error") or ""
            for want in ("SIGTERM", "did not finish", "cdk2x2_1536"):
                assert want in err, f"error does not mention {want!r}: {err!r}"
        finally:
            server.shutdown()


def test_a_signal_between_leases_is_still_a_clean_stop(monkeypatch):
    """The negative control on the fix: `main._stop_worker_processes` ends every
    finished run by sending SIGINT, and an idle worker must keep taking that quietly
    and returning. A fix that reported here would turn every normal run's shutdown
    into a failed job and a non-zero worker."""
    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        server, client = _controller(tmpdir)
        url = f"http://127.0.0.1:{server.port}"
        try:
            # No run exists, so the worker is between leases. Signal it from inside
            # the lease call itself rather than from a timer, so there is no race.
            def signal_instead_of_leasing(self, worker, batch_size):
                signal.raise_signal(signal.SIGINT)
                raise AssertionError("the handler did not raise")

            monkeypatch.setattr(ControllerClient, "lease", signal_instead_of_leasing)
            assert _drive_loop(
                monkeypatch, url,
                on_predict=lambda: (_ for _ in ()).throw(AssertionError("no job to run")),
            ) is None, "an idle worker signalled between leases must return, not raise"
        finally:
            server.shutdown()
