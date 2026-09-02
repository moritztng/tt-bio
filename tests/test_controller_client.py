"""ControllerClient's transient-read retry, and the one fleet preflight.

Both exist because a co-tenant filling the shared disk made the controller's store raise,
the handler answered 503, and every unretried poll killed its run (58 folds on 2026-08-03).
`predict` grew a private 5x backoff around `client.events`; boltzgen, rfd3 and pxdesign
never did. The retry moved onto `_request`, so the properties worth pinning are which calls
get replayed and which must not.

Stdlib only, no device, no network: urlopen is replaced by a script of outcomes.
"""

from __future__ import annotations

import ast
import io
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tt_bio.distributed import (  # noqa: E402
    ControllerClient,
    ControllerUnreachable,
    connect_controller,
)


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "http://c/x", code, "boom", {}, io.BytesIO(b"store is unhappy"))


class _Body(io.BytesIO):
    """Minimal stand-in for urlopen's context-manager response."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def urlopen(monkeypatch):
    """Script urlopen with a list of outcomes; record the requests it saw.

    Sleep is stubbed out: the backoff is 1+2+4+8 s and the point under test is the
    number of attempts, not the wall-clock.
    """
    monkeypatch.setattr("tt_bio.distributed.time.sleep", lambda _s: None)

    class Scripted:
        def __init__(self):
            self.outcomes: list = []
            self.methods: list[str] = []

        def install(self, *outcomes):
            self.outcomes = list(outcomes)
            self.methods = []            # each install is a fresh count
            monkeypatch.setattr(urllib.request, "urlopen", self)
            return self

        def __call__(self, req, timeout=None):
            self.methods.append(req.get_method())
            out = self.outcomes.pop(0) if self.outcomes else b'{"ok": true}'
            if isinstance(out, Exception):
                raise out
            return _Body(out)

        @property
        def attempts(self) -> int:
            return len(self.methods)

    return Scripted()


def test_a_transient_5xx_read_is_retried_and_succeeds(urlopen):
    urlopen.install(_http_error(503), _http_error(503), b'{"events": [], "status": "ok"}')
    client = ControllerClient("http://c")
    assert client.events("run1", 0) == {"events": [], "status": "ok"}
    assert urlopen.attempts == 3


def test_a_dropped_connection_is_retried(urlopen):
    urlopen.install(urllib.error.URLError("connection reset"), b'{"online_workers": 2}')
    assert ControllerClient("http://c").cluster() == {"online_workers": 2}
    assert urlopen.attempts == 2


def test_a_read_that_never_recovers_raises_after_the_attempt_budget(urlopen):
    urlopen.install(*[_http_error(503)] * 9)
    client = ControllerClient("http://c")
    with pytest.raises(RuntimeError, match="controller error 503"):
        client.events("run1", 0)
    assert urlopen.attempts == ControllerClient.READ_ATTEMPTS


def test_a_4xx_read_is_the_controllers_answer_and_is_not_retried(urlopen):
    """A missing run is settled. Retrying spends the whole backoff to be told twice."""
    urlopen.install(*[_http_error(404)] * 9)
    with pytest.raises(RuntimeError, match="controller error 404"):
        ControllerClient("http://c").events("gone", 0)
    assert urlopen.attempts == 1


def test_a_post_is_never_replayed(urlopen):
    """The safety half. /complete and /events(POST) are not idempotent: a replayed
    completion double-serves a job or duplicates a result row."""
    urlopen.install(*[_http_error(503)] * 9)
    client = ControllerClient("http://c")
    with pytest.raises(RuntimeError, match="controller error 503"):
        client.complete("run1", "w0", {"id": "j0"}, {})
    assert urlopen.attempts == 1
    assert urlopen.methods == ["POST"]


def test_every_read_method_goes_through_the_retry(urlopen):
    """Pins the GET/POST split against the method list, so a new endpoint added as a
    GET inherits the retry and one added as a POST does not silently get replayed."""
    reads = {
        "events": lambda c: c.events("r", 0),
        "cluster": lambda c: c.cluster(),
        "results": lambda c: c.results("r"),
        "run_jobs": lambda c: c.run_jobs("r"),
        "job_outputs": lambda c: c.job_outputs("r", "j"),
    }
    for name, call in reads.items():
        urlopen.install(_http_error(503), b'{"results": [], "jobs": [], "outputs": {}}')
        call(ControllerClient("http://c"))
        assert urlopen.attempts == 2, f"{name} did not retry"
        assert urlopen.methods == ["GET", "GET"], f"{name} is not a GET"


def test_connect_controller_returns_the_client_and_the_worker_count(urlopen):
    urlopen.install(b'{"online_workers": 4}')
    client, online = connect_controller("http://c")
    assert isinstance(client, ControllerClient)
    assert online == 4


def test_connect_controller_refuses_a_controller_with_no_workers(urlopen):
    """A run created against an empty controller queues forever and the caller polls
    forever. Refuse it up front, in the one place every fleet entrypoint shares."""
    urlopen.install(b'{"online_workers": 0}')
    with pytest.raises(ControllerUnreachable, match="No workers connected"):
        connect_controller("http://c")


def test_connect_controller_refuses_an_unreachable_controller_without_the_backoff(urlopen):
    """A preflight fails fast. Nothing has started, so a typo'd URL should say so now
    instead of spending 1+2+4+8 s of read backoff first; the retry is for a committed run.
    """
    urlopen.install(*[urllib.error.URLError("no route to host")] * 9)
    with pytest.raises(ControllerUnreachable, match="Cannot reach controller"):
        connect_controller("http://c")
    assert urlopen.attempts == 1


def test_the_client_connect_controller_returns_does_retry(urlopen):
    """The other half: once the preflight passes the run is committed, so the client
    handed back polls with the full budget."""
    urlopen.install(b'{"online_workers": 1}')
    client, _ = connect_controller("http://c")
    assert client.read_attempts == ControllerClient.READ_ATTEMPTS
    urlopen.install(_http_error(503), b'{"events": [], "status": "ok"}')
    assert client.events("r", 0)["status"] == "ok"
    assert urlopen.attempts == 2


def test_controller_unreachable_is_a_runtimeerror():
    """Callers that predate the class caught RuntimeError. Keep them working."""
    assert issubclass(ControllerUnreachable, RuntimeError)


def test_every_fleet_entrypoint_shares_the_one_preflight():
    """The guard for this sweep's finding: four copies of the preflight, two wordings and
    two exception types for one condition. A fifth copy is a fork, so fail on one.

    `_stream_via_controller` (predict --controller) is deliberately NOT here: it reads
    `online_workers` for the progress display and must stay lenient, because the platform
    submits through it and does its own refusing. It reads the count, it does not gate on it.
    """
    root = Path(__file__).resolve().parents[1] / "tt_bio"
    entrypoints = [
        root / "main.py",
        root / "rfd3" / "design.py",
        root / "pxdesign" / "design.py",
        root / "boltzgen" / "cli" / "boltzgen.py",
    ]
    offenders, unbound = [], []
    for path in entrypoints:
        text = path.read_text()
        assert "connect_controller(" in text, f"{path.name} lost the shared preflight"
        # Calling it is not enough: the first cut of this change called it in main.py
        # without importing it, and only a live `embed --controller` found the NameError.
        # The name has to be bound in the module that calls it.
        tree = ast.parse(text)
        imported = any(
            isinstance(node, ast.ImportFrom)
            and any(a.name == "connect_controller" for a in node.names)
            for node in ast.walk(tree))
        if not imported:
            unbound.append(path.name)
        if "No workers connected" in text or "Cannot reach controller" in text:
            offenders.append(path.name)
    assert not unbound, f"these call connect_controller without importing it: {unbound}"
    assert not offenders, (
        "these re-implement the controller preflight instead of calling "
        f"distributed.connect_controller: {offenders}")
