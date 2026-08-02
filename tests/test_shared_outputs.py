"""The co-located result transport: results must survive it, and only when co-location is real.

Embed results otherwise travel back as base64 inside JSON through one controller process
(~635 MB for a 1024-sequence esmc run), which is what leaves the cards idle for half the
wall. Workers that share the client's filesystem can skip that entirely -- but a worker
that only *appears* to share it must not, or its outputs vanish.
"""

from pathlib import Path

from tt_bio.main import _clear_shared_outputs, _offer_shared_outputs
from tt_bio.worker import SHARED_OUTPUT_PREFIX, _read_outputs, _shared_outputs_dir


def _payload(struct_dir):
    payload = {"config": {}}
    _offer_shared_outputs(payload, struct_dir)
    return payload


def test_offer_is_only_accepted_by_a_worker_that_sees_the_nonce(tmp_path):
    struct_dir = tmp_path / "results"
    cfg = _payload(struct_dir)["config"]

    assert _shared_outputs_dir(cfg) == struct_dir

    # A worker elsewhere: same path string, different filesystem -> no nonce, no sharing.
    elsewhere = dict(cfg)
    elsewhere["shared_outputs"] = {**cfg["shared_outputs"], "dir": str(tmp_path / "other")}
    (tmp_path / "other").mkdir()
    assert _shared_outputs_dir(elsewhere) is None


def test_shared_transport_delivers_the_same_bytes_without_base64(tmp_path):
    struct_dir = tmp_path / "results"
    cfg = _payload(struct_dir)["config"]
    out = tmp_path / "workdir"
    (out / "nested").mkdir(parents=True)
    (out / "a.npz").write_bytes(b"\x00binary-a")
    (out / "nested" / "b.npz").write_bytes(b"\x01binary-b")

    outputs = _read_outputs(out, _shared_outputs_dir(cfg))

    assert sorted(outputs) == ["a.npz", "nested/b.npz"]
    assert all(v.startswith(SHARED_OUTPUT_PREFIX) for v in outputs.values())
    # the payload never carries the bytes, and the files are already where the client wants them
    assert (struct_dir / "a.npz").read_bytes() == b"\x00binary-a"
    assert (struct_dir / "nested" / "b.npz").read_bytes() == b"\x01binary-b"


def test_without_sharing_it_still_base64s(tmp_path):
    out = tmp_path / "workdir"
    out.mkdir()
    (out / "a.npz").write_bytes(b"\x00binary-a")

    outputs = _read_outputs(out, None)

    import base64
    assert base64.b64decode(outputs["a.npz"]) == b"\x00binary-a"
    assert not outputs["a.npz"].startswith(SHARED_OUTPUT_PREFIX)


def test_nonce_is_not_left_behind_in_the_results(tmp_path):
    struct_dir = tmp_path / "results"
    payload = _payload(struct_dir)
    token = payload["config"]["shared_outputs"]["token"]
    assert (struct_dir / token).is_file()

    _clear_shared_outputs(payload, struct_dir)

    assert not (struct_dir / token).exists()
    assert list(struct_dir.iterdir()) == []


def test_a_token_cannot_escape_the_results_directory(tmp_path):
    cfg = {"shared_outputs": {"dir": str(tmp_path), "token": "../etc/passwd"}}
    assert _shared_outputs_dir(cfg) is None


def test_nonce_is_cleared_even_when_the_run_raises(tmp_path, monkeypatch):
    """A run that dies must not leave scaffolding in the user's results directory."""
    import tt_bio.main as m

    struct_dir = tmp_path / "results"
    seen = {}

    def boom(payload, sd):
        _offer_shared_outputs(payload, sd)
        seen["token"] = payload["config"]["shared_outputs"]["token"]

    payload = {"config": {}}
    boom(payload, struct_dir)
    assert (struct_dir / seen["token"]).is_file()

    try:
        raise RuntimeError("run died")
    except RuntimeError:
        m._clear_shared_outputs(payload, struct_dir)

    assert not (struct_dir / seen["token"]).exists()


def test_a_reported_but_missing_shared_file_is_not_silently_accepted(tmp_path, capsys):
    """The worker's scratch dir is gone by now, so a missing file is a lost output."""
    import tt_bio.main as m

    struct_dir = tmp_path / "results"
    struct_dir.mkdir()

    class _Client:
        def job_outputs(self, run_id, job_id):
            return {"a.npz": SHARED_OUTPUT_PREFIX + str(struct_dir / "a.npz")}

    m._write_job_outputs(_Client(), "run", "job", struct_dir)

    err = capsys.readouterr().err
    assert "a.npz" in err and "not there" in err
