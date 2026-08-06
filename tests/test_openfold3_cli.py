"""Host-only contract tests for the --model openfold3 CLI wiring (P13/S6).

Pins: the predict --model choice accepts openfold3; openfold3 is treated as an
MSA-dependent model by _resolve_msa_default (never silently single-sequence unless
explicitly asked); the worker resolves the OF3 checkpoint via $OF3_CKPT or the
cache and fails with a clear message otherwise. No device, no network.
"""
from __future__ import annotations

import pytest


def test_model_choice_accepts_openfold3():
    from tt_bio.main import predict

    model_opt = next(p for p in predict.params if p.name == "model")
    assert "openfold3" in model_opt.type.choices


def test_openfold3_is_msa_dependent(tmp_path):
    """No explicit source + no local DB -> falls back to the online server (True),
    exactly like boltz2/protenix-v2; single_sequence + explicit source is rejected."""
    import click

    from tt_bio.main import _resolve_msa_default

    use_server, db = _resolve_msa_default(
        "openfold3", False, None, None, False, tmp_path, None, "http://msa.example")
    assert use_server is True and db is None

    use_server, db = _resolve_msa_default(
        "openfold3", False, "/some/db", None, False, tmp_path, None, "http://msa.example")
    assert (use_server, db) == (False, "/some/db")

    with pytest.raises(click.BadParameter):
        _resolve_msa_default(
            "openfold3", True, "/some/db", None, True, tmp_path, None, "http://msa.example")


def test_worker_resolves_of3_checkpoint(tmp_path, monkeypatch):
    from tt_bio.worker import _ensure_local_artifacts

    ckpt = tmp_path / "of3-p2-155k.pt"
    ckpt.write_bytes(b"stub")
    monkeypatch.setenv("OF3_CKPT", str(ckpt))
    monkeypatch.setenv("BOLTZ_CACHE", str(tmp_path))
    cfg = {"model": "openfold3", "msa_dir": None}
    _ensure_local_artifacts(cfg)
    assert cfg["of3_ckpt"] == str(ckpt)
    assert cfg["msa_dir"]  # resolved to a writable dir


def test_worker_errors_clearly_without_checkpoint(tmp_path, monkeypatch):
    from tt_bio.worker import _ensure_local_artifacts

    monkeypatch.delenv("OF3_CKPT", raising=False)
    monkeypatch.setenv("BOLTZ_CACHE", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="OF3_CKPT"):
        _ensure_local_artifacts({"model": "openfold3", "msa_dir": None})
