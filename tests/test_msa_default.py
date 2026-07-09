"""Regression: the MSA-dependent models (boltz2, protenix-v2) must never silently
fold single-sequence.

The "~10 A Protenix-v2" result (docs/protenix-accuracy-investigation.md) was an
MSA-trained model folded single-sequence. ``predict`` now resolves an MSA source
by default via ``_resolve_msa_default``; these tests pin the precedence contract:

  1. --single_sequence      -> fold single-seq, no network, no notice
  2. explicit source        -> pass through unchanged
  3. local <cache>/msa_db   -> use it silently (no network, no notice)
  4. nothing                -> enable the online server + a privacy notice naming it

esmfold2 / esmfold2-fast are single-sequence by design and must pass through
untouched. Host-only — no device, no network.
"""
from __future__ import annotations

import click
import pytest

from tt_bio.main import _resolve_msa_default as _resolve

URL = "https://api.colabfold.com"


def resolve(model, use_msa_server=False, msa_db_path=None, msa_endpoint=None,
            single_sequence=False, cache="/nonexistent", controller=None, msa_server_url=URL):
    return _resolve(model, use_msa_server, msa_db_path, msa_endpoint,
                    single_sequence, cache, controller, msa_server_url)


def _ready_db(cache):
    """Create a fake ready local ColabFold DB under <cache>/msa_db."""
    db = cache / "msa_db"
    db.mkdir(parents=True)
    (db / "UNIREF30_READY").write_text("x")
    return db


@pytest.mark.parametrize("model", ["esmfold2", "esmfold2-fast"])
def test_single_sequence_models_untouched(model, tmp_path, capsys):
    """ESMFold2 variants are single-sequence by design: never auto-enable a source."""
    _ready_db(tmp_path)  # even with a local DB present, they must not pick it up
    assert resolve(model, cache=str(tmp_path)) == (False, None)
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("model", ["boltz2", "protenix-v2"])
def test_no_source_falls_back_online_with_notice(model, tmp_path, capsys):
    """No source and no local DB -> online server enabled + a one-line privacy notice."""
    use_msa_server, msa_db_path = resolve(model, cache=str(tmp_path))
    assert use_msa_server is True and msa_db_path is None
    assert URL in capsys.readouterr().out


@pytest.mark.parametrize("model", ["boltz2", "protenix-v2"])
def test_notice_names_the_actual_server(model, tmp_path, capsys):
    """The privacy notice must name the server actually contacted, not a hardcoded
    default — an honest disclosure when --msa_server_url is overridden."""
    resolve(model, cache=str(tmp_path), msa_server_url="https://msa.internal.example")
    assert "https://msa.internal.example" in capsys.readouterr().out


@pytest.mark.parametrize("model", ["boltz2", "protenix-v2"])
def test_local_db_used_silently(model, tmp_path, capsys):
    """A ready local DB is auto-detected and used with no network and no notice."""
    db = _ready_db(tmp_path)
    assert resolve(model, cache=str(tmp_path)) == (False, str(db))
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("model", ["boltz2", "protenix-v2"])
def test_explicit_sources_pass_through(model, tmp_path, capsys):
    """An explicit source is honored verbatim (and does not trigger the notice)."""
    _ready_db(tmp_path)  # present, but an explicit flag must win over auto-detect
    assert resolve(model, use_msa_server=True, cache=str(tmp_path)) == (True, None)
    assert resolve(model, msa_db_path="/db", cache=str(tmp_path)) == (False, "/db")
    # msa_endpoint alone counts as an explicit source: stay off both server and DB
    assert resolve(model, msa_endpoint="http://h:8765", cache=str(tmp_path)) == (False, None)
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("model", ["boltz2", "protenix-v2"])
def test_single_sequence_opt_out_silent(model, tmp_path, capsys):
    """--single_sequence skips both the local DB and the online fallback, silently."""
    _ready_db(tmp_path)
    assert resolve(model, single_sequence=True, cache=str(tmp_path)) == (False, None)
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("kwargs", [
    dict(use_msa_server=True),
    dict(msa_db_path="/db"),
    dict(msa_endpoint="http://h"),
])
def test_single_sequence_conflicts_error(kwargs, tmp_path):
    """--single_sequence combined with any explicit MSA source is a user error."""
    with pytest.raises(click.BadParameter):
        resolve("protenix-v2", single_sequence=True, cache=str(tmp_path), **kwargs)


def test_controller_skips_local_db(tmp_path, capsys):
    """In --controller mode the local host's DB is irrelevant (remote workers resolve
    on their own hosts): fall back online rather than auto-detecting a local DB."""
    _ready_db(tmp_path)
    assert resolve("boltz2", cache=str(tmp_path), controller="http://host:8765") == (True, None)
    assert URL in capsys.readouterr().out
