"""The offline-MSA preflight must fail the gate before it folds, not an hour in.

RELEASE_GATE_MSA_DIR is the gate's offline MSA path. A chain whose cached a3m is absent does
not fail as a missing input: the fold falls through to colabfold_search, which the gate hosts
do not have, so the leg dies with "colabfold_search not found" and the summary prints it as a
missed accuracy floor. On the first v0.6.4 gate run that cost 75 minutes and read as an
opendde-abag DockQ failure when the real cause was three a3m files.

Host-only: no device, no fold.
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def gate():
    path = REPO_ROOT / "scripts" / "release_gate.py"
    spec = importlib.util.spec_from_file_location("tt_bio_release_gate_msa", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _chain_hashes(path: Path) -> list:
    doc = yaml.safe_load(path.read_text()) or {}
    out = []
    for entry in doc.get("sequences") or []:
        seq = ((entry.get("protein") if isinstance(entry, dict) else None) or {}).get("sequence")
        if seq:
            out.append(hashlib.sha256(seq.encode()).hexdigest()[:16])
    return out


def test_no_msa_dir_is_a_noop(gate, monkeypatch):
    """Unset means fold against the server, so there is nothing to check."""
    monkeypatch.setattr(gate, "MSA_DIR", None)
    assert gate._preflight_msa_cache(["opendde-abag"]) is None


def test_missing_a3m_fails_before_any_fold(gate, monkeypatch, tmp_path):
    """The bug this exists for: opendde-abag selected, its chains not in the dir."""
    monkeypatch.setattr(gate, "MSA_DIR", str(tmp_path))
    with pytest.raises(SystemExit) as e:
        gate._preflight_msa_cache(["opendde-abag"])
    msg = str(e.value)
    assert "1ahw_abag.yaml" in msg
    for h in _chain_hashes(gate.OPENDDE_ABAG_DATA):
        assert h in msg, f"the message must name the file to seed: {h}.a3m"


def test_seeded_dir_passes(gate, monkeypatch, tmp_path):
    monkeypatch.setattr(gate, "MSA_DIR", str(tmp_path))
    for path in (gate.OPENDDE_ABAG_DATA, gate.DATA):
        for h in _chain_hashes(path):
            (tmp_path / f"{h}.a3m").write_text(">q\nSEQ\n")
    assert gate._preflight_msa_cache(["opendde-abag", "protenix-v2"]) is None


def test_empty_a3m_counts_as_missing(gate, monkeypatch, tmp_path):
    """A zero-byte a3m resolves to single-sequence in tt_bio.main, so it is not coverage."""
    monkeypatch.setattr(gate, "MSA_DIR", str(tmp_path))
    for h in _chain_hashes(gate.OPENDDE_ABAG_DATA):
        (tmp_path / f"{h}.a3m").write_text("")
    with pytest.raises(SystemExit):
        gate._preflight_msa_cache(["opendde-abag"])


def test_arm_not_selected_is_not_checked(gate, monkeypatch, tmp_path):
    """l1-budget folds a msa:empty target, so an empty dir must not block it."""
    monkeypatch.setattr(gate, "MSA_DIR", str(tmp_path))
    assert gate._preflight_msa_cache(["l1-budget", "capacity"]) is None
