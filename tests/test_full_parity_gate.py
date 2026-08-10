"""Device-free regression tests for scripts/full_parity_gate.py verdict logic.

Pins the false-pass found 2026-08-09/10: ``--seeds 5`` (a bare count, not a
comma-separated list) matched no fixture seed, so every leg reported
BLOCKED-REF-REGEN-NEEDED and the gate still printed GATE PASS with zero legs
scored. These tests run no device work: the seeds validation fires before any
fold, and the tally tests force the blocked path with a monkeypatched
``_incomplete_fixture_seeds``.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "full_parity_gate.py"


def _load():
    spec = importlib.util.spec_from_file_location("full_parity_gate", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    # @dataclass resolves cls.__module__ through sys.modules during exec_module
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _run_gate(args, tmp_path):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args, "--workdir", str(tmp_path),
         "--workers", "localhost:0"],
        cwd=REPO, capture_output=True, text=True, timeout=180)


def test_seeds_bare_count_rejected_not_false_pass(tmp_path):
    """The exact reported misuse: `--seeds 5` meaning "5 seeds" errors out loudly
    instead of blocking every leg and printing GATE PASS."""
    proc = _run_gate(["--leg", "openfold3-8hel-nomsa", "--seeds", "5"], tmp_path)
    assert proc.returncode != 0
    assert "GATE PASS" not in proc.stdout
    assert "--seeds" in proc.stderr
    assert "not a" in proc.stderr  # names the list-not-count semantics
    assert "openfold3-8hel-nomsa" in proc.stderr  # names the leg + its real fixture seeds


def test_seeds_non_integer_rejected(tmp_path):
    proc = _run_gate(["--leg", "openfold3-8hel-nomsa", "--seeds", "abc"], tmp_path)
    assert proc.returncode != 0
    assert "GATE PASS" not in proc.stdout
    assert "--seeds" in proc.stderr


def test_all_blocked_run_is_inconclusive_not_pass(tmp_path, monkeypatch, capsys):
    """Every leg blocked on reference regen => nothing scored => INCONCLUSIVE,
    exit nonzero. Reproduces the all-blocked condition through the same
    _incomplete_fixture_seeds path the typo took, with an otherwise valid seed."""
    mod = _load()
    ckpt = tmp_path / "of3-p2-155k.pt"
    ckpt.write_bytes(b"x")  # preflight only checks existence
    monkeypatch.setenv("OF3_CKPT", str(ckpt))
    monkeypatch.setattr(mod, "_incomplete_fixture_seeds",
                        lambda leg, seeds: [f"seed{s}" for s in seeds])
    monkeypatch.setattr(sys, "argv", ["full_parity_gate.py",
                                      "--leg", "openfold3-8hel-nomsa", "--seeds", "0",
                                      "--workdir", str(tmp_path),
                                      "--workers", "localhost:0"])
    rc = mod.main()
    out = capsys.readouterr().out
    assert rc != 0
    assert "GATE INCONCLUSIVE" in out
    assert "1/1 legs blocked" in out
    assert "GATE PASS" not in out


def test_scored_leg_plus_blocked_leg_still_passes(tmp_path, monkeypatch, capsys):
    """Normal case unchanged: a blocked leg still does NOT fail the gate when a
    sibling leg reaches a real scored verdict (rfd3-featurizer is card-free and
    runs its committed bit-exact reference for real here)."""
    mod = _load()
    ckpt = tmp_path / "of3-p2-155k.pt"
    ckpt.write_bytes(b"x")
    monkeypatch.setenv("OF3_CKPT", str(ckpt))
    monkeypatch.setattr(mod, "_incomplete_fixture_seeds",
                        lambda leg, seeds: ["seed0"] if leg.id == "openfold3-8hel-nomsa" else [])
    monkeypatch.setattr(sys, "argv", ["full_parity_gate.py",
                                      "--leg", "rfd3-featurizer",
                                      "--leg", "openfold3-8hel-nomsa", "--seeds", "0",
                                      "--workdir", str(tmp_path),
                                      "--workers", "localhost:0"])
    rc = mod.main()
    out = capsys.readouterr().out
    assert rc == 0
    assert "GATE PASS" in out
    assert "GATE INCONCLUSIVE" not in out
    assert "'PASS': 1" in out
    assert "'BLOCKED-REF-REGEN-NEEDED': 1" in out


def test_of3_ckpt_preflight(tmp_path, monkeypatch):
    """OpenFold3 legs fail fast at preflight when no checkpoint resolves, instead of
    dying inside tt_bio/worker.py after paying for fold setup."""
    mod = _load()
    leg = mod.LEGS_BY_ID["openfold3-8hel-nomsa"]
    monkeypatch.delenv("OF3_CKPT", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))  # no ~/.boltz or ~/of3-weights here
    problems = mod.preflight_check([leg])
    assert any("OF3_CKPT" in p and leg.id in p for p in problems)
    monkeypatch.setenv("OF3_CKPT", str(tmp_path / "nope.pt"))
    problems = mod.preflight_check([leg])
    assert any("OF3_CKPT" in p and "not an existing file" in p for p in problems)
    ckpt = tmp_path / "of3-p2-155k.pt"
    ckpt.write_bytes(b"x")
    monkeypatch.setenv("OF3_CKPT", str(ckpt))
    assert not any("OF3" in p for p in mod.preflight_check([leg]))
