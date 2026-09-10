"""Foldseek discovery: the path you asked for, or a message naming what was tried.

The candidate list used to carry one machine's conda prefix ahead of PATH, so a
packaged tt-bio could silently prefer that binary over the user's own.
"""
import os
import stat

import pytest

from tt_bio.saprot import find_foldseek


def _fake(path, tag="mine"):
    path.write_text(f"#!/bin/sh\necho {tag}\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


def test_no_machine_specific_path_is_compiled_in():
    src = __import__("tt_bio.saprot", fromlist=["x"]).__file__
    text = open(src).read()
    assert "/home/" not in text, "a developer's home path is shipped in the package"


def test_the_argument_wins_over_the_env_var(tmp_path, monkeypatch):
    arg = _fake(tmp_path / "arg_foldseek")
    monkeypatch.setenv("FOLDSEEK_BIN", _fake(tmp_path / "env_foldseek"))
    assert find_foldseek(arg) == arg


def test_the_env_var_wins_over_path(tmp_path, monkeypatch):
    env = _fake(tmp_path / "env_foldseek")
    other = tmp_path / "onpath"
    other.mkdir()
    _fake(other / "foldseek")
    monkeypatch.setenv("PATH", str(other))
    monkeypatch.setenv("FOLDSEEK_BIN", env)
    assert find_foldseek() == env


def test_path_is_the_fallback(tmp_path, monkeypatch):
    other = tmp_path / "onpath"
    other.mkdir()
    onpath = _fake(other / "foldseek")
    monkeypatch.setenv("PATH", str(other))
    monkeypatch.delenv("FOLDSEEK_BIN", raising=False)
    assert find_foldseek() == onpath


@pytest.mark.parametrize("how", ["arg", "env"])
def test_a_path_that_is_not_there_is_refused_by_name(tmp_path, monkeypatch, how):
    """Falling back to PATH would run a different binary than the one asked for."""
    other = tmp_path / "onpath"
    other.mkdir()
    _fake(other / "foldseek", "not-the-one-you-asked-for")
    monkeypatch.setenv("PATH", str(other))
    monkeypatch.delenv("FOLDSEEK_BIN", raising=False)
    missing = str(tmp_path / "nope" / "foldseek")
    if how == "env":
        monkeypatch.setenv("FOLDSEEK_BIN", missing)
        with pytest.raises(ValueError, match="FOLDSEEK_BIN"):
            find_foldseek()
    else:
        with pytest.raises(ValueError, match="--foldseek"):
            find_foldseek(missing)


def test_the_message_says_how_to_install_when_nothing_is_set(tmp_path, monkeypatch):
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    monkeypatch.delenv("FOLDSEEK_BIN", raising=False)
    with pytest.raises(ValueError) as e:
        find_foldseek()
    msg = str(e.value)
    assert "bioconda" in msg and "--foldseek" in msg and "FOLDSEEK_BIN" in msg
