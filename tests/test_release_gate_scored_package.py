"""The gate must say which tt_bio it imported, and warn when that is not its own checkout.

Device-free and torch-free. This exists because of a verdict it would have saved.
c13-land-first's 14-arm green gate (2026-09-18) authorised a default flip and was launched as
``cd <worktree> && /home/ttuser/tt-bio-dev/env/bin/python3 scripts/release_gate.py``. That venv
holds an EDITABLE install of /home/ttuser/tt-bio-dev, and sys.path[0] for a script run is the
SCRIPT's directory, never the cwd -- so the gate imported <shared>/tt_bio at 480ae2dfe, a tree
with no ``_B2_DIT_COND_HOIST`` line in it, while the launcher printed the flag block it grepped
out of the worktree. Reading a file on disk is not evidence about the file that was imported.

Run: python3 tests/test_release_gate_scored_package.py, or via pytest.
"""
import os
import sys
import types

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

import release_gate as rg


def _run(fake_init, capsys):
    """Run the preflight with tt_bio.__file__ forced to fake_init; return its output."""
    saved = sys.modules.get("tt_bio")
    stub = types.ModuleType("tt_bio")
    stub.__file__ = fake_init
    sys.modules["tt_bio"] = stub
    try:
        rg._preflight_scored_package()
    finally:
        if saved is None:
            sys.modules.pop("tt_bio", None)
        else:
            sys.modules["tt_bio"] = saved
    return capsys.readouterr().out


def test_it_always_names_the_package_it_scored(capsys):
    """Unconditional, so a green log proves WHICH tree went green."""
    own = os.path.join(REPO, "tt_bio", "__init__.py")
    out = _run(own, capsys)
    assert "scoring tt_bio from" in out
    assert os.path.join(REPO, "tt_bio") in out


def test_the_checkout_it_was_launched_from_draws_no_warning(capsys):
    out = _run(os.path.join(REPO, "tt_bio", "__init__.py"), capsys)
    assert "WARNING" not in out, out


def test_a_foreign_package_warns_and_names_both_trees_and_the_fix(capsys):
    """The negative control: the check is known to be able to fire.

    Without this the test above passes just as well against a preflight that warns never.
    """
    out = _run("/home/ttuser/tt-bio-dev/tt_bio/__init__.py", capsys)
    assert "WARNING" in out, out
    assert "/home/ttuser/tt-bio-dev/tt_bio" in out
    assert REPO in out, "name the checkout that was launched, not just the one imported"
    assert f"PYTHONPATH={REPO}" in out, "a warning without the remedy costs another run"


def test_it_runs_before_any_arm_folds():
    """Placement is the whole point: an epilogue never prints on a gate that dies mid-ladder."""
    src = open(os.path.join(REPO, "scripts", "release_gate.py"), encoding="utf-8").read()
    mine = src.index("    _preflight_scored_package()")
    for later in ("_preflight_eval_scorers(models)", "rows = []"):
        assert mine < src.index("    " + later), f"{later} must come after the banner"


def test_a_broken_import_degrades_to_a_warning_not_a_crash(capsys):
    """A banner must never be the thing that kills a 3h gate."""
    saved = sys.modules.get("tt_bio")
    stub = types.ModuleType("tt_bio")          # no __file__ at all
    sys.modules["tt_bio"] = stub
    try:
        rg._preflight_scored_package()
    finally:
        if saved is None:
            sys.modules.pop("tt_bio", None)
        else:
            sys.modules["tt_bio"] = saved
    assert "WARNING" in capsys.readouterr().out


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
