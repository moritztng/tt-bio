"""The gate's crash-resume must survive the gate's own run outputs.

Both defects here were found on 2026-09-19, after qb2 hard-reset at 16:21:00Z and killed a release
gate with nine arms green behind it. Neither arm list nor verdict was wrong; the resume simply could
not fire, so the answer to "did this tree pass the gate" cost a full re-run every time the box reset.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import gate_journal as gj  # noqa: E402


def _git(tmp, *args):
    subprocess.run(["git", *args], cwd=tmp, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@t")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "tt_bio").mkdir()
    (tmp_path / "tt_bio" / "__init__.py").write_text("")
    _git(tmp_path, "add", "tt_bio/__init__.py")
    _git(tmp_path, "commit", "-qm", "init")
    return tmp_path


def _dirty(repo_root):
    """release_gate._repo_dirty's predicate, against an arbitrary root."""
    out = subprocess.check_output(
        ["git", "status", "--porcelain", "--", "tt_bio", "scripts", "tests", "pyproject.toml"],
        cwd=repo_root, text=True, timeout=10)
    return bool(out.strip())


def test_run_outputs_in_repo_root_do_not_make_the_tree_dirty(repo):
    """The gate writes boltz2_results_prot/ and friends into the repo ROOT with --keep.

    Unscoped, `git status --porcelain` reports them and every arm after the first journals
    dirty=True, which gate_journal.resumable() refuses outright. Nineteen such paths were what
    stranded the nine green arms.
    """
    assert not _dirty(repo)
    for p in ("boltz2_results_prot", "rfd3_gate_designs", "pxdesign_gate.json"):
        (repo / p).mkdir(exist_ok=True) if not p.endswith(".json") else (repo / p).write_text("{}")
    # the unscoped predicate the gate used to have would now call this dirty
    assert subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip()
    # the scoped one must not
    assert not _dirty(repo), "gate run outputs must not cost the resume"


def test_a_real_source_edit_still_refuses_to_resume(repo):
    """The negative control: the scoping must not have made the predicate vacuous."""
    (repo / "tt_bio" / "__init__.py").write_text("# edited\n")
    assert _dirty(repo), "an edit under tt_bio/ must still be dirty"


def test_an_untracked_module_under_the_package_is_still_dirty(repo):
    """A stray module inside the package can shadow a real one, so it is a source change."""
    (repo / "tt_bio" / "shadow.py").write_text("x = 1\n")
    assert _dirty(repo)


def test_ingest_carries_members_so_a_multi_model_arm_can_resume(tmp_path):
    """fold-models scores nine models under one headline.

    Back-filled without members the record says members=['fold-models'], _resume_plan's subset test
    fails, and the ingest silently drops the most expensive arm in the gate while reporting PASS.
    """
    log = tmp_path / "gate.log"
    log.write_text("GATE PASS — all models cleared parse + ground-truth floor + geometry\n")
    journal = tmp_path / "j.jsonl"
    key = {f: v for f, v in zip(gj.KEY_FIELDS,
                                ("abc1234", False, "h", "p300c", False, False, "/pkg"))}
    models = ["boltz2", "esmfold2", "protenix-v2"]
    gj.ingest(log, key, journal, {"fold-models": models})
    rec = json.loads(journal.read_text().splitlines()[0])
    assert rec["arm"] == "fold-models"
    assert rec["members"] == sorted(models), "ingest dropped the arm's members"
    assert set(models) <= set(gj.resumable(journal, key)["fold-models"]["members"])


def test_ingest_without_members_is_the_old_broken_behaviour(tmp_path):
    """Negative control for the test above: no members means the arm covers only its own name."""
    log = tmp_path / "gate.log"
    log.write_text("GATE PASS — all models cleared parse + ground-truth floor + geometry\n")
    journal = tmp_path / "j.jsonl"
    key = {f: v for f, v in zip(gj.KEY_FIELDS,
                                ("abc1234", False, "h", "p300c", False, False, "/pkg"))}
    gj.ingest(log, key, journal)
    rec = json.loads(journal.read_text().splitlines()[0])
    assert rec["members"] == ["fold-models"]
