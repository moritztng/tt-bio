#!/usr/bin/env python3
"""Device-free control for provenance.check().

Uses REAL commits in this repository rather than fixtures, because the thing under test is a
`git diff` and a fixture would test the mock. Every case below names why it was picked.

  ONE_HEAD      59e2adced             the head the APB session 3 legs started on
  HARNESS_ONLY  a28ace13f             pass 21's guard fix: perf/ only, tt_bio/ untouched
  CODE_MOVED    8d72c4356 -> main     main's five new _MM_BLOCK keys, i.e. tt_bio/ DOES move

The last one is the negative control and it breaks exactly what the check reads: a real commit
pair whose diff under tt_bio/ is non-empty. The mutation test then shows the relaxation is
doing work -- plain head equality REJECTS the harness-only pair, which is the session this row
actually has on disk.
"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import provenance  # noqa: E402

REPO = str(Path(__file__).resolve().parents[2])
ONE_HEAD = "59e2adced7d6c40b53e7afb8f0823d2d3345e072"
HARNESS_ONLY = "a28ace13ffa0bfcf308af8a1d368aafd80761209"
CODE_MOVED = "8d72c4356335ef57f1963a611e7af85cf47632ad"


def _main_sha():
    r = subprocess.run(["git", "-C", REPO, "rev-parse", "origin/main"],
                       capture_output=True, text=True, check=True)
    return r.stdout.strip()


def test_single_head_passes():
    ok, lines = provenance.check([ONE_HEAD] * 12, REPO)
    assert ok, lines
    assert "one tree" in lines[0]


def test_harness_only_commit_does_not_break_a_session():
    ok, lines = provenance.check([ONE_HEAD] * 3 + [HARNESS_ONLY] * 5, REPO)
    assert ok, lines
    assert any("byte-identical" in x for x in lines), lines


def test_a_commit_that_moves_tt_bio_fails():
    ok, lines = provenance.check([CODE_MOVED, _main_sha()], REPO)
    assert not ok, lines
    assert any("tt_bio/tenstorrent.py" in x for x in lines), lines
    assert any("do not score" in x for x in lines), lines


def test_missing_head_fails():
    ok, lines = provenance.check([ONE_HEAD, None], REPO)
    assert not ok, lines


def test_mutation_head_equality_would_discard_the_real_session():
    """The check the campaign nearly used, run on the session it would have thrown away."""
    heads = [ONE_HEAD] * 3 + [HARNESS_ONLY] * 5
    assert len(set(heads)) != 1, "head-equality accepts this session"   # it does NOT
    assert provenance.check(heads, REPO)[0], "but the tt_bio-equality check keeps it"
    # and the relaxation is not vacuous: it still refuses the pair where tt_bio moved
    assert not provenance.check(
        [CODE_MOVED, _main_sha()], REPO)[0]


if __name__ == "__main__":
    # qb2's system python3 has no pytest, and this control is worth nothing if it only runs
    # somewhere else. Same functions either way.
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS %s" % name)
            except AssertionError as e:
                fails += 1
                print("FAIL %s: %s" % (name, e))
    print("%d failed" % fails)
    sys.exit(1 if fails else 0)
