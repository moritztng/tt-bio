"""The publish gate moves whole rows, and both count-bearing strings follow.

Two defects this file pins down, both found live on main. `perf/page_rows_pending.json` held
each of the three stripped rows twice, because a `--strip` run against a data file that had
been reverted after an earlier strip appended them again; a later `--restore` would have
published every one of them twice. And the strip regenerated the JSON subtitle while the
page's own meta description kept claiming eighteen models on a page drawing fifteen, because
nothing owned that string.

Each test runs the real script against a copy of the real page, so it exercises the CLI and
the file writes rather than a reimplementation of them.

The held row every test needs is built here rather than read from the page. Three of these
used to take `perf/page_rows_pending.json`'s first row and assert it was PXDesign, which tied
them to whether anything happened to be held that day: publishing the PXDesign row emptied the
file and they failed on an IndexError, having tested nothing about the guard. `make_held`
blanks one cell of a row the page already publishes and lets `--strip` move it, so the held
state is reproduced from any starting page and the round trip itself is under test.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
FILES = ("scripts/site_publish_guard.py", "site/data/perf-512aa.json",
         "site/benchmarks/index.html", "perf/page_rows_pending.json")
META = re.compile(r'<meta name="description" content="([^"]*)">')


@pytest.fixture()
def sandbox(tmp_path: Path) -> Path:
    for rel in FILES:
        dst = tmp_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / rel, dst)
    return tmp_path


def guard(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(root / "scripts" / "site_publish_guard.py"), *args],
                          capture_output=True, text=True, timeout=60)


def data(root: Path) -> dict:
    return json.loads((root / "site/data/perf-512aa.json").read_text())


def held(root: Path) -> list:
    return json.loads((root / "perf/page_rows_pending.json").read_text())["rows"]


def ids(root: Path, cat: str) -> list[str]:
    block = data(root)[cat]
    return [r["id"] for r in (block if isinstance(block, list) else block["models"])]


def visible(root: Path, cat: str) -> list[str]:
    """The rows the page draws. A hidden row stays in the file and does not count."""
    block = data(root)[cat]
    rows = block if isinstance(block, list) else block["models"]
    return [r["id"] for r in rows if not r.get("hidden")]


def meta(root: Path) -> str:
    return META.search((root / "site/benchmarks/index.html").read_text())[1]


BLOCKED = {"status": "blocked", "reason": "synthetic, to give this test a row to hold"}


def make_held(root: Path, cat: str = "design", row_id: str = "pxdesign",
              cell: str = "b200") -> dict:
    """Blank one cell of a published row and let `--strip` hold it. Returns the cell it replaced,
    so a test that wants the row to become restorable can put it back."""
    doc_path = root / "site/data/perf-512aa.json"
    doc = json.loads(doc_path.read_text())
    block = doc[cat]
    rows = block if isinstance(block, list) else block["models"]
    row = next(r for r in rows if r["id"] == row_id)
    good = row["cells"][cell]
    row["cells"][cell] = dict(BLOCKED)
    doc_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    assert "held %s/%s" % (cat, row_id) in guard(root, "--strip").stdout
    return good


def test_the_repo_itself_passes(sandbox: Path):
    assert guard(sandbox).returncode == 0


def test_a_complete_held_row_is_published_and_both_counts_follow(sandbox: Path):
    good = make_held(sandbox)
    f = sandbox / "perf/page_rows_pending.json"
    pending = json.loads(f.read_text())
    assert [h["row"]["id"] for h in pending["rows"]] == ["pxdesign"]
    pending["rows"][0]["row"]["cells"]["b200"] = good      # the cell arriving is what unblocks it
    f.write_text(json.dumps(pending, indent=2) + "\n")

    assert "restored design/pxdesign" in guard(sandbox, "--restore").stdout
    assert ids(sandbox, "design") == ["boltzgen", "rfd3", "pxdesign"]
    assert held(sandbox) == []
    # The counts follow the rows the page DRAWS, not the rows in the file. rfd3 is hidden,
    # so restoring pxdesign takes the visible design count from one to two. The literal
    # "three" that stood here went stale the day rfd3 was hidden; the returncode-0 check
    # below is what actually pins the counts to the rows.
    assert visible(sandbox, "design") == ["boltzgen", "pxdesign"]
    assert data(sandbox)["subtitle"].startswith("Eight structure-prediction models, two binder")
    assert meta(sandbox).startswith("Eleven open biomolecular models")
    assert guard(sandbox).returncode == 0


def test_an_incomplete_held_row_stays_held(sandbox: Path):
    make_held(sandbox)
    assert guard(sandbox, "--restore").stdout.strip() == "nothing to move"
    assert [h["row"]["id"] for h in held(sandbox)] == ["pxdesign"]
    assert "pxdesign" not in ids(sandbox, "design")


def test_strip_never_holds_the_same_row_twice(sandbox: Path):
    """The failure that put six rows in a three-row pending file: strip, revert, strip."""
    doc_path = sandbox / "site/data/perf-512aa.json"
    make_held(sandbox)                                     # the first strip
    doc = data(sandbox)
    doc["design"]["models"].append(held(sandbox)[0]["row"])  # the revert that did the damage
    doc_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")

    assert "already held pending" in guard(sandbox, "--strip").stdout

    assert [h["row"]["id"] for h in held(sandbox)] == ["pxdesign"]
    assert "pxdesign" not in ids(sandbox, "design")


def test_restore_does_not_publish_a_row_that_is_already_on_the_page(sandbox: Path):
    f = sandbox / "perf/page_rows_pending.json"
    pending = json.loads(f.read_text())
    doc = data(sandbox)
    pending["rows"] = [{"category": "affinity", "row": doc["affinity"]["models"][0]}]
    f.write_text(json.dumps(pending, indent=2) + "\n")

    assert "dropped duplicate held affinity/nesso1" in guard(sandbox, "--restore").stdout
    assert ids(sandbox, "affinity") == ["nesso1"]


def test_a_drifted_meta_description_fails_the_guard(sandbox: Path):
    page = sandbox / "site/benchmarks/index.html"
    page.write_text(META.sub('<meta name="description" content="Ninety models.">',
                             page.read_text(), count=1))
    out = guard(sandbox)
    assert out.returncode == 1
    assert "meta description does not match the published rows" in out.stderr
    assert guard(sandbox, "--strip").returncode == 0
    assert guard(sandbox).returncode == 0
