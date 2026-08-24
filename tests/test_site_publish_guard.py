"""The publish gate moves whole rows, and both count-bearing strings follow.

Two defects this file pins down, both found live on main. `perf/page_rows_pending.json` held
each of the three stripped rows twice, because a `--strip` run against a data file that had
been reverted after an earlier strip appended them again; a later `--restore` would have
published every one of them twice. And the strip regenerated the JSON subtitle while the
page's own meta description kept claiming eighteen models on a page drawing fifteen, because
nothing owned that string.

Each test runs the real script against a copy of the real page, so it exercises the CLI and
the file writes rather than a reimplementation of them.
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


def meta(root: Path) -> str:
    return META.search((root / "site/benchmarks/index.html").read_text())[1]


def test_the_repo_itself_passes(sandbox: Path):
    assert guard(sandbox).returncode == 0


def test_a_complete_held_row_is_published_and_both_counts_follow(sandbox: Path):
    f = sandbox / "perf/page_rows_pending.json"
    pending = json.loads(f.read_text())
    row = pending["rows"][0]
    assert row["row"]["id"] == "pxdesign", "this test assumes PXDesign is the held row"
    row["row"]["cells"]["b200"] = {"status": "measured", "s_per_design": 99.0, "ref": "synthetic"}
    f.write_text(json.dumps(pending, indent=2) + "\n")

    assert "restored design/pxdesign" in guard(sandbox, "--restore").stdout
    assert ids(sandbox, "design") == ["boltzgen", "rfd3", "pxdesign"]
    assert held(sandbox) == []
    assert data(sandbox)["subtitle"].startswith("Eight structure-prediction models, three binder")
    assert meta(sandbox).startswith("Eighteen open biomolecular models")
    assert guard(sandbox).returncode == 0


def test_an_incomplete_held_row_stays_held(sandbox: Path):
    assert guard(sandbox, "--restore").stdout.strip() == "nothing to move"
    assert [h["row"]["id"] for h in held(sandbox)] == ["pxdesign"]
    assert "pxdesign" not in ids(sandbox, "design")


def test_strip_never_holds_the_same_row_twice(sandbox: Path):
    """The failure that put six rows in a three-row pending file: strip, revert, strip."""
    doc_path = sandbox / "site/data/perf-512aa.json"
    pending_path = sandbox / "perf/page_rows_pending.json"
    doc = data(sandbox)
    doc["design"]["models"].append(json.loads(pending_path.read_text())["rows"][0]["row"])
    doc_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    before = doc_path.read_text()
    pending_path.write_text(json.dumps({"rows": []}) + "\n")

    assert "held design/pxdesign" in guard(sandbox, "--strip").stdout
    doc_path.write_text(before)  # the revert that did the damage
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
    assert "meta description does not match the published row counts" in out.stderr
    assert guard(sandbox, "--strip").returncode == 0
    assert guard(sandbox).returncode == 0
