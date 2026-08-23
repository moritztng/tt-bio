"""The benchmarks page actually draws every row site/data/perf-512aa.json carries.

`site/benchmarks/index.html` is plain JS with no build step, so nothing checked that a new
data block reached the page. It cost a real defect: adding the `affinity` category left a
`design`-only variable behind in `statics()` and the page died on load with
``ReferenceError: design is not defined`` -- every chart blank, no test red. This runs the
page's own script against the real JSON under a small DOM stub and asserts the rows land in
the tables, the charts and the scope and methods lists.

Skipped where node is not installed. It is a lint, not a browser: it proves the render path
executes and the numbers reach the DOM, not that the SVG looks right.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PAGE = REPO / "site" / "benchmarks" / "index.html"
DATA = REPO / "site" / "data" / "perf-512aa.json"
HARNESS = REPO / "site" / "benchmarks" / "render_check.js"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_page_renders_every_data_row():
    proc = subprocess.run(["node", str(HARNESS)], cwd=REPO, capture_output=True, text=True,
                          timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_every_category_block_is_shaped_like_the_others():
    """A category the renderer draws needs cond, note, methods and models with cells.

    CATEGORIES in the page names the blocks it reads; a block that is present but missing a
    key renders an empty paragraph rather than failing, so check the data side here.
    """
    d = json.loads(DATA.read_text())
    page = PAGE.read_text()
    keys = [line.split('key: "')[1].split('"')[0]
            for line in page.splitlines() if "key: \"" in line and "label:" in line]
    assert keys, "CATEGORIES not found in the page"
    for k in keys:
        block = d.get(k)
        assert block, f"page draws category {k!r} but the data has no such block"
        for field in ("cond", "note", "methods", "models"):
            assert block.get(field), f"{k}.{field} missing"
        for m in block["models"]:
            for field in ("id", "name", "target", "batch", "cells"):
                assert field in m, f"{k} model {m.get('id')} missing {field}"
            for cell, c in m["cells"].items():
                if c.get("status") == "measured":
                    assert "s_per_fold" in c or "s_per_design" in c, \
                        f"{k}/{m['id']}/{cell} measured with no seconds"
                    assert c.get("ref"), f"{k}/{m['id']}/{cell} measured with no provenance"
                else:
                    assert c.get("detail") or c.get("reason"), \
                        f"{k}/{m['id']}/{cell} is not measured and says nothing about why"
