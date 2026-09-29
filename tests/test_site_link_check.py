"""The published site's local links resolve, and the check refuses the links that once 404ed.

The logo on /benchmarks/ pointed at `../assets/` and the 404 page's "Overview" at `/`, the root
of moritztng.github.io. Both went live because nothing read an href. These tests run the real
check on the real site/ and on a copy carrying each of those links back.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "site_link_check.py"


def run(site: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), str(site)], capture_output=True, text=True)


def test_live_site_links_resolve():
    r = run(REPO / "site")
    assert r.returncode == 0, r.stderr


def test_publish_workflow_runs_it():
    assert "scripts/site_link_check.py" in (REPO / ".github/workflows/pages.yml").read_text()


@pytest.mark.parametrize("page,good,bad", [
    ("benchmarks/index.html", 'class="brand" href="../"', 'class="brand" href="../assets/"'),
    ("404.html", 'href="/tt-bio/">Overview', 'href="/">Overview'),
    ("index.html", 'href="benchmarks/#methods"', 'href="benchmarks/#no-such-id"'),
])
def test_refuses_a_broken_link(tmp_path: Path, page: str, good: str, bad: str):
    site = tmp_path / "site"
    shutil.copytree(REPO / "site", site)
    text = (site / page).read_text()
    assert good in text
    (site / page).write_text(text.replace(good, bad, 1))
    r = run(site)
    assert r.returncode == 1 and page in r.stderr, r.stderr
