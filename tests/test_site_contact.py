"""The tt-bio.com contact endpoint's own tests (site-contact/contact.test.mjs), run under node.

They need node 22 for node:sqlite, which backs a D1-shaped shim so the rate-limit SQL runs for
real. Skipped where node is missing or older.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tests.test_perf_page_renders import _node

REPO = Path(__file__).resolve().parents[1]


def _major(node: str) -> int:
    return int(subprocess.run([node, "--version"], capture_output=True, text=True).stdout[1:].split(".")[0])


@pytest.mark.skipif(_node() is None or _major(_node()) < 22, reason="needs node >= 22")
def test_contact_endpoint():
    proc = subprocess.run([_node(), "--test", "site-contact/contact.test.mjs"], cwd=REPO,
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr
