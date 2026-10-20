"""A device test that calls `ttnn.open_device` itself must ask tt_bio for the p300 descriptor first.

A lone p300 chip is a CUSTOM topology to tt-metal, and a bare open dies at once with "Custom
fabric mesh graph descriptor path must be specified for CUSTOM cluster type". The CLI sets the
descriptor per worker; a test fixture that opens the card directly inherits nothing. Two such
fixtures passed every p150a and Wormhole run and errored on the first p300c shard that drew them
(gate rel-ea58ac531, 2026-10-20).
"""
from __future__ import annotations

import re
from pathlib import Path

TESTS = Path(__file__).parent
OPEN = re.compile(r"^\s*\w+\s*=\s*ttnn\.open_device\(", re.M)
SETS = ("ensure_p300_mesh_descriptor()", "TT_MESH_GRAPH_DESC_PATH")


def test_every_direct_device_open_sets_the_descriptor_first():
    bare = []
    for f in sorted(TESTS.glob("test_*.py")):
        src = f.read_text()
        m = OPEN.search(src)
        if not m:
            continue
        first = min((src.find(s) for s in SETS if s in src), default=-1)
        if first < 0 or first > m.start():
            bare.append(f.name)
    assert not bare, f"these open a card without the p300 descriptor and error on a p300c: {bare}"
