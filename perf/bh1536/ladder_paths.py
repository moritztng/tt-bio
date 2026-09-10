#!/usr/bin/env python3
"""Where one ladder's evidence lands, so the same harness can walk two cards.

The ladder was written for qb1 card 0 (p150a) and is now also walked on qb2 card 1 (p300c).
`size_limits.py` treats the two boards as the same DRAM shape, which makes a p150a ceiling a
prediction for p300c and not a result -- so the second card needs its own evidence file, not a
second copy of the harness. `--out_tag p300c` writes `results.p300c.jsonl` /
`sweep.p300c.log`; no tag keeps the original names.
"""
import os
import socket
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def tag_from_env(explicit: str = "") -> str:
    return explicit or os.environ.get("TT_BIO_LADDER_TAG", "")


def _suffixed(stem: str, ext: str, tag: str) -> Path:
    return ROOT / (f"{stem}.{tag}{ext}" if tag else f"{stem}{ext}")


def results_path(tag: str = "") -> Path:
    return _suffixed("results", ".jsonl", tag)


def sweep_path(tag: str = "") -> Path:
    return _suffixed("sweep", ".log", tag)


def runs_dir(tag: str = "") -> Path:
    return ROOT / "runs" / (tag or "default")


def where() -> dict:
    """Host identity recorded with every row: a ceiling belongs to a card, not to a repo."""
    return {"host": socket.gethostname()}
