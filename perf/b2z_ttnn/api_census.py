#!/usr/bin/env python3
"""Which of the ttnn symbols tt-bio actually calls still exist on the installed stack.

Static census of `ttnn.<attr>` over tt_bio/, resolved against the live module. Existence only --
a symbol that survived with a changed signature still shows PRESENT here and has to be caught by
running the fold. Run it once per venv and diff.
"""
from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PAT = re.compile(r"\bttnn\.([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)")


def census() -> dict[str, int]:
    counts: dict[str, int] = {}
    for p in (REPO / "tt_bio").rglob("*.py"):
        for m in PAT.finditer(p.read_text(errors="replace")):
            counts[m.group(1)] = counts.get(m.group(1), 0) + 1
    return counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    import ttnn

    counts = census()
    missing, present = {}, {}
    for name, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        obj, ok = ttnn, True
        for part in name.split("."):
            if not hasattr(obj, part):
                ok = False
                break
            obj = getattr(obj, part)
        (present if ok else missing)[name] = n

    res = {"ttnn": md.version("ttnn"), "python": sys.executable,
           "n_symbols": len(counts), "n_missing": len(missing),
           "missing": missing, "present_count": len(present)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("ttnn", "n_symbols", "n_missing", "missing")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
