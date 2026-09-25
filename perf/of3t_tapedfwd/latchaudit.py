#!/usr/bin/env python3
"""Every refusal-cache write in the engine, classified by whether it can cache a DECLINE.

The recorded defect this row was sent to re-check -- one taped call switching fused triangle
attention off for the whole process -- is a general shape, not one bug: a memo set that
conflates a DEVICE REFUSAL (permanent, cacheable) with a CALL-STATE DECLINE (this tape, this
mode) lets one call retire a config for the rest of the process. A device refusal arrives as
an exception; a decline arrives as a None. So the classification is lexical and checkable:
is the `.add()` inside an `except` handler or not?

The four that are not are not automatically defects -- a config that came back None for a key
that already carries the shapes and the dtype is a function of the key, not of the caller --
but they are the four a reviewer has to read, and naming them is the point.

No card, no imports of the package, AST only.

    python3 perf/of3t_tapedfwd/latchaudit.py --json perf/of3t_tapedfwd/out/LATCHES.json
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FILES = ("tt_bio/tenstorrent.py", "tt_bio/triatt_sdpa.py", "tt_bio/esmc.py")


def audit():
    rows = []
    for rel in FILES:
        tree = ast.parse((REPO / rel).read_text())
        spans = [(h.lineno, max(getattr(x, "end_lineno", h.lineno) for x in ast.walk(h)))
                 for h in ast.walk(tree) if isinstance(h, ast.ExceptHandler)]
        for n in ast.walk(tree):
            if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and n.func.attr == "add" and isinstance(n.func.value, ast.Name)
                    and ("OVER_L1" in n.func.value.id or "REFUSED" in n.func.value.id)):
                rows.append({"file": rel, "line": n.lineno, "set": n.func.value.id,
                             "in_except": any(a <= n.lineno <= b for a, b in spans)})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    rows = audit()
    out = {"doc": "refusal-cache writes, split by device refusal vs call-state decline",
           "total": len(rows),
           "in_except_cacheable": sum(1 for r in rows if r["in_except"]),
           "not_in_except_must_be_read": [r for r in rows if not r["in_except"]],
           "all": rows}
    print(json.dumps(out, indent=1))
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
