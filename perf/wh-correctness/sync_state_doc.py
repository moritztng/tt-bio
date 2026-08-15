#!/usr/bin/env python3
"""Drop the generated composition matrix into the state doc, in place of the marker.

The table in the deliverable is the JSONL and nothing else. Hand-transcribing it once is
a typo; hand-transcribing it after every re-run is a stale cell nobody notices. This
replaces everything between the marker and the next blank-line-delimited paragraph.
"""
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOC = Path("/home/moritz/.coworker/state/wh-correctness-input-sweep.md")
MARK = "<!-- MATRIX-4A -->"

MARK_4B = "<!-- MATRIX-4B -->"


def gen(group: str) -> str:
    return subprocess.run([sys.executable, str(HERE / "matrix.py"), "--markdown",
                           "--group", group],
                          capture_output=True, text=True, check=True).stdout.strip()


def splice(text: str, mark: str, table: str, end_anchor: str) -> str:
    i = text.index(mark)
    j = text.index(end_anchor, i)
    return text[:i] + mark + "\n\n" + table + "\n\n" + text[j:]


t = DOC.read_text()
comp = gen("composition")
t = splice(t, MARK, comp, "**Capability rejects")
if MARK_4B in t:
    t = splice(t, MARK_4B, gen("size"), "**Over-cap rejects")
DOC.write_text(t)
print(f"state doc updated: {len(re.findall(r'^[|] [a-z]', comp, re.M))} composition rows"
      + (", size table too" if MARK_4B in t else ""))
