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

table = subprocess.run([sys.executable, str(HERE / "matrix.py"), "--markdown",
                        "--group", "composition"],
                       capture_output=True, text=True, check=True).stdout.strip()
t = DOC.read_text()
i = t.index(MARK)
# Everything from the marker to the next "**Capability rejects" paragraph is generated.
j = t.index("**Capability rejects", i)
DOC.write_text(t[:i] + MARK + "\n\n" + table + "\n\n" + t[j:])
n_cells = len(re.findall(r"^\| \w", table, re.M))
print(f"state doc updated: {n_cells} composition rows")
