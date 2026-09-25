#!/usr/bin/env python3
"""sha256 over the ATOM/HETATM records of every CIF under a results dir, one line per file.
Header blocks carry dates and versions, the atom records carry the structure."""
import hashlib, sys
from pathlib import Path
for d in sys.argv[1:]:
    for f in sorted(Path(d).rglob("*.cif")):
        h = hashlib.sha256()
        for line in f.read_text().splitlines():
            if line.startswith(("ATOM", "HETATM")):
                h.update(line.encode() + b"\n")
        print(h.hexdigest()[:16], f.relative_to(d))
