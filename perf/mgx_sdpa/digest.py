"""sha256 of every output file a run wrote, keyed by path relative to its out_dir.

    python3 perf/mgx_sdpa/digest.py <label> <out_dir> >> digests.jsonl

Two trees that should agree bit for bit are compared by running the same CLI into two out_dirs
and diffing these rows. Timings and logs are left out, since they differ run to run.
"""
import hashlib
import json
import sys
from pathlib import Path

SKIP = {".log", ".txt"}
label, root = sys.argv[1], Path(sys.argv[2])
files = {}
for p in sorted(root.rglob("*")):
    if p.is_file() and p.suffix not in SKIP and p.name not in ("results.json", "designs.json"):
        files[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
print(json.dumps({"label": label, "out_dir": str(root), "files": files}))
