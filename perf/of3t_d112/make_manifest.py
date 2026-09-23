#!/usr/bin/env python3
"""Manifest the campaign's 0.4.3 / 0.5.0 upstream references: what, where, and the hash.

A reference that only exists as a directory somebody remembers is what D112 is. The location
stops a prune; the manifest is what makes a REBUILD checkable, which is the part that survives
losing every copy. So this writes, for each entry, the digest under a stated rule and the
provenance of the number it has to agree with.
"""
import hashlib
import json
import platform
import socket
import subprocess
import sys
from pathlib import Path

TREE_RULE = ("sha256 of the sorted per-file sha256 hex strings, concatenated, of every .py "
             "under the package root (of3t-trunk043ref/tree_digest.py, A24-AMENDMENT)")


def sha256_file(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def tree_digest(root: Path):
    per = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
           for p in sorted(root.rglob("*.py"))}
    return len(per), hashlib.sha256("".join(sorted(per.values())).encode()).hexdigest()


def main(root: Path, expect: Path | None):
    known = json.loads(expect.read_text()) if expect and expect.exists() else {}
    out = {"host": socket.gethostname(), "root": str(root),
           "python": platform.python_version(),
           "tree_digest_rule": TREE_RULE, "packages": [], "files": []}
    for name in ("of3pkg043", "of3pkg050"):
        pkg = root / name / "openfold3"
        if not pkg.is_dir():
            continue
        n, d = tree_digest(pkg)
        rec = {"name": name, "package_root": str(pkg), "n_py_files": n, "tree_sha256": d}
        if name in known:
            rec["expected"] = known[name]
            rec["agrees"] = d.startswith(known[name])
        out["packages"].append(rec)
    for p in sorted(root.rglob("*")):
        if not p.is_file() or "openfold3" in p.parts or p.name == "MANIFEST.json":
            continue
        out["files"].append({"path": str(p.relative_to(root)), "bytes": p.stat().st_size,
                             "sha256": sha256_file(p)})
    (root / "MANIFEST.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"packages": out["packages"], "n_files": len(out["files"])}, indent=1))
    bad = [p["name"] for p in out["packages"] if p.get("agrees") is False]
    if bad:
        print(f"DIGEST DISAGREES with the recorded value: {bad}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    e = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    sys.exit(main(Path(sys.argv[1]), e))
