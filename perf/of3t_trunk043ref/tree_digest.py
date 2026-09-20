#!/usr/bin/env python3
"""Pin a release tree by what it computes, not by what it calls itself (A24-AMENDMENT).

A digest pins the bytes, and for a reference the bytes ARE the function. `PKG-INFO` is a label a
build wrote; it agrees here and it is not what decides anything.

The whole-tree digest is sha256 over the sorted per-file sha256 of every `.py` under the package
root, hex strings concatenated with no separator. That rule is stated because a digest quoted
without its rule cannot be reproduced -- `of3t-auxheads043` published 8f035f4e for the same 0.4.3
sdist under a different rule, and neither number is wrong, they are answers to different
questions. The per-file digests below are the part that is rule-independent.
"""
import hashlib
import json
import sys
from pathlib import Path

# the files that decide the pairformer trunk's function between these two revisions
WITNESS = [
    "core/model/latent/base_blocks.py",
    "core/model/latent/pairformer.py",
    "core/model/layers/triangular_attention.py",
    "core/model/layers/attention_pair_bias.py",
    "core/model/primitives/attention.py",
    "entry_points/parameters.py",
]


def digest(root: Path) -> dict:
    files = sorted(root.rglob("*.py"))
    per = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    whole = hashlib.sha256("".join(sorted(per.values())).encode()).hexdigest()
    return {"root": str(root), "n_py_files": len(files), "tree_sha256": whole,
            "rule": "sha256 of the sorted per-file sha256 hex strings, concatenated, of every "
                    ".py under the package root",
            "witness_files": {w: per.get(w, "ABSENT") for w in WITNESS}}


if __name__ == "__main__":
    out = [digest(Path(a)) for a in sys.argv[1:]]
    print(json.dumps(out, indent=2))
