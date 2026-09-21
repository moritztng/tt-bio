#!/usr/bin/env python3
"""Which model modules import a shared triangle-attention block, enumerated from the code.

The campaign's supplied table missed BoltzGen (which reaches the block through an ALIAS) and this
row has now caught a missing model three times. So this stops reading lists: it walks every
`tt_bio/**/*.py` that is model code, resolves `from ...tenstorrent import X [as Y]` through the
IMPORTED name rather than the local one, and reports every module that pulls in a block the two
triangle-attention levers can reach.

Excluded: `_vendor/` and `*reference.py` (torch references, not device code).
"""
import ast
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
BLOCKS = {"TriangleAttention", "PairformerModule", "PairformerLayer", "Pairformer",
          "TriangleMultiplication", "Transition", "AttentionPairBias", "OuterProductMean"}

hits = defaultdict(set)
scanned = 0
for p in sorted((ROOT / "tt_bio").rglob("*.py")):
    s = str(p)
    if "/_vendor/" in s or p.name.endswith("reference.py") or "/tests/" in s:
        continue
    scanned += 1
    try:
        tree = ast.parse(p.read_text(errors="replace"))
    except SyntaxError:
        continue
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and "tenstorrent" in node.module:
            for al in node.names:
                if al.name in BLOCKS:
                    hits[str(p.relative_to(ROOT))].add(
                        al.name if al.asname is None else f"{al.name} as {al.asname}")

print(f"scanned {scanned} model modules under {ROOT}/tt_bio (excluding _vendor, *reference.py)\n")
if not hits:
    print("no module imports a shared block")
for f in sorted(hits):
    print(f"  {f:46s} {', '.join(sorted(hits[f]))}")

print("\npackages with NO shared-block import (levers cannot reach them by this route):")
pkgs = defaultdict(bool)
for p in sorted((ROOT / "tt_bio").rglob("*.py")):
    s = str(p)
    if "/_vendor/" in s or p.name.endswith("reference.py") or "/tests/" in s:
        continue
    rel = p.relative_to(ROOT)
    pkg = rel.parts[1] if len(rel.parts) > 2 else rel.name
    pkgs[pkg] |= str(rel) in hits
for pkg in sorted(k for k, v in pkgs.items() if not v):
    print(f"  {pkg}")
