#!/usr/bin/env python3
"""Undo the nested PairformerLayer route that a 3-way merge builds from two resolver generations.

Rows based on the pass-425 composition (59646c2bb) carry its route, `if not wide and trans_mask_z
is None:`. The current resolver writes `if not ops.taping() and trans_mask_z is None:`. Merging
such a row into a fresh composition does not conflict: git nests the old route inside the new
route's `else:`, so a taped, unmasked, not-wide step takes main's in-place block again, the one no
training arm verified (D264's routing half, reintroduced by the merge rather than by any row).

This keeps the outer route and the inner `else:` body, which is the tree of3t-ieatom scored
(06b91c5ce). compose_verify.sh then checks the engine equals that tree.

Usage: collapse_nested_pairformer_route.py <tenstorrent.py>  ->  0 collapsed, 2 nothing to do.
"""
import ast
import pathlib
import sys
import textwrap

p = pathlib.Path(sys.argv[1])
s = p.read_text()
inner = "        else:\n            if not wide and trans_mask_z is None:\n"
i = s.find(inner)
if i < 0 or "        if not ops.taping() and trans_mask_z is None:\n" not in s[:i]:
    sys.exit(2)
j = s.index("            else:\n", i + len(inner))
# the inner else body runs to the first line indented less than 16 spaces
body_start = j + len("            else:\n")
lines = s[body_start:].split("\n")
n = 0
for n, ln in enumerate(lines):
    if ln.strip() and not ln.startswith(" " * 16):
        break
body = "\n".join(lines[:n])
out = s[:i] + "        else:\n" + textwrap.indent(textwrap.dedent(body), " " * 12).rstrip(" ") \
    + "\n" + "\n".join(lines[n:])
ast.parse(out)
if "if not wide and trans_mask_z is None:" in out:
    sys.exit(2)
p.write_text(out)
