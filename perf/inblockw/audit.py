#!/usr/bin/env python3
"""Inventory every `ttnn.linear(` in tt_bio/ and whether it passes `core_grid=`.

A line grep miscounts: these calls span up to nine lines. This does a balanced-paren scan
from each match, so a call is classified on its whole argument list.
"""
import collections
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "tt_bio")
per = collections.Counter()
per_cg = collections.Counter()
sites = collections.defaultdict(list)
total = with_cg = 0

for f in sorted(root.rglob("*.py")):
    src = f.read_text()
    for m in re.finditer(r"ttnn\.linear\(", src):
        i, depth = m.end(), 1
        while i < len(src) and depth:
            depth += (src[i] == "(") - (src[i] == ")")
            i += 1
        call = src[m.start():i]
        line = src[:m.start()].count("\n") + 1
        total += 1
        per[str(f)] += 1
        if "core_grid" in call:
            with_cg += 1
            per_cg[str(f)] += 1
            sites[str(f)].append(line)

print(f"total ttnn.linear: {total}   with core_grid: {with_cg}")
for k in sorted(per):
    print(f"  {k:44s} {per[k]:3d} sites, {per_cg[k]:3d} with core_grid")
print()
for k in sorted(sites):
    print(f"{k}: {sites[k]}")
