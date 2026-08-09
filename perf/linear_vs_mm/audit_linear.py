#!/usr/bin/env python3
"""Enumerate every ttnn.linear( call site in tt_bio/, whether it passes core_grid, and the
enclosing class/function. Pure source analysis, no device."""
import io, os, re, sys, tokenize

ROOT = sys.argv[1] if len(sys.argv) > 1 else "tt_bio"


def enclosing(lines, idx):
    """Walk backwards for the nearest def/class at lower indent."""
    line = lines[idx]
    ind = len(line) - len(line.lstrip())
    fn = cls = None
    for j in range(idx - 1, -1, -1):
        l = lines[j]
        if not l.strip():
            continue
        i = len(l) - len(l.lstrip())
        if i < ind:
            m = re.match(r"\s*def\s+(\w+)", l)
            if m and fn is None:
                fn, ind = m.group(1), i
                continue
            m = re.match(r"\s*class\s+(\w+)", l)
            if m:
                cls = m.group(1)
                break
            ind = i
    return cls, fn


def call_text(src, start):
    """Return the full text of the call starting at index of '(' in src."""
    depth, i = 0, start
    while i < len(src):
        if src[i] == "(":
            depth += 1
        elif src[i] == ")":
            depth -= 1
            if depth == 0:
                return src[start : i + 1]
        i += 1
    return src[start : start + 400]


rows = []
for dirpath, _, files in os.walk(ROOT):
    for f in sorted(files):
        if not f.endswith(".py"):
            continue
        p = os.path.join(dirpath, f)
        src = open(p, encoding="utf-8").read()
        lines = src.split("\n")
        for m in re.finditer(r"ttnn\.linear\s*\(", src):
            lno = src.count("\n", 0, m.start()) + 1
            txt = call_text(src, m.end() - 1)
            flat = re.sub(r"\s+", " ", txt)
            cg = "core_grid" in flat
            cls, fn = enclosing(lines, lno - 1)
            rows.append((p, lno, cls, fn, cg, flat[:220]))

print(f"total ttnn.linear call sites: {len(rows)}")
print(f"with explicit core_grid:     {sum(1 for r in rows if r[4])}")
print(f"without core_grid:           {sum(1 for r in rows if not r[4])}")
print()
byfile = {}
for r in rows:
    byfile.setdefault(r[0], []).append(r)
for p, rs in byfile.items():
    print(f"### {p}  ({len(rs)} sites, {sum(1 for r in rs if r[4])} with core_grid)")
    for p_, lno, cls, fn, cg, flat in rs:
        print(f"  {lno:5d}  cg={'Y' if cg else 'n'}  {cls}.{fn}")
        print(f"         {flat}")
    print()
