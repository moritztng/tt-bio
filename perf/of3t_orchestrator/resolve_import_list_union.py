#!/usr/bin/env python3
"""Resolve a conflict that sits inside one parenthesised `from X import (...)` list.

main keeps landing imports into the same lists the rows extend (pass 420: MGX added five names to
`protenix.py`'s `from .tenstorrent import (...)`, of3t-softmax added `softmax_ckc`). Both sides
must be nothing but identifiers and commas, optionally closed by `)`. The resolution is HEAD's
names in HEAD's order, then the row's extra names, closed the way both sides close. The file must
parse afterwards, so a list that is not what it looks like refuses.

Usage: resolve_import_list_union.py <file>  ->  0 resolved, 2 not this shape (caller must stop).
"""
import ast
import pathlib
import re
import sys

p = pathlib.Path(sys.argv[1])
s = p.read_text()
m = re.search(r"<<<<<<< [^\n]*\n(.*?)=======\n(.*?)>>>>>>> [^\n]*\n", s, re.S)
if not m:
    sys.exit(2)
IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\s+as\s+[A-Za-z_][A-Za-z0-9_]*)?$")


def names(block):
    body = block.rstrip()
    closed = body.endswith(")")
    body = body[:-1] if closed else body
    out = [t.strip() for t in body.replace("\n", " ").split(",") if t.strip()]
    if not out or not all(IDENT.match(t) for t in out):
        return None, None
    return out, closed


head, h_closed = names(m.group(1))
row, r_closed = names(m.group(2))
if head is None or row is None or h_closed != r_closed:
    sys.exit(2)
# The line before the conflict must be inside an open `import (`.
before = s[:m.start()]
if before.rfind("import (") < before.rfind(")"):
    sys.exit(2)
merged = head + [n for n in row if n not in head]
indent = re.match(r"(\s*)", m.group(1)).group(1)
lines, cur = [], indent
for n in merged:
    piece = n + ", "
    if len(cur) + len(piece) > 99 and cur.strip():
        lines.append(cur.rstrip())
        cur = indent
    cur += piece
cur = cur.rstrip().rstrip(",") + (")" if h_closed else ",")
lines.append(cur)
new = s[:m.start()] + "\n".join(lines) + "\n" + s[m.end():]
if "<<<<<<< " not in new:
    try:
        ast.parse(new)
    except SyntaxError:
        sys.exit(2)
p.write_text(new)
