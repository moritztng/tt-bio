#!/usr/bin/env python3
"""Resolve a `__slots__` conflict by unioning the entries, and refuse the property collision.

Two rows add a slot to the same class and conflict on one line. The union is almost always
right: slots are a declaration of storage, not a policy, and each side added one because its own
code assigns it. What is NOT right is a blind union, because a slot and a property cannot share
a name -- the class body's property descriptor and the slot descriptor collide, and the one
defined later silently wins.

That is live here, not hypothetical. D126 renamed `autograd.Tensor`'s `value` slot to `_value`
and put a `value` PROPERTY over it so the setter can re-key `_PARAMS`; `of3t-gradients` is based
on a main from before that and still declares a plain `value` slot. Unioning the names would
reintroduce `value` as a slot alongside the property. So: if a side's entry has a `_`-prefixed
twin on the other side AND the class defines a property of the bare name, the bare name is
dropped and the underscore form kept, which is the form the property reads.

Refuses with exit 2 on anything that is not one `__slots__` tuple against another.

Usage: resolve_slots_union.py <file> <their-ref>  ->  0 resolved, 2 not this shape.
"""
from __future__ import annotations

import ast
import pathlib
import re
import sys

SLOTS = re.compile(r'^(\s*)__slots__\s*=\s*\((.*)\)\s*$')


def _entries(line: str):
    """(indent, [names]) for a one-line `__slots__` tuple, or None."""
    m = SLOTS.match(line.rstrip("\n"))
    if not m:
        return None
    try:
        names = ast.literal_eval("(" + m.group(2) + ")")
    except (ValueError, SyntaxError):
        return None
    if not isinstance(names, tuple) or not all(isinstance(n, str) for n in names):
        return None
    return m.group(1), list(names)


def _properties(text: str, at: int) -> set:
    """Names defined as a property in the class the hunk at offset `at` belongs to.

    Read from the file with the conflict blanked out, so the other side's lines cannot make it
    unparseable. If it will not parse at all, return the empty set and let the collision check
    fall through to the `_`-twin rule alone.
    """
    blanked = re.sub(r"^(<<<<<<< |=======$|>>>>>>> ).*$", "", text, flags=re.M)
    try:
        tree = ast.parse(blanked)
    except SyntaxError:
        return set()
    line = text[:at].count("\n") + 1
    best, out = None, set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ClassDef) and n.lineno <= line:
            if best is None or n.lineno > best.lineno:
                best = n
    if best is None:
        return out
    for n in best.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for d in n.decorator_list:
                name = d.attr if isinstance(d, ast.Attribute) else getattr(d, "id", None)
                if name in ("property", "setter", "cached_property"):
                    out.add(n.name)
    return out


def resolve(path: pathlib.Path, theirs_ref: str) -> int:
    s = path.read_text()
    start, mid, end = "<<<<<<< HEAD\n", "=======\n", f">>>>>>> {theirs_ref}\n"
    pos = 0
    while True:
        i = s.find(start, pos)
        if i < 0:
            return 2
        j, k = s.find(mid, i), -1
        if j >= 0:
            k = s.find(end, j)
        if j < 0 or k < 0:
            return 2
        pos = k + len(end)
        ours, theirs = s[i + len(start):j], s[j + len(mid):k]
        a, b = _entries(ours), _entries(theirs)
        if a is None or b is None:
            continue                      # not this hunk; the caller's chain tries the others
        indent, head = a
        merged = list(head) + [n for n in b[1] if n not in head]
        props = _properties(s, i)
        dropped = [n for n in merged if not n.startswith("_")
                   and ("_" + n) in merged and n in props]
        if dropped:
            merged = [n for n in merged if n not in dropped]
        body = indent + "__slots__ = (" + ", ".join(f'"{n}"' for n in merged) + ")\n"
        if len(body) > 100:               # the file's own line budget; a wrap needs a human
            print(f"  {path}: the unioned __slots__ line is {len(body)} chars -- too long to "
                  f"write mechanically", file=sys.stderr)
            return 2
        new = s[:i] + body + s[k + len(end):]
        path.write_text(new)
        note = f", dropped {', '.join(dropped)} (a property of that name exists)" if dropped else ""
        print(f"  {path.name}: __slots__ unioned to {len(merged)} entries{note}")
        return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        raise SystemExit(1)
    raise SystemExit(resolve(pathlib.Path(sys.argv[1]), sys.argv[2]))
