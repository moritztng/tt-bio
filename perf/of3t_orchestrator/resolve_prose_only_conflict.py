#!/usr/bin/env python3
"""Resolve a merge conflict whose two sides differ only in PROSE -- comments and docstrings.

Rows based on an older `wk/of3t` reflow a comment, or carry an earlier wording of a docstring
that main has since sharpened. `of3t-crop768` conflicts with main in four hunks across two files
and every one of them is a rewrapped comment or a superseded docstring sentence; the code is
character-identical. That is not a disagreement and it should not stop the composition.

The rule is HEAD's prose, and the safety is that it is CHECKED rather than argued: both sides are
reconstructed in full, parsed, stripped of every docstring, and compared as ASTs. Comments never
enter an AST, so the comparison sees exactly the thing that matters. **If the two sides differ
anywhere in executable code, the ASTs differ and this refuses** -- which is the right outcome,
because then it is a real disagreement and the caller must stop.

HEAD wins the prose for the same reason it wins a kwarg collision: HEAD is main, main ships, and
a row's superseded wording is not worth a merge conflict. The row keeps its prose on its own
branch; nothing is lost that a reader of `wk/of3t` needed.

Usage: resolve_prose_only_conflict.py <file> <their-ref>  ->  0 resolved, 2 not this shape.
"""
from __future__ import annotations

import ast
import pathlib
import re
import sys


def sides(text: str, theirs_ref: str):
    """(ours, theirs) with every conflict hunk resolved to that side, or None if unbalanced."""
    pat = re.compile(r"^<<<<<<< HEAD\n(.*?)^=======\n(.*?)^>>>>>>> "
                     + re.escape(theirs_ref) + r"\n", re.S | re.M)
    hunks = list(pat.finditer(text))
    if not hunks:
        return None
    if "<<<<<<<" in pat.sub("", text) or ">>>>>>>" in pat.sub("", text):
        return None                                   # a marker this pattern did not consume
    return pat.sub(lambda m: m.group(1), text), pat.sub(lambda m: m.group(2), text), len(hunks)


def skeleton(src: str):
    """The module's AST with every docstring removed, as a comparable string, or None."""
    try:
        mod = ast.parse(src)
    except SyntaxError:
        return None
    for node in ast.walk(mod):
        body = getattr(node, "body", None)
        if not isinstance(body, list) or not body:
            continue
        if not isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        first = body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) \
                and isinstance(first.value.value, str):
            body.pop(0)
            if not body:
                body.append(ast.Pass())
    return ast.dump(ast.fix_missing_locations(mod), include_attributes=False)


def resolve(path: pathlib.Path, theirs_ref: str) -> int:
    text = path.read_text()
    got = sides(text, theirs_ref)
    if got is None:
        return 2
    ours, theirs, n = got
    a, b = skeleton(ours), skeleton(theirs)
    if a is None or b is None:
        print(f"  {path}: one side does not parse on its own", file=sys.stderr)
        return 2
    if a != b:
        return 2                                      # a real code difference; not this rule
    path.write_text(ours)
    print(f"  {path.name}: {n} prose-only hunk(s), HEAD's wording kept -- the two sides' code "
          f"is identical after stripping docstrings")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        raise SystemExit(1)
    raise SystemExit(resolve(pathlib.Path(sys.argv[1]), sys.argv[2]))
