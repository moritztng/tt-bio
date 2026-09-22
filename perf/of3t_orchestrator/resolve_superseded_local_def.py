#!/usr/bin/env python3
"""Take HEAD when a row's LOCAL copy of a function has been superseded by a unified one on main.

`of3t-rebase` added `sample_ranking_score` to `tt_bio/openfold3_fold.py` and called it. Main has
since landed the unified rule -- `7848e2d17 rank: one sample-ranking rule for the AF3-lineage
models`, plus `6c7ea5661` and `ae1d90ee0` refining it to rank on the full-precision score -- so
HEAD has no local def and calls `rank.ranking_score` instead. Two hunks, one change, and the
campaign's standing rule is UNIFIED NEVER PER-MODEL: HEAD wins, both times.

The resolution is to take HEAD everywhere in the file, which is only safe if the row's name
really is gone rather than half-removed. So it is checked rather than argued:

  * every conflict hunk is resolved to HEAD;
  * the result parses;
  * each name that ONLY the row's side defined has **zero** remaining references in the file --
    a dangling call is how "take HEAD" silently produces a NameError at fold time;
  * and the row's side must actually have defined something HEAD does not, otherwise this is
    not a superseded-definition conflict and some other rule owns it.

Refuses with exit 2 on anything else. Usage:
  resolve_superseded_local_def.py <file> <their-ref>   ->  0 resolved, 2 not this shape.
"""
from __future__ import annotations

import ast
import pathlib
import re
import sys


def _sides(s, start, mid, end):
    out, pos = [], 0
    while True:
        i = s.find(start, pos)
        if i < 0:
            return out
        j = s.find(mid, i)
        k = s.find(end, j) if j >= 0 else -1
        if j < 0 or k < 0:
            return out
        out.append((i, j, k))
        pos = k + len(end)


def _defs(chunks):
    """Names defined across a list of conflict-hunk sides.

    Each side is parsed ON ITS OWN and dedented. Concatenating them does not parse -- one hunk
    is a whole `def`, the next is an indented fragment of a call -- and a SyntaxError here
    silently returns "defines nothing", which made this resolver refuse the very conflict it was
    written for.
    """
    import textwrap
    out = set()
    for c in chunks:
        try:
            tree = ast.parse(textwrap.dedent(c))
        except SyntaxError:
            continue
        out |= {n.name for n in ast.walk(tree)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    return out


def resolve(path: pathlib.Path, theirs_ref: str) -> int:
    s = path.read_text()
    start, mid, end = "<<<<<<< HEAD\n", "=======\n", f">>>>>>> {theirs_ref}\n"
    hunks = _sides(s, start, mid, end)
    if not hunks:
        return 2

    only_theirs = (_defs([s[j + len(mid):k] for _i, j, k in hunks])
                   - _defs([s[i + len(start):j] for i, j, _k in hunks]))
    if not only_theirs:
        return 2                        # not a superseded-definition conflict

    out, pos = [], 0
    for i, j, k in hunks:
        out.append(s[pos:i])
        out.append(s[i + len(start):j])          # HEAD
        pos = k + len(end)
    out.append(s[pos:])
    new = "".join(out)
    if "<<<<<<<" in new or ">>>>>>>" in new:
        return 2
    try:
        ast.parse(new)
    except SyntaxError as e:
        print(f"  {path}: taking HEAD does not parse -- {e}", file=sys.stderr)
        return 1
    for name in sorted(only_theirs):
        if re.search(rf"\b{re.escape(name)}\b", new):
            print(f"  {path}: HEAD drops `{name}` but the file still references it -- taking "
                  f"HEAD would leave a dangling call, so this is not a clean supersession",
                  file=sys.stderr)
            return 2
    path.write_text(new)
    print(f"  {path.name}: {len(hunks)} hunk(s) resolved to HEAD -- "
          f"{', '.join(sorted(only_theirs))} is the row's local copy of something main has since "
          f"unified, and no reference to it survives")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        raise SystemExit(1)
    raise SystemExit(resolve(pathlib.Path(sys.argv[1]), sys.argv[2]))
