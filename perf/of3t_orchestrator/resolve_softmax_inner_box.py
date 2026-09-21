#!/usr/bin/env python3
"""Resolve the softmax-backward conflict between the BOX memory policy and the SHARED helper.

Two independent repairs landed on the same three lines and neither knows about the other.

  * the BOX policy re-reads the softmax output inside the closure -- `y = box[0]` -- so `free`
    may evict it to DRAM and write the new handle back into the box, instead of pinning it in
    L1 with `evictable = False`;
  * D56 replaced the inlined `sum_j g_j y_j` (and its `TT_BIO_SOFTMAX_BW_RENORM` branch) with
    the shared `softmax_bw_inner` helper, because *"a repair applied to one of two identical
    expressions is the kind of half-fix that reads as fixed"*.

They compose: read the handle, then call the helper. Taking either side alone is wrong and one
way is worse than wrong -- with the box on the surviving lines outside the hunk and the helper
line taken from the other side, `y` is never bound and the backward raises NameError the first
time it runs, which no collection-only compose check can see.

This replaces `resolve_d116_softmax_inner.py`'s code half, which matched one literal hunk in one
orientation. The orientation flipped the moment D56 landed on main: at d116 the BOX was HEAD and
the helper was the row, and for every row based on a pre-D56 main it is the other way round. A
literal is a new file per row; the shape is the same every time.

Refuses, with exit 2, anything that is not exactly this shape -- the caller then stops the
compose, which is the right outcome for a real disagreement.

Usage: resolve_softmax_inner_box.py <file> <their-ref>  ->  0 resolved, 2 not this shape.
"""
from __future__ import annotations

import ast
import pathlib
import re
import sys
import textwrap

HELPER = "softmax_bw_inner"


def _stmts(text: str):
    """Parse a conflict side as a statement list, or None. Comments are kept separately."""
    body = textwrap.dedent(text)
    try:
        return ast.parse(body).body
    except SyntaxError:
        return None


def _is_box_read(node) -> bool:
    """`y = box[0]` -- a single Name target assigned one subscript of a Name called `box`."""
    return (isinstance(node, ast.Assign) and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Subscript)
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == "box")


def _helper_target(nodes):
    """The name assigned from a `softmax_bw_inner(...)` call, or None."""
    for n in nodes:
        if not (isinstance(n, ast.Assign) and len(n.targets) == 1
                and isinstance(n.targets[0], ast.Name) and isinstance(n.value, ast.Call)):
            continue
        f = n.value.func
        name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
        if name == HELPER:
            return n.targets[0].id
    return None


def _inline_target(nodes):
    """The name the inlined reduction assigns, or None if the side is not that shape.

    The inline side is one assignment of a reduction, optionally followed by the renorm `if`
    that re-assigns the same name. Anything else is a real disagreement.
    """
    if not nodes:
        return None
    first = nodes[0]
    if not (isinstance(first, ast.Assign) and len(first.targets) == 1
            and isinstance(first.targets[0], ast.Name) and isinstance(first.value, ast.Call)):
        return None
    target = first.targets[0].id
    if HELPER in ast.dump(first):
        return None
    for extra in nodes[1:]:
        if not isinstance(extra, ast.If):
            return None
        for sub in ast.walk(extra):
            if isinstance(sub, ast.Name) and isinstance(getattr(sub, "ctx", None), ast.Store) \
                    and sub.id != target:
                return None
    return target


def resolve(path: pathlib.Path, theirs_ref: str) -> int:
    s = path.read_text()
    start, mid, end = "<<<<<<< HEAD\n", "=======\n", f">>>>>>> {theirs_ref}\n"
    if start not in s or end not in s or s.count(start) != 1:
        return 2
    i = s.index(start)
    j = s.index(mid, i)
    k = s.index(end, j)
    sides = {"ours": s[i + len(start):j], "theirs": s[j + len(mid):k]}
    parsed = {w: _stmts(t) for w, t in sides.items()}
    if any(v is None for v in parsed.values()):
        return 2

    # Exactly one side calls the helper; the other inlines the same reduction.
    helper_side = [w for w, n in parsed.items() if _helper_target(n) is not None]
    if len(helper_side) != 1:
        return 2
    hw = helper_side[0]
    iw = "theirs" if hw == "ours" else "ours"
    target = _helper_target(parsed[hw])
    box_reads = [n for n in parsed[iw] if _is_box_read(n)]
    rest = [n for n in parsed[iw] if not _is_box_read(n)]
    if len(box_reads) > 1 or _inline_target(rest) != target:
        print(f"  {path}: a softmax-inner conflict that is not inline-against-helper -- one "
              f"inlined `{target}` and at most one box read are required", file=sys.stderr)
        return 2
    if box_reads and any(_is_box_read(n) for n in parsed[hw]):
        return 2                                   # helper side already has it; nothing to merge
    # The inline side may carry the TT_BIO_SOFTMAX_BW_RENORM branch. Dropping it is only safe if
    # the helper carries the same branch, which is D56's own claim -- so read it rather than
    # trust it. The helper lives in `tt_bio/autograd.py` of the tree being composed.
    if any(isinstance(n, ast.If) for n in rest):
        src = path.parent / "autograd.py" if path.name != "autograd.py" else path
        try:
            helper_src = ast.parse(src.read_text())
        except (OSError, SyntaxError):
            print(f"  {path}: cannot read {src} to check the helper carries the renorm branch",
                  file=sys.stderr)
            return 2
        fn = next((n for n in helper_src.body
                   if isinstance(n, ast.FunctionDef) and n.name == HELPER), None)
        if fn is None or "RENORM" not in ast.dump(fn):
            print(f"  {path}: the inline side has a renorm branch and {HELPER} does not carry "
                  f"one -- taking the helper would drop behaviour", file=sys.stderr)
            return 2

    # The box read keeps its own source line, comment and all; the helper call keeps its.
    box_line = [ln for ln in sides[iw].splitlines() if re.match(r"\s*\w+\s*=\s*box\[", ln)]
    helper_lines = [ln for ln in sides[hw].splitlines() if ln.strip()]
    body = "\n".join(box_line + helper_lines)
    new = s[:i] + body + "\n" + s[k + len(end):]
    if "<<<<<<<" in new or ">>>>>>>" in new:
        return 2
    try:
        ast.parse(new)
    except SyntaxError as e:
        print(f"  {path}: the composed backward does not parse -- {e}", file=sys.stderr)
        return 1
    if HELPER not in body or (box_reads and "box[" not in body):
        print(f"  {path}: the resolution dropped one side", file=sys.stderr)
        return 2
    path.write_text(new)
    where = "HEAD" if hw == "ours" else "theirs"
    print(f"  {path.name}: softmax inner composed -- {HELPER} from {where}"
          + (f", the box read from {'HEAD' if iw == 'ours' else 'theirs'}" if box_reads
             else ", no box read in the hunk"))
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        raise SystemExit(1)
    raise SystemExit(resolve(pathlib.Path(sys.argv[1]), sys.argv[2]))
