#!/usr/bin/env python3
"""Resolve a merge conflict whose two sides are both KEYWORD-ARGUMENT TAILS of one call.

Why this is generic and the per-row cases are not. On 2026-09-21 main gained M18's
`tri_att_sdpa_hifi=...` at OpenFold3's four Pairformer-family sites. Every concluded row that had
also appended a kwarg at one of those sites -- of3t-confidence, of3t-gradients, and any that
follow -- now conflicts on the SAME shape: two lists of keyword arguments that mostly agree.
Writing one case per row would mean a new case every time main touches a shared signature.

The rule, and both halves matter:

  * UNION the argument NAMES. Each side appended something optional and meant to keep it.
  * On a name present in BOTH with different values, **HEAD wins**. HEAD is the composition's
    base, which is main, and main is what ships. This is not a tie-break for tidiness: at
    `openfold3_confidence.py` the contested name is `scale_pair_bias` -- main ships **False** and
    a row's branch carries **True**, which is **D1**, a repair HELD because applying it measured
    **0.149 A worse at rank 0** and which pin 9629 asks Moritz to decide. A union that took the
    row's value would apply a held repair inside the composition.

PRECONDITION, checked and refused rather than assumed: both sides must end the call with `)` and
every top-level segment must be a `name=value`. A conflict that is not purely a kwarg tail is a
real disagreement and must stop the compose, which is what the caller does when this exits 2.

usage: resolve_kwarg_tail_conflict.py <file> <their-ref>     e.g. origin/wk/of3t-gradients
exit 0 resolved | 2 precondition not met (caller should fail) | 1 error
"""
from __future__ import annotations

import ast
import pathlib
import re
import sys
import textwrap


def split_kwargs(text: str):
    """name -> `name=value` source, splitting only at top-level commas, or None."""
    t = text.rstrip()
    # Two shapes occur. A TAIL ends the call with ')'. A FRAGMENT is an interior slice of the
    # argument list -- git conflicts on the lines that differ, not on the whole call -- and ends
    # with a comma. Both are safe to union; what is refused is anything with a positional in it.
    closing = t.endswith(")")
    if closing:
        t = t[:-1]
    out, depth, cur = {}, 0, ""
    for ch in t.replace("\n", " ") + ",":
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            x = cur.strip()
            if x:
                if "=" not in x:
                    return None                          # a positional: not a pure kwarg tail
                out[x.split("=", 1)[0].strip()] = x
            cur = ""
        else:
            cur += ch
    if not out:
        return None
    return out, closing


def resolve(path: pathlib.Path, theirs_ref: str) -> int:
    s = path.read_text()
    start = "<<<<<<< HEAD\n"
    mid = "=======\n"
    end = f">>>>>>> {theirs_ref}\n"
    if start not in s or end not in s:
        print(f"  {path}: no <<<<<<< HEAD / >>>>>>> {theirs_ref} pair", file=sys.stderr)
        return 2
    if s.count(start) != 1:
        print(f"  {path}: {s.count(start)} conflict hunks; this rule takes exactly one",
              file=sys.stderr)
        return 2
    i = s.index(start)
    j = s.index(mid, i)
    k = s.index(end, j)
    ours, theirs = s[i + len(start):j].rstrip(), s[j + len(mid):k].rstrip()
    # An IMPORT conflict is the other shape main's shared-signature changes produce: HEAD adds
    # the new symbol to a `from .x import (...)`, the row added a different import beside it.
    # Unioning imports cannot change behaviour except by binding a name, and an unused binding
    # is harmless -- so this one is safe in a way a content union is not. Both sides must parse
    # as nothing but imports, or it falls through to the kwarg rule and then to a hard failure.
    def only_imports(text):
        try:
            mod = ast.parse(textwrap.dedent(text))
        except SyntaxError:
            return None
        if not mod.body or not all(isinstance(n, (ast.Import, ast.ImportFrom)) for n in mod.body):
            return None
        return [ln for ln in text.splitlines() if ln.strip()]

    ia, ib = only_imports(ours), only_imports(theirs)
    if ia is not None and ib is not None:
        # Merge `from X import a, b` with `from X import a, c` into one line rather than keeping
        # both. Keeping both parses and even runs -- the second rebinds the same names -- but it
        # is not what either side wrote, and "compiles and is not what anyone meant" is the
        # failure this whole file exists to avoid.
        mod_a = ast.parse(textwrap.dedent(ours))
        mod_b = ast.parse(textwrap.dedent(theirs))
        froms: dict = {}
        plains: list = []
        for node in list(mod_a.body) + list(mod_b.body):
            if isinstance(node, ast.ImportFrom):
                key = (node.level, node.module or "")
                names = froms.setdefault(key, [])
                for al in node.names:
                    if (al.name, al.asname) not in [(x.name, x.asname) for x in names]:
                        names.append(al)
            else:
                text = "import " + ", ".join(
                    al.name + (f" as {al.asname}" if al.asname else "") for al in node.names)
                if text not in plains:
                    plains.append(text)
        lines = list(plains)
        for (level, module), names in froms.items():
            rendered = ", ".join(al.name + (f" as {al.asname}" if al.asname else "")
                                 for al in sorted(names, key=lambda x: x.name))
            lines.append(f"from {'.' * level}{module} import {rendered}")
        new = s[:i] + "\n".join(lines) + "\n" + s[k + len(end):]
        try:
            ast.parse(new)
        except SyntaxError as e:
            print(f"  {path}: the merged imports do not parse -- {e}", file=sys.stderr)
            return 1
        path.write_text(new)
        print(f"  {path.name}: union of {len(lines)} import line(s)")
        return 0

    pa, pb = split_kwargs(ours), split_kwargs(theirs)
    if pa is None or pb is None:
        print(f"  {path}: not a pure keyword-argument tail or fragment on both sides",
              file=sys.stderr)
        return 2
    (a, close_a), (b, close_b) = pa, pb
    if close_a != close_b:
        print(f"  {path}: one side closes the call and the other does not", file=sys.stderr)
        return 2

    merged = dict(b)
    merged.update(a)                                     # HEAD wins every collision
    order = list(a) + [n for n in b if n not in a]
    indent = re.match(r"\s*", ours).group(0)
    body = ",\n".join(indent + merged[n] for n in order) + (")" if close_a else ",")
    new = s[:i] + body + "\n" + s[k + len(end):]
    try:
        ast.parse(new)                                   # a lost bracket fails HERE, not on a card
    except SyntaxError as e:
        print(f"  {path}: the merged call does not parse -- {e}", file=sys.stderr)
        return 1
    path.write_text(new)
    took = [n for n in a if n in b and a[n] != b[n]]
    print(f"  {path.name}: union of {len(merged)} kwargs"
          + (f"; HEAD kept {', '.join(a[n] for n in took)}" if took else "; no collisions"))
    return 0


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__.strip().splitlines()[-2])
        return 1
    return resolve(pathlib.Path(sys.argv[1]), sys.argv[2])


if __name__ == "__main__":
    sys.exit(main())
