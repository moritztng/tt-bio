#!/usr/bin/env python3
"""Resolve an add/add conflict on a TEST file by keeping both suites, under two names.

Two rows independently wrote a test file with the same name. That is a filename collision, not
a disagreement, and merging the bodies is the wrong move: at pass 358
`tests/test_training_tape_rebind.py` arrived from `origin/main` (D126 -- the tape registry
follows a `value` replacement, checked with `parameter_for(t.value)`) and from
`origin/wk/of3t-apbback` (via of3t-rebind -- `Parameters.rebind()` puts the optimizer's tensors
back into the MODEL SLOT, checked with `getattr(model, name)`). Those are two different
contracts and both should hold. Nine hunks of interleaved prose and assertions is not something
to union; the two files are.

So: HEAD keeps the path, and theirs is written beside it as `<base>__<row>.py`. Nothing is
dropped and nothing is silently merged. If the row's suite does not actually pass on the
composition, the compose's own collection check is what says so -- this resolver only refuses
to lose it.

Refuses with exit 2 unless every one of these holds:
  * the path is `tests/test_*.py`;
  * the conflict is add/add -- there is no merge base for it, so there is no shared history to
    union along;
  * both sides parse;
  * the sibling path is free.

Usage: resolve_addadd_test_file.py <file> <their-ref>  ->  0 resolved, 2 not this shape.
"""
from __future__ import annotations

import ast
import pathlib
import subprocess
import sys


def _stage(path: str, n: int):
    r = subprocess.run(["git", "show", f":{n}:{path}"], capture_output=True)
    return r.stdout.decode() if r.returncode == 0 else None


def resolve(path: pathlib.Path, theirs_ref: str) -> int:
    p = path.as_posix()
    if not (p.startswith("tests/") and path.name.startswith("test_") and path.suffix == ".py"):
        return 2
    base, ours, theirs = _stage(p, 1), _stage(p, 2), _stage(p, 3)
    if base is not None or ours is None or theirs is None:
        return 2                                   # not add/add
    row = theirs_ref.rsplit("/", 1)[-1].replace("wk-", "").replace("of3t-", "")
    sibling = path.with_name(f"{path.stem}__{row}.py")
    if sibling.exists():
        print(f"  {p}: {sibling} already exists -- two rows would land on it", file=sys.stderr)
        return 2
    for label, text in (("HEAD", ours), (theirs_ref, theirs)):
        try:
            ast.parse(text)
        except SyntaxError as e:
            print(f"  {p}: the {label} side does not parse -- {e}", file=sys.stderr)
            return 2
    path.write_text(ours)
    sibling.write_text(theirs)
    subprocess.run(["git", "add", p, sibling.as_posix()], check=True)
    print(f"  {path.name}: add/add kept as TWO suites -- HEAD at {p}, "
          f"{theirs_ref.rsplit('/', 1)[-1]}'s at {sibling.as_posix()}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        raise SystemExit(1)
    raise SystemExit(resolve(pathlib.Path(sys.argv[1]), sys.argv[2]))
