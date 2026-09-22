#!/usr/bin/env python3
"""Does this call pass this keyword? Answered by AST, because grep answers a different question.

WHY THIS EXISTS
---------------
Three times in a fortnight this campaign asserted something about source it had matched with a
text pattern, and three times the pattern was answering a narrower question than the claim:

  * D153  a substring `of3t_rebase` read as a path, when two of the hits were `origin/wk/of3t-rebase`
          (a branch) and `perf/of3t_rebase/*.json` (a repo-relative path). Two namespaces wrongly
          reported as broken, one of them the orchestrator's own.
  * D158  `attention.py:314` resolved by basename to tt-bio's `attention.py`, when the citation was
          about UPSTREAM's file of the same name. A stale-citation report that was not one.
  * D55   `site_softmax(... )` reported as passing no `compute_kernel_config`, because the call
          spans two lines and grep stopped at the first. A row was dispatched to add an argument
          that was already there.

Each time the AST was available and each time the pattern was faster to type. So: make the correct
query the fast one.

USAGE
-----
    kwarg_at_site.py <file-or-dir> <callee> [kwarg]

`callee` matches the last component, so `softmax` matches `ttnn.softmax` and `site_softmax` matches
only itself. With `kwarg`, each call is marked PASSES or MISSING for that keyword; without, the
full keyword list is printed. Exit is 0 whatever it finds -- this is a QUERY, not a check, and a
tool that fails the build on a question is a tool people stop asking.

WHAT IT STILL CANNOT TELL YOU
-----------------------------
That the keyword's VALUE is meaningful. `of3t-fwdkcfg`'s case turned on exactly this: every site
passes `compute_kernel_config`, and its value is `softmax_ckc(token)`, which returns `None` unless
a per-site lever is on. PASSES means the argument is at the call, nothing more. Read the value.
"""
from __future__ import annotations

import ast
import pathlib
import sys


def calls(path: pathlib.Path, callee: str):
    try:
        tree = ast.parse(path.read_text(errors="replace"))
    except SyntaxError:
        return
    parents = {}
    for n in ast.walk(tree):
        for c in ast.iter_child_nodes(n):
            parents[c] = n

    def enclosing(n):
        while n in parents:
            n = parents[n]
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return n.name
        return "<module>"

    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        f = n.func
        name = f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else None)
        if name != callee:
            continue
        kws = [k.arg for k in n.keywords if k.arg]
        val = {k.arg: ast.unparse(k.value) for k in n.keywords if k.arg}
        yield n.lineno, enclosing(n), kws, val


def main(argv):
    if len(argv) < 3:
        print(__doc__.strip().split("USAGE")[1].strip()[:200], file=sys.stderr)
        return 2
    target, callee = pathlib.Path(argv[1]), argv[2]
    kwarg = argv[3] if len(argv) > 3 else None
    files = sorted(target.rglob("*.py")) if target.is_dir() else [target]

    rows = []
    for f in files:
        for lineno, encl, kws, val in calls(f, callee):
            rows.append((str(f), lineno, encl, kws, val))
    if not rows:
        print("no call to %r under %s" % (callee, target))
        return 0

    if kwarg:
        n_pass = sum(1 for *_x, kws, _v in rows if kwarg in kws)
        print("%d call(s) to %r; %d pass %r, %d do not"
              % (len(rows), callee, n_pass, kwarg, len(rows) - n_pass))
        for f, ln, encl, kws, val in rows:
            mark = "PASSES " if kwarg in kws else "MISSING"
            extra = ("  = " + val[kwarg]) if kwarg in val else ""
            print("  %s %s:%d  in %s%s" % (mark, f, ln, encl, extra))
        print("\nPASSES means the argument is AT THE CALL. It says nothing about its value -- "
              "read that separately (D55 turned on a passed argument whose value was None).")
    else:
        print("%d call(s) to %r" % (len(rows), callee))
        for f, ln, encl, kws, _v in rows:
            print("  %s:%d  in %-28s kwargs: %s" % (f, ln, encl, ", ".join(kws) or "(none)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
