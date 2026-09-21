#!/usr/bin/env python3
"""Show, rather than argue, that this row's shipped-file changes move no default.

Three files on the shared path carry an import move and nothing else:

  `tt_bio/openfold3_fold.py`            this row's own: `from .protenix import ConfidenceHead`
                                        leaves `_confidence()` for module scope, because a
                                        tt-bio module first imported INSIDE a taped block is
                                        never shimmed and every ttnn call in it runs raw
                                        (rule C of `tape_reach_guard.py`).
  `tt_bio/openfold3_confidence.py`      PROTOCOL A2's first site, `import ttnn` out of
  `tt_bio/openfold3_host_prep.py`       `forward()` / `run_input_atom_encoder()`.

The A2 pair was landed by 5a2efa001 on `wk/of3t`, which this row is based on; it is checked
here because deliverable 3 owes the statement whether or not this row typed the diff, and
because **the pair is still unfixed on `origin/main`** -- `git show origin/main:` has both
imports inside their functions. The claim below is about the campaign branch.

Each file is compared across the COMMIT THAT MOVED ITS IMPORT, not against the working tree.
`openfold3_confidence.py` has changed since 5a2efa001 for reasons belonging to other rows, so
a working-tree diff would fold their edits into this row's claim and read DIFFERENT for the
wrong reason -- which it did, before this was written the other way round.

Two checks, both executable:

  1. With every `import` statement stripped, the module's AST before and after the move is
     IDENTICAL. An import move cannot change a default, a threshold or a branch, and this
     says so about these diffs rather than about import moves in general.
  2. In the tree under test, the names resolve to the same objects, and the two A2 modules
     hold the real `ttnn` as a module global -- which is the whole point, since that global
     is what `tape()` rebinds.
"""
from __future__ import annotations

import ast
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

MOVES = [
    ("origin/wk/of3t", None, "tt_bio/openfold3_fold.py"),
    ("5a2efa001^", "5a2efa001", "tt_bio/openfold3_confidence.py"),
    ("5a2efa001^", "5a2efa001", "tt_bio/openfold3_host_prep.py"),
]


class StripImports(ast.NodeTransformer):
    def visit_Import(self, node):
        return None

    def visit_ImportFrom(self, node):
        return None


def bodyless(src):
    t = StripImports().visit(ast.parse(src))
    return ast.dump(ast.fix_missing_locations(t), include_attributes=False)


def at(rev, f):
    return (open(os.path.join(REPO, f)).read() if rev is None
            else subprocess.check_output(["git", "-C", REPO, "show", f"{rev}:{f}"], text=True))


ok = True
for base, head, f in MOVES:
    old, new = at(base, f), at(head, f)
    same = bodyless(old) == bodyless(new)
    n_old = sum(1 for n in ast.walk(ast.parse(old))
                if isinstance(n, (ast.Import, ast.ImportFrom)))
    n_new = sum(1 for n in ast.walk(ast.parse(new))
                if isinstance(n, (ast.Import, ast.ImportFrom)))
    ok &= same
    print(f"{f:34s} {base:16s} -> {head or 'working tree':16s} AST-without-imports "
          f"{'IDENTICAL' if same else 'DIFFERENT'}, imports {n_old} -> {n_new}")

import ttnn                                                            # noqa: E402
import tt_bio.openfold3_confidence as C                                # noqa: E402
import tt_bio.openfold3_fold as FOLD                                   # noqa: E402
import tt_bio.openfold3_host_prep as H                                 # noqa: E402
from tt_bio.protenix import ConfidenceHead as B                        # noqa: E402

assert os.path.dirname(os.path.dirname(C.__file__)) == REPO, (
    f"tt_bio resolved to {C.__file__}, not this worktree -- the check would be about the "
    f"shared checkout instead of the tree that carries the change")
print(f"tt_bio resolved to {os.path.dirname(C.__file__)}")
print(f"openfold3_fold.ConfidenceHead is protenix.ConfidenceHead: {FOLD.ConfidenceHead is B}")
reach = {m.__name__.split(".")[-1]: getattr(m, "ttnn", None) is ttnn for m in (C, H)}
print(f"module-global ttnn, so tape()'s _swap reaches them: {reach}")

assert ok, "a move is more than an import move; this is not a no-default-moved change"
assert FOLD.ConfidenceHead is B
assert all(reach.values())
print("PASS: no executable default moved, and both A2 modules are reachable by the tape")
