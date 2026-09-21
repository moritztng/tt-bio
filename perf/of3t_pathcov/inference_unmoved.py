#!/usr/bin/env python3
"""Show, rather than argue, that this row's one shipped-file change moves no default.

The change is in `tt_bio/openfold3_fold.py`: `from .protenix import ConfidenceHead` moves out of
`_confidence()` and up to module scope, because a tt-bio module first imported INSIDE a taped
block is never shimmed and every ttnn call in it runs raw (rule C of `tape_reach_guard.py`).

Two checks, both executable:

  1. With every `import` statement stripped, the module's AST before and after is IDENTICAL.
     An import move cannot change a default, a threshold or a branch, and this is what says so
     about this diff rather than about import moves in general.
  2. The name resolves to the same object it resolved to before, so no call site sees a
     different class.
"""
from __future__ import annotations

import ast
import subprocess
import sys

BASE = sys.argv[1] if len(sys.argv) > 1 else "origin/wk/of3t"
FILE = "tt_bio/openfold3_fold.py"


class StripImports(ast.NodeTransformer):
    def visit_Import(self, node):
        return None

    def visit_ImportFrom(self, node):
        return None


def bodyless(src):
    t = StripImports().visit(ast.parse(src))
    return ast.dump(ast.fix_missing_locations(t), include_attributes=False)


old = subprocess.check_output(["git", "show", f"{BASE}:{FILE}"], text=True)
new = open(FILE).read()
same = bodyless(old) == bodyless(new)
print(f"AST with imports stripped, {BASE} vs working tree: "
      f"{'IDENTICAL' if same else 'DIFFERENT'}")

n_old = sum(1 for n in ast.walk(ast.parse(old)) if isinstance(n, (ast.Import, ast.ImportFrom)))
n_new = sum(1 for n in ast.walk(ast.parse(new)) if isinstance(n, (ast.Import, ast.ImportFrom)))
print(f"import statements: {n_old} -> {n_new}")

from tt_bio.openfold3_fold import ConfidenceHead as A      # noqa: E402
from tt_bio.protenix import ConfidenceHead as B            # noqa: E402
print(f"openfold3_fold.ConfidenceHead is protenix.ConfidenceHead: {A is B}")

assert same, "the diff is more than an import move; this is not a no-default-moved change"
assert A is B
print("PASS: no executable default moved")
