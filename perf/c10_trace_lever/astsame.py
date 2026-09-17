"""Is a diff comment-only? Compare two Python sources as ASTs with docstrings stripped.

The 14.8813 s number of record was measured on 5e1886b4f. This tree is two commits later, and
both commits only rewrote prose in comments and module docstrings. Asserting that by eye is not
evidence, so the harness checks it: identical ASTs after docstring removal means identical
executable code, and a real one-token change breaks the check (negative controls in
test_controls.py).
"""
from __future__ import annotations
import ast


def strip_docstrings(tree):
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = node.body
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                and isinstance(body[0].value.value, str):
            node.body = body[1:] or [ast.Pass()]
    return tree


def code_fingerprint(source: str) -> str:
    return ast.dump(strip_docstrings(ast.parse(source)), annotate_fields=True,
                    include_attributes=False)


def equivalent(a: str, b: str) -> bool:
    return code_fingerprint(a) == code_fingerprint(b)
