"""AST census of ttnn.linear / ttnn.matmul call sites in tt_bio/tenstorrent.py.

Reports, for every site, whether it passes core_grid explicitly and which enclosing
class/function it belongs to. A site is keyed by its 1-based line number, which is what the
runtime arm switch keys on too, so the two agree by construction.
"""
from __future__ import annotations
import ast, json, sys
from pathlib import Path

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "tt_bio/tenstorrent.py")


def call_name(node: ast.Call) -> str | None:
    f = node.func
    if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == "ttnn":
        if f.attr in {"linear", "matmul"}:
            return f.attr
    return None


def main() -> int:
    tree = ast.parse(SRC.read_text())
    scope: list[tuple[int, int, str]] = []
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            scope.append((n.lineno, n.end_lineno, n.name))
    out = []
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        kind = call_name(n)
        if kind is None:
            continue
        kws = {k.arg for k in n.keywords if k.arg}
        starred = any(k.arg is None for k in n.keywords)
        enclosing = [name for lo, hi, name in scope if lo <= n.lineno <= hi]
        out.append(dict(line=n.lineno, kind=kind, core_grid="core_grid" in kws,
                        starred=starred, kwargs=sorted(kws), scope=enclosing[-3:]))
    out.sort(key=lambda r: r["line"])
    json.dump(dict(source=str(SRC), total=len(out),
                   explicit=sum(r["core_grid"] for r in out),
                   bare=sum(not r["core_grid"] for r in out), sites=out),
              sys.stdout, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
