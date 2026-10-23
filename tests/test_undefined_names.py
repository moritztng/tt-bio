"""No function in tt-bio's own modules reads a global name nothing defines.

A hunk meant for T1's runner landed in trimul_tail.gin_moved as `if split:`; `split` exists only in T1's
scope, so every Wormhole trimul that took gin_moved raised NameError on its first call. Python reports
such names only when the line runs, and no card-free test reached it. symtable sees them statically.

Scope: tt_bio minus `_vendor` and `boltzgen`, which stay close to their upstream sources (jaxtyping
shape strings in `_vendor` also read as names).
"""
import ast
import builtins
import pathlib
import symtable

ROOT = pathlib.Path(__file__).resolve().parents[1] / "tt_bio"
SKIP = {"_vendor", "boltzgen"}


def _undefined(path):
    src = path.read_text()
    tree = ast.parse(src)
    if any(isinstance(n, ast.ImportFrom) and any(a.name == "*" for a in n.names) for n in ast.walk(tree)):
        return []
    top = symtable.symtable(src, str(path), "exec")
    known = set(dir(builtins)) | {"__file__", "__builtins__", "__path__"}
    known |= {s.get_name() for s in top.get_symbols() if s.is_assigned() or s.is_imported() or s.is_namespace()}
    known |= {name for n in ast.walk(tree) if isinstance(n, ast.Global) for name in n.names}
    out = []

    def walk(t):
        for s in t.get_symbols():
            if s.is_referenced() and (t.get_type() == "module" or s.is_global()) and s.get_name() not in known:
                out.append(f"{path.relative_to(ROOT.parent)}: {t.get_name()} reads `{s.get_name()}`")
        for c in t.get_children():
            walk(c)

    walk(top)
    return out


def test_no_undefined_globals():
    files = [p for p in sorted(ROOT.rglob("*.py")) if not SKIP & set(p.relative_to(ROOT).parts)]
    assert len(files) > 50
    assert [u for p in files for u in _undefined(p)] == []
