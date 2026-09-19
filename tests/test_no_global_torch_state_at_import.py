"""No test module may mutate process-wide torch state at import time.

Found 2026-09-18 by running the WHOLE suite instead of `tests/test_train*.py`, which is what
every training row (and the orchestrator) had been running. `tests/test_tenstorrent.py` and
`tests/test_protenix.py` each called ``torch.set_grad_enabled(False)`` at module level. That is
a process-wide switch and nothing restores it, so from the moment either module was IMPORTED,
autograd was off for the rest of the session.

The two casualties were `tests/test_train_interface.py`'s gradcheck invariants -- the tests that
check a reference is verified before the device is blamed, and that a kinked op needs a gate
mask. They need ``requires_grad=True`` to build a graph, so they failed in every full-suite run
and passed in isolation. Three things made that invisible for a day:

* the training rows ran `tests/test_train*.py`, which never imports the polluting modules;
* a module's import runs even when every test in it SKIPS, so a host with no card disabled
  autograd for the whole session while contributing no coverage at all -- the pollution is worse
  on the machine that cannot run the tests that wanted it;
* earlier orchestrator runs used an interpreter without ttnn, where the affected tests skipped.

So the defect class is "global state set at import, restored never", and the fix is to scope it:

    @pytest.fixture(autouse=True)
    def _grad_off():
        with torch.no_grad():
            yield

Static and importless -- it parses, it does not run the modules it judges, because importing
them is the very thing that causes the damage.
"""

import ast
from pathlib import Path

TESTS = Path(__file__).resolve().parent

# Process-wide torch switches with no automatic restore. Each one silently changes the meaning
# of every test that runs after it in the same process.
FORBIDDEN_AT_IMPORT = (
    "set_grad_enabled",
    "set_default_dtype",
    "set_default_device",
    "set_float32_matmul_precision",
)


def _import_time_statements(module: ast.Module):
    """Statements that execute on import: top level, and inside top-level if/try/with/for.

    Deliberately does NOT descend into a function or class body -- a call inside a fixture is
    the fix, not the defect, so a checker that flagged it would push people back to the
    module-level form to keep the gate quiet.
    """
    stack = list(module.body)
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, (ast.If, ast.Try, ast.With, ast.For, ast.While)):
            for attr in ("body", "orelse", "finalbody", "handlers"):
                for child in getattr(node, attr, []) or []:
                    if isinstance(child, ast.ExceptHandler):
                        stack.extend(child.body)
                    else:
                        stack.append(child)


def _walk_outside_callables(node: ast.AST):
    """Every node under ``node`` except inside a function or class body.

    ``ast.walk`` descends into everything, which is why the first draft of this checker flagged
    its own recommended fix: a top-level ``@pytest.fixture`` is a top-level statement, so
    walking it reached the ``torch.no_grad()`` inside. Its control caught that, which is the
    only reason it is not in the shipped gate.
    """
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return
    yield node
    for child in ast.iter_child_nodes(node):
        yield from _walk_outside_callables(child)


def _offenders(source: str) -> list:
    found = []
    for stmt in _import_time_statements(ast.parse(source)):
        for node in _walk_outside_callables(stmt):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr in FORBIDDEN_AT_IMPORT:
                owner = getattr(func.value, "id", "") or getattr(func.value, "attr", "")
                if owner == "torch":
                    found.append((node.lineno, f"torch.{func.attr}"))
    return sorted(set(found))


def test_no_test_module_disables_autograd_or_repins_dtype_for_the_whole_process():
    bad = {}
    for path in sorted(TESTS.glob("test_*.py")):
        if path.resolve() == Path(__file__).resolve():
            continue
        try:
            source = path.read_text(encoding="utf-8")
        except OSError:
            continue
        hits = _offenders(source)
        if hits:
            bad[path.name] = hits
    assert not bad, (
        "these test modules mutate process-wide torch state at IMPORT time, so they change the "
        f"meaning of every test that runs after them in the same process: {bad}. This exact "
        "defect made tests/test_train_interface.py's two gradcheck invariants fail in every "
        "full-suite run and pass in isolation, for a day, and it fires even when every test in "
        "the offending module skips -- import happens anyway. Scope it to the tests that want "
        "it:\n\n"
        "    @pytest.fixture(autouse=True)\n"
        "    def _grad_off():\n"
        "        with torch.no_grad():\n"
        "            yield\n")


def test_the_control_the_checker_separates_import_time_from_fixture_scoped():
    """Each case is a way this gate could be wrong in one direction or the other."""
    module_level = "import torch\ntorch.set_grad_enabled(False)\n"
    inside_top_level_if = (
        "import torch\n"
        "if True:\n"
        "    torch.set_grad_enabled(False)\n")
    inside_try = (
        "import torch\n"
        "try:\n"
        "    torch.set_default_dtype(torch.float64)\n"
        "except Exception:\n"
        "    pass\n")
    fixture_scoped = (
        "import torch\n"
        "import pytest\n"
        "@pytest.fixture(autouse=True)\n"
        "def _grad_off():\n"
        "    with torch.no_grad():\n"
        "        yield\n")
    in_a_test_body = (
        "import torch\n"
        "def test_x():\n"
        "    torch.set_grad_enabled(False)\n")
    unrelated_owner = (
        "import numpy as torch_like\n"
        "torch_like.set_grad_enabled(False)\n")

    assert _offenders(module_level), "the plain module-level form must be caught"
    assert _offenders(inside_top_level_if), (
        "a top-level `if` body still runs on import, so it must be caught -- otherwise the "
        "defect just moves one indent to the right")
    assert _offenders(inside_try), "a top-level try body runs on import too"
    assert not _offenders(fixture_scoped), (
        "the FIX must not trip the gate; a checker that flags it pushes people back to the "
        "module-level form to keep the gate quiet")
    assert not _offenders(in_a_test_body), "a call inside a test is that test's own business"
    assert not _offenders(unrelated_owner), (
        "only torch's process-wide setters count; a same-named method on another object is not "
        "this defect")
