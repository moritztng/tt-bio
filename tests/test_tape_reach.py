"""The two silent ways the tape can fail to be installed, pinned as tests.

`tt_bio.autograd.tape()` works by rebinding the module-global name `ttnn` in every tt_bio
module that holds one. Both of its failure modes are silent, which is why they are tests
rather than comments:

1. A module that writes `import ttnn` INSIDE a function binds the real module into function
   scope at call time. `_swap` never sees it, the module's ttnn calls run untaped, and they
   receive raw handles rather than raising -- so the tape's "an op it cannot follow is loud"
   guarantee does not reach that file at all. `openfold3_confidence` and
   `openfold3_host_prep` were in that state and 16 call sites ran untaped.
2. `_Ttnn.__getattr__` asks whether an attribute is a nested namespace. It used to ask with
   `isinstance(attr, type(ttnn))`, reading `ttnn` off its own module global, so a second
   proxy rebinding it made every `ttnn.experimental.*` verb come back raw.

The first test is a census, not a ban: a function-local import is fine in a module that does
no device work under a tape, and several exist for profiling and for deferring the import
cost. What is not fine is a NEW one appearing unnoticed in a module the tape must reach, so
the set is pinned and a change has to be argued for here.
"""
import ast
import pathlib
import types

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent / "tt_bio"

# Function-local `import ttnn`, by module, as of the OF3 tape census. Each is host-side or
# diagnostic: `main`/`worker` defer the import cost off the CLI path, `protenix`,
# `esmfold2_runtime`, `opendde` and `token_axis` sync or move host tensors outside any tape,
# and `mm_generic` reads the wheel's own install path. Adding an entry means asserting the
# same of a new one.
ALLOWED = {
    "esmfold2_runtime.py", "main.py", "mm_generic.py", "opendde.py", "protenix.py",
    "token_axis.py", "worker.py",
    # `train/` reaches ttnn from outside any forward: `tensors.py` is the host-transfer pair
    # and says in its own docstring that the import is deferred so the module imports without
    # the wheel, and `mesh.py` all-reduces GRADIENTS between chips, after the backward has
    # closed. Neither is a call site a tape has to follow.
    "tensors.py", "mesh.py",
    # A kernel-source patcher, not a model path.
    "patch_trimul_tail.py",
}


def _local_ttnn_imports(path):
    tree = ast.parse(path.read_text(), str(path))
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.Import) and any(a.name == "ttnn" for a in sub.names):
                out.append(sub.lineno)
    return out


def test_no_new_function_local_ttnn_import():
    found = {}
    for path in sorted(ROOT.rglob("*.py")):
        if "_vendor" in path.parts:
            continue
        lines = _local_ttnn_imports(path)
        if lines:
            found[path.name] = lines
    unexpected = {k: v for k, v in found.items() if k not in ALLOWED}
    assert not unexpected, (
        f"function-local `import ttnn` in {unexpected}: `tape()` rebinds the MODULE global, "
        f"so these call sites run untaped and receive raw handles without raising. Move the "
        f"import to module scope, or add the module to ALLOWED with the reason it does no "
        f"device work under a tape.")


def test_proxy_namespace_test_survives_a_rebound_global():
    """`_Ttnn` must find nested namespaces without reading its own module global."""
    ttnn = pytest.importorskip("ttnn")
    from tt_bio import taped_ttnn as tp

    prev = tp.ttnn
    try:
        tp.ttnn = object()          # what a second proxy installing itself looks like
        shim = tp._Ttnn(ttnn)
        assert isinstance(shim.experimental, tp._Ttnn), (
            "ttnn.experimental came back as something other than a proxy namespace, so every "
            "experimental verb under it would run untaped")
    finally:
        tp.ttnn = prev


def test_module_type_is_what_the_proxy_asks_for():
    from tt_bio import taped_ttnn as tp
    src = pathlib.Path(tp.__file__).read_text()
    assert "isinstance(attr, types.ModuleType)" in src
    assert "isinstance(attr, type(ttnn))" not in src
    assert isinstance(types.ModuleType, type)
