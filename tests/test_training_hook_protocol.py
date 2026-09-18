"""The grad-hook protocol is a contract between rows, so it is gated rather than remembered.

`tt_bio/dispatch.py:OpSurface` offers every decorated op to an installed hook as
`hook(name, shipped, args, kwargs)`, returning `None` to decline. Before that unification each
op surface held an object exposing named `linear`/`layer_norm` methods, and two rows built the
same machinery independently against the older shape.

That is how this gate came to exist. `train-c-interface` merged `train-a1-defork` at the commit
before the unification, earned a passing test run against it, and its `tt_bio/train/lora.py`
installs two hooks -- the LoRA census and the adapter -- that expose `linear`/`layer_norm` and
define no `__call__`. Under the current protocol both raise `TypeError: object is not callable`
on the first op they see, and the failure surfaces only when the two branches meet on `main`.
Every number in that row's verdict was true of the tree it was measured on; what was false was
that the tree still existed.

So the contract is checked here, on the merged tree, by the party that owns the merge queue.
The check is deliberately static and importless: `dispatch.py` imports nothing but `functools`,
which is what lets this run on a host with no wheel and no card.
"""

import ast
import importlib.util
import inspect
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TT_BIO = ROOT / "tt_bio"
DISPATCH = TT_BIO / "dispatch.py"


def _load_dispatch():
    """Import `tt_bio/dispatch.py` by path, so `tt_bio/__init__` (and ttnn) stay out of it."""
    spec = importlib.util.spec_from_file_location("_tt_bio_dispatch_under_test", DISPATCH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _hook_arity():
    """The live protocol's arity, read off `OpSurface.dispatching` instead of hard-coded.

    A row that legitimately changes the protocol should see this gate move with it, not fail
    against a number frozen in a test.
    """
    mod = _load_dispatch()
    src = inspect.getsource(mod.OpSurface.dispatching)
    tree = ast.parse(src.lstrip())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "hook":
            return len(node.args)
    pytest.fail("OpSurface.dispatching no longer calls `hook(...)`; the protocol moved and this "
                "gate cannot tell what it moved to. Update the gate with the row that moved it.")


def _installed_hook_classes():
    """Every class whose instance is handed to a `set_grad_hook` call anywhere under `tt_bio/`.

    Matches `X.set_grad_hook(Name(...))` and the `rec = Name(...)` / `set_grad_hook(rec)` form,
    which is what the LoRA census actually writes.
    """
    found = []
    for path in sorted(TT_BIO.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        classes = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
        if not classes:
            continue
        # local name -> class it was constructed from
        built = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                callee = node.value.func
                name = callee.id if isinstance(callee, ast.Name) else None
                if name in classes:
                    for tgt in node.targets:
                        if isinstance(tgt, ast.Name):
                            built[tgt.id] = name
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            if node.func.attr != "set_grad_hook" or not node.args:
                continue
            arg = node.args[0]
            cls = None
            if isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name):
                cls = arg.func.id
            elif isinstance(arg, ast.Name):
                cls = built.get(arg.id)
            if cls in classes:
                found.append((path.relative_to(ROOT), cls, classes[cls], node.lineno))
    return found


@pytest.mark.skipif(not DISPATCH.exists(),
                    reason="tt_bio/dispatch.py is not on this tree yet (train-a1-defork "
                           "has not landed), so there is no protocol to check against")
def test_every_installed_grad_hook_implements_the_current_protocol():
    arity = _hook_arity()
    hooks = _installed_hook_classes()
    if not hooks:
        pytest.skip("no class-based grad hook is installed anywhere under tt_bio/")
    broken = []
    for rel, cls, node, lineno in hooks:
        methods = {m.name for m in node.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))}
        if "__call__" not in methods:
            broken.append(f"{rel}:{lineno}: {cls} is installed as a grad hook but defines no "
                          f"__call__, so the protocol's {arity}-argument call raises TypeError. "
                          f"It exposes {sorted(methods - {'__init__'})}, which is the retired "
                          f"named-method protocol.")
    assert not broken, (
        "grad hooks built to the retired protocol are installed on this tree:\n  "
        + "\n  ".join(broken)
        + f"\n\nThe live contract is hook(name, shipped, args, kwargs) -> None to decline "
          f"({arity} positional arguments), per tt_bio/dispatch.py. Give each class "
          f"__call__(self, name, shipped, args, kwargs) dispatching on `name`.")


@pytest.mark.skipif(not DISPATCH.exists(), reason="tt_bio/dispatch.py is not on this tree yet")
def test_the_retired_named_method_hook_actually_breaks():
    """The negative control: a hook of the old shape must fail, or the gate above reads nothing.

    Without this, the assertion could pass because the protocol is lenient rather than because
    the tree is clean.
    """
    mod = _load_dispatch()
    surface = mod.OpSurface("gate.control")

    @surface.dispatching
    def linear(x, w, bias=None, **kw):
        return "SHIPPED"

    class RetiredShape:
        """Verbatim in shape to the LoRA census: named methods, no __call__."""

        def __init__(self):
            self.seen = []

        def linear(self, x, w, bias, *, activation=None, **kw):
            self.seen.append("linear")
            return None

        def layer_norm(self, x, weight, bias, **kw):
            return None

    surface.set_grad_hook(RetiredShape())
    with pytest.raises(TypeError, match="not callable"):
        linear("x", "w")

    # And the current shape must work, so the control is discriminating rather than just noisy.
    calls = []

    def current(name, shipped, args, kwargs):
        calls.append(name)
        return None  # decline, so the shipped op runs and the value is production's

    surface.set_grad_hook(current)
    assert linear("x", "w") == "SHIPPED"
    assert calls == ["linear"]
