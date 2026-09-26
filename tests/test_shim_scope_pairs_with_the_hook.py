"""The grad hook and the ttnn shim go in together, or a shipped module breaks on its own output.

The hook makes `ops.linear` and `ops.layer_norm` hand back an `autograd.Tensor`. The shim is
what makes every other verb in the same module accept one. Shipped modules mix the two freely:
`OF3DiffusionConditioning._pair` is a handful of `ops` calls and a dozen raw `ttnn.` ones, so
with the hook on and the shim off the first raw verb downstream of an `ops` call is handed an
`autograd.Tensor` and pybind refuses it.

That is where `tt_bio.train.recipes.train_loop` died on the OpenFold3 forward -- in
`lora.walked_weights`' discovery pass, before step 0, the first time anything ran the shipped
training loop against the shipped OF3 forward. The discovery pass deliberately does not open a
tape (a tape would route MORE than a training step does, which `walked_weights` documents), so
the fix is the shim without the tape: `taped_ttnn.shim_scope`.
"""
import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_the_discovery_forward_runs_inside_shim_scope():
    """Static, so it runs on a host with no card: the `with` around the discovery forward."""
    src = (ROOT / "tt_bio" / "train" / "lora.py").read_text()
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "walked_weights")
    withs = [n for n in ast.walk(fn) if isinstance(n, ast.With)
             and any(isinstance(i.context_expr, ast.Call)
                     and getattr(i.context_expr.func, "id", None) == "shim_scope"
                     for i in n.items)]
    assert withs, ("walked_weights runs its discovery forward without shim_scope; the first "
                   "raw ttnn verb after an ops call will be handed an autograd.Tensor")
    calls = [n for w in withs for n in ast.walk(w)
             if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "forward"]
    assert calls, "shim_scope is opened but the discovery forward is not inside it"


def test_shim_scope_swaps_a_shipped_module_and_puts_it_back():
    """Behavioural, and card-free: ttnn imports without a device."""
    ttnn = pytest.importorskip("ttnn")
    import tt_bio.openfold3_diffusion as od
    from tt_bio.taped_ttnn import shim_scope, taped_ttnn

    assert od.ttnn is ttnn
    with shim_scope():
        assert od.ttnn is taped_ttnn(), \
            "a shipped module still holds the real ttnn inside shim_scope"
    assert od.ttnn is ttnn, "shim_scope left the shim installed"


def test_shim_scope_is_reentrant_and_leaves_an_open_tape_alone():
    pytest.importorskip("ttnn")
    import tt_bio.openfold3_diffusion as od
    from tt_bio.taped_ttnn import shim_scope, tape, taped_ttnn

    with tape():
        inner = od.ttnn
        with shim_scope():
            assert od.ttnn is inner is taped_ttnn()
        assert od.ttnn is inner, "the inner scope tore down the tape's own shim"
