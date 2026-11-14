"""No function on the taped gradient path defines a class.

A class defined inside a function is a new class per call, and a class is always a reference cycle
(it sits in its own `__mro__`). Refcounting never frees it, so whatever its methods close over waits
for the cyclic collector. When that is a device tensor it stays on the card meanwhile: the per-call
sink in `triatt_qkv_heads` held three pair tensors per Evoformer block that way and ran BindCraft 2
out of DRAM at 480 tokens on Wormhole (04b88f631). A gc probe cannot see it either, because ttnn
tensors are not GC-tracked. So this is a source check. No device.
"""
import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1] / "tt_bio"
FILES = ["autograd.py", "taped_ttnn.py", "lnbw.py", "af2.py"]


@pytest.mark.parametrize("name", FILES)
def test_no_class_inside_a_function(name):
    tree = ast.parse((ROOT / name).read_text())
    found = [f"{fn.name}:{node.lineno} class {node.name}"
             for fn in ast.walk(tree) if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
             for node in ast.walk(fn) if isinstance(node, ast.ClassDef)]
    assert not found, f"{name} defines a class per call (hoist it to module level): {found}"
