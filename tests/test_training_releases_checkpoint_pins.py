"""A loop that runs a backward must release the pins that backward's forward took.

`autograd.checkpoint` pins every input of every checkpointed segment into the module-global
`_CKPT_PINS`, and it has to: the recompute happens inside the backward, so the pin must
outlive the forward and nothing inside `checkpoint` can know when it is dead. The release is
therefore owed by whatever drives the loop, and `release_pins()` is how it pays.

`train_loop` did not pay it. At crop 384 the OpenFold3 composition takes 120 pins a step --
48 pairformer blocks and 24 DiT blocks -- holding 1.02 GB of card DRAM that the next step
cannot have. Six steps stand at 8.1 GB under a backward whose own transient is ~23 GB, and
the seventh is refused a 906 MB buffer. Every OF3T training claim was capped at six steps by
it, which at two to six steps is fewer steps than the data draw needs to mean anything.

Static and importless, like `test_training_on_step_callback.py`: it has to run on a host with
no wheel and no card.
"""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "tt_bio"


def _backward_call(node) -> bool:
    if not isinstance(node, ast.Call):
        return False
    f = node.func
    return (isinstance(f, ast.Name) and f.id == "backward") or \
           (isinstance(f, ast.Attribute) and f.attr == "backward")


def _releases(node) -> bool:
    return any(isinstance(n, ast.Call)
               and getattr(n.func, "id", getattr(n.func, "attr", None)) == "release_pins"
               for n in ast.walk(node))


def _loops_that_backward():
    """Every `for`/`while` body in the package that drives a backward, innermost first."""
    out = []
    for path in sorted(ROOT.rglob("*.py")):
        if "_vendor" in path.parts:
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, (ast.For, ast.While)):
                continue
            if any(_backward_call(n) for n in ast.walk(node)):
                out.append((path.relative_to(ROOT.parent), node))
    return out


def test_the_package_has_a_loop_that_backwards():
    found = _loops_that_backward()
    assert found, ("no loop in tt_bio drives a backward, so this guard is watching nothing. "
                   "Either the training loop moved or the AST shape changed")


def test_every_loop_that_backwards_releases_its_pins():
    missing = []
    for rel, node in _loops_that_backward():
        # The innermost loop containing the backward owns the release; an outer loop that
        # contains it satisfies the rule through the inner one.
        inner = [n for n in ast.walk(node)
                 if isinstance(n, (ast.For, ast.While)) and n is not node
                 and any(_backward_call(c) for c in ast.walk(n))]
        if inner:
            continue
        if not _releases(node):
            missing.append(f"{rel}:{node.lineno}")
    assert not missing, (
        "these loops run a backward and never call release_pins(), so every checkpointed "
        "segment's input stays pinned and allocated for the rest of the process: "
        + ", ".join(missing))


def test_train_loop_releases_after_the_backward_and_before_the_next_sample():
    src = (ROOT / "train" / "recipes.py").read_text()
    fn = next(n for n in ast.parse(src).body
              if isinstance(n, ast.FunctionDef) and n.name == "train_loop")
    bwd = [n.lineno for n in ast.walk(fn) if _backward_call(n)]
    rel = [n.lineno for n in ast.walk(fn)
           if isinstance(n, ast.Call)
           and getattr(n.func, "id", getattr(n.func, "attr", None)) == "release_pins"]
    step = [n.lineno for n in ast.walk(fn)
            if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "step"]
    assert bwd and rel and step, f"backward {bwd}, release_pins {rel}, opt.step {step}"
    assert min(bwd) < min(rel) < min(step), (
        f"release_pins at {min(rel)} must come after the backward at {min(bwd)} -- the "
        f"recompute reads the pinned values -- and before the optimizer at {min(step)}, so "
        f"one sample's pins are gone before the next sample builds its own")
