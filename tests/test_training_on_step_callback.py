"""`train_loop` hands each step out as it happens, not only at the end.

`history` is returned when the loop returns, so a run killed at step 90 of 100 returns nothing
and the 90 steps it did are gone. That is the normal way a long training arm ends on this fleet:
an OOM kill, a wedged card, a supersession. The curve is the deliverable of a training-outcome
grade, so it has to be on disk while the run is alive.

Static and importless, like `test_training_hook_protocol.py`: `recipes.py` reaches ttnn through
`.optim`, and this check has to run on a host with no wheel and no card.
"""
import ast
from pathlib import Path

RECIPES = Path(__file__).resolve().parents[1] / "tt_bio" / "train" / "recipes.py"


def _train_loop():
    tree = ast.parse(RECIPES.read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "train_loop":
            return node
    raise AssertionError("tt_bio/train/recipes.py declares no train_loop")


def test_on_step_is_a_keyword_with_a_default_of_none():
    fn = _train_loop()
    names = [a.arg for a in fn.args.kwonlyargs]
    assert "on_step" in names, f"train_loop takes no on_step; kwonly args are {names}"
    default = fn.args.kw_defaults[names.index("on_step")]
    assert isinstance(default, ast.Constant) and default.value is None, \
        "on_step must default to None, so a caller that does not pass one is unaffected"


def test_on_step_is_called_after_the_record_exists_and_before_the_checkpoint():
    fn = _train_loop()
    calls = [n for n in ast.walk(fn)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == "on_step"]
    assert calls, "on_step is declared but never called; the curve would still be lost"
    lines = sorted(n.lineno for n in ast.walk(fn)
                   if isinstance(n, ast.Call)
                   and isinstance(getattr(n.func, "attr", None), str)
                   and n.func.attr == "append")
    saves = sorted(n.lineno for n in ast.walk(fn)
                   if isinstance(n, ast.Call)
                   and getattr(n.func, "attr", None) == "save")
    assert lines and saves
    call = calls[0].lineno
    assert lines[0] < call < saves[0], (
        f"on_step at line {call} must sit between history.append ({lines[0]}) and "
        f"ckpt.save ({saves[0]}): after the record exists, before the step can die in a write")
