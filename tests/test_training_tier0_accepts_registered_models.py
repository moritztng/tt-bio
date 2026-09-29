"""Every model with a registered featuriser must be a name `--model` accepts.

This closes a blind spot in `test_training_catalogue_reachability.py`, which is the gate next
door and which I wrote one pass earlier. That one checks *featuriser implies registered*. It
does not check *registered implies reachable*, and the difference is not academic: on
`wk/train-b3-train` the ABodyBuilder3 featuriser exists, `catalogue.register("abodybuilder3",
...)` runs, `catalogue.names()` returns `['abodybuilder3']` from Python -- and the documented
Tier-0 command still refuses it:

    $ tt-bio finetune data/ --model abodybuilder3 ...
    Error: Invalid value for '--model': 'abodybuilder3' is not one of 'protenix-v2', 'openfold3'.

`--model` is `click.Choice(ADAPTABLE)` and `ADAPTABLE` (cli.py:30) never learned the name. Click
validates a Choice during parameter parsing, so the body's `abb3_dataset.register()` -- which is
deliberately late, to keep `--dry-run` free of the featuriser's imports, and that reasoning is
sound -- is never reached for the one model that has a featuriser. So the capability is real, the
Python path works, and the entry point the README names rejects it. One line of tuple.

That is the fourth time this campaign has shipped a capability its own documented entry point
refuses: the README's data-parallelism claim, the dry run's missing duration, the missing
registration, and now the Choice list. The shape is always the same -- the thing works one tier
down from where it is advertised -- so it is worth a gate per tier boundary rather than one
general gate, because each boundary fails in its own way.

Static and importless.
"""

import ast
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "tt_bio"
CLI = PKG / "train" / "cli.py"

pytestmark = pytest.mark.skipif(
    not CLI.is_file(), reason="tt_bio/train/cli.py is not on this tree yet")


def _label(path: Path) -> str:
    """Repo-relative where possible, absolute otherwise -- the control scans a tmp tree.

    Third time this exact trap has bitten a gate of mine: `Path.relative_to` raises rather than
    falling back when the path is outside the repo, and every one of these gates has a control
    that deliberately scans a temporary directory.
    """
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return os.fspath(path)


def _adaptable(source: str) -> tuple:
    """The `--model` Choice list, read from its assignment rather than hard-coded here."""
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", "") == "ADAPTABLE" for t in node.targets):
            value = node.value
            if isinstance(value, (ast.Tuple, ast.List)):
                return tuple(e.value for e in value.elts
                             if isinstance(e, ast.Constant) and isinstance(e.value, str))
    raise AssertionError(
        "tt_bio/train/cli.py no longer defines ADAPTABLE as a literal tuple. If the --model "
        "choices moved or became computed, re-cut this gate against the new source rather than "
        "deleting it -- a computed list derived FROM the registry would make this gate "
        "unnecessary, which is the better fix.")


def _registered_names(package: Path) -> dict:
    """`{name: "file:line"}` for every `catalogue.register("<literal>", ...)` under ``package``.

    Only literal first arguments are readable statically. A registration built from a variable
    is invisible here, and that is a real limit of this gate rather than something to paper
    over: the control below pins it so nobody mistakes silence for coverage.
    """
    found = {}
    for path in sorted(package.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, OSError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            func = node.func
            is_register = (
                (isinstance(func, ast.Attribute) and func.attr == "register"
                 and getattr(func.value, "id", "") == "catalogue")
                or (isinstance(func, ast.Name) and func.id == "register"))
            if not is_register:
                continue
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                found.setdefault(first.value,
                                 f"{_label(path)}:{node.lineno}")
    return found


def test_every_registered_featuriser_is_a_model_the_cli_will_accept():
    registered = _registered_names(PKG)
    if not registered:
        pytest.skip("nothing calls catalogue.register with a literal name yet, so no model is "
                    "reachable at Tier 0 by construction -- that gap is the neighbouring gate's")
    adaptable = _adaptable(CLI.read_text(encoding="utf-8"))
    unreachable = {n: where for n, where in registered.items() if n not in adaptable}
    assert not unreachable, (
        f"these models have a registered featuriser but are not a value `--model` accepts, so "
        f"`tt-bio finetune --model <them>` is refused by click BEFORE the registration in the "
        f"command body ever runs: {unreachable}. ADAPTABLE is {list(adaptable)} at "
        f"tt_bio/train/cli.py. The capability is real and reachable from Python; only the "
        f"documented entry point refuses it. Add the name to ADAPTABLE -- or better, derive the "
        f"choices from the registry so the two cannot disagree again.")


def test_the_control_the_gate_reads_both_lists_and_knows_its_own_limit():
    adaptable_src = 'ADAPTABLE = ("protenix-v2", "openfold3")\n'
    assert _adaptable(adaptable_src) == ("protenix-v2", "openfold3")
    assert _adaptable('ADAPTABLE = ["a", "b"]\n') == ("a", "b")

    import tempfile

    with tempfile.TemporaryDirectory() as d:
        pkg = Path(d) / "tt_bio"
        pkg.mkdir()
        (pkg / "a.py").write_text(
            'from tt_bio.train import catalogue\n'
            'catalogue.register("abodybuilder3", None)\n')
        # A bare `register` imported from the catalogue counts too.
        (pkg / "b.py").write_text(
            'from .catalogue import register\n'
            'register("openbind-0", None)\n')
        # A non-literal name is INVISIBLE to a static reader. Pinned so the limit is explicit.
        (pkg / "c.py").write_text(
            'from tt_bio.train import catalogue\n'
            'name = "computed"\n'
            'catalogue.register(name, None)\n')
        names = _registered_names(pkg)

    assert "abodybuilder3" in names, "an attribute call on `catalogue` must be read"
    assert "openbind-0" in names, "a bare imported `register` must be read"
    assert "computed" not in names, (
        "a registration built from a variable cannot be read statically; this assertion exists "
        "so the limit is recorded rather than discovered later as a false green")
