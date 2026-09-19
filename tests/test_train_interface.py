"""The four cut lines, the escape hatch, and the five invariants -- each as a test that can fail.

A tier boundary that is only described drifts. Every test here asserts the boundary rather than
documenting it, and each one is named after the one-line test in ``tt_bio/train/__init__.py``'s
table so the table cannot quietly disagree with the code.

    tier  cut line                                         test
    0     no callables in the signature                    test_tier0_*
    1     no `for` over steps in user code                 test_tier1_*
    2     no ttnn call in user code                        test_tier2_*
    3     ttnn appears here                                test_tier3_*

Host-only by construction. The escape-hatch check reads bytecode rather than importing the
recipe, so it runs on a machine with no ttnn wheel and no card -- which matters, because a gate
that only runs where a card is free is a gate that runs rarely. The arms that genuinely need a
device are marked and skipped, and they are marked as needing one rather than as optional.
"""
from __future__ import annotations

import ast
import dis
import importlib.util
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN = REPO_ROOT / "tt_bio" / "train"

def _has_ttnn() -> bool:
    """Is the wheel importable. Tolerates a module already in ``sys.modules``.

    ``find_spec`` raises ValueError rather than returning None for a module whose ``__spec__``
    is unset, which is what a hand-installed stand-in looks like. A gate that crashes at
    collection time on an unusual environment gates nothing.
    """
    import sys

    if "ttnn" in sys.modules:
        return True
    try:
        return importlib.util.find_spec("ttnn") is not None
    except (ImportError, ValueError):
        return False


HAS_TTNN = _has_ttnn()
needs_device = pytest.mark.skipif(not HAS_TTNN, reason="needs the ttnn wheel and a card")


def _module_ast(name: str) -> ast.Module:
    return ast.parse((TRAIN / name).read_text(), filename=name)


def _func_source(name: str, fn: str) -> str:
    """A function's source text, straight off the file. No import, so no ttnn."""
    text = (TRAIN / name).read_text()
    for node in ast.parse(text, filename=name).body:
        if isinstance(node, ast.FunctionDef) and node.name == fn:
            return ast.get_source_segment(text, node)
    raise AssertionError(f"{name} defines no function {fn}")


def _global_loads(source: str) -> set:
    """Every name the compiled code will look up in globals, nested code objects included.

    ``LOAD_GLOBAL`` rather than ``co_names`` or an AST walk: ``co_names`` also carries
    attribute names, so ``objectives.objective`` would look like a global called
    ``objective``, and an AST free-variable analysis has to re-derive scoping rules CPython
    already applied. This asks the compiler what it will actually reach for.
    """
    code = compile(source, "<recipe>", "exec")
    out, stack = set(), [code]
    while stack:
        c = stack.pop()
        for ins in dis.get_instructions(c):
            if ins.opname == "LOAD_GLOBAL":
                out.add(ins.argval)
        stack += [k for k in c.co_consts if hasattr(k, "co_code")]
    return out


def _program(code) -> list:
    """A code object as a comparable program: opcodes and the names they touch.

    Not raw ``co_code``. Two compilations of the SAME text legitimately differ there -- a
    nested code object embeds its ``co_filename``, jump targets shift with it, and CPython
    picks between ``LOAD_METHOD`` and ``LOAD_GLOBAL``+``LOAD_ATTR`` for an attribute call
    depending on how the enclosing code was compiled. Measured on this recipe: 383
    instructions either way, identical names and identical nested consts, with three
    positional differences and one such pair. Asserting raw equality would be asserting an
    artifact, and it would fail on a correct recipe the first time CPython changed a
    specialisation. What has to hold is that it is the same program.
    """
    collapse = {"LOAD_METHOD": "LOAD_ATTR"}
    out = []
    for ins in dis.get_instructions(code):
        name = collapse.get(ins.opname, ins.opname)
        if hasattr(ins.argval, "co_code"):
            out.append((name, f"<code {ins.argval.co_name}>"))
            out += [("  nested", x) for x in _program(ins.argval)]
        elif isinstance(ins.argval, (str, int, float, bool, type(None), tuple, frozenset)):
            # `argrepr` carries the jump offset for a branch; `argval` does not, which is why
            # this reads argval and drops anything that is not a name or a literal.
            out.append((name, None if ins.opname.startswith(("JUMP", "POP_JUMP", "FOR_ITER",
                                                             "SEND")) else ins.argval))
        else:
            out.append((name, None))
    return out


def _tier2_names() -> tuple:
    """``TIER2`` read out of the source, so this test needs no ttnn to know the vocabulary."""
    for node in _module_ast("__init__.py").body:
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", None) == "TIER2" for t in node.targets):
            return tuple(ast.literal_eval(node.value))
    raise AssertionError("tt_bio/train/__init__.py defines no TIER2 tuple")


# --------------------------------------------------------------- tier 0: no callables

def test_tier0_signature_has_no_callables():
    """Every Tier-0 knob is expressible on a command line, so its legality is decidable.

    This is the property that lets `--dry-run` answer on a laptop. A callable in the
    signature would make it undecidable, because the only way to learn what a callable does
    is to run it -- and running it is what `--dry-run` exists to avoid.
    """
    from tt_bio.train.cli import finetune

    offenders = []
    for p in finetune.params:
        if callable(p.default):
            offenders.append(f"{p.name} defaults to a callable")
        if p.type is None or not hasattr(p.type, "convert"):
            offenders.append(f"{p.name} has no command-line type")
    assert not offenders, (
        f"Tier 0 took something that is not a flag: {offenders}. The cut line is that a "
        f"Tier-0 run's legality is decidable before a device opens; a callable is not.")


def test_tier0_dry_run_opens_no_device():
    """`--dry-run` answers without importing ttnn. The cut line, executed."""
    import sys

    from click.testing import CliRunner

    from tt_bio.train.cli import finetune

    before = set(sys.modules)
    res = CliRunner().invoke(finetune, [
        str(REPO_ROOT), "--model", "protenix-v2", "--out", "/tmp/x",
        "--global-batch", "8", "--steps", "10", "--tokens", "256", "--dry-run"])
    assert res.exit_code == 0, res.output
    assert "fits" in res.output and "5.06 GB" in res.output, res.output
    newly = {m for m in set(sys.modules) - before if m.split(".")[0] == "ttnn"}
    assert not newly, f"--dry-run imported {newly}; it must answer before a device opens"


def test_tier0_refuses_a_measured_oom_rather_than_estimating():
    from click.testing import CliRunner

    from tt_bio.train.cli import finetune

    res = CliRunner().invoke(finetune, [
        str(REPO_ROOT), "--model", "protenix-v2", "--out", "/tmp/x",
        "--global-batch", "8", "--steps", "10", "--tokens", "384", "--dry-run"])
    assert res.exit_code != 0
    assert "75,497,472" in res.output, res.output


def test_tier0_recipe_names_match_the_recipes_module():
    """The flag's allowed values are pinned by hand so `--dry-run` need not import the tape.

    A hand-copied list is a duplication, so it gets a test rather than a comment. Reading the
    names out of the source keeps this arm host-only, which is the point of the duplication.
    """
    from tt_bio.train.cli import RECIPE_NAMES

    text = (TRAIN / "recipes.py").read_text()
    for node in ast.parse(text).body:
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") == "_RECIPES":
            shipped = tuple(ast.literal_eval(k) for k in node.value.keys)
            assert sorted(RECIPE_NAMES) == sorted(shipped), (
                f"tt_bio/train/cli.py pins {RECIPE_NAMES} but recipes.py ships {shipped}")
            return
    raise AssertionError("recipes.py defines no annotated _RECIPES mapping to check against")


def test_tier0_a_model_with_no_featuriser_refuses_by_name():
    """The README's claim, asserted: a run stops with a named error, before a device opens.

    What is missing is the featuriser, not the interface, and the message has to say which --
    a user who reads "not implemented" goes looking in the wrong place.
    """
    import sys

    from click.testing import CliRunner

    from tt_bio.train.cli import finetune

    before = set(sys.modules)
    res = CliRunner().invoke(finetune, [
        str(REPO_ROOT), "--model", "protenix-v2", "--out", "/tmp/tt-bio-train-test",
        "--global-batch", "8", "--steps", "2", "--tokens", "256"])
    assert res.exit_code == 1, res.output
    assert res.exception is None or isinstance(res.exception, SystemExit), res.exception
    assert "FEATURISER" in res.output and "catalogue.register" in res.output, res.output
    assert not {m for m in set(sys.modules) - before if m.split(".")[0] == "ttnn"}, (
        "the featuriser refusal opened a device stack it did not need")


def test_tier0_verb_is_registered_lazily_by_path():
    """`tt-bio --help` and `tt-bio predict` must not import the training stack to exist.

    Checked as AST because importing `tt_bio.main` needs the full inference dependency set.
    What matters here is that the registration is a STRING: an import statement in main.py
    would pull the tape in for every command, and the opt-in invariant test would catch that
    -- this asserts the mechanism that keeps it true rather than just its absence.
    """
    main = (REPO_ROOT / "tt_bio" / "main.py").read_text()
    tree = ast.parse(main)
    lazy = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
                getattr(x, "id", None) == "LAZY" for x in node.targets):
            lazy = ast.literal_eval(node.value)
    assert lazy == {"finetune": "tt_bio.train.cli:finetune"}, (
        f"the lazy command map is {lazy!r}. Tier 0's verb has to be registered by dotted "
        f"path, not imported, or every tt-bio command pays for the tape.")
    mod, attr = lazy["finetune"].split(":")
    target = importlib.import_module(mod)
    assert hasattr(target, attr), f"{mod} defines no {attr}"


@pytest.mark.skipif(importlib.util.find_spec("einops") is None,
                    reason="needs tt-bio's full inference dependency set")
def test_tier0_verb_appears_on_the_cli_without_importing_the_tape():
    """Listing the commands must not import the training CLI. Naming one must.

    In a subprocess, because the claim is about what `list_commands` imports and any earlier
    test that imports `tt_bio.train.cli` for its own reasons answers it first. Run in-process
    this passes or fails on collection order, which is how it came to fail on a tree where
    nothing was wrong with the CLI.
    """
    probe = textwrap.dedent("""
        import sys
        import click
        from tt_bio.main import cli
        ctx = click.Context(cli)
        assert "finetune" in cli.list_commands(ctx)
        assert "tt_bio.train.cli" not in sys.modules, (
            "listing the commands imported the training CLI; the whole point of the dotted "
            "path is that naming a command is what loads it")
        assert cli.get_command(ctx, "finetune").name == "finetune"
        assert cli.get_command(ctx, "preflight").name == "preflight"
        assert "tt_bio.train.cli" in sys.modules, (
            "naming the command did not load it, so the dotted path resolves to nothing")
        print("OK")
    """)
    r = subprocess.run([sys.executable, "-c", probe], cwd=REPO_ROOT,
                       capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip().endswith("OK"), r.stdout + r.stderr


# --------------------------------------------------------------- tier 1: no user `for`


def test_tier1_entry_point_owns_no_loop():
    """Tier 1 has no `for` over steps, and Tier 2 does. Both halves, so neither can drift.

    If the loop ever moves up into `finetune`, there are two implementations of it and the
    escape hatch stops being the same code -- which is the failure mode this whole file
    exists to prevent.
    """
    tier1 = _func_source("loop.py", "finetune")
    assert not [n for n in ast.walk(ast.parse(tier1)) if isinstance(n, (ast.For, ast.While))], (
        "tt_bio/train/loop.py:finetune grew a loop. Tier 1 resolves a recipe and calls it; "
        "the loop belongs to the Tier-2 recipe, and having it in both makes them two "
        "implementations that agree only today.")
    tier2 = _func_source("recipes.py", "train_loop")
    assert [n for n in ast.walk(ast.parse(tier2)) if isinstance(n, ast.For)], (
        "the Tier-2 recipe has no `for` over steps, so there is nothing for a Tier-2 user to "
        "own and the tier boundary is in the wrong place.")


def test_tier1_objective_is_a_name_not_a_callable():
    from tt_bio.train import objectives

    assert objectives.names(), "no objective rows registered"
    with pytest.raises(KeyError, match="no objective"):
        objectives.objective("does-not-exist")


# --------------------------------------------------------------- tier 2: no ttnn

def test_tier2_recipe_makes_no_ttnn_call():
    src = _func_source("recipes.py", "train_loop")
    assert "ttnn" not in src, "the Tier-2 recipe names ttnn; that is Tier 3's side of the line"
    assert "ttnn" not in _global_loads(src)


# --------------------------------------------------------------- tier 3: ttnn is here

def test_tier3_is_where_ttnn_appears():
    """The positive marker. A boundary asserted only by absence can be satisfied by a stub."""
    autograd = (REPO_ROOT / "tt_bio" / "autograd.py").read_text()
    assert "\nimport ttnn\n" in autograd, (
        "tt_bio/autograd.py no longer imports ttnn, so either Tier 3 moved or the tier table "
        "in tt_bio/train/__init__.py is wrong. One of the two needs fixing.")
    from tt_bio.train import checks as gc

    assert set(gc.OP_CLASSES) == {"eltwise", "reduction", "kink"}


# --------------------------------------------------------------- the escape hatch

def test_escape_hatch_recipe_uses_only_tier2_names():
    """THE load-bearing test. A recipe may reach for nothing but Tier 2, builtins and its args.

    r4's finding, mechanised: an escape hatch is only real if the tier above is expressible in
    the tier below's public API, and it only stays real if a test asserts it. The moment a
    recipe needs a private helper this fails, and then the helper becomes public or the recipe
    changes. There is no third outcome.
    """
    import builtins

    allowed = set(_tier2_names()) | set(dir(builtins))
    reached = _global_loads(_func_source("recipes.py", "train_loop"))
    leaked = sorted(reached - allowed)
    assert not leaked, (
        f"the Tier-1 recipe reaches globals that are not in TIER2: {leaked}. Either make them "
        f"public by adding them to tt_bio/train/__init__.py's TIER2 tuple -- a deliberate "
        f"widening of the contract -- or rewrite the recipe without them. What must not happen "
        f"is the hatch quietly depending on something a Tier-2 user cannot import.")


def test_escape_hatch_negative_control_a_private_reach_fails():
    """The check must be able to FAIL. A gate nobody has seen fail is not known to be a gate."""
    import builtins

    allowed = set(_tier2_names()) | set(dir(builtins))
    broken = "def r(x):\n    return _private_helper(x) + AdamW\n"
    leaked = sorted(_global_loads(broken) - allowed)
    assert leaked == ["_private_helper"], (
        f"the escape-hatch check did not catch a deliberate private reach; it found {leaked}. "
        f"Fix the check, not the control.")


@needs_device
def test_escape_hatch_imports_resolve_to_tier2_objects():
    """A recipe's imports must land on the same objects TIER2 promises, not on lookalikes.

    The bytecode check above proves the recipe reaches only TIER2 *names*. This proves the
    names in the recipe module are the TIER2 *objects*, which is what stops an import from
    a private module shadowing a public name.
    """
    import tt_bio.train as T
    from tt_bio.train import recipes

    for name in _tier2_names():
        if hasattr(recipes, name):
            assert getattr(recipes, name) is T.__getattr__(name), (
                f"recipes.{name} is not tt_bio.train.{name}; the recipe would be written "
                f"against something a Tier-2 user cannot reach by that name")


@needs_device
def test_escape_hatch_source_is_the_shipped_recipe():
    from tt_bio.train import recipes

    for name in recipes.names():
        src = recipes.source(name)
        assert src.strip().startswith("def "), src[:80]
        # rstrip: inspect.getsource keeps the trailing newline, ast.get_source_segment does
        # not. The text is what matters, not which of the two put a newline at the end.
        assert src.rstrip() == _func_source("recipes.py",
                                            recipes.recipe(name).__name__).rstrip(), (
            "recipes.source() returned text that is not the shipped function's own source")


# --------------------------------------------------------------- the five invariants

def test_invariant_gradcheck_bars_are_per_op_class_off_measured_floors():
    from tt_bio.train.checks import BARS, FD_BAR, MASK_BAR

    # One bar for all ops is wrong twice over: eltwise is fp32-exact at 3.0e-07 while a
    # reduction rounds like a matmul at 7.05e-03, and a kinked op is not mantissa-bounded.
    assert BARS[("eltwise", "float32")][0] == 3.0e-06
    assert BARS[("reduction", "float32")][0] == 2.5e-02
    assert BARS[("eltwise", "bfloat16")][0] == BARS[("reduction", "bfloat16")][0] == 1.0e-02
    assert BARS[("kink", "float32")][0] is None, (
        "a kinked op got an rel_L2 bar. A1 measured linear_relu at 0.0625 against 0.01 with "
        "HiFi4 reaching only 0.0128; it is scored on direction and its gate mask instead.")
    assert BARS[("eltwise", "float32")][0] < BARS[("reduction", "float32")][0]
    assert FD_BAR == 2e-6 and MASK_BAR == 0.99
    for (cls, dt), (_, _, why) in BARS.items():
        assert why.strip(), f"bar {(cls, dt)} carries no derivation; a bar without one drifts"


def test_invariant_gradcheck_verifies_its_reference_before_the_device():
    """Level 1 first: a reference that fails finite differences reports itself, not the device."""
    torch = pytest.importorskip("torch")
    import numpy as np

    from tt_bio.train.checks import gradcheck

    x = torch.tensor(np.random.default_rng(0).standard_normal((8, 4)), requires_grad=True)
    rep = gradcheck("eltwise-mul", ref_loss=lambda: (x * x).sum(), ref_params=[x],
                    tt_grads={"x": (2 * x).detach().numpy()},
                    op_class="eltwise", dtype="float32")
    assert rep.reference_verified and rep.fd_worst < 2e-6
    assert rep.passed

    # the control that must fail, so we know the check can
    x2 = torch.tensor(np.random.default_rng(0).standard_normal((8, 4)), requires_grad=True)
    bad = gradcheck("eltwise-mul-broken", ref_loss=lambda: (x2 * x2).sum(), ref_params=[x2],
                    tt_grads={"x": (2.2 * x2).detach().numpy()},
                    op_class="eltwise", dtype="float32")
    assert not bad.passed, "a 10 % wrong gradient passed; the check cannot fail"


def test_invariant_a_kinked_op_needs_a_gate_mask():
    torch = pytest.importorskip("torch")
    import numpy as np

    from tt_bio.train.checks import gradcheck

    x = torch.tensor(np.random.default_rng(0).standard_normal((8, 4)), requires_grad=True)
    with pytest.raises(ValueError, match="gate_mask"):
        gradcheck("relu", ref_loss=lambda: torch.relu(x).sum(), ref_params=[x],
                  tt_grads={"x": (x > 0).double().numpy()}, op_class="kink", dtype="float32")


def test_invariant_the_optimizer_refuses_a_bf16_master():
    from tt_bio.train import AdamW

    for bad in ("bfloat16", "bf16", "float16", "bfloat8_b"):
        with pytest.raises(ValueError, match="fp32 master"):
            AdamW({}, master_dtype=bad)
    assert AdamW({}, master_dtype="float32").master_dtype == "float32"


def test_invariant_the_step_control_is_cumulative_not_per_step():
    """Asserting per-step survival fails a healthy run; the cumulative ratio does not.

    `perf/ptxft`'s own measurement is the reason: with an fp32 master behind a bf16 device
    copy, a step below bf16 spacing is SUPPOSED to round to zero.
    """
    from tt_bio.train import AdamW
    from tt_bio.train.optim import DISPLACEMENT_BAND

    assert DISPLACEMENT_BAND == (0.9, 1.1)
    opt = AdamW({})
    with pytest.raises(RuntimeError, match="nothing has stepped"):
        opt.check_displacement()
    src = _func_source("optim.py", "check_displacement") if False else \
        (TRAIN / "optim.py").read_text()
    assert "displacement()" in src and "cumulative" in src.lower()


def test_invariant_plan_returns_unmeasured_rather_than_guessing():
    from tt_bio.train import plan
    from tt_bio.train.dryrun import UNMEASURED

    # measured: a replica at 256 aa is 5.06 GB of 34.23 GB
    fits = plan(tokens=256, chips=1, global_batch=8)
    assert fits.verdict == "fits" and abs(fits.replica_gb - 5.06) < 1e-9
    assert abs(fits.occupancy - 0.1478) < 5e-4

    # a measured OOM is a refusal, not an extrapolation
    assert plan(tokens=384, chips=1).verdict == "refused"
    assert plan(tokens=512, chips=1).verdict == "refused"

    # above the measured crop, and above a LoRA adapter, it says so
    assert plan(tokens=768, chips=1).verdict == UNMEASURED
    assert plan(tokens=256, chips=1, frozen_trunk=False).verdict == UNMEASURED

    # one, two and four chips are measured; three and eight are not, and there the step time
    # is withheld rather than scaled. Four arrived by measurement (train-w-fourchip: 3.686x at
    # 92.2 %), so what this pins is that the widths WITHOUT a measurement still refuse.
    assert plan(tokens=256, chips=2, seconds_per_step_1chip=2.0).seconds_per_step is not None
    four = plan(tokens=256, chips=4, seconds_per_step_1chip=2.0)
    assert abs(four.seconds_per_step - 2.0 / 3.686) < 1e-9
    assert any("train-w-fourchip" in s for s in four.sources)
    for unmeasured_width in (3, 8):
        p = plan(tokens=256, chips=unmeasured_width, seconds_per_step_1chip=2.0)
        assert p.seconds_per_step is None
        assert any(UNMEASURED in s for s in p.sources)
    for p in (plan(tokens=768, chips=1), plan(tokens=256, chips=1, frozen_trunk=False)):
        assert not p.measured and p.seconds_per_step is None and p.replica_gb is None


def test_invariant_the_optimizer_refuses_an_unreduced_step():
    from tt_bio.train import AdamW, Mesh, UnreducedGradients

    wide = Mesh({"dp": [0, 1]}).axis("dp")
    with pytest.raises(UnreducedGradients, match="diverging"):
        AdamW({}, data_parallel=wide).step()
    # a narrow axis is not a special case: it steps, and rejects replicas it cannot use
    narrow = Mesh({"dp": [0]}).axis("dp")
    AdamW({}, data_parallel=narrow).step()
    with pytest.raises(ValueError, match="no data-parallel axis"):
        AdamW({}, data_parallel=narrow).step(replicas={"w": [1, 2]})


def test_invariant_global_batch_is_never_derived_from_device_count():
    from tt_bio.train import Mesh, batches

    ax = Mesh({"dp": [0, 1]}).axis("dp")
    with pytest.raises(ValueError, match="does not divide"):
        list(batches(64, global_batch=7, steps=1, data_parallel=ax))
    got = list(batches(64, global_batch=8, steps=2, data_parallel=ax))
    assert [b.global_batch for b in got] == [8, 8]
    assert all(b.micro_batch == 4 for b in got), "micro-batch follows from global, not vice versa"
    # the same global batch on a wider axis is the same recipe, only sharded differently
    one = list(batches(64, global_batch=8, steps=2, seed=3))
    two = list(batches(64, global_batch=8, steps=2, seed=3, data_parallel=ax))
    assert [b.indices for b in one] == [b.indices for b in two]


def test_invariant_provenance_samples_the_clock_during_the_run():
    from tt_bio.train import provenance as pv

    with pv.during(seed=11, config={"k": 1}) as prov:
        pass
    assert prov.seed == 11 and prov.git_sha != "" and prov.seconds is not None
    assert "samples" in prov.aiclk, "no clock sampling happened at all"
    # the accuracy triple: a bar, the seed floor beside it, and no implied pass when unmeasured
    assert prov.passed is None and "not measured" in prov.summary()
    prov.deviation_a = 0.4
    assert prov.passed is True and "1.84 A" in prov.summary()
    prov.deviation_a = 0.7
    assert prov.passed is False
    assert pv.ARTIFACT_CLOCK_MHZ == 1200 and (pv.KILL_BAR_A, pv.SEED_FLOOR_A) == (0.60, 1.84)


def test_the_naming_collision_is_resolved():
    """`tt_bio.finetune` the MODULE is gone; `tt_bio.train.finetune` the FUNCTION is the one name.

    Both existing at once is the hazard this resolves, and the orchestrator's
    `test_the_training_package_defines_no_forward` is the gate that makes it mechanical:
    `AdamW` and `LoraConfig` under `tt_bio/train/` collide with a surviving top-level
    `finetune.py`, so leaving the module in place fails a test somebody else owns.
    """
    assert not (REPO_ROOT / "tt_bio" / "finetune.py").exists(), (
        "tt_bio/finetune.py is back. It shadows tt_bio.train.finetune, and its AdamW and "
        "LoraConfig collide with tt_bio/train/'s.")
    import tt_bio.train as T

    # Every folded name still resolves to a module under tt_bio/train/, and `finetune` is
    # now unambiguously the Tier-1 function. Checked through the resolution table rather
    # than with hasattr, so this arm needs no ttnn: hasattr would import lora.py.
    for name, where in (("AdamW", "optim"), ("af3_lr", "optim"), ("LoraConfig", "lora"),
                        ("lora_factors", "lora"), ("save_adapter", "checkpoint"),
                        ("load_adapter", "checkpoint"), ("to_host", "tensors"),
                        ("finetune", "loop")):
        assert T._WHERE[name] == where, (
            f"{name} resolves to {T._WHERE.get(name)!r}, not {where!r}; the fold moved it "
            f"somewhere else or it did not survive")
    for name in ("AdamW", "af3_lr", "save_adapter", "to_host"):
        assert hasattr(T, name)   # the ttnn-free half, resolved for real


@needs_device
def test_the_two_recipe_readers_agree():
    """The CLI reads recipe text off the file; `recipes.source()` reads it through `inspect`.

    Two readers exist so `--show-recipe` needs no wheel. Two readers of the same thing is a
    duplication, so it gets a test rather than a comment.
    """
    from tt_bio.train import recipes
    from tt_bio.train.cli import _recipe_text

    for name in recipes.names():
        assert _recipe_text(name).rstrip() == recipes.source(name).rstrip()


def test_show_recipe_needs_no_wheel():
    """Printing a program is a text operation. It must work with no ttnn and no card."""
    import sys

    from click.testing import CliRunner

    from tt_bio.train.cli import finetune

    before = set(sys.modules)
    res = CliRunner().invoke(finetune, ["--show-recipe"])
    assert res.exit_code == 0, res.output
    assert res.output.startswith("def train_loop("), res.output[:120]
    assert "for batch in" in res.output, "the printed body has no loop to own"
    assert not {m for m in set(sys.modules) - before if m.split(".")[0] == "ttnn"}
    bad = CliRunner().invoke(finetune, ["--show-recipe", "nope"])
    assert bad.exit_code != 0 and "recipes are ['default']" in bad.output, bad.output


@needs_device
def test_escape_hatch_source_execs_against_tier2_and_is_the_same_program():
    """Exec the returned source in a namespace holding NOTHING but Tier 2, and compare.

    This is the executable form of the hatch. The bytecode-name check above proves the recipe
    reaches only Tier-2 names; this proves the text actually RUNS with only those names bound
    -- an import the recipe module happens to have would not save it here -- and that what
    comes out is the same program the Tier-1 call runs, instruction for instruction.

    Needs the wheel only because importing `recipes` imports the tape.
    """
    import inspect

    import tt_bio.train as T
    from tt_bio.train import recipes

    for name in recipes.names():
        shipped = recipes.recipe(name)
        ns = T.tier2()
        exec(compile(recipes.source(name), "<hatch>", "exec"), ns)
        rebuilt = ns[shipped.__name__]
        assert inspect.signature(rebuilt) == inspect.signature(shipped)
        assert _program(rebuilt.__code__) == _program(shipped.__code__), (
            f"recipe {name!r} exec'd from source() is not the same program as the one Tier 1 "
            f"runs. Since Tier 1 calls this very function, that can only mean source() is "
            f"returning something else.")


@needs_device
def test_escape_hatch_negative_control_exec_fails_without_a_private_name():
    """The exec arm must reject a body that needs something Tier 2 does not export."""
    import tt_bio.train as T

    ns = T.tier2()
    with pytest.raises(NameError):
        exec(compile("def r():\n    return _private_helper(1)\n", "<bad>", "exec"), ns)
        ns["r"]()
