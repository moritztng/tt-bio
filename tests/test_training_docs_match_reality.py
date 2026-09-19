"""The README's works-today list, checked against what the code actually does.

Done-definition item 5 is that a third party can follow the docs and train something, so the
README is a deliverable rather than a description of one. This gate exists because the first
time anyone read it as a user, it was wrong in the one place that mattered most.

The README lists "single-box data parallelism up to 4 chips" under **what works today**. The
Tier-1 recipe -- the thing `tt-bio finetune` and `train.finetune(...)` actually run -- refuses
any data-parallel axis wider than one chip, and says so itself: "this recipe is a single process,
so it only ever holds one replica's gradient ... The launcher for it is not built." Verified by
execution against a 2-chip mesh, which raises before a device is opened.

Both statements are defensible in isolation. The optimizer API really does support data
parallelism -- `data_parallel=`, `step(replicas=...)` and the `UnreducedGradients` guard are all
built and tested. What is missing is the multi-process launcher, so DP is reachable by someone
who writes their own Tier-2 loop AND their own launcher, and unreachable at the Tier-0 and
Tier-1 entry points the README is describing at that point. "Works today" in a section about a
command line means the command line does it.

That distinction is exactly what a user cannot see and what Moritz asked for by name -- "very
easy to train a model across multiple cards" -- so the gate keys on the CODE and only then reads
the prose: while the recipe refuses a wide axis, the works-today sentence may not claim data
parallelism. Delete the refusal and the claim becomes true and this test goes quiet on its own.

Static and importless, so it runs with no wheel and no card.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
RECIPES = ROOT / "tt_bio" / "train" / "recipes.py"

pytestmark = pytest.mark.skipif(
    not RECIPES.is_file(),
    reason="tt_bio/train/recipes.py is not on this tree yet (the training interface has not "
           "landed), so there is no works-today claim to check against")


SUGGESTED = (
    "the interface, the dry run, the LoRA adapters, the optimizer, gradient checking and "
    "checkpoints, all on one chip. Data parallelism is built in the optimizer and needs a "
    "multi-process launcher that is not written yet, so a multi-chip run is Tier 2 plus your own "
    "launcher rather than a flag"
)


def _recipe_refuses_a_wide_dp_axis() -> str:
    """The recipe's own refusal, if it has one. Returns the message or ''."""
    src = RECIPES.read_text()
    # Find the guard, then take a fixed window after it rather than trying to match to the
    # closing paren: the message itself contains `opt.step(replicas=...)`, so a lazy `\)` match
    # truncates mid-message and loses the very words the control below checks for.
    m = re.search(r"if\s+dp\.width\s*>\s*1\s*:", src)
    if not m:
        return ""
    window = src[m.start():m.start() + 1200]
    return window if "NotImplementedError" in window else ""


def _recipe_hands_a_wide_dp_axis_to_the_launcher() -> str:
    """The dispatch that replaced the refusal, if it is there. Returns the window or ''.

    The other half of keying on code. While the recipe refused a wide axis, the honest README
    was one that did not claim data parallelism, and the control below proved the gate could
    see that refusal. The launcher landed, so the fact that decides the same question is now
    the recipe handing a wide axis to it, and the control proves the gate can see THAT. Either
    way the gate reads `tt_bio/train/recipes.py`; what it reads there changed once.
    """
    src = RECIPES.read_text()
    m = re.search(r"if\s+dp\.width\s*>\s*1\s+and\s+launcher\.driving\(\)\s*:", src)
    if not m:
        return ""
    window = src[m.start():m.start() + 400]
    return window if "launcher.drive(" in window else ""


def _works_today_sentence() -> str:
    if not README.is_file():
        return ""
    txt = README.read_text()
    m = re.search(r"\*\*What works today:\*\*(.{0,600}?)\*\*What does not", txt, re.S)
    return m.group(1) if m else ""


def test_the_readme_does_not_claim_data_parallelism_works_while_the_recipe_refuses_it():
    refusal = _recipe_refuses_a_wide_dp_axis()
    claim = _works_today_sentence()
    if not claim:
        pytest.skip("the README has no '**What works today:**' sentence to check")
    claims_dp = re.search(r"data[- ]parallel(ism)?", claim, re.I)
    if not refusal:
        # The launcher landed, or the guard moved. Either way the claim is no longer contradicted
        # by this file and the gate has nothing to say.
        pytest.skip("tt_bio/train/recipes.py no longer refuses a >1-wide dp axis, so the "
                    "works-today claim is not contradicted here")
    assert not claims_dp, (
        "README's '**What works today:**' sentence claims data parallelism, but the Tier-1 recipe "
        "in tt_bio/train/recipes.py refuses any dp axis wider than one chip -- so `tt-bio "
        "finetune` and `train.finetune(...)`, which is what that part of the README is about, "
        "raise NotImplementedError on it. The recipe's own message says the launcher is not "
        "built.\n\n"
        "The optimizer's DP support IS real (data_parallel=, step(replicas=...), "
        "UnreducedGradients), so the fix is to say where it is reachable from rather than to drop "
        "it. Suggested replacement for the works-today list:\n\n    " + SUGGESTED + "\n\n"
        "Delete the recipe's refusal instead -- i.e. build the launcher -- and this test goes "
        "quiet by itself.")


def test_the_control_the_gate_still_reads_the_code_and_not_the_prose():
    """Negative control: the gate must key on the code, not merely on the README's wording.

    Without this, the assertion above could be passing on some tree because the regex never
    matches anything rather than because doc and code agree.

    One of the two facts about `recipes.py` has to be findable: either it still refuses a wide
    dp axis, in which case the works-today claim must not mention data parallelism, or it
    hands one to the launcher, in which case the claim is true. Neither found means the gate
    is reading nothing and the assertion above is passing for free -- which is the failure
    this control exists to make loud. `train-d-dp-launcher` built the launcher and removed the
    refusal, and the arm below moved with it rather than being deleted.
    """
    refusal = _recipe_refuses_a_wide_dp_axis()
    dispatch = _recipe_hands_a_wide_dp_axis_to_the_launcher()
    assert refusal or dispatch, (
        "tt_bio/train/recipes.py neither refuses a >1-wide dp axis nor hands one to "
        "tt_bio.train.launcher, so the gate above cannot tell an honest README from a lucky "
        "one -- it is matching nothing and skipping. Re-read the recipe and update whichever "
        "pattern moved.")
    if refusal:
        assert "NotImplementedError" in refusal and "launcher" in refusal, (
            "the refusal matched but does not look like the one this gate was written "
            "against; re-read it before trusting the assertion above.")
    else:
        assert "launcher.drive(" in dispatch, (
            "the dispatch matched but does not call launcher.drive(), so a wide axis reaches "
            "no launcher and the works-today claim is false again.")


# ---------------------------------------------------------------------------
# The second claim in the same block, found by running the command the README
# prints (train-orchestrator pass 32).
#
# The README's dry-run example carries the comment "will this fit, and how long? answered
# without opening a card". Run it and the answer is half an answer:
#
#     plan: fits at 256 tokens on 1 chip(s) -- 5.06 GB of 34.23 GB (14.8 %)
#       fits, at the measured 256 aa replica size.
#       - replica: train-r5 REPLICA-FITS: 0.929 GB bf16 weights ...
#
# Nothing about how long. And `plan()` is right to withhold it: no Protenix-v2 training step
# has been measured on this hardware, `seconds_per_step` stays None unless the caller passes
# its own `seconds_per_step_1chip`, and dryrun.py's whole stated design is that UNMEASURED is
# a first-class answer rather than a projection with a plausible shape. That part is correct
# and must not change.
#
# The defect is that the user is told nothing. `cli.py` prints the UNMEASURED note only when
# `not fit.measured`, which is a property of the WHOLE verdict -- and the verdict here is
# "fits", because the memory half IS measured. So a per-half UNMEASURED is invisible, and a
# reader cannot tell the difference between "we answered how long" and "we silently skipped
# the question". Silence where the doc promised an answer is the same defect class as the
# data-parallelism claim above, just quieter.
#
# The fix belongs in the dry-run print path and is a few lines: answer the duration question
# explicitly, as UNMEASURED with dryrun.py's own reason, whenever `seconds_per_step is None`.
# `tt_bio/train/cli.py` has a live owner (`train-b3-train`), so this gate states the contract
# and the owner lands it rather than the orchestrator reaching across the partition.

CLI = ROOT / "tt_bio" / "train" / "cli.py"


def _readme_dryrun_block() -> str:
    text = README.read_text(encoding="utf-8")
    for block in re.findall(r"```bash\n(.*?)```", text, re.S):
        if "--dry-run" in block:
            return block
    return ""


def _dry_run_branch_source(cli_text: str) -> str:
    """The source of the `if dry_run:` statement inside the finetune command, or ""."""
    import ast

    tree = ast.parse(cli_text)
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if isinstance(test, ast.Name) and test.id == "dry_run":
            return ast.get_source_segment(cli_text, node) or ""
    return ""


def _plan_call_source(cli_text: str) -> str:
    """The source of the `plan(...)` call inside the finetune command, or "".

    The print path and the call site are two separate ways to break the same promise, and only
    one of them is visible in the branch. A dry run can say "duration: UNMEASURED" forever while
    `plan()` is never given the one argument that would let it answer -- which is what `main`
    does at cli.py:219 -- so the gate reads both.
    """
    import ast

    tree = ast.parse(cli_text)
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "plan"):
            return ast.get_source_segment(cli_text, node) or ""
    return ""


@pytest.mark.skipif(not CLI.is_file(), reason="tt_bio/train/cli.py is not on this tree yet")
def test_the_dry_run_answers_the_duration_question_the_readme_promises_it_answers():
    block = _readme_dryrun_block()
    if "how long" not in block:
        pytest.skip("the README's dry-run example no longer promises a duration, so there is "
                    "nothing for the command to owe -- the claim was withdrawn instead of "
                    "delivered, which is a legitimate fix and this gate goes quiet for it")

    branch = _dry_run_branch_source(CLI.read_text(encoding="utf-8"))
    assert branch, ("could not find an `if dry_run:` branch in tt_bio/train/cli.py, so this "
                    "gate cannot see what the dry run prints. Re-cut the gate rather than "
                    "deleting it")
    assert "seconds_per_step" in branch, (
        "the README's dry-run example promises 'will this fit, and how long?' and the command "
        "prints only the fit. `plan()` withholding an unmeasured step time is CORRECT and stays "
        "-- what is missing is saying so: cli.py's dry-run branch prints the UNMEASURED note "
        "only when `not fit.measured`, which is true of the whole verdict, so a measured-memory "
        "/ unmeasured-duration plan says nothing about duration at all. Answer it explicitly "
        "when `fit.seconds_per_step is None`, quoting dryrun.py's own reason. Owner of "
        "tt_bio/train/cli.py: train-b3-train. Alternative fix: drop 'and how long' from the "
        "README comment, which makes this gate skip.")

    call = _plan_call_source(CLI.read_text(encoding="utf-8"))
    assert call, ("could not find a `plan(...)` call in tt_bio/train/cli.py, so this gate cannot "
                  "see whether the command gives plan() a way to answer. Re-cut it rather than "
                  "deleting it")
    assert "seconds_per_step_1chip" in call, (
        "the dry run says it answers 'how long' and `plan()` is called without "
        "`seconds_per_step_1chip`, so the answer can only ever be UNMEASURED. `dryrun.py` prints "
        "a step time the moment it is given one; the command has to pass the user's measurement "
        "through. A print path that explains the silence is not the same as a command that can "
        "break it.")


def test_the_control_the_duration_gate_reads_the_print_path_and_not_the_module():
    """A `seconds_per_step` anywhere else in cli.py must not satisfy the gate.

    Without this control the check would pass the moment the string appeared in an import, a
    docstring or an unrelated branch -- which is how a gate comes to pass on a tree that still
    has the defect. Pass 25's own gate needed the same control for the same reason.
    """
    defective = (
        "def finetune(dry_run, fit):\n"
        "    fit = plan(tokens=256)  # seconds_per_step lives here, not in the branch\n"
        "    if dry_run:\n"
        "        click.echo('fits')\n"
        "        return\n"
    )
    fixed = (
        "def finetune(dry_run, fit):\n"
        "    fit = plan(tokens=256)\n"
        "    if dry_run:\n"
        "        if fit.seconds_per_step is None:\n"
        "            click.echo('duration: UNMEASURED -- ' + fit.why)\n"
        "        return\n"
    )
    assert "seconds_per_step" not in _dry_run_branch_source(defective)
    assert "seconds_per_step" in _dry_run_branch_source(fixed)


def test_the_control_the_call_site_half_of_the_duration_gate_can_fail():
    """Both directions for the second half, for the reason the first half has a control.

    `main` at 638187138 is the defective shape verbatim: a dry run that prints, and a `plan()`
    call with no way to answer. A gate that could not fail on it would be decoration.
    """
    defective = (
        "def finetune(tokens, chips):\n"
        "    fit = plan(tokens=tokens or 256, chips=chips, frozen_trunk=True)\n"
        "    # seconds_per_step_1chip is mentioned here and nowhere that matters\n"
    )
    fixed = (
        "def finetune(tokens, chips, seconds_per_step):\n"
        "    fit = plan(tokens=tokens or 256, chips=chips, frozen_trunk=True,\n"
        "               seconds_per_step_1chip=seconds_per_step)\n"
    )
    assert "seconds_per_step_1chip" not in _plan_call_source(defective)
    assert "seconds_per_step_1chip" in _plan_call_source(fixed)
