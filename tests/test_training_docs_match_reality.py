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


def test_the_control_a_recipe_without_the_refusal_clears_the_gate():
    """Negative control: the gate must key on the code, not merely on the README's wording.

    Without this, the assertion above could be passing on some tree because the regex never
    matches anything rather than because doc and code agree.
    """
    refusal = _recipe_refuses_a_wide_dp_axis()
    assert refusal, (
        "the dp-width refusal was not found in tt_bio/train/recipes.py, so the gate above cannot "
        "tell an honest README from a lucky one. If the launcher was built and the guard removed, "
        "delete this control with it; if the guard was only reworded, update the pattern.")
    assert "NotImplementedError" in refusal and "launcher" in refusal, (
        "the refusal matched but does not look like the one this gate was written against; "
        "re-read it before trusting the assertion above.")
