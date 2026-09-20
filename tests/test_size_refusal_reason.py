"""The size guard's refusal has to be recorded with its reason, not with a bracket.

`tt_bio.size_limits.check` ends in a multi-line `raise SizeTooLargeError(`, so a traceback
contains the class name TWICE: once on the source line of the raise, which has no message
after it, and once on the final line, which has all of it. `_size_limit_refusal` took the
first match and split on the name, so every refused rung recorded the reason `"("`.

That is not cosmetic. A refused rung IS the measurement -- it is the record of the size at
which this model stops accepting work on this card -- and the ladder comparator prints the
stored string back at whoever reads the baseline. opendde's p150a fragment recorded
`{"1152": "(", "1280": "(", "1408": "(", "1536": "("}` on 2026-09-20: four rungs whose
refusal reason is one bracket, with the ceiling, the architecture and the alternatives that
`check` went to the trouble of composing thrown away.
"""
import importlib.util
import pathlib
import traceback

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def rg():
    spec = importlib.util.spec_from_file_location(
        "release_gate_refusal_reason", REPO_ROOT / "scripts" / "release_gate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _real_traceback_text():
    """A genuine traceback out of the shipped guard, not a hand-typed imitation.

    Written this way so the test tracks the raise site: if someone reflows the `raise` onto
    one line, or the message moves, this still asserts what a real fold log carries.
    """
    from tt_bio import size_limits
    try:
        size_limits.check("opendde", 1152, arch="blackhole")
    except size_limits.SizeTooLargeError:
        return traceback.format_exc()
    pytest.skip("opendde no longer refuses 1152 residues on blackhole")


def test_the_reason_is_the_guards_message_not_the_raise_sites_bracket(rg):
    text = _real_traceback_text()
    assert "raise SizeTooLargeError(" in text, "the raise site decoy is gone; test is stale"

    reason = rg._size_limit_refusal(text)

    assert reason is not None
    assert reason != "("
    # The three things a refusal is unactionable without: the model, the limit and the arch.
    assert "opendde" in reason
    assert "1024" in reason
    assert "blackhole" in reason


def test_a_log_with_no_refusal_line_is_not_a_refusal(rg):
    """A fold that died some other way must fall through to the error path.

    The regex is the whole discriminator now, so a log that merely mentions the class -- a
    source listing, a `grep`ped comment, a log line about handling one -- must not be read as
    the guard having refused this rung.
    """
    assert rg._size_limit_refusal("Traceback\n    raise SizeTooLargeError(\nMemoryError\n") is None
    assert rg._size_limit_refusal("# see SizeTooLargeError for the guard\n") is None
    assert rg._size_limit_refusal("") is None


def test_a_chained_raise_records_the_exception_that_was_raised(rg):
    """`raise X from Y` prints the cause FIRST. Taking the last match takes the real one."""
    text = (
        "Traceback (most recent call last):\n"
        "  File \"a.py\", line 1, in <module>\n"
        "tt_bio.size_limits.SizeTooLargeError: the cause\n"
        "\nThe above exception was the direct cause of the following exception:\n\n"
        "Traceback (most recent call last):\n"
        "  File \"b.py\", line 2, in <module>\n"
        "tt_bio.size_limits.SizeTooLargeError: what was actually raised\n"
    )
    assert rg._size_limit_refusal(text) == "what was actually raised"
