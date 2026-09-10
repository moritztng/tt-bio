"""_err_text: the job-row error has to survive truncation AND say which op asked.

A recorded OOM that names bytes but no site cannot be acted on. esmfold2 at 1536 tokens on
Blackhole was refused 4831838208 B with no attributable frame anywhere in the row, which is what
this pair of properties exists to prevent.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tt_bio import size_limits
from tt_bio.worker import _err_text, _origin_frame

_OOM = ("TT_FATAL @ /project/tt_metal/impl/allocator/bank_manager.cpp:439: false\ninfo:\n"
        "Out of Memory: Not enough space to allocate 4831838208 B DRAM buffer across 8 banks, "
        "where each bank needs to store 603979776 B, but bank size is 4278190016 B "
        "(allocated: 3579670528 B, free: 698519488 B, largest free block: 504088512 B)\n"
        "backtrace:\n" + "\n".join(f" --- frame {i}" for i in range(200)))


def _failed_inside_tt_bio():
    """A real exception raised by a real tt_bio frame, not a callback defined in this file.

    `describe_device_oom` runs a regex over its argument, so None makes re.finditer raise from
    inside size_limits.py, in `_last_refusal` -- the frame that owns the pattern. Monkeypatching
    a test-local function into a tt_bio module would put the test file in the traceback instead
    and the assertions below would pass on nothing.
    """
    try:
        size_limits.describe_device_oom(None)
    except TypeError as e:
        return e
    raise AssertionError("describe_device_oom(None) did not raise")


def test_origin_names_the_deepest_tt_bio_frame():
    origin = _origin_frame(_failed_inside_tt_bio())
    assert origin is not None
    assert origin.startswith("size_limits.py:")
    assert " in _last_refusal" in origin


def test_origin_is_none_when_no_tt_bio_frame_is_involved():
    """The negative control. A frame attributed to tt_bio when none ran would be a false lead."""
    try:
        raise ValueError("raised right here in the test, no tt_bio frame below it")
    except ValueError as e:
        assert _origin_frame(e) is None


def test_err_text_keeps_the_origin_after_truncation():
    """The origin is appended after the length budget, so the elision can never eat it."""
    big = RuntimeError(_OOM).with_traceback(_failed_inside_tt_bio().__traceback__)
    out = _err_text(big, limit=200)
    assert " ... " in out                   # the message itself really was truncated
    assert out.splitlines()[-1] == f"[tt_bio origin: {_origin_frame(big)}]"
    assert "size_limits.py:" in out.splitlines()[-1]


def test_err_text_still_keeps_both_ends_of_the_message():
    """The property the 2000-char budget existed for: head and tail both survive."""
    out = _err_text(RuntimeError(_OOM), limit=2000)
    assert out.startswith("TT_FATAL @")
    assert out.rstrip().endswith("frame 199")
