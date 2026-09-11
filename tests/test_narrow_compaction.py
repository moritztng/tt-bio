"""Compaction between narrowing retries: off by default, and it runs when it is on.

Why it exists. On a Wormhole Galaxy chip the retry ladder runs against a device that each
attempt leaves fuller: measured on OpenDDE at 1088 tokens against an 8192-row alignment,
allocated per bank went 856 -> 874 -> 946 MiB across three narrowing attempts while the largest
free run fell 179 -> 101 -> 58 MiB, and the second attempt was refused **by 768 bytes** -- it
needed 101014272 B per bank against a largest free block of 101013504 B, with 199 MiB per bank
free. The memory was there and it was not in one piece.

The default stays False because reallocation moves every later allocation, which is a perf
question on every board including Blackhole, and this branch is not allowed to change a
shipped number. So the first test below is the one that matters for anyone merging this.
"""
from __future__ import annotations

import pytest

ttnn = pytest.importorskip("ttnn")
from tt_bio import tenstorrent as T  # noqa: E402

_OOM = RuntimeError("Out of Memory: Not enough space to allocate 100 B DRAM buffer "
                    "across 12 banks")


def _ladder(monkeypatch, on: bool, fail_times: int):
    monkeypatch.setattr(T, "_NARROW_COMPACT", on)
    calls = []

    def run(blk):
        calls.append(("run", blk))
        if len([c for c in calls if c[0] == "run"]) <= fail_times:
            raise _OOM
        return "ok"

    out = T._with_dram_narrowing(run, 1024, lambda b: b // 2,
                                 compact=lambda: calls.append(("compact", None)))
    return out, calls


def test_compaction_is_off_by_default():
    """The shipped default, read at import. Nothing on any board moves because of this branch."""
    assert T._NARROW_COMPACT is False


def test_with_the_flag_off_no_compaction_runs(monkeypatch):
    out, calls = _ladder(monkeypatch, on=False, fail_times=3)
    assert out == "ok"
    assert [c for c in calls if c[0] == "compact"] == []
    assert [b for k, b in calls if k == "run"] == [1024, 512, 256, 128]


def test_with_the_flag_on_it_compacts_once_per_retry(monkeypatch):
    """The negative control for the test above: same ladder, same refusals, and the only
    difference is the flag."""
    out, calls = _ladder(monkeypatch, on=True, fail_times=3)
    assert out == "ok"
    assert len([c for c in calls if c[0] == "compact"]) == 3
    assert calls[1] == ("compact", None), "compaction has to happen BEFORE the narrower retry"


def test_a_run_that_succeeds_first_time_never_compacts(monkeypatch):
    _out, calls = _ladder(monkeypatch, on=True, fail_times=0)
    assert calls == [("run", 1024)]


def test_a_failing_reallocate_is_not_swallowed(monkeypatch):
    """`reallocate` frees its input before it allocates, so a refusal inside compaction has
    already destroyed the operand. Retrying against a half-freed set would report a bookkeeping
    error instead of the allocator refusal that caused it."""
    monkeypatch.setattr(T, "_NARROW_COMPACT", True)

    def boom():
        raise RuntimeError("Out of Memory: Not enough space to allocate 1 B DRAM buffer "
                           "across 12 banks")

    with pytest.raises(RuntimeError, match="Out of Memory"):
        T._with_dram_narrowing(lambda b: (_ for _ in ()).throw(_OOM), 1024,
                               lambda b: b // 2, compact=boom)
