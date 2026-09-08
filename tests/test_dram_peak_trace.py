"""`TT_BIO_DRAM_PEAK` is a footprint census, not a progress trace, and the difference cost a pass.

A tag only writes when its OWN high-water mark rises. The trunk's tags recur once per recycling
cycle, so cycle 1 and cycle 10 write the same single line and a fold that is grinding in the trunk
is indistinguishable from one that has finished with it -- which is exactly the read that failed on
the 896 aa rung (state/opendde-l1-clash-to-1024.md, Pass 10: the census ended at "pairformer block
46 done" for both a fold that had moved on and one that had not).

`TT_BIO_DRAM_PEAK_TRACE=1` writes every sample instead, with elapsed seconds and a per-tag sample
count, which is what localises a stall to a cycle. It must not disturb the census line itself: the
release gate's capacity leg matches on the "[DRAM] tag: N GiB used (of M GiB) maxfree=..." prefix,
so the trace fields are a SUFFIX and nothing else moves.

Host-only: the memory view is stubbed, no device.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from unittest import mock

import pytest

from tt_bio import tenstorrent as T

# The prefix the release gate's capacity leg reads.
GATE = re.compile(r"^\[DRAM\] (?P<tag>.+): \d+\.\d{3} GiB used \(of 12\.0 GiB\) maxfree=\d+MiB/bank")

#: The patterns that actually parse these files today, verified by reading them rather than by
#: trusting the comment in `dram_peak`. Every one stops at "GiB used" and takes a max over lines,
#: so a suffix cannot move a number and extra non-peak lines in trace mode cannot either.
#: Asserted as CONTENT, not existence: if a consumer rewrites its pattern, this fails here instead
#: of silently in a gate leg.
CONSUMERS = {
    "scripts/release_gate.py": r'r"^\[DRAM\] .*?: ([0-9.]+) GiB used"',
    "perf/bh1024/record.py": r'r"\[DRAM\] .*?: ([\d.]+) GiB used"',
    "perf/whb2/wh_cap.sh": r"'[0-9]+\.[0-9]+ GiB used'",
    "perf/whb2/wh_leverD.sh": r"'[0-9]+\.[0-9]+ GiB used'",
}


def test_every_consumer_of_this_file_stops_at_gib_used():
    """The suffix is only safe because nothing downstream reads past that field."""
    root = Path(T.__file__).resolve().parents[1]
    for rel, needle in CONSUMERS.items():
        text = (root / rel).read_text()
        assert needle in text, f"{rel} no longer parses with {needle}"
TRACE_SUFFIX = re.compile(r" t=\+\d+\.\d+s n=(?P<n>\d+)$")


class _MemoryView:
    """Just the fields `dram_peak` reads."""
    total_bytes_per_bank = 1 << 30
    num_banks = 12
    largest_contiguous_bytes_free_per_bank = 1 << 20

    def __init__(self, used_per_bank):
        self.total_bytes_free_per_bank = self.total_bytes_per_bank - used_per_bank


def _sample(path, trace, used_per_bank, tag="trunk msa block"):
    env = {"TT_BIO_DRAM_PEAK": str(path)}
    if trace:
        env["TT_BIO_DRAM_PEAK_TRACE"] = "1"
    with mock.patch.dict(os.environ, env, clear=False), \
         mock.patch.object(T, "get_device", lambda: None), \
         mock.patch.object(T.ttnn, "get_memory_view",
                           lambda d, b, u=used_per_bank: _MemoryView(u)):
        T.dram_peak(tag)


@pytest.fixture
def fresh(monkeypatch):
    monkeypatch.setattr(T, "_DRAM_PEAK", {})
    monkeypatch.setattr(T, "_DRAM_PEAK_N", {})
    monkeypatch.setattr(T, "_DRAM_PEAK_T0", None)
    monkeypatch.delenv("TT_BIO_DRAM_PEAK_TRACE", raising=False)


# A tag sampled three times while its footprint FALLS: the second and third samples are the
# ones the census throws away.
FALLING = [300 << 20, 100 << 20, 50 << 20]


def test_the_census_hides_every_sample_after_the_peak(fresh, tmp_path):
    """The defect, stated as a property rather than as a story."""
    path = tmp_path / "census.txt"
    for used in FALLING:
        _sample(path, False, used)
    lines = path.read_text().splitlines()
    assert len(lines) == 1                      # three samples in, one line out
    assert GATE.match(lines[0])
    assert not TRACE_SUFFIX.search(lines[0])    # and no trace fields by default


def test_the_trace_keeps_every_sample_and_numbers_them(fresh, tmp_path):
    path = tmp_path / "trace.txt"
    for used in FALLING:
        _sample(path, True, used)
    lines = path.read_text().splitlines()
    assert len(lines) == len(FALLING)
    for i, line in enumerate(lines, 1):
        m = TRACE_SUFFIX.search(line)
        assert m and int(m.group("n")) == i      # the per-tag counter is what separates cycles


def test_the_trace_does_not_move_the_line_the_release_gate_matches(fresh, tmp_path):
    """Same sample, both modes: everything before the suffix must be byte-identical."""
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    _sample(a, False, 300 << 20)
    T._DRAM_PEAK.clear()
    _sample(b, True, 300 << 20)
    census, traced = a.read_text().strip(), b.read_text().strip()
    assert traced.startswith(census)
    assert TRACE_SUFFIX.search(traced[len(census):] or traced)


def test_a_tag_that_never_rises_still_traces(fresh, tmp_path):
    """The stall case: a recurring tag at a flat footprint writes nothing in census mode and one
    line per cycle in trace mode."""
    path = tmp_path / "flat.txt"
    for _ in range(4):
        _sample(path, True, 100 << 20)
    assert len(path.read_text().splitlines()) == 4


def test_the_lazy_transition_branch_tags_on_entry():
    """A branch that only tags after its loop is invisible while it runs.

    The eager row branch calls `dram_peak` before its loop, so a slow call there shows up as a tag
    with no successor. The lazy branch (H > SEQ_LEN_MORE_CHUNKING) tagged only on the way out, so a
    Transition grinding over the 8192-row MSA representation wrote nothing at all and the trace
    stopped dead -- which is how the 896 aa fold read as "no tag reached in 13 minutes" while it was
    inside exactly one such call (state/opendde-l1-clash-to-1024.md, Pass 14).
    """
    src = open(T.__file__).read()
    enter = src.index('transition4d loop enter (lazy')
    done = src.index('transition4d loop done (lazy')
    assert enter < done, "the entry tag must precede the loop, not follow it"
    # and it must sit inside the large-sequence branch, above the row loop
    branch = src.index("if H > SEQ_LEN_MORE_CHUNKING:", src.index("def __call__", src.index("class Transition") if "class Transition" in src else 0))
    assert branch < enter < src.index("for s in range(0, H, transition_h_chunk_size)")
