"""A DRAM refusal narrows the fp32-softmax row block instead of ending the fold.

`_FP32_SOFTMAX_BLOCK_BYTES` is an absolute budget and DRAM occupancy is not, so the two can
disagree and the budget loses. Measured 2026-09-02 on the GWH02 galaxy: OpenFold3 at 736 aa
against a 14190-row alignment asks for one unblocked fp32 score copy of 6 379 012 096 B --
under the 8 GiB budget, so nothing blocks it -- while the MSA track already holds enough that
the allocator has 358 MB per bank free. The budget was tuned against what allocates on an empty
device, and every model on this path meets it with whatever its own trunk already holds.

Host-only: these cover the decision, not the kernels.
"""
from __future__ import annotations

import pytest

from tt_bio import tenstorrent as tt


@pytest.fixture(autouse=True)
def clean():
    tt._FP32_SOFTMAX_DRAM_ROW_CAP.clear()
    before = dict(tt.FP32_SOFTMAX_STATS)
    yield
    tt._FP32_SOFTMAX_DRAM_ROW_CAP.clear()
    tt.FP32_SOFTMAX_STATS.update(before)


class TestRefusalDetection:
    def test_allocator_refusal_is_recognised(self):
        assert tt._fp32_softmax_dram_oom(RuntimeError(
            "TT_FATAL: Out of Memory: Not enough space to allocate 6379012096 B DRAM buffer "
            "across 12 banks"))

    def test_any_other_runtime_error_is_not(self):
        """A compile error retried at half the block would be reported as a smaller-is-fine pass."""
        for msg in ("Kernel compilation failed", "shape mismatch", "watcher detected a hang"):
            assert not tt._fp32_softmax_dram_oom(RuntimeError(msg))


class TestNarrowing:
    def test_halves_to_a_tile_multiple(self):
        assert tt._fp32_softmax_dram_narrow(("k",), 736) == 352      # 368 -> floor 32
        assert tt._fp32_softmax_dram_narrow(("k",), 352) == 160

    def test_reaches_one_tile_row_and_stops_there(self):
        blk, seen = 736, []
        for _ in range(12):
            blk = tt._fp32_softmax_dram_narrow(("k",), blk)
            seen.append(blk)
            if blk <= 32:
                break
        assert seen[-1] == 32, seen
        # Strictly decreasing: a narrow that returned its own input would spin forever, because
        # the retry loop only stops on blk <= 32 or a non-OOM error.
        assert all(b < a for a, b in zip([736] + seen, seen)), seen

    def test_a_small_block_still_makes_progress(self):
        assert tt._fp32_softmax_dram_narrow(("k",), 48) == 32
        assert tt._fp32_softmax_dram_narrow(("k",), 32) == 32

    def test_cap_is_per_shape_class_and_only_ever_tightens(self):
        tt._fp32_softmax_dram_narrow(("a",), 736)
        tt._fp32_softmax_dram_narrow(("b",), 128)
        assert tt._FP32_SOFTMAX_DRAM_ROW_CAP == {("a",): 352, ("b",): 64}
        tt._fp32_softmax_dram_narrow(("a",), 1024)     # a LOOSER cap must not win
        assert tt._FP32_SOFTMAX_DRAM_ROW_CAP[("a",)] == 352

    def test_every_narrow_is_counted(self):
        n = tt.FP32_SOFTMAX_STATS["dram_narrowed"]
        tt._fp32_softmax_dram_narrow(("a",), 736)
        tt._fp32_softmax_dram_narrow(("a",), 352)
        assert tt.FP32_SOFTMAX_STATS["dram_narrowed"] == n + 2


class TestNeutralityWhenNothingIsRefused:
    def test_no_refusal_leaves_no_cap_and_no_count(self):
        """The whole point of the memo: a fold that never gets refused takes the old path.

        Every size that folded before this change still picks its block from the byte budget
        alone, so `dram_narrowed == 0` on such a run is the check that the change was inert.
        """
        assert tt._FP32_SOFTMAX_DRAM_ROW_CAP == {}
        assert tt.FP32_SOFTMAX_STATS["dram_narrowed"] == 0


class TestRetryControlFlow:
    """The retry itself, with the block runner stubbed: no device, no ttnn tensors.

    The blocking loop in `_fp32_softmax_attention` needs real tensors, so before this the only
    coverage of what happens after a refusal was a fold on a Galaxy chip.
    """

    def test_a_call_that_fits_runs_once_and_narrows_nothing(self):
        seen = []
        out = tt._fp32_softmax_with_narrowing(lambda b: seen.append(b) or "o", 736, ("k",))
        assert out == "o" and seen == [736]
        assert tt.FP32_SOFTMAX_STATS["dram_narrowed"] == 0
        assert tt._FP32_SOFTMAX_DRAM_ROW_CAP == {}

    def test_one_refusal_halves_the_block_and_the_call_succeeds(self):
        seen = []

        def run(b):
            seen.append(b)
            if b == 736:
                raise RuntimeError("Out of Memory: Not enough space to allocate 6379012096 B")
            return "o"

        assert tt._fp32_softmax_with_narrowing(run, 736, ("k",)) == "o"
        assert seen == [736, 352]
        assert tt._FP32_SOFTMAX_DRAM_ROW_CAP[("k",)] == 352

    def test_it_keeps_halving_while_dram_keeps_refusing(self):
        seen = []

        def run(b):
            seen.append(b)
            if b > 64:
                raise RuntimeError("Out of Memory: Not enough space to allocate 1 B")
            return "o"

        assert tt._fp32_softmax_with_narrowing(run, 736, ("k",)) == "o"
        assert seen == [736, 352, 160, 64]

    def test_an_unshrinkable_block_re_raises_instead_of_spinning(self):
        calls = []

        def run(b):
            calls.append(b)
            raise RuntimeError("Out of Memory: Not enough space to allocate 1 B")

        with pytest.raises(RuntimeError, match="Out of Memory"):
            tt._fp32_softmax_with_narrowing(run, 736, ("k",))
        assert calls[-1] == 32 and len(calls) < 12, calls

    def test_a_non_allocator_error_propagates_untouched(self):
        """Retrying a compile error at half the block would report it as smaller-is-fine."""
        def run(b):
            raise RuntimeError("Kernel compilation failed")

        with pytest.raises(RuntimeError, match="compilation"):
            tt._fp32_softmax_with_narrowing(run, 736, ("k",))
        assert tt.FP32_SOFTMAX_STATS["dram_narrowed"] == 0
        assert tt._FP32_SOFTMAX_DRAM_ROW_CAP == {}
