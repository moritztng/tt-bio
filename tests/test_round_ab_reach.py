"""The A/B's lever counters must move with the lever's EFFECT, not with what it was handed.

This file exists because of one lost card session. On 2026-09-26 at 04:04Z a 28-round A/B ran
to completion and reported nothing: the softmax arm was counted as the dtype of `y` read
BEFORE `softmax_bw_dx`, and the lever narrows `y` INSIDE it, so the counter was structurally
incapable of separating the arms. `check_fired` then raised "wired but inert" -- a verdict about
the instrument, printed as a verdict about the lever.

Every test here keeps a control that fails on the old instrument, because a counter that cannot
move is indistinguishable from a lever that did not fire unless something pins the difference.
"""
import collections
import sys

import pytest

sys.path.insert(0, "perf/bcx_bwbytes")
sys.path.insert(0, ".")


class FakeDType:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return f"DataType.{self.name}"


F32, BF16 = FakeDType("FLOAT32"), FakeDType("BFLOAT16")


class FakeTensor:
    def __init__(self, dtype):
        self.dtype = dtype


@pytest.fixture
def wired():
    """Install the harness's levers over stubs, and put every patched module attribute back.

    The stubs stand in for the engine calls, not for the harness: `install_levers` wraps whatever
    `ag.softmax_bw_dx` and `R.permute_via_reblock` are at install time, so a stub that behaves
    like the real one exercises the real wrapper.
    """
    import round_ab
    from tt_bio import autograd as ag, reblock_permute as R

    saved = {
        "dx": ag.softmax_bw_dx, "pvr": R.permute_via_reblock,
        "back": R.reblock_permute_back, "fwd": R.reblock_permute,
        "tree": ag._pairwise_sum0, "use": ag._use_tree,
        "dtype": ag.SOFTMAX_BW_DTYPE, "route": ag.SOFTMAX_BW_ROUTE,
    }

    def fake_dx(y, g, dim=-1, config=None):
        """The real `softmax_bw_dx`'s decision structure: narrow, count, route."""
        if ag.SOFTMAX_BW_DTYPE == "bf16":
            if y.dtype is not BF16:
                y = FakeTensor(BF16)
                ag.SOFTMAX_BW_DTYPE_STATS["narrowed_y"] += 1
            if g.dtype is not BF16:
                ag.SOFTMAX_BW_DTYPE_STATS["narrowed_g"] += 1
        if ag.SOFTMAX_BW_ROUTE == "moreh":
            ag.SOFTMAX_BW_DTYPE_STATS["moreh"] += 1
        return FakeTensor(y.dtype)

    def fake_pvr(x, dims, memory_config=None):
        """Declines the way `eligible_back` declines: through `_reject`, into `REJECTS`."""
        R._reject("back_window_DRAM", [1, 64, 224, 224])
        return x

    ag.softmax_bw_dx, R.permute_via_reblock = fake_dx, fake_pvr
    ag.SOFTMAX_BW_DTYPE_STATS.clear()
    R.REJECTS.clear()
    round_ab.COUNT.clear()
    round_ab.ARM["on"] = False
    yield round_ab, ag, R
    ag.softmax_bw_dx, R.permute_via_reblock = saved["dx"], saved["pvr"]
    R.reblock_permute_back, R.reblock_permute = saved["back"], saved["fwd"]
    ag._pairwise_sum0, ag._use_tree = saved["tree"], saved["use"]
    ag.SOFTMAX_BW_DTYPE, ag.SOFTMAX_BW_ROUTE = saved["dtype"], saved["route"]
    ag.SOFTMAX_BW_DTYPE_STATS.clear()
    R.REJECTS.clear()
    round_ab.COUNT.clear()
    round_ab.ARM["on"] = False


def _both_arms(round_ab, apply, call):
    for on in (False, True):
        round_ab.ARM["on"] = on
        apply()
        call()


def test_the_softmax_counter_separates_the_arms_on_the_returned_dtype(wired):
    round_ab, ag, R = wired
    apply = round_ab.install_levers(1 << 30, precision=True, moreh=False)
    _both_arms(round_ab, apply,
               lambda: ag.softmax_bw_dx(FakeTensor(F32), FakeTensor(F32)))

    assert round_ab.COUNT["softmax_bw:off:FLOAT32"] == 1
    assert round_ab.COUNT["softmax_bw:on:BFLOAT16"] == 1
    assert round_ab.COUNT["softmax_bw_narrowed:on"] == 1
    assert round_ab.COUNT["softmax_bw_narrowed:off"] == 0


def test_the_input_dtype_counter_is_the_control_and_cannot_separate(wired):
    """The lost session's instrument, kept and pinned as a control.

    Both arms are handed fp32 by construction, so `softmax_bw_in:` reads identically whatever
    the lever does. If this ever separates, the operands changed and the other test's evidence
    needs re-reading; if `softmax_bw:` ever stops separating where this one does not, the
    wrapper has drifted back to counting the input.
    """
    round_ab, ag, R = wired
    apply = round_ab.install_levers(1 << 30, precision=True, moreh=False)
    _both_arms(round_ab, apply,
               lambda: ag.softmax_bw_dx(FakeTensor(F32), FakeTensor(F32)))

    assert round_ab.COUNT["softmax_bw_in:off:FLOAT32"] == 1
    assert round_ab.COUNT["softmax_bw_in:on:FLOAT32"] == 1
    ins = {k for k in round_ab.COUNT if k.startswith("softmax_bw_in:")}
    assert {k.split(":")[-1] for k in ins} == {"FLOAT32"}


def test_the_moreh_route_is_counted_where_it_is_taken(wired):
    round_ab, ag, R = wired
    apply = round_ab.install_levers(1 << 30, precision=True, moreh=True)
    _both_arms(round_ab, apply,
               lambda: ag.softmax_bw_dx(FakeTensor(F32), FakeTensor(F32)))

    assert round_ab.COUNT["softmax_bw_moreh:on"] == 1
    assert round_ab.COUNT["softmax_bw_moreh:off"] == 0
    round_ab.check_fired(round_ab.COUNT,
                         [{"round": 3, "levers_on": False}, {"round": 4, "levers_on": True}],
                         precision=True, moreh=True)


def test_a_declined_permute_is_counted_with_the_clause_that_declined_it(wired):
    """`reblock_back:on == 0` alone cannot say whether the tape asked. This can."""
    round_ab, ag, R = wired
    apply = round_ab.install_levers(1 << 30, precision=False, moreh=False)
    _both_arms(round_ab, apply,
               lambda: R.permute_via_reblock(FakeTensor(BF16), [0, 2, 3, 1]))

    assert round_ab.COUNT["pvr_ask:on:0.2.3.1"] == 1
    assert round_ab.COUNT["pvr_reject:on:back_window_DRAM:1x64x224x224"] == 1
    assert round_ab.COUNT["reblock_back:on"] == 0


def test_a_bf16_return_with_no_narrowing_raises(wired):
    """The other way the lever can be inert: the dtype moved for some reason of its own while
    `SOFTMAX_BW_DTYPE` never reached the call. Counted separately so it cannot pass as a win."""
    round_ab = wired[0]
    c = collections.Counter({"softmax_bw:on:BFLOAT16": 40, "softmax_bw:off:FLOAT32": 40})
    with pytest.raises(RuntimeError, match="did not reach the call"):
        round_ab.check_fired(c, [{"round": 3, "levers_on": True}], precision=True)


def test_the_moreh_arm_serving_nothing_raises(wired):
    round_ab = wired[0]
    c = collections.Counter({"softmax_bw:on:BFLOAT16": 40, "softmax_bw:off:FLOAT32": 40,
                             "softmax_bw_narrowed:on": 40})
    with pytest.raises(RuntimeError, match="served no call on the ON arm"):
        round_ab.check_fired(c, [{"round": 3, "levers_on": True}], precision=True, moreh=True)
