"""`meter.lever_reach` refuses a round whose levers stopped matching what the arm armed.

The failure it exists for is silent: a lever that goes inert part-way through an arm leaves the
process running, the stamp at startup still says the arm was armed, and the median blends two
arms into a null. So the test that matters is the one where the flag CHANGES under a live
`REACH` callable, not the one where it was never set.
"""
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "perf" / "bcx_round"))
import meter as M                                                      # noqa: E402


@pytest.fixture
def armed():
    """Every lever on, restored afterwards.

    Add a lever to `meter.LEVERS` and this fixture has to arm it too, or every test in this
    file fails with the inert-lever refusal the file exists to check. That coupling is
    deliberate: the refusal is what stops an arm measuring a lever it did not arm.
    """
    from tt_bio import fanin_l1, genq, mm_layout, reblock_permute, tenstorrent, triatt_bw
    from tt_bio.af2 import AF2PairBlock
    saved = (genq.compact(), reblock_permute.TAPED_MOVE, mm_layout.MM_LAYOUT,
             triatt_bw.FUSED, tenstorrent._TRIATT_FUSED_HIFI, AF2PairBlock.rne_kernel,
             fanin_l1.FANIN_L1)
    genq.set_compact(True)
    reblock_permute.TAPED_MOVE = True
    mm_layout.MM_LAYOUT = True
    triatt_bw.FUSED = True
    tenstorrent._TRIATT_FUSED_HIFI = True
    AF2PairBlock.rne_kernel = True
    fanin_l1.FANIN_L1 = True
    yield {k: True for k in M.LEVERS}
    (genq_on, reblock_permute.TAPED_MOVE, mm_layout.MM_LAYOUT, triatt_bw.FUSED,
     tenstorrent._TRIATT_FUSED_HIFI, AF2PairBlock.rne_kernel, fanin_l1.FANIN_L1) = saved
    genq.set_compact(genq_on)


def test_stamps_every_lever_when_the_arm_holds(armed):
    out = M.lever_reach(armed)()
    assert out == {"lever_" + k: True for k in M.LEVERS}
    assert len(M.LEVERS) == 7


def test_refuses_the_round_when_a_lever_goes_inert(armed):
    check = M.lever_reach(armed)
    check()                                        # the arm is armed at the first boundary
    from tt_bio import reblock_permute
    reblock_permute.TAPED_MOVE = False             # and inert at the second
    with pytest.raises(SystemExit) as exc:
        check()
    assert "taped_channel_move" in str(exc.value)
    assert '"armed": true' in str(exc.value) and '"read": false' in str(exc.value)


def test_refuses_an_arm_that_was_never_armed(armed):
    """The other direction: an `off` arm whose flag is still on from somewhere else."""
    with pytest.raises(SystemExit) as exc:
        M.lever_reach({**armed, "mm_layout": False})()
    assert "mm_layout" in str(exc.value)


def test_a_reach_error_does_not_hide_an_inert_lever(armed):
    """`meter._reach` swallows Exception so an instrument cannot kill a round. SystemExit is
    not Exception, which is what makes this check fail closed rather than land in
    `reach_error`."""
    saved = list(M.REACH)
    M.REACH[:] = [M.lever_reach({**armed, "rne_kernel": False})]
    try:
        with pytest.raises(SystemExit):
            M._reach()
    finally:
        M.REACH[:] = saved
