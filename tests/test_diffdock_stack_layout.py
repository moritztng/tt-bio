"""The stack edge layout, which `run_stack` and `ResidentStack` now share.

`device.stack_layout` is the one place a convolution stack's padded edge blocks are laid out.
Both the uploading form and the card-resident form read it, so the properties the kernels rely
on -- every block starting on a multiple of 32, dead rows pointing at the pad node, real rows
kept in order -- are asserted here once, card-free.
"""

import pytest

torch = pytest.importorskip("torch")

from tt_bio.diffdock.device import pad32, stack_layout  # noqa: E402

N_P = 256


def _ei(e):
    return torch.stack([torch.arange(e) % (N_P - 1), (torch.arange(e) * 7) % (N_P - 1)])


@pytest.mark.parametrize("e,plan", [
    (100, [(100, None)]),
    (100, [(100, (40,)), (64, None)]),
    (4096, [(4096, (2272,)), (4096, (2272,)), (2272, None)]),
    (37, [(37, (11, 23))]),
])
def test_blocks_are_32_aligned_and_in_order(e, plan):
    ei, pos, blocks = stack_layout(_ei(e), N_P, plan)
    assert pos[0] == 0 and pos[e] == ei.shape[1]
    for lo, hi, extra in blocks:
        assert pad32(hi - lo) == (hi - lo) + extra
        assert pos[lo] % 32 == 0
        assert torch.equal(ei[:, pos[lo]:pos[lo] + hi - lo], _ei(e)[:, lo:hi])
        dead = ei[:, pos[lo] + hi - lo:pos[lo] + hi - lo + extra]
        assert (dead == N_P - 1).all(), "a dead row must point at the pad node, both ends"


def test_every_cut_and_split_lands_on_a_block_boundary():
    plan = [(4096, (2272,)), (2272, None)]
    _, pos, _ = stack_layout(_ei(4096), N_P, plan)
    for cut, splits in plan:
        for b in (cut, *(splits or ())):
            assert pos[b] % 32 == 0, f"boundary {b} is not 32-aligned in the padded layout"


def test_an_unsplit_stack_is_one_padded_block():
    ei, pos, blocks = stack_layout(_ei(100), N_P, [(100, None)])
    assert len(blocks) == 1 and ei.shape[1] == pad32(100) == 128
