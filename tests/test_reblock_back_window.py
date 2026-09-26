"""`eligible_back`'s N floor is a WINDOW someone drew, and it is now named and overridable.

The clause reads `N < BACK_N_MIN` and shipped as a literal 256, chosen for the trimul's DRAM
chunk at 512 aa and up. BC2's design step is 211 aa, which buckets to 224, one bucket under it --
and `perf/bcx_bwbytes/probe.py` measures the kernel `torch.equal` to `ttnn.permute` and 5.16x /
6.16x faster at exactly [1, 64, 224, 224] and [1, 128, 224, 224], DRAM to DRAM. So the floor
declines a shape where the kernel is both correct and much faster.

Card-free: `eligible_back` only reads shape, dtype, layout and the two memory configs.
"""
import sys

import pytest
import ttnn

sys.path.insert(0, ".")


class FakeTensor:
    def __init__(self, shape, mc):
        self.shape = shape
        self.dtype = ttnn.bfloat16
        self.layout = ttnn.TILE_LAYOUT
        self._mc = mc

    def memory_config(self):
        return self._mc


@pytest.fixture
def R():
    from tt_bio import reblock_permute as R
    prev = R.BACK_N_MIN
    R.REJECTS.clear()
    yield R
    R.BACK_N_MIN = prev
    R.REJECTS.clear()


def _t(n, c=64):
    return FakeTensor([1, c, n, n], ttnn.DRAM_MEMORY_CONFIG)


def test_the_shipped_window_declines_bc2s_bucket(R):
    assert R.BACK_N_MIN == 256
    assert R.eligible_back(_t(224), ttnn.DRAM_MEMORY_CONFIG) is False
    assert any(k[0].startswith("back_window") for k in R.REJECTS), R.REJECTS


def test_lowering_the_window_admits_it_and_nothing_else_changes(R):
    """The control for the override: the SAME tensor, admitted only because the floor moved."""
    R.BACK_N_MIN = 224
    assert R.eligible_back(_t(224), ttnn.DRAM_MEMORY_CONFIG) is True
    assert R.eligible_back(_t(192), ttnn.DRAM_MEMORY_CONFIG) is False


def test_the_window_never_admits_a_ragged_n(R):
    """224 is a multiple of 32 and 211 is not. Lowering the floor must not reach past the
    kernel's own tile requirement -- the back direction declines ragged shapes by design."""
    R.BACK_N_MIN = 32
    assert R.eligible_back(_t(211), ttnn.DRAM_MEMORY_CONFIG) is False
    assert any(k[0] == "back_ragged" for k in R.REJECTS), R.REJECTS


def test_the_window_is_read_at_call_time_not_import_time(R):
    """The A/B flips it per round, so a value captured at import would separate no arms."""
    t = _t(224)
    R.BACK_N_MIN = 256
    assert R.eligible_back(t, ttnn.DRAM_MEMORY_CONFIG) is False
    R.BACK_N_MIN = 224
    assert R.eligible_back(t, ttnn.DRAM_MEMORY_CONFIG) is True
