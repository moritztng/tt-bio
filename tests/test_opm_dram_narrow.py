"""OuterProductMean's token-row block: the refusal memo and the block arithmetic.

The block loop itself needs real ttnn tensors, but the two host-side decisions do not: how far
a refusal narrows the block, and which block the byte budget asks for at a given token width.
`_with_dram_narrowing`, the retry both this and the fp32-softmax tail run through, is covered in
tests/test_fp32_softmax_dram_narrow.py.
"""
import pytest

import tt_bio.tenstorrent as tt


@pytest.fixture(autouse=True)
def _clean():
    tt._OPM_DRAM_ROW_CAP.clear()
    tt.OPM_ROW_STATS.update(whole=0, blocked=0, dram_narrowed=0)
    yield
    tt._OPM_DRAM_ROW_CAP.clear()
    tt.OPM_ROW_STATS.update(whole=0, blocked=0, dram_narrowed=0)


class TestNarrow:
    def test_it_halves_and_remembers_the_shape_class(self):
        key = (1024, 32, 32, 1024)
        assert tt._opm_dram_narrow(key, 1024) == 512
        assert tt._OPM_DRAM_ROW_CAP[key] == 512
        assert tt.OPM_ROW_STATS["dram_narrowed"] == 1

    def test_a_looser_cap_never_wins(self):
        key = (1024, 32, 32, 1024)
        tt._opm_dram_narrow(key, 256)
        tt._opm_dram_narrow(key, 1024)
        assert tt._OPM_DRAM_ROW_CAP[key] == 128

    def test_one_shape_class_does_not_narrow_another(self):
        tt._opm_dram_narrow((800, 32, 32, 800), 800)
        assert (1024, 32, 32, 1024) not in tt._OPM_DRAM_ROW_CAP

    def test_it_floors_at_one_tile_row(self):
        key = (64, 32, 32, 64)
        assert tt._opm_dram_narrow(key, 48) == 32
        assert tt._opm_dram_narrow(key, 32) == 32


class TestBlockArithmetic:
    """The block the byte budget asks for, once the token count clears the chunking gate.

    The refused sizes are the measured ones: OpenBind-0 at 864 padded tokens asked for
    8 790 736 896 B of fp32 score, and OpenFold3 at 800 and 1024 died on 800*800*1024*2 and
    1024*1024*1024*2 of OPM z. Whatever the block is, the tensor it implies has to fit the budget.
    """

    @staticmethod
    def _rows(tokens, c_a=32, c_b=32):
        return max(32, min(tt.OPM_CHUNK_SIZE,
                           (tt.OPM_Z_BUDGET_BYTES // (c_a * c_b * tokens * 2)) // 32 * 32))

    @pytest.mark.parametrize("tokens", (640, 800, 864, 896, 1024))
    def test_the_block_keeps_z_under_budget(self, tokens):
        rows = self._rows(tokens)
        assert rows % 32 == 0 and rows >= 32
        assert rows * 32 * 32 * tokens * 2 <= tt.OPM_Z_BUDGET_BYTES

    def test_the_block_shrinks_as_the_token_width_grows(self):
        assert self._rows(1024) < self._rows(640) <= tt.OPM_CHUNK_SIZE

    def test_the_unblocked_z_at_1024_is_the_buffer_that_was_refused(self):
        """The tensor this exists for, named by its own byte count."""
        assert 1024 * 32 * 32 * 1024 * 2 == 2147483648
        assert self._rows(1024) * 32 * 32 * 1024 * 2 == 268435456
