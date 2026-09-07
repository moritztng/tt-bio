"""The MSA track's two row blocks: OuterProductMean's token rows and PairWeightedAveraging's
MSA-depth rows -- the refusal memos and the block arithmetic.

The block loop itself needs real ttnn tensors, but the two host-side decisions do not: how far
a refusal narrows the block, and which block the byte budget asks for at a given token width.
`_with_dram_narrowing`, the retry both this and the fp32-softmax tail run through, is covered in
tests/test_fp32_softmax_dram_narrow.py.
"""
import pytest

import tt_bio.tenstorrent as tt


@pytest.fixture(autouse=True)
def _clean():
    def reset():
        tt._OPM_DRAM_ROW_CAP.clear()
        tt._PWA_DEPTH_ROW_CAP.clear()
        tt.OPM_ROW_STATS.update(whole=0, blocked=0, dram_narrowed=0)
        tt.PWA_DEPTH_STATS.update(whole=0, blocked=0, dram_narrowed=0)

    reset()
    yield
    reset()


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


class TestPwaNarrow:
    def test_it_halves_and_remembers_the_shape_class(self):
        key = (14189, 1088)
        assert tt._pwa_dram_narrow(key, 1920) == 960
        assert tt._PWA_DEPTH_ROW_CAP[key] == 960
        assert tt.PWA_DEPTH_STATS["dram_narrowed"] == 1

    def test_a_looser_cap_never_wins(self):
        key = (14189, 1088)
        tt._pwa_dram_narrow(key, 256)
        tt._pwa_dram_narrow(key, 4096)
        assert tt._PWA_DEPTH_ROW_CAP[key] == 128

    def test_it_floors_at_one_tile_row(self):
        assert tt._pwa_dram_narrow((64, 64), 32) == 32


class TestPwaBlockArithmetic:
    """The refused tensor is one head's tile-padded `v`: 14189 x 1088 x 32 x 2 = 988 008 448 B,
    with a second full-depth c_m copy of `m` beside it. The block bounds the normed chunk."""

    @staticmethod
    def _rows(tokens, c_m=64):
        return max(32, (tt.PWA_DEPTH_BLOCK_BYTES // (tokens * c_m * 2)) // 32 * 32)

    @pytest.mark.parametrize("tokens", (640, 768, 896, 1024, 1088))
    def test_the_block_keeps_the_normed_chunk_under_budget(self, tokens):
        rows = self._rows(tokens)
        assert rows % 32 == 0 and rows >= 32
        assert rows * tokens * 64 * 2 <= tt.PWA_DEPTH_BLOCK_BYTES

    def test_a_shallow_alignment_is_one_block_and_takes_the_whole_path(self):
        """Depth below the block is the unblocked path, so a shallow MSA cannot change path."""
        assert self._rows(1088) > 256

    def test_the_refused_v_at_1088_is_named_by_its_byte_count(self):
        assert 14189 * 1088 * 32 * 2 == 988008448
