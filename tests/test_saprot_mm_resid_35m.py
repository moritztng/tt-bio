"""35M's mm_resid blocks (saprot._MM_RESID_SMALL) and mm_generic.build's refusal of a subblock that does not tile the
block (the kernels ran 5,4,2,2,2 and wrote non-finite output, ls118). CPU only."""
import pytest

pytest.importorskip("ttnn")
from tt_bio import mm_generic as MG  # noqa: E402
from tt_bio import saprot as sp  # noqa: E402


def test_build_refuses_a_subblock_that_does_not_tile_the_block():
    with pytest.raises(RuntimeError, match="does not tile"):
        MG.build(None, None, None, [None], ((5, 4, 2, 2, 2), (8, 8)), (None,) * 4)


def test_small_blocks_tile_and_cover_35m():
    assert set(sp._MM_RESID_SMALL) == {(640, 480), (1920, 480)}
    for m, k, n, sh, sw in sp._MM_RESID_SMALL.values():
        assert m % sh == 0 and n % sw == 0 and sh * sw <= 4
    assert sp._SAPROT_MM_RESID_35M is False
