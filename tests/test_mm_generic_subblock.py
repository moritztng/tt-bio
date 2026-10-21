"""mm_generic.build refuses a subblock that does not tile the block (the kernels ran 5,4,2,2,2 and wrote non-finite
output, ls118). CPU only."""
import pytest

pytest.importorskip("ttnn")
from tt_bio import mm_generic as MG  # noqa: E402


def test_build_refuses_a_subblock_that_does_not_tile_the_block():
    with pytest.raises(RuntimeError, match="does not tile"):
        MG.build(None, None, None, [None], ((5, 4, 2, 2, 2), (8, 8)), (None,) * 4)

