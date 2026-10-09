"""OuterProductMean flattens b's (D, J, S) to (D*J, S) as a tile view when J is whole tiles."""
import pytest
import torch
import ttnn

from tt_bio import tenstorrent as T


@pytest.mark.device
@pytest.mark.parametrize("j,s", [(736, 9984), (736, 9947), (64, 97), (40, 97)])
def test_flat_b_is_the_row_major_reshape(j, s):
    """`opm_flat_b`'s tile view holds exactly the values of the row-major (D*J, S) reshape."""
    dev = T.get_device()
    torch.manual_seed(0)
    b = torch.randn(32, j, s).bfloat16()
    t = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    out = T.opm_flat_b(t)
    assert tuple(out.shape) == (32 * j, s) and out.layout == ttnn.TILE_LAYOUT
    assert torch.equal(ttnn.to_torch(out), b.reshape(32 * j, s))
