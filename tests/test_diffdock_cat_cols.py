"""`cat_cols` places single columns exactly, where a column concat rounds and then throws.

Measured on chip 21 (anx-glx, ttnn 0.68, fp32 tile): nine one-wide columns concatenated come
back 8.4e-4 off at 99,008 rows and the concat throws outright at 286,624 (circular buffers
2,404,096 B against 1,572,864 B of L1). A gated DiffDock complex carries 286,624 main-stack
edge rows, so this is a size users reach, not a corner.
"""
import pytest

torch = pytest.importorskip("torch")


@pytest.mark.device
@pytest.mark.parametrize("rows", [4096, 286624])
def test_columns_are_placed_exactly_at_every_row_count(rows):
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import ttnn

    from tt_bio.diffdock.device_loop import cat_cols, upload
    from tt_bio.tenstorrent import get_device

    dev = get_device()
    want = torch.randn(rows, 9) * 30
    got = ttnn.to_torch(cat_cols([upload(dev, want[:, k]) for k in range(9)])).float()
    assert got.shape[1] == 9
    assert torch.equal(got[:, :9], want)


@pytest.mark.device
def test_one_column_is_returned_as_it_is():
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()

    from tt_bio.diffdock.device_loop import cat_cols, upload
    from tt_bio.tenstorrent import get_device

    col = upload(get_device(), torch.randn(64))
    assert cat_cols([col]) is col
