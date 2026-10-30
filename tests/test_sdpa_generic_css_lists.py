"""sdpa_generic.css_lists: each row is its count, then its key chunk ids ascending, zero after.

Opens no device.
"""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("ttnn")

from tt_bio import sdpa_generic as SG                                       # noqa: E402


def test_rows_list_the_occupied_chunks_in_order():
    g = torch.Generator().manual_seed(0)
    occ = torch.rand(95, 95, generator=g) < 0.1
    occ[torch.arange(95), torch.arange(95)] = True
    rows = SG.css_lists(occ)
    width = SG.css_width(95)
    assert rows.shape == (1, 1, 95, width) and width % 16 == 0 and width > 95
    for i in range(95):
        ks = torch.nonzero(occ[i]).flatten().tolist()
        assert rows[0, 0, i].tolist() == [len(ks)] + ks + [0] * (width - 1 - len(ks))


def test_an_empty_q_chunk_is_refused():
    occ = torch.eye(4, dtype=torch.bool)
    occ[2, 2] = False
    with pytest.raises(AssertionError):
        SG.css_lists(occ)
