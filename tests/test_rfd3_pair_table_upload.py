"""The rfd3 P_LL gather table goes up in wide chunks slice_written into a preallocated table.

Every row the gather can index must come back exactly as the host table holds it, for a table smaller
than one wide page, an odd row count, and one spanning several chunks with a padded last chunk.
"""
import pytest
import torch
import ttnn

from tt_bio.main import ensure_p300_mesh_descriptor
from tt_bio.rfd3 import model as M

pytestmark = pytest.mark.device


@pytest.mark.parametrize("n_atoms,chunk", [(3, None), (37, None), (301, 1 << 14)])
def test_pair_table_rows_equal_host(n_atoms, chunk, monkeypatch):
    ensure_p300_mesh_descriptor()
    from tt_bio.tenstorrent import get_device
    dev = get_device()
    if chunk:
        monkeypatch.setattr(M, "_TABLE_CHUNK_ROWS", chunk)
    flat = torch.randn(n_atoms * n_atoms, 16)
    table = M._upload_pair_table(flat, dev)
    idx = torch.arange(flat.shape[0], dtype=torch.int32).reshape(1, -1)
    rows = ttnn.to_torch(ttnn.embedding(ttnn.from_torch(idx, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev,
                                                        dtype=ttnn.uint32), table, layout=ttnn.ROW_MAJOR_LAYOUT))
    assert torch.equal(rows.reshape(-1, 16), flat.to(torch.bfloat16))
