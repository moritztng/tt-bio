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


@pytest.mark.parametrize("n_atoms,rows,chunk", [(37, 37, None), (301, 96, None), (301, 301, 1 << 14),
                                                (1546, 1546, None), (1547, 1547, None), (1547, 333, None),
                                                (1914, 1914, None), (1914, 511, None), (2458, 2458, None)])
def test_pair_table_written_from_tiled_blocks_equals_host(n_atoms, rows, chunk, monkeypatch):
    """TokenInitializer's on-card table (tiled row blocks of P_LL at a padded row stride) holds every pair row the
    gather indexes exactly as the host does. 1547 atoms hung the card and 1914 wrote wrong rows when the table's
    row stride was the atom count; 2458 is a block above one write chunk (one slice_write of 6M rows hangs)."""
    ensure_p300_mesh_descriptor()
    from tt_bio.tenstorrent import get_device
    dev = get_device()
    if chunk:
        monkeypatch.setattr(M, "_TABLE_CHUNK_ROWS", chunk)
    p = torch.randn(n_atoms, n_atoms, 16).to(torch.bfloat16)
    stride = M._pair_row_stride(n_atoms)
    table = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, M._pair_table_rows(n_atoms * stride)[1], 16]),
                                           ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, dev)
    for a0 in range(0, n_atoms, rows):
        block = ttnn.from_torch(p[a0:a0 + rows].unsqueeze(0), layout=ttnn.TILE_LAYOUT, device=dev,
                                dtype=ttnn.bfloat16)
        M._write_pair_rows(table, block, a0 * stride, stride)
        ttnn.deallocate(block)
    got = ttnn.to_torch(table).reshape(-1, 16)[:n_atoms * stride].reshape(n_atoms, stride, 16)[:, :n_atoms]
    assert torch.equal(got, p)
