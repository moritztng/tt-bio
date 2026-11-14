"""gather_tiled `fma=(a, inv)` is a + table[idx] * inv as close to a float64 reference as the gather + mask_add pair it
replaces, and `rows` computes exactly the listed units of the full call (RFD3 process_z under RFD3_PZ_ROWS)."""
import pytest
import torch
import ttnn

from tt_bio import gather_tiled as G
from tt_bio import pair_swiglu_ws as W
from tt_bio.main import ensure_p300_mesh_descriptor
from tt_bio.rfd3 import model as M

pytestmark = pytest.mark.device


def _up(t, dev, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16):
    return ttnn.from_torch(t, dtype=dtype, layout=layout, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)


@pytest.mark.parametrize("I,copies,n_rows", [(85, 1, None), (85, 12, 500), (70, 12, 0), (70, 1, 3)])
def test_fma_and_rows(I, copies, n_rows):
    ensure_p300_mesh_descriptor()
    dev = M.get_device()
    g = torch.Generator().manual_seed(I + copies)
    N, C = 65 * 65, 128
    tab = torch.randn(N, C, generator=g).to(torch.bfloat16)
    idx = torch.randint(0, N, (1, I, I), generator=g)
    a = torch.randn(1, I, I, C, generator=g).to(torch.bfloat16)
    inv = (0.5 + torch.rand(1, I, I, 1, generator=g)).to(torch.bfloat16)
    rm = _up(tab, dev, ttnn.ROW_MAJOR_LAYOUT)
    if copies > 1:
        rm = G.spread(rm, copies)
    di = _up(G.pad_index(idx, I, I), dev, ttnn.ROW_MAJOR_LAYOUT, ttnn.uint32)
    da, dinv = _up(a, dev), _up(inv, dev)

    ref = a.double() + tab.double()[idx] * inv.double()
    em = G.gather_tiled(di, rm, 1, I, copies=copies)
    old = ttnn.to_torch(ttnn.addcmul(da, em, dinv, value=1.0)).double()
    full = ttnn.to_torch(G.gather_tiled(di, rm, 1, I, copies=copies, fma=(da, dinv))).double()
    e_old, e_new = (old - ref).abs().max().item(), (full - ref).abs().max().item()
    assert e_new <= max(e_old, 1e-6) * 1.01, (e_new, e_old)
    if n_rows is None:
        return

    JT = -(-I // 32)
    units = I * JT
    rows = torch.randperm(units, generator=g)[:n_rows]
    lst = W.row_list(dev, [1, I, JT * 32, C], rows)
    part = ttnn.to_torch(G.gather_tiled(di, rm, 1, I, copies=copies, fma=(da, dinv), rows=lst))
    pad = torch.zeros(1, I, JT * 32, C, dtype=torch.float64)
    pad[..., :I, :] = full
    pp = torch.zeros(1, I, JT * 32, C, dtype=part.dtype)
    pp[..., :I, :] = part
    sel = pad.view(units, 32, C)[rows]
    assert torch.equal(pp.double().view(units, 32, C)[rows], sel)
    ttnn.deallocate(lst[0])
