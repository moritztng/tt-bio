"""pair_swiglu_ws `rows`: the listed row tiles equal the full call's bit for bit, every other row tile keeps what the
output held, in place (resid is x) and out of place (resid_src)."""
import pytest
import torch
import ttnn

from tt_bio import pair_swiglu_ws as W
from tt_bio.main import ensure_p300_mesh_descriptor
from tt_bio.rfd3 import model as M

pytestmark = pytest.mark.device


def _up(t, dev, dtype=ttnn.bfloat16):
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)


@pytest.mark.parametrize("hidden,n_rows", [(512, 37), (256, 300), (512, 0), (256, 2200)])
def test_rows_equal_full(hidden, n_rows):
    ensure_p300_mesh_descriptor()
    dev = M.get_device()
    g = torch.Generator().manual_seed(hidden + n_rows)
    x = torch.randn(1, 70, 1024, 128, generator=g)            # 2240 row tiles
    w1, w2, w3 = (torch.randn(a, b, generator=g) / a ** 0.5 for a, b in ((128, hidden), (128, hidden), (hidden, 128)))
    gamma = 1 + 0.1 * torch.randn(1, 128, generator=g)
    w12, w3d = _up(torch.cat([w1, w2], 1), dev, ttnn.bfloat8_b), _up(w3, dev, ttnn.bfloat8_b)
    ln = (_up(gamma, dev), None, 1e-6)
    kw = dict(ln=ln, pack_silu=True, fidelity=ttnn.MathFidelity.HiFi2)
    full = _up(x, dev)
    W.pair_swiglu(full, w12, w3d, resid=full, **kw)
    full = ttnn.to_torch(full).view(-1, 32, 128)
    rows = torch.randperm(2240, generator=g)[:n_rows]
    lst = W.row_list(dev, _up(x, dev), rows)
    keep = torch.ones(2240, dtype=torch.bool)
    keep[rows] = False

    xin = _up(x, dev)                                           # in place: x + f(x) on the listed rows only
    W.pair_swiglu(xin, w12, w3d, resid=xin, rows=lst, **kw)
    got = ttnn.to_torch(xin).view(-1, 32, 128)
    ref = x.to(torch.bfloat16).float().view(-1, 32, 128)
    assert torch.equal(got[rows], full[rows])
    assert torch.equal(got[keep], ref[keep])

    stale = torch.randn(1, 70, 1024, 128, generator=g)        # out of place: out = x + f(x), out's other rows stay
    xs, out = _up(x, dev), _up(stale, dev)
    W.pair_swiglu(xs, w12, w3d, resid=out, resid_src=xs, rows=lst, **kw)
    got = ttnn.to_torch(out).view(-1, 32, 128)
    assert torch.equal(got[rows], full[rows])
    assert torch.equal(got[keep], stale.to(torch.bfloat16).float().view(-1, 32, 128)[keep])
    assert torch.equal(ttnn.to_torch(xs).view(-1, 32, 128), ref)
