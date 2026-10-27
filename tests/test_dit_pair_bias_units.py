"""dit_pair_bias `units`: the listed 32 x 32 output tiles equal the full call's bit for bit, every other tile keeps
what `outs` held (the pair memo's bias buffers)."""
import pytest
import torch
import ttnn

from tt_bio import dit_pair_bias as D
from tt_bio.main import ensure_p300_mesh_descriptor
from tt_bio.rfd3 import model as M

pytestmark = pytest.mark.device


def _up(t, dev):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)


@pytest.mark.parametrize("n_out,n_units", [(2, 37), (1, 300), (2, 0), (2, 484)])
def test_units_equal_full(n_out, n_units):
    ensure_p300_mesh_descriptor()
    dev = M.get_device()
    g = torch.Generator().manual_seed(n_out * 1000 + n_units)
    I, n, c, nt = 685, 704, 128, 22                             # the page fixture's PF / DiT shapes: 484 tiles
    z = torch.randn(1, I, n, c, generator=g)
    wt = torch.randn(1, 1, 32, c, generator=g) / c ** 0.5
    m = torch.full((1, 1, n, n), -1e4)
    m[..., :I, :I] = 0.0
    zd, wd, md = _up(z, dev), _up(wt, dev), _up(m, dev)
    full = D.dit_pair_bias(zd, wd, md, 4.9, n_out=n_out)
    full = [ttnn.to_torch(o) for o in (full if n_out > 1 else [full])]
    units = torch.randperm(nt * nt, generator=g)[:n_units].sort().values
    stale = [torch.randn(1, 32 // n_out, n, n, generator=g) for _ in range(n_out)]
    outs = [_up(s, dev) for s in stale]
    D.dit_pair_bias(zd, wd, md, 4.9, n_out=n_out, units=units.tolist(), outs=outs)
    sel = torch.zeros(nt * nt, dtype=torch.bool)
    sel[units] = True
    sel = sel.view(nt, nt).repeat_interleave(32, 0).repeat_interleave(32, 1)
    for got, ref, old in zip(outs, full, stale):
        got = ttnn.to_torch(got)
        assert torch.equal(got[..., sel], ref[..., sel])
        assert torch.equal(got[..., ~sel], old.to(torch.bfloat16)[..., ~sel])
