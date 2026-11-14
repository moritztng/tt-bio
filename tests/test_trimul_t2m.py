"""T2m (`trimul_t2m`): the trimul output tail T2 reading x where the einsum left it.

The last channel chunk stays channel-major, [1, C, N, N]; channels before it are already in the output
layout. T2m must equal T2 on the moved x bit for bit (the gather and the tile transpose are data moves,
the arithmetic is T2's), and so carries T2's float64 error.

Run: TT_VISIBLE_DEVICES=<card> python3 -m pytest tests/test_trimul_t2m.py
"""
import pytest
import torch
import ttnn

from tt_bio import mm_generic as MG
from tt_bio import tenstorrent as T
from tt_bio import trimul_tail as TTL
from tt_bio.af2 import compute_kernel_config


def _dev(t, dev):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)


def _rel_rms(a, b):
    return ((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt()).item()


@pytest.mark.parametrize("gate", [False, True])
@pytest.mark.parametrize("c,c_off,n", [(128, 0, 128), (256, 0, 128), (256, 128, 96), (256, 0, 512)])
def test_t2m_equals_t2_on_the_moved_x(c, c_off, n, gate, monkeypatch):
    monkeypatch.setattr(TTL, "T2", True)
    monkeypatch.setattr(TTL, "T2M", True)
    dev = T.get_device()
    gen = torch.Generator().manual_seed(7 * c + c_off + n + gate)
    h = {"x": torch.randn(1, n, n, c, generator=gen) * 2 + torch.randn(1, n, n, 1, generator=gen),
         "g": torch.randn(1, n, n, c, generator=gen), "z": torch.randn(1, n, n, c, generator=gen),
         "w": torch.randn(c, c, generator=gen) / c ** 0.5, "wg": torch.randn(c, c, generator=gen) / c ** 0.5,
         "gamma": 1 + 0.1 * torch.randn(c, generator=gen), "beta": 0.1 * torch.randn(c, generator=gen)}
    h = {k: v.to(torch.bfloat16) for k, v in h.items()}
    f = {k: v.double() for k, v in h.items()}
    mu = f["x"].mean(-1, keepdim=True)
    xn = (f["x"] - mu) / torch.sqrt((f["x"] - mu).pow(2).mean(-1, keepdim=True) + 1e-5) * f["gamma"] + f["beta"]
    ref = f["z"] + (xn @ f["w"]) * torch.sigmoid(f["g"] @ f["wg"] if gate else f["g"])

    d = {k: _dev(v, dev) for k, v in h.items() if k != "z"}
    kcfg = T.trunk_compute_kernel_config(compute_kernel_config())
    ckc, grid = MG.ckc_args(kcfg), tuple(T.COMPUTE_GRID_MAIN)
    wg = d["wg"] if gate else None

    z_t2 = _dev(h["z"], dev)
    y = TTL.fused_tail_ln(d["x"], d["gamma"], d["beta"], d["w"], d["g"], z_t2, ckc, grid, wg=wg)
    assert y is not None, TTL.REJECTS
    t2 = ttnn.to_torch(y).double().reshape(ref.shape)

    # x as the channel loop leaves it: [0, c_off) moved, the rest NaN so a stray read shows
    prev = h["x"].clone()
    prev[..., c_off:] = float("nan")
    chunk = _dev(h["x"][..., c_off:].permute(0, 3, 1, 2).contiguous(), dev)
    z_m = _dev(h["z"], dev)
    y = TTL.fused_tail_ln(_dev(prev, dev), d["gamma"], d["beta"], d["w"], d["g"], z_m, ckc, grid, wg=wg,
                          moved=(chunk, c_off))
    assert y is not None, (TTL.T2M_WHY, TTL.REJECTS)
    t2m = ttnn.to_torch(y).double().reshape(ref.shape)
    assert torch.isfinite(t2m).all()
    assert torch.equal(t2m, t2), (_rel_rms(t2m, ref), _rel_rms(t2, ref))
    assert _rel_rms(t2m, ref) < 5e-3
