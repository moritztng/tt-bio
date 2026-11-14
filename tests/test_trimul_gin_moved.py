"""`trimul_tail.gin_moved` (the trimul_gin lever's kernel) against the route it replaces and float64.

The resident split (`fused_tail(split=2)`) plus two channel moves gives a and b as [B, C, S, S]; gin_moved
writes the same tensors from one kernel. Both are one bf16 rounding of the same fp32 sums, so they must
sit equally close to float64 and differ from each other by at most a bf16 ULP or two, with and without
the pair mask folded into a.

Run: TT_VISIBLE_DEVICES=<card> python3 -m pytest tests/test_trimul_gin_moved.py
"""
import pytest
import torch
import ttnn

from tt_bio import mm_generic as MG
from tt_bio import tenstorrent as T
from tt_bio import trimul_tail as TTL
from tt_bio.af2 import compute_kernel_config

pytestmark = pytest.mark.device

CZ, C = 256, 128    # protenix-v2's c_z; one chunk of a and of b


def _dev(t, dev):
    return ttnn.from_torch(t.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)


def _rel_rms(a, b):
    return ((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt()).item()


@pytest.mark.parametrize("n,masked", [(64, True), (160, True), (160, False)])
def test_gin_moved_matches_split_route_and_float64(n, masked, monkeypatch):
    dev = T.get_device()
    # Production's fidelity (trunk_hifi3): HiFi4 with fp32 accumulation returns wrong values on some
    # row blocks on Wormhole, in both routes, which this test would otherwise be measuring.
    monkeypatch.setattr(T, "_TRUNK_MATH_FIDELITY", "hifi3")
    monkeypatch.setattr(TTL, "GIN_MOVE", True)
    monkeypatch.setattr(TTL, "EPI", 1)       # the resident split route needs the lean epilogue
    g = torch.Generator().manual_seed(n)
    x = torch.randn(1, n, n, CZ, generator=g).to(torch.bfloat16)
    wp = (torch.randn(CZ, 2 * C, generator=g) / 16).to(torch.bfloat16)
    wg = (torch.randn(CZ, 2 * C, generator=g) / 16).to(torch.bfloat16)
    m = (torch.rand(1, n, n, generator=g) > 0.1).to(torch.bfloat16) if masked else None

    d = torch.float64
    x64 = x.to(d).reshape(n * n, CZ)
    ab = ((x64 @ wp.to(d)) * torch.sigmoid(x64 @ wg.to(d))).reshape(1, n, n, 2 * C)
    ref = [ab[..., :C], ab[..., C:]]
    if m is not None:
        ref[0] = ref[0] * m.to(d)[..., None]
    ref = [r.permute(0, 3, 1, 2) for r in ref]

    dx, dwp, dwg, dwpT, dwgT = (_dev(t, dev) for t in (x, wp, wg, wp.t(), wg.t()))
    dm = None if m is None else _dev(m, dev)
    ckc = MG.ckc_args(T.trunk_compute_kernel_config(compute_kernel_config()))
    grid = tuple(T.COMPUTE_GRID_MAIN)

    assert TTL.gin_moved_ok(dx, dwpT, dm)
    moved = TTL.gin_moved(dx, dwpT, dwgT, ckc, grid, mask=dm)
    assert moved is not None
    split = TTL.fused_tail(dx, dx, dwp, dwg, ckc, grid, split=2, mask=dm)
    assert split is not None, TTL.REJECTS
    split_moved = [T._channel_move(o, ttnn.DRAM_MEMORY_CONFIG) for o in split]

    for mv, sp, r in zip(moved, split_moved, ref):
        mv, sp = ttnn.to_torch(mv).to(d), ttnn.to_torch(sp).to(d)
        assert mv.shape == r.shape
        e_mv, e_sp = _rel_rms(mv, r), _rel_rms(sp, r)
        assert e_sp < 1e-2, e_sp
        assert e_mv <= 1.05 * e_sp + 1e-4, (e_mv, e_sp)
        # no isolated gross element hiding under a good rms (the WH HiFi4 erratum looked like this)
        assert (mv - r).abs().max().item() < 0.1
        assert _rel_rms(mv, sp) < 4e-3


@pytest.mark.parametrize("masked", [True, False])
def test_gin_moved_channel_slices_equal_one_launch(masked, monkeypatch):
    """ESMFold2's c_z 256 / hidden 256 overflows one launch's L1 budget, so gin_moved runs it as two
    channel slices. Per channel the arithmetic is the one-launch kernel's: same bits, same float64 error."""
    dev = T.get_device()
    monkeypatch.setattr(T, "_TRUNK_MATH_FIDELITY", "hifi3")
    monkeypatch.setattr(TTL, "GIN_MOVE", True)
    n, c = 128, 256
    g = torch.Generator().manual_seed(7)
    x = torch.randn(1, n, n, CZ, generator=g).to(torch.bfloat16)
    wp = (torch.randn(CZ, 2 * c, generator=g) / 16).to(torch.bfloat16)
    wg = (torch.randn(CZ, 2 * c, generator=g) / 16).to(torch.bfloat16)
    m = (torch.rand(1, n, n, generator=g) > 0.1).to(torch.bfloat16) if masked else None
    dx = _dev(x, dev)
    dm = None if m is None else _dev(m, dev)
    ckc = MG.ckc_args(T.trunk_compute_kernel_config(compute_kernel_config()))
    grid = tuple(T.COMPUTE_GRID_MAIN)

    ns = TTL.gin_move_split(dx, c, dm)
    assert ns == 2, ns
    w = c // ns
    sl = lambda t, s: torch.cat([t[:, s * w:(s + 1) * w], t[:, c + s * w:c + (s + 1) * w]], dim=-1)  # noqa: E731
    sliced = TTL.gin_moved(dx, [_dev(sl(wp, s).t(), dev) for s in range(ns)],
                           [_dev(sl(wg, s).t(), dev) for s in range(ns)], ckc, grid, mask=dm)
    monkeypatch.setattr(TTL, "_gin_move_tiles", lambda kt, ct2, mask: 0)   # idle device: one launch fits
    whole = TTL.gin_moved(dx, _dev(wp.t(), dev), _dev(wg.t(), dev), ckc, grid, mask=dm)
    assert sliced is not None and whole is not None

    d = torch.float64
    x64 = x.to(d).reshape(n * n, CZ)
    ab = ((x64 @ wp.to(d)) * torch.sigmoid(x64 @ wg.to(d))).reshape(1, n, n, 2 * c)
    ref = [ab[..., :c], ab[..., c:]]
    if m is not None:
        ref[0] = ref[0] * m.to(d)[..., None]
    for s_, w_, r in zip(sliced, whole, ref):
        s_, w_ = ttnn.to_torch(s_).to(d), ttnn.to_torch(w_).to(d)
        assert torch.isfinite(s_).all()
        assert torch.equal(s_, w_)
        assert _rel_rms(s_, r.permute(0, 3, 1, 2)) < 1e-2


@pytest.mark.parametrize("n,cz,c", [(160, 256, 128), (96, 256, 256)])
def test_gin_fused_gate_matches_float64(n, cz, c, monkeypatch):
    # trimul_gin_fuse: p * sigmoid(g) in DST. Against float64 it must be no worse than the unfused kernel;
    # (96, 256, 256) is ESMFold2's in-projection shape, which the fused gate runs in one launch.
    dev = T.get_device()
    monkeypatch.setattr(T, "_TRUNK_MATH_FIDELITY", "hifi3")
    monkeypatch.setattr(TTL, "GIN_MOVE", True)
    g = torch.Generator().manual_seed(n)
    x = torch.randn(1, n, n, cz, generator=g).to(torch.bfloat16)
    wp = (torch.randn(cz, 2 * c, generator=g) / 16).to(torch.bfloat16)
    wg = (torch.randn(cz, 2 * c, generator=g) / 16).to(torch.bfloat16)
    d = torch.float64
    x64 = x.to(d).reshape(n * n, cz)
    ab = ((x64 @ wp.to(d)) * torch.sigmoid(x64 @ wg.to(d))).reshape(1, n, n, 2 * c)
    ref = [ab[..., :c].permute(0, 3, 1, 2), ab[..., c:].permute(0, 3, 1, 2)]
    dx, dwpT, dwgT = (_dev(t, dev) for t in (x, wp.t(), wg.t()))
    ckc = MG.ckc_args(T.trunk_compute_kernel_config(compute_kernel_config()))
    grid = tuple(T.COMPUTE_GRID_MAIN)
    outs = {}
    for fuse in (False, True):
        monkeypatch.setattr(TTL, "_lever", lambda name, fuse=fuse: fuse and name == "trimul_gin_fuse")
        ns = TTL.gin_move_split(dx, c)
        assert ns is not None
        if fuse:
            assert ns == 1
        wpT = dwpT if ns == 1 else [_dev(t, dev) for t in _slices(wp.t(), c, ns)]
        wgT = dwgT if ns == 1 else [_dev(t, dev) for t in _slices(wg.t(), c, ns)]
        outs[fuse] = [ttnn.to_torch(o).to(d) for o in TTL.gin_moved(dx, wpT, wgT, ckc, grid)]
    for base, fused, r in zip(outs[False], outs[True], ref):
        e0, e1 = _rel_rms(base, r), _rel_rms(fused, r)
        assert e1 <= 1.05 * e0 + 1e-4, (e1, e0)
        assert (fused - r).abs().max().item() < 0.1


def _slices(wT, c, n):
    # gin_move_split's layout: slice s = rows s of a's n parts, then rows s of b's n parts.
    h = c // n
    return [torch.cat([wT[s * h:(s + 1) * h], wT[c + s * h:c + (s + 1) * h]]) for s in range(n)]
