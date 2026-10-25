"""T2CM with the g / z reads on the reader (`trimul_tail.T2_ZR`) against T2_ZR off: the same compute on the same
operands, only the RISC that reads them moves, so the outputs must be bit-identical. Covers the plain form
(ESMFold2 / OpenFold3), the GATE form, and AF2's form with both biases, the dropout mask and a separate `out`.

Run: TT_VISIBLE_DEVICES=<card> python3 -m pytest tests/test_trimul_t2zr.py
"""
import pytest
import torch

ttnn = pytest.importorskip("ttnn")

from tt_bio import mm_generic as MG  # noqa: E402
from tt_bio import tenstorrent as T  # noqa: E402
from tt_bio import trimul_tail as TTL  # noqa: E402
from tt_bio.af2 import compute_kernel_config  # noqa: E402


def _dev(t, dev):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)


@pytest.mark.parametrize("form,K,n", [("plain", 256, 128), ("plain", 128, 160), ("plain", 256, 512),
                                      ("gate", 128, 96), ("af2", 128, 128)])
def test_zr_is_bit_exact(form, K, n, monkeypatch):
    dev = T.get_device()
    monkeypatch.setattr(TTL, "T2", True)
    monkeypatch.setattr(TTL, "T2CM", True)
    g = torch.Generator().manual_seed(n + K)
    bf = lambda *s, sc=1.0: (torch.randn(*s, generator=g) * sc).to(torch.bfloat16).float()  # noqa: E731
    xcm, gin, z = bf(1, K, n, n), bf(1, n, n, K), bf(1, n, n, K)
    ln_w, ln_b = 1 + 0.1 * bf(K), 0.1 * bf(K)
    wp, wg = bf(K, K, sc=K ** -0.5), bf(K, K, sc=K ** -0.5)
    ckc = MG.ckc_args(T.trunk_compute_kernel_config(compute_kernel_config()))
    ops = [_dev(t, dev) for t in (xcm, ln_w, ln_b, wp, gin)]
    kw = {}
    if form != "plain":
        kw["wg"] = _dev(wg, dev)
    if form == "af2":
        kw["bp"], kw["bg"] = TTL.bias_rows(_dev(bf(K), dev)), TTL.bias_rows(_dev(bf(K), dev))
        kw["mask"] = _dev((torch.rand(1, 1, n, K, generator=g) > 0.1).float(), dev)
    outs = {}
    for zr in (False, True):
        monkeypatch.setattr(TTL, "T2_ZR", zr)
        resid = _dev(z, dev)
        if form == "af2":
            kw["out"] = ttnn.allocate_tensor_on_device(resid.shape, resid.dtype, ttnn.TILE_LAYOUT, dev,
                                                       ttnn.DRAM_MEMORY_CONFIG)
        y = TTL.fused_tail_ln_cm(*ops, resid, ckc, tuple(T.COMPUTE_GRID_MAIN), **kw)
        assert y is not None, TTL.REJECTS
        outs[zr] = ttnn.to_torch(y).float()
        if form == "af2":    # `out` leaves resid as it was
            assert torch.equal(ttnn.to_torch(resid).float(), z)
        ttnn.deallocate(y)
        if y is not resid:
            ttnn.deallocate(resid)
    assert torch.isfinite(outs[False]).all()
    assert not torch.equal(outs[False], z)
    assert torch.equal(outs[True], outs[False]), (outs[True] - outs[False]).abs().max().item()
