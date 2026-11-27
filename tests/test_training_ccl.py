"""CPU unit tests for tt_bio.train.ccl (exact fp32 2D hierarchical all-reduce helpers)."""
import sys
import pytest
import torch

from tt_bio.train import ccl


def test_ccl_import_does_not_import_ttnn():
    assert "ttnn" not in sys.modules


def test_flat_2d_shape_rfd4_galaxy():
    b, ch, r, c = ccl.flat_2d_shape(178_448_304, (8, 4), cols=10944)
    assert (b, ch) == (1, 1)
    assert r == 16384 and c == 10944
    assert r % (32 * 8 * 4) == 0
    assert c % 32 == 0
    pad_frac = (r * c - 178_448_304) / 178_448_304
    assert 0.0 <= pad_frac < 0.005


def test_configure_refuses_too_small_payload():
    with pytest.raises(ValueError, match="8192"):
        ccl.configure_fp32_ring_fabric(4352)


@pytest.mark.parametrize("rm,cm", [(8, 4), (4, 4), (2, 4), (1, 4)])
def test_2d_hierarchical_fp32_sum_within_order_floor_and_broken_arm_fails(rm, cm):
    n = rm * cm
    g = torch.Generator().manual_seed(42 + n)
    x = torch.randn(n, 64, 64, generator=g, dtype=torch.float32)
    x *= torch.pow(10.0, -6 * torch.rand(n, 64, 1, generator=g))

    ser, floor = ccl.fp32_reduction_order_floor(x)
    assert floor > 0.0

    # Exact 2D order: sum along axis 0 (0..rm-1) then axis 1 (0..cm-1)
    x2d = x.view(rm, cm, 64, 64)
    s0 = x2d[0].clone()
    for i in range(1, rm):
        s0 = s0 + x2d[i]
    s1 = s0[0].clone()
    for j in range(1, cm):
        s1 = s1 + s0[j]

    err = float((s1 - ser).abs().max())
    assert err <= floor, f"2D exact sum exceeded floor: {err} > {floor}"

    # Broken arm: rank N-1 dropped must FAIL the floor decisively
    xb = x.clone()
    xb[n - 1] = 0.0
    xb2d = xb.view(rm, cm, 64, 64)
    b0 = xb2d[0].clone()
    for i in range(1, rm):
        b0 = b0 + xb2d[i]
    b1 = b0[0].clone()
    for j in range(1, cm):
        b1 = b1 + b0[j]
    err_broken = float((b1 - ser).abs().max())
    assert err_broken > floor * 1e4, f"broken arm did not fail decisively: {err_broken} vs {floor}"
