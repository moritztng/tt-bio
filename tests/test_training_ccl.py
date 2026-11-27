"""CPU unit tests for tt_bio.train.ccl (exact fp32 2D hierarchical all-reduce helpers)."""
import math
import sys
import pytest
import torch

from tt_bio.train import ccl

RFD4_PARAM_SHAPE_COUNTS = [
    ((6144, 768), 14),
    ((768, 3072), 14),
    ((2304, 768), 14),
    ((768, 768), 38),
    ((768, 256), 84),
    ((256, 384), 56),
    ((384, 384), 15),
    ((768, 384), 7),
    ((384, 768), 7),
    ((384, 256), 16),
    ((768, 897), 1),
    ((1024, 256), 2),
    ((256, 512), 3),
    ((256, 609), 1),
    ((768,), 151),
    ((147, 768), 1),
    ((12, 384), 14),
    ((1473, 32), 1),
    ((384, 107), 1),
    ((2304,), 14),
    ((32, 768), 1),
    ((384,), 52),
    ((256,), 44),
    ((3, 768), 1),
    ((128, 16), 1),
    ((44, 32), 1),
    ((7, 32), 1),
    ((6, 32), 1),
    ((11, 16), 1),
    ((4, 16), 1),
]


def test_ccl_import_does_not_import_ttnn():
    import subprocess
    from pathlib import Path
    out = subprocess.run(
        [sys.executable, "-c", "import sys, tt_bio.train.ccl; assert 'ttnn' not in sys.modules"],
        capture_output=True, text=True, check=False,
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    assert out.returncode == 0, out.stderr


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
def test_plan_reduction_groups_covers_all_558_rfd4_params(rm, cm):
    shapes = {}
    idx = 0
    for shp, cnt in RFD4_PARAM_SHAPE_COUNTS:
        for j in range(cnt):
            shapes[f"p{idx:03d}_{j}"] = shp
            idx += 1
    assert len(shapes) == 558
    total_elems = sum(math.prod(s) for s in shapes.values())
    assert total_elems == 178_448_304

    plan = ccl.plan_reduction_groups(shapes, (rm, cm))
    # Top 4 width groups (768, 3072, 256, 384) cover >99% of elements (512 also promotes on <=16 chips)
    assert [wg.cols for wg in plan.width_groups][:4] == [768, 3072, 256, 384]
    seen = set()
    w_elems = 0
    row_quantum = 32 * rm * cm
    for wg in plan.width_groups:
        assert wg.cols % 32 == 0
        assert wg.padded_rows >= wg.total_rows
        assert wg.padded_rows % row_quantum == 0
        off = 0
        for nm, shp, r_off, rows in wg.entries:
            assert nm not in seen
            seen.add(nm)
            assert r_off == off
            assert rows * wg.cols == math.prod(shp)
            assert rows % 32 == 0
            off += rows
            w_elems += math.prod(shp)
        assert off == wg.total_rows
    assert w_elems / total_elems > 0.99

    s_elems = 0
    for sg in plan.small_groups:
        off = 0
        for nm, shp, b_off, b in sg.entries:
            assert nm not in seen
            seen.add(nm)
            assert b_off == off
            assert b * sg.m * sg.n == math.prod(shp)
            off += b
            s_elems += math.prod(shp)
        assert off == sg.total_batch

    assert seen == set(shapes)
    assert w_elems + s_elems == total_elems


def test_plan_reduction_groups_respects_replicated_names():
    shapes = {"host_glue.w": (768, 897), "trunk.w": (6144, 768), "norm.b": (768,)}
    plan = ccl.plan_reduction_groups(shapes, (8, 4), replicated_names={"host_glue.w"})
    assert plan.skipped == ("host_glue.w",)
    all_planned = {e[0] for wg in plan.width_groups for e in wg.entries} | {
        e[0] for sg in plan.small_groups for e in sg.entries
    }
    assert all_planned == {"trunk.w", "norm.b"}


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
