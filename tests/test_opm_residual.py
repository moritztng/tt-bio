"""`OuterProductMean(..., residual=z)` is `z + OuterProductMean(...)`, bit for bit, whole and blocked.

The Protenix/OpenDDE MSA block adds the OPM update to z3 through the op itself, so a row-blocked
update adds each block to its own rows and frees the old z3 at the join. At 1536 tokens and
c_z=384 the residual, a whole OPM output and their sum were three 1811939328 B buffers at once.
"""
import pytest
import torch
import ttnn

from tt_bio import tenstorrent as T

pytestmark = pytest.mark.device

N, S, C_M, C, C_Z = 64, 80, 64, 32, 128


@pytest.fixture(scope="module")
def dev():
    return T.get_device()


def _opm(dev):
    torch.manual_seed(0)
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)
    sd = {"norm.weight": 1 + 0.1 * torch.randn(C_M), "norm.bias": 0.1 * torch.randn(C_M),
          "proj_a.weight": torch.randn(C, C_M) / C_M ** 0.5,
          "proj_b.weight": torch.randn(C, C_M) / C_M ** 0.5,
          "proj_o.weight": torch.randn(C_Z, C * C) / C,
          "proj_o.bias": 0.1 * torch.randn(C_Z)}
    return T.OuterProductMean(sd, ckc, scale_bias=True)


@pytest.mark.parametrize("rows", [None, 32])
@pytest.mark.parametrize("chunked", [False, True])
def test_residual_is_the_callers_add(dev, rows, chunked):
    opm = _opm(dev)
    ft = lambda x: ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    mh, zh = torch.randn(1, S, N, C_M), torch.randn(1, N, N, C_Z)
    m = (lambda: [ft(mh[:, s:s + 32]) for s in range(0, S, 32)]) if chunked else (lambda: ft(mh))
    if rows:
        T._OPM_DRAM_ROW_CAP[(N, C, C, N)] = rows      # as a refusal leaves it: row-blocked
    try:
        ref = ttnn.to_torch(ttnn.add(ft(zh), opm(m(), None, None)))
        z = ft(zh)
        out = opm(m(), None, None, residual=z)
    finally:
        T._OPM_DRAM_ROW_CAP.clear()
    assert tuple(out.shape) == (1, N, N, C_Z)
    assert torch.equal(ttnn.to_torch(out), ref)
    assert z.is_allocated() == (not rows)             # a blocked join consumes the residual


def test_parked_chunks_are_the_device_chunks(dev):
    """A chunk list parked on the host is uploaded a chunk at a time, same bytes as on device."""
    opm = _opm(dev)
    ft = lambda x: ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    mh, zh = torch.randn(1, S, N, C_M), torch.randn(1, N, N, C_Z)
    ref = ttnn.to_torch(opm([ft(mh[:, s:s + 32]) for s in range(0, S, 32)], None, None,
                            residual=ft(zh)))
    parked = [ttnn.from_device(ft(mh[:, s:s + 32])) for s in range(0, S, 32)]
    out = opm(parked, None, None, residual=ft(zh))
    assert torch.equal(ttnn.to_torch(out), ref)
    assert all(p.storage_type() != ttnn.StorageType.DEVICE for p in parked)   # still parked
