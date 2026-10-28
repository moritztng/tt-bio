"""RFD3_DIT_MM2D: the DiT linears through the pinned 2D config are as close to a float64 reference as ttnn's own pick,
with and without the fused silu, at the page shape (685 tokens) and a ragged short one."""
import pytest
import torch
import ttnn

from tt_bio.main import ensure_p300_mesh_descriptor
from tt_bio.rfd3 import model as M

pytestmark = pytest.mark.device


def _up(t, dev):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)


@pytest.mark.parametrize("rows", [685, 77])
@pytest.mark.parametrize("k,n", sorted(M._MM2D_SHAPES))
@pytest.mark.parametrize("silu", [False, True])
def test_mm2d_matches_default(monkeypatch, rows, k, n, silu):
    ensure_p300_mesh_descriptor()
    dev = M.get_device()
    ckc = M._default_compute_kernel_config()
    g = torch.Generator().manual_seed(rows + k + n)
    x = torch.randn(1, rows, k, generator=g).bfloat16()
    w = (torch.randn(k, n, generator=g) / k ** 0.5).bfloat16()
    ref = x.double() @ w.double()
    ref = torch.nn.functional.silu(ref) if silu else ref
    xd, wd = _up(x, dev), _up(w, dev)

    monkeypatch.setattr(M, "_MM2D_ENV", False)
    assert M._mm2d_config(xd, wd) is None
    base = ttnn.to_torch(ttnn.linear(xd, wd, compute_kernel_config=ckc, dtype=ttnn.bfloat16,
                                     activation="silu" if silu else None, core_grid=M.BATCH_INVARIANT_GRID))
    monkeypatch.setattr(M, "_MM2D_ENV", True)
    pc = M._mm2d_config(xd, wd, silu=silu)
    assert pc is not None
    got = ttnn.to_torch(ttnn.linear(xd, wd, compute_kernel_config=ckc, dtype=ttnn.bfloat16, program_config=pc))

    e_base = (base.double()[..., :rows, :] - ref).abs().max().item()
    e_got = (got.double()[..., :rows, :] - ref).abs().max().item()
    assert e_got <= 1.5 * e_base + 1e-3, (e_got, e_base)
