"""ESMFold2's outer product runs on permute_back_pk.tile_rows at every L, not only a tile-aligned one.

tile_rows needs L % 32 == 0. Every other length fell back to `_rows`, whose 5-D permute of the
[B, L*32, L*32] product took ~22 s to set up the first time a process saw that L: a fresh 585-aa
fold spent 25 s before its first trunk loop against 3 s at 576 aa, and each worker of a multi-card
run paid it again for every new protein length. The caller now pads b's L axis with zero columns
to the next tile and `_rows_pk` slices them off. Column j of the product reads only b[:, :, j], so
the real columns are the ones `_rows` computes.
"""
import pytest
import torch
import ttnn

pytestmark = pytest.mark.device


@pytest.mark.parametrize("L,M", [(45, 8), (77, 33), (64, 8)])
def test_pk_at_an_untiled_length_equals_the_tile_path(L, M, monkeypatch):
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    from tt_bio import esmfold2 as E
    from tt_bio.tenstorrent import get_device
    dev = get_device()
    g = torch.Generator().manual_seed(L * 100 + M)
    sd = {"norm.weight": 1 + 0.1 * torch.randn(128, generator=g), "norm.bias": 0.1 * torch.randn(128, generator=g),
          "W.weight": 0.1 * torch.randn(64, 128, generator=g), "Wout.weight": 0.05 * torch.randn(256, 1024, generator=g),
          "Wout.bias": 0.1 * torch.randn(256, generator=g)}
    ck = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                          fp32_dest_acc_en=True, packer_l1_acc=False)
    opm = E.OuterProductMean(sd, ck)
    m = ttnn.from_torch(torch.randn(1, L, M, 128, generator=g), layout=ttnn.TILE_LAYOUT, device=dev,
                        dtype=ttnn.bfloat16)
    recip = ttnn.from_torch(0.5 + torch.rand(1, L, L, 1, generator=g), layout=ttnn.TILE_LAYOUT, device=dev,
                            dtype=ttnn.bfloat16)
    from tt_bio import permute_back_pk as PK
    served = []
    real_tile_rows = PK.tile_rows
    monkeypatch.setattr(PK, "tile_rows", lambda *a, **k: served.append(1) or real_tile_rows(*a, **k))
    out = {}
    for path in ("tile", "pk"):
        monkeypatch.setattr(E, "_OPM_PATH", path)
        out[path] = ttnn.to_torch(opm(m, recip)).float()
    assert out["pk"].shape == out["tile"].shape == (1, L, L, 256)
    assert served, "tile_rows did not serve the pk path"
    assert torch.equal(out["pk"], out["tile"])
