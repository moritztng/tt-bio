"""ttnn's float32 -> bf16 host conversion rounds exactly like torch's, so a host tensor may be cast in
torch before `ttnn.from_torch` (`protenix._KeyedWeights._up`, the Protenix MSA feature) with the same
device tensor and a 4x faster tilize. Checked on the values that tell rounding modes apart: exact
ties, the neighbours of ties, subnormals, the largest finite values, infinities and zeros.

ttnn opens the cluster for a host tilize, so run it pinned:
TT_VISIBLE_DEVICES=<card> python3 -m pytest -s tests/test_bf16_host_round_hw.py
"""
import pytest
import torch
import ttnn

from tt_bio.main import ensure_p300_mesh_descriptor

pytestmark = pytest.mark.device


def _values():
    g = torch.Generator().manual_seed(0)
    bits = torch.randint(-2 ** 31, 2 ** 31 - 1, (1 << 20,), generator=g, dtype=torch.int64).to(torch.int32)
    rnd = bits.view(torch.float32)
    rnd = rnd[torch.isfinite(rnd)]
    hi = torch.randint(-2 ** 15, 2 ** 15, (1 << 16,), generator=g, dtype=torch.int32) << 16
    ties = (hi | 0x8000).view(torch.float32)                         # exactly halfway
    near = torch.cat([(hi | 0x7FFF).view(torch.float32), (hi | 0x8001).view(torch.float32)])
    sub = (torch.randint(1, 1 << 23, (1 << 14,), generator=g, dtype=torch.int32)).view(torch.float32)
    edge = torch.tensor([0.0, -0.0, float("inf"), float("-inf"), 3.3895314e38, -3.3895314e38,
                         torch.finfo(torch.float32).max, torch.finfo(torch.float32).tiny])
    v = torch.cat([rnd, ties, near, sub, -sub, edge])
    v = v[torch.isfinite(v) | torch.isinf(v)]
    n = -(-v.numel() // 1024) * 1024
    return torch.cat([v, torch.zeros(n - v.numel())]).reshape(-1, 32, 32)


def test_torch_and_ttnn_round_float32_to_bf16_alike():
    ensure_p300_mesh_descriptor()
    v = _values()
    a = ttnn.to_torch(ttnn.from_torch(v, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16))
    b = ttnn.to_torch(ttnn.from_torch(v.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16))
    same = (a.view(torch.int16) == b.view(torch.int16))
    assert bool(same.all()), f"{int((~same).sum())} of {v.numel()} differ, e.g. {v[~same][:8].tolist()}"
