"""`autograd._split_k`: a long-K, small-output ``a^T @ b`` as K chunks summed in fp32.

The card-free half checks which calls it takes and how it cuts them; the device half checks the
result against float64 next to ttnn's own plan.

Run: python3 -m pytest tests/test_split_k.py            (card-free)
     TT_VISIBLE_DEVICES=<card> python3 -m pytest tests/test_split_k.py -m device
"""
import pytest
import torch
import ttnn

from tt_bio import autograd as ag


class _Op:
    def __init__(self, shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
        self.shape, self.dtype, self.layout = shape, dtype, layout

    def is_sharded(self):
        return False


@pytest.mark.parametrize("a,b,kw", [
    ((28672, 64), (28672, 224), {}),                                   # no transpose_a
    ((28672, 64), (28672, 224), {"transpose_a": True, "transpose_b": True}),
    ((28672, 64), (28672, 224), {"transpose_a": True, "program_config": object()}),
    ((28672, 64), (28672, 224), {"transpose_a": True, "core_grid": object()}),
    ((1024, 64), (1024, 224), {"transpose_a": True}),                  # K below the floor
    ((28672, 256), (28672, 224), {"transpose_a": True}),               # output above the cap
    ((28672, 64), (28672, 200), {"transpose_a": True}),                # not whole tiles
    ((2, 28672, 64), (2, 28672, 224), {"transpose_a": True}),          # batched
])
def test_declines_leave_the_call_alone(a, b, kw):
    assert ag._split_k(_Op(a), _Op(b), kw) is None


def test_cut_is_a_fixed_number_of_whole_tile_chunks(monkeypatch):
    calls = []
    monkeypatch.setattr(ag, "bmm_program_config", lambda a3, b3, ta, **kw: ("plan", a3.shape, b3.shape))
    monkeypatch.setattr(ttnn, "reshape", lambda t, s: _Op(list(s), t.dtype))
    monkeypatch.setattr(ttnn, "matmul", lambda a3, b3, **kw: calls.append(("mm", a3.shape, b3.shape, kw)) or
                        _Op([a3.shape[0], a3.shape[2], b3.shape[2]], kw["dtype"]))
    monkeypatch.setattr(ttnn, "sum", lambda t, dim: calls.append(("sum", t.shape, dim)) or _Op(t.shape[1:], t.dtype))
    monkeypatch.setattr(ttnn, "typecast", lambda t, d: calls.append(("cast", d)) or _Op(t.shape, d))
    monkeypatch.setattr(ttnn, "deallocate", lambda t: None)
    before = ag.SPLIT_K_STATS["served"]
    out = ag._split_k(_Op((28672, 64)), _Op((28672, 224)), {"transpose_a": True, "dtype": ttnn.float32})
    assert out.shape == [64, 224] and out.dtype == ttnn.float32
    # Kt = 896 -> 64 chunks of 14 tiles; the count does not depend on the card's grid.
    assert calls[0][:3] == ("mm", [64, 448, 64], [64, 448, 224])
    assert calls[0][3]["dtype"] == ttnn.float32 and calls[0][3]["transpose_a"]
    assert calls[1] == ("sum", [64, 64, 224], 0)
    calls.clear()
    out = ag._split_k(_Op((8192, 64)), _Op((8192, 224)), {"transpose_a": True})
    assert calls[0][:3] == ("mm", [64, 128, 64], [64, 128, 224])
    assert calls[-1] == ("cast", ttnn.bfloat16) and out.dtype == ttnn.bfloat16
    assert ag.SPLIT_K_STATS["served"] == before + 2


@pytest.mark.device
@pytest.mark.parametrize("K", [8192, 28672])
def test_matches_float64_at_least_as_well_as_ttnn(K):
    from tt_bio import tenstorrent as T
    dev = T.get_device()
    torch.manual_seed(K)
    x, g = torch.randn(K, 64), torch.randn(K, 224)
    a = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    b = ttnn.from_torch(g, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    cfg = ag.precise_config()
    ref = ttnn.to_torch(a).double().T @ ttnn.to_torch(b).double()
    rel = lambda y: float((ttnn.to_torch(y).double() - ref).norm() / ref.norm())
    own = ttnn.matmul(a, b, transpose_a=True, compute_kernel_config=cfg, dtype=ttnn.float32)
    got = ag._split_k(a, b, {"transpose_a": True, "compute_kernel_config": cfg, "dtype": ttnn.float32})
    assert got is not None and list(got.shape) == [64, 224]
    assert rel(got) <= rel(own) * 1.05 + 1e-7, (rel(got), rel(own))
