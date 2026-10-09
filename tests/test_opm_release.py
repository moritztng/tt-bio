"""OuterProductMean frees its operands after a whole contraction, and survives a refused relayout.

The whole path contracts `z = a b^T` once, frees `a` and `b`, then relayouts z. When the first relayout copy is
refused, the call cannot contract again, so it relayouts the same z in row blocks. Both must give the float64
answer, and the whole path must not be slower to reach it.
"""
import pytest
import torch
import ttnn

from tt_bio import tenstorrent as T

N, C_M, C, C_Z, DEPTH = 64, 64, 32, 128, 2079


def _ref(sd, mh):
    x = torch.nn.functional.layer_norm(mh[0].double(), (C_M,), sd["norm.weight"].double(),
                                       sd["norm.bias"].double(), 1e-5)
    a, b = x @ sd["proj_a.weight"].double().t(), x @ sd["proj_b.weight"].double().t()
    o = torch.einsum("sic,sjd->ijcd", a, b).reshape(N, N, C * C)
    return (o @ sd["proj_o.weight"].double().t() + sd["proj_o.bias"].double()) / mh.shape[1]


@pytest.fixture(autouse=True)
def _clean():
    def reset():
        T._OPM_DRAM_ROW_CAP.clear()
        T.OPM_ROW_STATS.update(whole=0, blocked=0, dram_narrowed=0, join_split=0)
    reset()
    yield
    reset()


@pytest.mark.device
@pytest.mark.parametrize("chunked", [False, True])
def test_refused_relayout_after_release(monkeypatch, chunked):
    dev = T.get_device()
    torch.manual_seed(0)
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)
    sd = {"norm.weight": 1 + 0.1 * torch.randn(C_M), "norm.bias": 0.1 * torch.randn(C_M),
          "proj_a.weight": torch.randn(C, C_M) / C_M ** 0.5,
          "proj_b.weight": torch.randn(C, C_M) / C_M ** 0.5,
          "proj_o.weight": torch.randn(C_Z, C * C) / C,
          "proj_o.bias": 0.1 * torch.randn(C_Z)}
    opm = T.OuterProductMean(sd, ckc, scale_bias=True)
    mh = torch.randn(1, DEPTH, N, C_M)
    ft = lambda x: ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    m = lambda: [ft(mh[:, s:s + 512]) for s in range(0, DEPTH, 512)] if chunked else ft(mh)
    ref = _ref(sd, mh.bfloat16().float())
    rel = lambda o: float((o - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt())

    whole = ttnn.to_torch(opm(m(), None, None)).double().reshape(N, N, C_Z)
    assert T.OPM_ROW_STATS["whole"] == 1 and T.OPM_ROW_STATS["blocked"] == 0

    real, refused = T.ttnn.to_layout, []

    def to_layout(t, layout, *a, **kw):
        # Refuse the whole z's first copy (rows = N * C), once, the way the allocator does.
        if layout == ttnn.ROW_MAJOR_LAYOUT and tuple(t.shape) == (N * C, C * N) and not refused:
            refused.append(1)
            raise RuntimeError("TT_FATAL: Out of Memory: Not enough space to allocate (test)")
        return real(t, layout, *a, **kw)

    monkeypatch.setattr(T.ttnn, "to_layout", to_layout)
    out = ttnn.to_torch(opm(m(), None, None)).double().reshape(N, N, C_Z)
    monkeypatch.setattr(T.ttnn, "to_layout", real)
    assert refused and T.OPM_ROW_STATS["dram_narrowed"] == 1 and T.OPM_ROW_STATS["blocked"] == 1
    assert torch.isfinite(out).all()
    assert rel(whole) < 2e-2 and rel(out) < 1.25 * rel(whole) + 1e-3, (rel(whole), rel(out))
    # The relayout moves numbers; only proj_o's row blocking can round differently.
    assert float((out - whole).abs().max()) < 5e-2


# Device-free: the same op on a torch-backed stand-in for every ttnn call it makes, so a refusal can be
# forced at each allocation in turn. A freed tensor raises when read, which is how 0d2737a5b died on
# Wormhole (3 of 88 folds at ~800 tokens): the whole z's TILE relayout or permute refused after `a` and
# `b` were freed, and the retry read the freed operand.
class _Buf:
    def __init__(self):
        self.live = True


class _T:
    def __init__(self, data, layout=ttnn.TILE_LAYOUT, buf=None, dtype=ttnn.bfloat16):
        self._d, self.layout, self.dtype = data, layout, dtype
        self.buf = buf or _Buf()

    @property
    def d(self):
        if not self.buf.live:
            raise RuntimeError("read of a deallocated tensor")
        return self._d

    @property
    def shape(self):
        return tuple(self._d.shape)

    @property
    def padded_shape(self):
        s = list(self._d.shape)
        return tuple(s[:-2] + [-(-v // 32) * 32 for v in s[-2:]])

    def device(self):
        return _FakeTTNN.dev

    def storage_type(self):
        return ttnn.StorageType.DEVICE

    def __getitem__(self, idx):
        return _FakeTTNN.new(self.d[idx].clone(), self.layout)


class _Dev:
    def __init__(self, arch):
        self._arch = arch

    def arch(self):
        return self._arch

    def compute_with_storage_grid_size(self):
        return ttnn.CoreCoord(8, 8)


class _FakeTTNN:
    """Every allocation goes through `new`, which refuses the `refuse_at`-th one with the allocator's text."""
    dev = None

    def __init__(self, arch, refuse_at=None):
        type(self).dev = _Dev(arch)
        self.n, self.refuse_at, self.refused, self.first_contraction = 0, refuse_at, [], None
        type(self).new = self._new

    def __getattr__(self, name):
        return getattr(ttnn, name)

    def _new(self, data, layout=ttnn.TILE_LAYOUT):
        i, self.n = self.n, self.n + 1
        if i == self.refuse_at:
            self.refused.append(i)
            raise RuntimeError("TT_FATAL: Out of Memory: Not enough space to allocate (test)")
        return _T(data.contiguous(), layout)

    def deallocate(self, t, *a, **kw):
        t.buf.live = False

    def reshape(self, t, shape):
        return _T(t.d.reshape(shape), t.layout, t.buf, t.dtype)

    def to_layout(self, t, layout, *a, **kw):
        return self.new(t.d.clone(), layout)

    def permute(self, t, dims, *a, **kw):
        return self.new(t.d.permute(dims))

    def transpose(self, t, d0, d1, *a, **kw):
        return self.new(t.d.transpose(d0, d1))

    def reallocate(self, t, *a, **kw):
        out = self.new(t.d.clone(), t.layout)
        t.buf.live = False
        return out

    def concat(self, parts, dim, *a, **kw):
        return self.new(torch.cat([p.d for p in parts], dim))

    def zeros(self, shape, *a, **kw):
        return self.new(torch.zeros(shape, dtype=torch.float64))

    def layer_norm(self, x, weight, bias, epsilon, **kw):
        return self.new(torch.nn.functional.layer_norm(x.d, x.shape[-1:], weight.d, bias.d, epsilon))

    def linear(self, x, w, bias=None, **kw):
        return self.new(x.d @ w.d + (0 if bias is None else bias.d))

    def matmul(self, x, y, transpose_b=False, **kw):
        if transpose_b and self.first_contraction is None:
            self.first_contraction = self.n
        return self.new(x.d @ (y.d.transpose(-1, -2) if transpose_b else y.d))

    def multiply(self, t, s, **kw):
        return self.new(t.d * (s.d if isinstance(s, _T) else s))

    def multiply_(self, t, s, **kw):
        t.d.mul_(s.d if isinstance(s, _T) else s)
        return t

    def add(self, x, y, **kw):
        return self.new(x.d + y.d)

    def add_(self, x, y, **kw):
        x.d.add_(y.d)
        return x


def _fake_opm(sd):
    opm = object.__new__(T.OuterProductMean)
    w = lambda k: _T(sd[k].t().contiguous())
    opm.scale_bias, opm.compute_kernel_config, opm._o_folded = True, None, {}
    opm.norm_weight, opm.norm_bias = _T(sd["norm.weight"]), _T(sd["norm.bias"])
    opm.a_weight, opm.b_weight, opm.o_weight = w("proj_a.weight"), w("proj_b.weight"), w("proj_o.weight")
    opm.a_bias = opm.b_bias = None
    opm.o_bias = _T(sd["proj_o.bias"])
    return opm


@pytest.mark.parametrize("arch", ["BLACKHOLE", "WORMHOLE_B0"])
@pytest.mark.parametrize("chunked", [False, True])
@pytest.mark.parametrize("with_residual", [False, True])
def test_a_refusal_at_every_allocation_after_the_contraction_finishes(monkeypatch, arch, chunked,
                                                                      with_residual):
    depth = 40
    g = torch.Generator().manual_seed(0)
    r = lambda *s: torch.randn(*s, generator=g, dtype=torch.float64)
    sd = {"norm.weight": 1 + 0.1 * r(C_M), "norm.bias": 0.1 * r(C_M),
          "proj_a.weight": r(C, C_M) / C_M ** 0.5, "proj_b.weight": r(C, C_M) / C_M ** 0.5,
          "proj_o.weight": r(C_Z, C * C) / C, "proj_o.bias": 0.1 * r(C_Z)}
    mh, res = r(1, depth, N, C_M), r(1, N, N, C_Z)
    want = _ref(sd, mh) + (res[0] if with_residual else 0)

    def call(refuse_at=None):
        fake = _FakeTTNN(getattr(ttnn.Arch, arch), refuse_at)
        monkeypatch.setattr(T, "ttnn", fake)
        T._OPM_DRAM_ROW_CAP.clear()
        x = [_T(mh[:, s:s + 16].clone()) for s in range(0, depth, 16)] if chunked else _T(mh.clone())
        resid = _T(res.clone()) if with_residual else None
        out = T.OuterProductMean.__call__(_fake_opm(sd), x, None, None, residual=resid)
        monkeypatch.setattr(T, "ttnn", ttnn)
        return out.d.reshape(N, N, C_Z), fake

    clean, fake = call()
    assert torch.allclose(clean, want, rtol=1e-9, atol=1e-9)
    assert T.OPM_ROW_STATS["whole"] == 1 and T.OPM_ROW_STATS["blocked"] == 0
    first, total = fake.first_contraction, fake.n
    assert first is not None and total - first >= 6
    for k in range(first, total):
        T.OPM_ROW_STATS.update(whole=0, blocked=0, dram_narrowed=0, join_split=0)
        out, f = call(k)
        assert f.refused == [k], (k, f.refused)
        assert T.OPM_ROW_STATS["dram_narrowed"] == 1 and T.OPM_ROW_STATS["blocked"] == 1, k
        assert torch.allclose(out, want, rtol=1e-9, atol=1e-9), (k, float((out - want).abs().max()))
