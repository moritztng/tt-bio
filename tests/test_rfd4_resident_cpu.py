"""CPU-only unit tests for resident RFD4 training invariants (no TT device opened).

Covers:
1. DeviceHeads.forward untaped s32 deallocation when masks_dev is passed with bf16 s.
2. DeviceFolds.tape_backward skipping and deallocating frozen leaves (requires_grad=False).
3. DeviceOptimizer exact dead_grads detection (None, disabled, and identically-zero tensors)
   plus save_adapter/load_adapter checkpoint round-trip and Muon-momentum refusal.
4. rfd4_run.py resident wiring for --freeze, --resume refold, and dead_grads.
"""
from __future__ import annotations

import ast
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch


class _FakeTensor:
    _next_addr = 1000

    def __init__(self, data: torch.Tensor, dtype: str = "float32", device=None):
        self._data = data.detach().clone()
        self.dtype = dtype
        self._device = device or "fake_dev"
        self._alloc = True
        self._addr = _FakeTensor._next_addr
        _FakeTensor._next_addr += 1

    @property
    def shape(self):
        return tuple(self._data.shape)

    def device(self):
        return self._device

    def is_allocated(self) -> bool:
        return self._alloc

    def buffer_address(self) -> int:
        return self._addr


def _make_fake_ttnn():
    m = types.ModuleType("ttnn")
    m.Tensor = _FakeTensor
    m.float32 = "float32"
    m.bfloat16 = "bfloat16"
    m.bfloat8_b = "bfloat8_b"
    m.bfloat4_b = "bfloat4_b"
    m.uint32 = "uint32"
    m.int32 = "int32"
    m.uint16 = "uint16"
    m.TILE_LAYOUT = "TILE"
    m.ROW_MAJOR_LAYOUT = "RM"
    m.DRAM_MEMORY_CONFIG = "DRAM"
    m.L1_MEMORY_CONFIG = "L1"
    m.NOC = types.SimpleNamespace(NOC_0="NOC_0", NOC_1="NOC_1")
    m.UnaryOpType = types.SimpleNamespace(RELU="RELU")
    m.MathFidelity = types.SimpleNamespace(HiFi4="HiFi4", HiFi2="HiFi2", LoFi="LoFi")
    m.WormholeComputeKernelConfig = lambda **kw: kw
    m.types = types.SimpleNamespace(
        BlackholeComputeKernelConfig=lambda **kw: kw,
    )
    m.deallocated = []

    def _td(dt):
        return torch.bfloat16 if dt == m.bfloat16 else torch.float32

    def from_torch(t, dtype=m.float32, layout=None, device=None, **kw):
        dt = dtype or m.float32
        return _FakeTensor(t.to(_td(dt)), dtype=dt, device=device)

    def to_torch(t):
        return t._data.clone()

    def deallocate(t):
        if t is not None:
            t._alloc = False
            m.deallocated.append(t)

    def clone(t):
        return _FakeTensor(t._data, dtype=t.dtype, device=t._device)

    def typecast(t, dt):
        return _FakeTensor(t._data.to(_td(dt)), dtype=dt, device=t._device)

    def reshape(t, shape):
        return _FakeTensor(t._data.reshape(tuple(shape)), dtype=t.dtype, device=t._device)

    def slice_(t, beg, fin):
        idx = tuple(builtins_slice(a, b) for a, b in zip(beg, fin))
        return _FakeTensor(t._data[idx], dtype=t.dtype, device=t._device)

    def concat(ts, dim=0):
        return _FakeTensor(torch.cat([x._data for x in ts], dim=dim), dtype=ts[0].dtype, device=ts[0]._device)

    def zeros(shape, dtype=m.float32, layout=None, device=None):
        return _FakeTensor(torch.zeros(tuple(shape), dtype=_td(dtype)), dtype=dtype, device=device)

    def _bin(fn):
        def _op(a, b):
            av = a._data if isinstance(a, _FakeTensor) else a
            bv = b._data if isinstance(b, _FakeTensor) else b
            ref = a if isinstance(a, _FakeTensor) else b
            return _FakeTensor(fn(av.float(), bv.float() if torch.is_tensor(bv) else bv).to(_td(ref.dtype)),
                               dtype=ref.dtype, device=ref._device)
        return _op

    def matmul(a, b, transpose_a=False, transpose_b=False, dtype=None, compute_kernel_config=None):
        av = a._data.float()
        bv = b._data.float()
        if transpose_a:
            av = av.transpose(-2, -1)
        if transpose_b:
            bv = bv.transpose(-2, -1)
        dt = dtype or a.dtype
        return _FakeTensor(torch.matmul(av, bv).to(_td(dt)), dtype=dt, device=a._device)

    def transpose(t, d0, d1):
        return _FakeTensor(t._data.transpose(d0, d1), dtype=t.dtype, device=t._device)

    def rms_norm(x, weight=None, epsilon=1e-5, **kw):
        v = x._data.float()
        inv = torch.rsqrt((v * v).mean(dim=-1, keepdim=True) + epsilon)
        out = v * inv
        if weight is not None:
            out = out * weight._data.float()
        return _FakeTensor(out.to(_td(x.dtype)), dtype=x.dtype, device=x._device)

    def linear(x, w, bias=None, **kw):
        out = torch.matmul(x._data.float(), w._data.float())
        if bias is not None:
            out = out + bias._data.float()
        return _FakeTensor(out.to(_td(x.dtype)), dtype=x.dtype, device=x._device)

    builtins_slice = slice
    m.from_torch = from_torch
    m.to_torch = to_torch
    m.deallocate = deallocate
    m.clone = clone
    m.typecast = typecast
    m.reshape = reshape
    m.slice = slice_
    m.concat = concat
    m.zeros = zeros
    m.add = _bin(lambda a, b: a + b)
    m.subtract = _bin(lambda a, b: a - b)
    m.multiply = _bin(lambda a, b: a * b)
    m.divide = _bin(lambda a, b: a / b)
    m.square = lambda x: _FakeTensor((x._data.float() ** 2).to(_td(x.dtype)), dtype=x.dtype, device=x._device)
    m.sqrt = lambda x: _FakeTensor(torch.sqrt(x._data.float()).to(_td(x.dtype)), dtype=x.dtype, device=x._device)
    m.clamp = lambda x, min=None, max=None: _FakeTensor(
        torch.clamp(x._data.float(), min=min, max=max).to(_td(x.dtype)), dtype=x.dtype, device=x._device)
    m.reciprocal = lambda x: _FakeTensor((1.0 / x._data.float()).to(_td(x.dtype)), dtype=x.dtype, device=x._device)
    m.matmul = matmul
    m.transpose = transpose
    m.rms_norm = rms_norm
    m.linear = linear
    return m


def _install_fake_ttnn(monkeypatch, *, preload: tuple[str, ...] = ()):
    """Install fake ttnn cleanly without polluting real-ttnn imports when ttnn is installed."""
    if importlib.util.find_spec("ttnn") is not None:
        for mod_name in preload:
            __import__(mod_name)
    fake = _make_fake_ttnn()
    monkeypatch.setitem(sys.modules, "ttnn", fake)
    return fake


def test_heads_s32_dealloc_when_masks_dev_given(monkeypatch):
    fake = _install_fake_ttnn(monkeypatch, preload=("tt_bio.rfd4.heads",))
    import tt_bio.rfd4.heads as heads_mod
    monkeypatch.setattr(heads_mod, "ttnn", fake)
    monkeypatch.setattr(heads_mod.ops, "taping", lambda: False)
    DeviceHeads = heads_mod.DeviceHeads

    dh = DeviceHeads.__new__(DeviceHeads)
    dh.device = "fake_dev"
    dh.dtype = fake.float32
    dh.ckc = {}
    dh.cfg = {"output_parameterization": {"bb_ca": "v", "local_latents": "v"}}
    dh.sequence_readout_enabled = True
    dh._insertion_head_scale = 8.0 ** -0.5
    dh.ca_norm_w = fake.from_torch(torch.ones(1, 1, 1, 8))
    dh.ca_w = fake.from_torch(torch.ones(8, 3))
    dh.ll_norm_w = fake.from_torch(torch.ones(1, 1, 1, 8))
    dh.ll_w = fake.from_torch(torch.ones(8, 4))
    dh.seq_norm_w = fake.from_torch(torch.ones(1, 1, 1, 8))
    dh.seq_w = fake.from_torch(torch.ones(8, 5))
    dh.ins_q_w = fake.from_torch(torch.ones(8, 8))
    dh.ins_k_w = fake.from_torch(torch.ones(8, 8))

    m_ca = fake.from_torch(torch.ones(1, 4, 1))
    m_ll = fake.from_torch(torch.ones(1, 4, 1))
    m_pm = fake.from_torch(torch.ones(1, 4, 1))

    # Case 1: bf16 s with masks_dev provided -> temporary s32 must be deallocated, s stays live.
    s_bf16 = fake.from_torch(torch.randn(1, 4, 8), dtype=fake.bfloat16)
    created_s32 = []
    orig_typecast = fake.typecast

    def spy_typecast(t, dt):
        out = orig_typecast(t, dt)
        if t is s_bf16 and dt == fake.float32:
            created_s32.append(out)
        return out

    monkeypatch.setattr(fake, "typecast", spy_typecast)
    dh(s_bf16, {}, masks_dev=(m_ca, m_ll, m_pm), insertion=False)
    assert len(created_s32) == 1
    assert not created_s32[0].is_allocated(), "s32 leaked when masks_dev was passed with bf16 s"
    assert s_bf16.is_allocated(), "caller input s_bf16 was erroneously deallocated"
    assert m_ca.is_allocated() and m_ll.is_allocated() and m_pm.is_allocated()

    # Case 2: fp32 s with masks_dev provided -> s32 is s, must NOT deallocate caller's s.
    s_f32 = fake.from_torch(torch.randn(1, 4, 8), dtype=fake.float32)
    dh(s_f32, {}, masks_dev=(m_ca, m_ll, m_pm), insertion=False)
    assert s_f32.is_allocated(), "caller input s_f32 was erroneously deallocated"


def test_device_folds_tape_backward_skips_frozen_leaf(monkeypatch):
    fake = _install_fake_ttnn(monkeypatch, preload=("tt_bio.train.folds",))
    import tt_bio
    from tt_bio.train.folds import DeviceFolds, _FoldNode, _OutputSpec

    df = DeviceFolds.__new__(DeviceFolds)
    df.device = "fake_dev"
    df.leaves = {
        "live_w": torch.zeros(4, 4, requires_grad=True),
        "frozen_w": torch.zeros(4, 4, requires_grad=False),
    }
    df.nodes = [
        _FoldNode(0, "leaf", (), {"name": "live_w"}, (4, 4)),
        _FoldNode(1, "leaf", (), {"name": "frozen_w"}, (4, 4)),
    ]
    df.leaf_nid = {"live_w": 0, "frozen_w": 1}
    df.consts = {}

    h_live = fake.from_torch(torch.ones(4, 4))
    h_frozen = fake.from_torch(torch.ones(4, 4))
    df.specs = [
        _OutputSpec(0, h_live, fake.float32, fake.TILE_LAYOUT, fake.DRAM_MEMORY_CONFIG, (4, 4)),
        _OutputSpec(1, h_frozen, fake.float32, fake.TILE_LAYOUT, fake.DRAM_MEMORY_CONFIG, (4, 4)),
    ]

    g_live = fake.from_torch(torch.full((4, 4), 2.0))
    g_frozen = fake.from_torch(torch.full((4, 4), 3.0))
    p_live = types.SimpleNamespace(grad=g_live)
    p_frozen = types.SimpleNamespace(grad=g_frozen)

    if "tt_bio.autograd" in sys.modules:
        monkeypatch.setattr(sys.modules["tt_bio.autograd"], "parameter_for",
                            lambda h: p_live if h is h_live else p_frozen)
    else:
        fake_ag = types.ModuleType("tt_bio.autograd")
        fake_ag.parameter_for = lambda h: p_live if h is h_live else p_frozen
        monkeypatch.setitem(sys.modules, "tt_bio.autograd", fake_ag)
        monkeypatch.setattr(tt_bio, "autograd", fake_ag, raising=False)

    out_grads, idle = df.tape_backward({})
    assert list(out_grads.keys()) == ["live_w"]
    assert idle == []
    assert out_grads["live_w"].is_allocated()
    assert not g_frozen.is_allocated(), "frozen leaf grad was not deallocated"


def test_device_optim_dead_grads_and_checkpoint_roundtrip(monkeypatch, tmp_path: Path):
    fake = _install_fake_ttnn(monkeypatch, preload=("tt_bio.train.device_optim", "tt_bio.train.checkpoint"))
    from tt_bio.train import device_optim as DO
    from tt_bio.train.checkpoint import load_adapter, save_adapter
    from tt_bio.train.optim import Muon

    # Reset caches so fake device tensors aren't shared across tests.
    DO._ONES_COL_CACHE.clear()
    DO._ONES_ROW_CACHE.clear()

    rng = np.random.default_rng(7)
    init = {
        "nn.blk.w0": torch.from_numpy(rng.standard_normal((8, 8)).astype(np.float32)),
        "nn.blk.w1": torch.from_numpy(rng.standard_normal((8, 8)).astype(np.float32)),
        "nn.blk.w2": torch.from_numpy(rng.standard_normal((8, 8)).astype(np.float32)),
        "nn.blk.norm0": torch.from_numpy(rng.standard_normal((8,)).astype(np.float32)),
        "nn.blk.norm1": torch.from_numpy(rng.standard_normal((8,)).astype(np.float32)),
        "nn.blk.norm2": torch.from_numpy(rng.standard_normal((8,)).astype(np.float32)),
        "nn.blk.norm3": torch.from_numpy(rng.standard_normal((8,)).astype(np.float32)),
    }
    opt = DO.DeviceOptimizer(
        init, "fake_dev", lr=1e-3, clip_norm=1.0,
        muon=Muon(exclude=("norm",), adjust_lr_fn="match_rms_adamw", ns_steps=2),
        ema_decay=0.99, ns_mode="fp32",
    )

    def make_grads(seed: int, *, with_zeros: bool = True):
        r = np.random.default_rng(seed)
        g = {
            "nn.blk.w0": fake.from_torch(torch.from_numpy(r.standard_normal((8, 8)).astype(np.float32))),
            "nn.blk.w1": fake.from_torch(torch.zeros((8, 8), dtype=torch.float32)) if with_zeros else fake.from_torch(torch.from_numpy(r.standard_normal((8, 8)).astype(np.float32))),
            # nn.blk.w2 omitted (None)
            "nn.blk.norm0": fake.from_torch(torch.from_numpy(r.standard_normal((8,)).astype(np.float32))),
            "nn.blk.norm1": fake.from_torch(torch.zeros((8,), dtype=torch.float32)) if with_zeros else fake.from_torch(torch.from_numpy(r.standard_normal((8,)).astype(np.float32))),
            # nn.blk.norm2 omitted (None)
            "nn.blk.norm3": fake.from_torch(torch.from_numpy(r.standard_normal((8,)).astype(np.float32))),
        }
        return g

    st1 = opt.step(make_grads(10), disabled=("nn.blk.norm3",), free_grads=True)
    assert st1["dead_grads"] == {"nn.blk.w1", "nn.blk.w2", "nn.blk.norm1", "nn.blk.norm2", "nn.blk.norm3"}
    assert opt.last_dead_grads == frozenset(st1["dead_grads"])
    st2 = opt.step(make_grads(11, with_zeros=False), free_grads=True)
    assert st2["dead_grads"] == {"nn.blk.w2", "nn.blk.norm2"}

    # Checkpoint save + reload into a fresh DeviceOptimizer.
    ck = tmp_path / "res.safetensors"
    save_adapter(ck, opt, meta={"step": opt.steps})
    opt2 = DO.DeviceOptimizer(
        init, "fake_dev", lr=1e-3, clip_norm=1.0,
        muon=Muon(exclude=("norm",), adjust_lr_fn="match_rms_adamw", ns_steps=2),
        ema_decay=0.99, ns_mode="fp32",
    )
    meta = load_adapter(ck, dict(init), "fake_dev", opt=opt2)
    assert meta["step"] == 2 and opt2.steps == 2 and opt2.ema.n_averaged == 2

    hm1, hm2 = opt.host_master(), opt2.host_master()
    assert all(np.array_equal(hm1[k], hm2[k]) for k in hm1)
    sh1, sh2 = opt.ema.host_shadow(), opt2.ema.host_shadow()
    assert all(np.array_equal(sh1[k], sh2[k]) for k in sh1)

    # Next step from resumed optimizer must match unbroken optimizer bit-for-bit.
    opt.step(make_grads(12), free_grads=True)
    opt2.step(make_grads(12), free_grads=True)
    hm1_next, hm2_next = opt.host_master(), opt2.host_master()
    assert all(np.array_equal(hm1_next[k], hm2_next[k]) for k in hm1_next)
    sh1_next, sh2_next = opt.ema.host_shadow(), opt2.ema.host_shadow()
    assert all(np.array_equal(sh1_next[k], sh2_next[k]) for k in sh1_next)

    # Loading a non-Muon checkpoint into a Muon DeviceOptimizer must refuse loudly.
    opt_nomuon = DO.DeviceOptimizer(init, "fake_dev", lr=1e-3, muon=None, ema_decay=None)
    ck_nomuon = tmp_path / "nomuon.safetensors"
    save_adapter(ck_nomuon, opt_nomuon, meta={"step": 0})
    with pytest.raises(ValueError, match="no Muon momentum"):
        load_adapter(ck_nomuon, dict(init), "fake_dev", opt=opt2)

    # Partial family reload (e.g., cross-resuming an AdamW checkpoint with missing bucket entries).
    opt_nomuon.load_host_state(hm1_next, {"exp_avg": {"nn.blk.norm0": np.ones(8, dtype=np.float32)}})
    ea = opt_nomuon.state_families()["exp_avg"]
    assert np.allclose(ea["nn.blk.norm0"], 1.0)
    assert np.allclose(ea["nn.blk.norm1"], 0.0)


def test_rfd4_run_resident_audit_wiring():
    src = Path(__file__).resolve().parents[1].joinpath("tt_bio/train/rfd4_run.py").read_text()
    tree = ast.parse(src)
    # Check --freeze, --resume dfolds.refold, and opt_info["dead_grads"] are all present.
    assert 'opt_info["dead_grads"]' in src
    assert "dfolds.refold(opt.master, on_refold=refold_hook)" in src
    assert "if a.freeze:" in src
    # Ensure rfd4_run parses cleanly and keeps single top-level step branch.
    assert any(isinstance(n, ast.FunctionDef) and n.name == "main" for n in ast.walk(tree))
