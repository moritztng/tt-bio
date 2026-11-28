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
import importlib.machinery
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
    m.__spec__ = importlib.machinery.ModuleSpec("ttnn", loader=None)
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

    def permute(t, dims):
        return _FakeTensor(t._data.permute(*dims).contiguous(), dtype=t.dtype, device=t._device)

    def to_layout(t, layout):
        return _FakeTensor(t._data, dtype=t.dtype, device=t._device)

    def rsqrt(x, fast_and_approximate_mode=True):
        return _FakeTensor(torch.rsqrt(x._data.float()).to(_td(x.dtype)), dtype=x.dtype, device=x._device)

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
    m.permute = permute
    m.to_layout = to_layout
    m.rsqrt = rsqrt
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


def test_refold_hook_populates_empty_t1_cache_on_resume(monkeypatch):
    """After --resume, tri._t1_w is empty before step 1; refold_hook must populate it
    from dev_masters so _t1_in never lazily reads stale step-0 host weights."""
    import tt_bio.train.folds as F

    fake_ttnn = _make_fake_ttnn()
    monkeypatch.setitem(sys.modules, "ttnn", fake_ttnn)
    monkeypatch.setattr(F, "_dev_concat_f32", lambda xs, dim: _FakeTensor(
        torch.cat([x._data for x in xs], dim=dim), "float32"))
    monkeypatch.setattr(F, "_dev_transpose_2d", lambda x: _FakeTensor(x._data.t().contiguous(), "float32"))

    tri = types.SimpleNamespace(_t1_w={})
    pu = types.SimpleNamespace(tri_out=tri, tri_in=None)
    trunk = types.SimpleNamespace(
        pair_updates={0: pu},
        blocks=(),
        _levers=frozenset({"trimul_t1", "trimul_t1w8"}),
    )
    hook = F.make_rfd4_refold_hook(trunk)
    pfx = "nn.pair_update_layers.0.tri_mult_out."
    dev_masters = {
        pfx + "linear_a_p.weight": _FakeTensor(torch.full((4, 4), 2.0), "float32"),
        pfx + "linear_b_p.weight": _FakeTensor(torch.full((4, 4), 3.0), "float32"),
        pfx + "linear_a_g.weight": _FakeTensor(torch.full((4, 4), 4.0), "float32"),
        pfx + "linear_b_g.weight": _FakeTensor(torch.full((4, 4), 5.0), "float32"),
        pfx + "linear_g.weight": _FakeTensor(torch.full((4, 4), 6.0), "float32"),
    }
    hook(dev_masters)
    assert True in tri._t1_w, "refold_hook left _t1_w empty on resume"
    wp, wg, wgo = tri._t1_w[True]
    assert wp.shape == (8, 4) and torch.allclose(wp._data[:4], torch.full((4, 4), 2.0))
    assert torch.allclose(wp._data[4:], torch.full((4, 4), 3.0))
    assert wg.shape == (8, 4) and torch.allclose(wg._data[:4], torch.full((4, 4), 4.0))
    assert torch.allclose(wg._data[4:], torch.full((4, 4), 5.0))
    assert wgo is not None and torch.allclose(wgo._data, torch.full((4, 4), 6.0))
    # Without trimul_t1 in trunk._levers, empty _t1_w stays empty (no extra DRAM allocated).
    tri2 = types.SimpleNamespace(_t1_w={})
    trunk2 = types.SimpleNamespace(pair_updates={0: types.SimpleNamespace(tri_out=tri2, tri_in=None)},
                                   blocks=(), _levers=frozenset())
    F.make_rfd4_refold_hook(trunk2)(dev_masters)
    assert tri2._t1_w == {}


def test_device_insertion_loss_small_and_near_zero(monkeypatch):
    """Exact float64 insertion CE must hold on small (~4e-3) and near-zero (<1e-6)
    losses without SFPU negative-CE bias or p-onehot cancellation, and do 0 PCIe
    when no query rows are active."""
    import contextlib
    fake = _install_fake_ttnn(monkeypatch, preload=("tt_bio.train.rfd4_heads",))
    if "tt_bio.autograd" not in sys.modules:
        import tt_bio
        fake_ag = types.ModuleType("tt_bio.autograd")
        fake_ag.no_grad = contextlib.nullcontext
        monkeypatch.setitem(sys.modules, "tt_bio.autograd", fake_ag)
        monkeypatch.setattr(tt_bio, "autograd", fake_ag, raising=False)
    import tt_bio.train.rfd4_heads as RH
    from tt_bio.train import rfd4_objective as RO

    monkeypatch.setattr(RH, "ttnn", fake)
    reads, writes = [], []
    orig_to, orig_from = fake.to_torch, fake.from_torch

    def spy_to(t):
        reads.append(t.shape)
        return orig_to(t)

    def spy_from(t, *a, **kw):
        writes.append(tuple(t.shape))
        return orig_from(t, *a, **kw)

    monkeypatch.setattr(fake, "to_torch", spy_to)
    monkeypatch.setattr(fake, "from_torch", spy_from)

    n_tok, n_pad = 30, 32
    pad = np.zeros((1, n_tok), dtype=bool)
    pad[0, :26] = True
    z_np = np.full((1, n_pad, n_pad), -8.0, dtype=np.float32)

    # 1. Inactive batch: 0 device ops, 0 PCIe transfers.
    target_none = np.full((1, n_tok), -1, dtype=np.int64)
    z_tt = orig_from(torch.from_numpy(z_np))
    loss0, dz0 = RH.device_insertion_loss(
        z_tt, {"insertion_source_idx": target_none, "padding_mask": pad},
    )
    assert np.array_equal(loss0, np.zeros(1, dtype=np.float64)) and dz0 is None
    assert reads == [] and writes == []

    # 2. Small-loss (~4e-3) and near-zero (<1e-6) peaked logits (5 active rows).
    target = np.full((1, n_tok), -1, dtype=np.int64)
    active_rows = [20, 21, 22, 23, 24]
    for k, r in enumerate(active_rows):
        target[0, r] = k  # indexed candidate keys 0..4
    for label, runner_up in (("small_4e-3", -5.47), ("near_zero_1e-7", -16.1), ("exact_zero", -40.0)):
        z_case = np.full((1, n_pad, n_pad), -30.0, dtype=np.float32)
        for k, r in enumerate(active_rows):
            z_case[0, r, k] = 0.0
            z_case[0, r, (k + 1) % 5] = runner_up
        z_tt = orig_from(torch.from_numpy(z_case))
        loss_got, dz_tt = RH.device_insertion_loss(
            z_tt, {"insertion_source_idx": target, "padding_mask": pad}, weight=0.5,
        )
        loss_ref, dz_ref = RO.insertion_loss(
            z_case[:, :n_tok, :n_tok].astype(np.float64),
            {"insertion_source_idx": target, "padding_mask": pad},
        )
        assert float(loss_got[0]) >= 0.0, f"{label}: negative CE loss {loss_got[0]}"
        abs_err = abs(float(loss_got[0]) - float(loss_ref[0]))
        rel_err = abs_err / max(abs(float(loss_ref[0])), 1e-12)
        assert abs_err < 1e-12 and rel_err < 1e-12, f"{label}: loss mismatch {loss_got[0]} vs {loss_ref[0]}"

        dz_got = orig_to(dz_tt).numpy()
        assert np.all(dz_got[:, n_tok:, :] == 0.0) and np.all(dz_got[:, :, n_tok:] == 0.0)
        dz_want = (dz_ref * 0.5).astype(np.float64)
        dz_rel = float(np.linalg.norm(dz_got[:, :n_tok, :n_tok] - dz_want) / max(np.linalg.norm(dz_want), 1e-12))
        assert dz_rel < 1e-6, f"{label}: dz rel_l2={dz_rel:.3e}"

        # Broken-arm check: -5e-4 SFPU bias on small_4e-3 or x1.10 dz must fail the bar loudly.
        if label == "small_4e-3":
            bad_loss = float(loss_got[0]) - 5e-4
            assert abs(bad_loss - float(loss_ref[0])) / float(loss_ref[0]) > 1e-2
            bad_dz_rel = float(np.linalg.norm(dz_got[:, :n_tok, :n_tok] * 1.10 - dz_want) / np.linalg.norm(dz_want))
            assert bad_dz_rel > 1e-2

    # 3. Winner-mask scaling + SPMD multi-chip path (world == 2) + shape guard.
    import tt_bio.train.mesh_dp as mdp
    b_global = 2
    z_2 = np.full((b_global, n_pad, n_pad), -30.0, dtype=np.float32)
    target_2 = np.full((b_global, n_tok), -1, dtype=np.int64)
    pad_2 = np.broadcast_to(pad, (b_global, n_tok)).copy()
    for bi in range(b_global):
        for k, r in enumerate(active_rows):
            target_2[bi, r] = k
            z_2[bi, r, k] = 0.0
            z_2[bi, r, (k + 1) % 5] = -5.47 - 0.5 * bi
    sharded_in = orig_from(torch.from_numpy(z_2[:1]))
    monkeypatch.setattr(mdp, "mesh_world", lambda d: 2)
    monkeypatch.setattr(mdp, "read_sharded", lambda t, d: torch.from_numpy(z_2))
    put_seen = []

    def fake_put_sharded(t, d, **kw):
        put_seen.append(t.clone())
        return orig_from(t[:1])

    monkeypatch.setattr(mdp, "put_sharded", fake_put_sharded)
    loss_2, _ = RH.device_insertion_loss(
        sharded_in, {"insertion_source_idx": target_2, "padding_mask": pad_2},
        win=np.array([1]), weight=0.5,
    )
    loss_2_ref, dz_2_ref = RO.insertion_loss(
        z_2[:, :n_tok, :n_tok].astype(np.float64),
        {"insertion_source_idx": target_2, "padding_mask": pad_2},
    )
    assert np.allclose(loss_2, loss_2_ref, atol=1e-12, rtol=1e-12)
    dz_uploaded = put_seen[-1].numpy()
    assert np.all(dz_uploaded[0] == 0.0), "loser rank must get zero insertion dz"
    assert np.allclose(dz_uploaded[1, :n_tok, :n_tok], dz_2_ref[1] * 0.5, atol=1e-6, rtol=1e-5)

    with pytest.raises(ValueError, match="batch leading dim"):
        RH.device_insertion_loss(
            sharded_in, {"insertion_source_idx": target, "padding_mask": pad},
        )


def test_device_heads_split2_mm_precision_and_dealloc(monkeypatch):
    """_split2_mm must beat bf16-unpacked fp32 matmul by >100x on the 3-matmul
    insertion chain and free every intermediate when untaped."""
    fake = _install_fake_ttnn(monkeypatch, preload=("tt_bio.rfd4.heads",))
    import tt_bio.rfd4.heads as heads_mod
    monkeypatch.setattr(heads_mod, "ttnn", fake)
    monkeypatch.setattr(heads_mod.ops, "taping", lambda: False)

    # Simulate Blackhole ttnn.matmul unpacking float32 inputs as bfloat16.
    def bh_matmul(a, b, transpose_a=False, transpose_b=False, dtype=None, compute_kernel_config=None):
        av = a._data.to(torch.bfloat16).float()
        bv = b._data.to(torch.bfloat16).float()
        if transpose_a:
            av = av.transpose(-2, -1)
        if transpose_b:
            bv = bv.transpose(-2, -1)
        dt = dtype or a.dtype
        return _FakeTensor(torch.matmul(av, bv).to(torch.bfloat16 if dt == fake.bfloat16 else torch.float32),
                           dtype=dt, device=a._device)

    monkeypatch.setattr(fake, "matmul", bh_matmul)

    rng = np.random.default_rng(123)
    s_np = rng.standard_normal((1, 16, 64)).astype(np.float32) * 0.5
    wq_np = rng.standard_normal((64, 64)).astype(np.float32) * 0.1
    wk_np = rng.standard_normal((64, 64)).astype(np.float32) * 0.1

    q_ref = s_np.astype(np.float64) @ wq_np.astype(np.float64)
    k_ref = s_np.astype(np.float64) @ wk_np.astype(np.float64)
    ins_ref = np.matmul(q_ref, np.swapaxes(k_ref, -2, -1)) * (64.0 ** -0.5)

    s_t = fake.from_torch(torch.from_numpy(s_np))
    wq_t = fake.from_torch(torch.from_numpy(wq_np))
    wk_t = fake.from_torch(torch.from_numpy(wk_np))

    # Broken arm: raw bh_matmul chain on float32 hits >1e-3 rel_l2 error.
    q_raw = bh_matmul(s_t, wq_t, dtype=fake.float32)
    k_raw = bh_matmul(s_t, wk_t, dtype=fake.float32)
    ins_raw = bh_matmul(q_raw, k_raw, transpose_b=True, dtype=fake.float32)._data.numpy() * (64.0 ** -0.5)
    raw_rel = float(np.linalg.norm(ins_raw - ins_ref) / np.linalg.norm(ins_ref))
    assert raw_rel > 1e-3, f"expected raw bf16-unpacked matmul chain to exceed 1e-3, got {raw_rel:.3e}"

    # Fixed arm: _split2_mm chain achieves <5e-5 rel_l2 and leaks zero intermediates.
    created = []
    orig_tc, orig_sub, orig_add, orig_mm = fake.typecast, fake.subtract, fake.add, fake.matmul

    def _track(fn):
        def _w(*a, **kw):
            out = fn(*a, **kw)
            created.append(out)
            return out
        return _w

    monkeypatch.setattr(fake, "typecast", _track(orig_tc))
    monkeypatch.setattr(fake, "subtract", _track(orig_sub))
    monkeypatch.setattr(fake, "add", _track(orig_add))
    monkeypatch.setattr(fake, "matmul", _track(orig_mm))

    q_s = heads_mod._split2_mm(s_t, wq_t, fake.float32, {})
    k_s = heads_mod._split2_mm(s_t, wk_t, fake.float32, {})
    ins_s = heads_mod._split2_mm(q_s, k_s, fake.float32, {}, transpose_b=True)
    ins_got = ins_s._data.numpy() * (64.0 ** -0.5)
    split_rel = float(np.linalg.norm(ins_got - ins_ref) / np.linalg.norm(ins_ref))
    assert split_rel < 5e-5, f"_split2_mm chain rel_l2={split_rel:.3e} >= 5e-5"

    live = [t for t in created if t.is_allocated()]
    assert set(live) == {q_s, k_s, ins_s}, f"leaked intermediates: {len(live)} live tensors"
    assert s_t.is_allocated() and wq_t.is_allocated() and wk_t.is_allocated()


def test_device_insertion_loss_active_row_fast_path_edge_cases(monkeypatch):
    """Active-row fast path in device_insertion_loss must match RO.insertion_loss bit-exactly
    across inactive batch items, masked-key targets, and all-masked keys."""
    fake = _install_fake_ttnn(monkeypatch, preload=("tt_bio.train.rfd4_heads",))
    import tt_bio.train.rfd4_heads as RH
    import tt_bio.train.rfd4_objective as RO
    import tt_bio.train.mesh_dp as mdp
    monkeypatch.setattr(RH, "ttnn", fake)
    monkeypatch.setattr(mdp, "mesh_world", lambda d: 1)

    rng = np.random.default_rng(99)
    b, n_tok, n_pad = 4, 10, 16
    z_np = rng.standard_normal((b, n_pad, n_pad)).astype(np.float32) * 2.0
    target = np.full((b, n_tok), -1, dtype=np.int64)
    pad = np.ones((b, n_tok), dtype=bool)
    # b=0: inactive (all -1)
    # b=1: normal active rows
    target[1, [6, 7]] = [1, 3]
    pad[1, 9] = False
    # b=2: active row whose target points to a masked key (pad=False or target>=0)
    target[2, [4, 5]] = [9, 5]
    pad[2, 9] = False
    # b=3: all keys masked (pad=False everywhere) with an active row
    pad[3, :] = False
    target[3, 2] = 0

    z_tt = fake.from_torch(torch.from_numpy(z_np))
    loss_got, dz_tt = RH.device_insertion_loss(
        z_tt, {"insertion_source_idx": target, "padding_mask": pad}, weight=0.75,
    )
    loss_ref, dz_ref = RO.insertion_loss(
        z_np[:, :n_tok, :n_tok].astype(np.float64),
        {"insertion_source_idx": target, "padding_mask": pad},
    )
    assert np.array_equal(loss_got, loss_ref), f"loss diff={np.max(np.abs(loss_got - loss_ref))}"
    dz_got = dz_tt._data.numpy()
    dz_want = np.zeros((b, n_pad, n_pad), dtype=np.float32)
    dz_want[:, :n_tok, :n_tok] = (dz_ref * 0.75).astype(np.float32)
    assert np.array_equal(dz_got, dz_want), f"dz diff={np.max(np.abs(dz_got - dz_want))}"


def test_exact_fp32_norm_adaln_on_card(monkeypatch):
    """_exact_fp32_row_mean + _rsqrt_exact_f32 + _norm_adaln_on_card match float64 and leak zero tensors."""
    import contextlib
    fake = _install_fake_ttnn(monkeypatch, preload=("tt_bio.train.rfd4_pair", "tt_bio.train.device_optim"))
    if "tt_bio.autograd" not in sys.modules:
        import tt_bio
        fake_ag = types.ModuleType("tt_bio.autograd")
        fake_ag.no_grad = contextlib.nullcontext
        monkeypatch.setitem(sys.modules, "tt_bio.autograd", fake_ag)
        monkeypatch.setattr(tt_bio, "autograd", fake_ag, raising=False)
    import tt_bio.train.rfd4_pair as RP
    monkeypatch.setattr(RP, "ttnn", fake)

    rng = np.random.default_rng(7)
    b, n, c = 2, 4, 384
    x_np = rng.standard_normal((b, n, n, c)).astype(np.float32)
    # Include 400x dynamic range across channels (matching trained RFD4 ln_out weights).
    lnw_np = (1.0 + 0.1 * rng.standard_normal((1, 1, 1, c))).astype(np.float32)
    lnw_np[..., 0] = 20.0
    gam_np = rng.standard_normal((b, 1, 1, c)).astype(np.float32)
    bet_np = rng.standard_normal((b, 1, 1, c)).astype(np.float32)
    e1, e2 = 1e-5, 1e-5

    x64 = x_np.astype(np.float64)
    r1 = 1.0 / np.sqrt(np.mean(x64 * x64, axis=-1, keepdims=True) + e1)
    a64 = x64 * r1 * lnw_np.astype(np.float64)
    r2 = 1.0 / np.sqrt(np.mean(a64 * a64, axis=-1, keepdims=True) + e2)
    z64 = a64 * r2 * gam_np.astype(np.float64) + bet_np.astype(np.float64)

    x_tt = fake.from_torch(torch.from_numpy(x_np))
    lnw_tt = fake.from_torch(torch.from_numpy(lnw_np))
    gam_tt = fake.from_torch(torch.from_numpy(gam_np))
    bet_tt = fake.from_torch(torch.from_numpy(bet_np))

    # Simulate inaccurate SFPU rsqrt seed (~2e-4 rel error) AND BH fp32 permute truncation
    # so any regression back to raw fp32 permute or unrefined rsqrt fails this gate.
    def approx_rsqrt(s, fast_and_approximate_mode=True):
        exact = torch.rsqrt(s._data.float())
        return _FakeTensor((exact * (1.0 + 2e-4)).to(torch.float32), dtype=fake.float32, device=s._device)

    orig_permute = fake.permute

    def bh_permute(t, dims):
        out = orig_permute(t, dims)
        if t.dtype == fake.float32:
            # Real BH float32 permute truncates mantissa bits (~5.9e-4 rel_l2).
            out._data = out._data.to(torch.bfloat16).to(torch.float32)
        return out

    monkeypatch.setattr(fake, "rsqrt", approx_rsqrt)
    monkeypatch.setattr(fake, "permute", bh_permute)

    z_tt = RP._norm_adaln_on_card(x_tt, lnw_tt, gam_tt, bet_tt, e1, e2, c, {}, cg=None)
    z_got = z_tt._data.numpy().astype(np.float64)
    rel = float(np.linalg.norm(z_got - z64) / np.linalg.norm(z64))
    assert rel < 1e-6, f"_norm_adaln_on_card rel_l2={rel:.3e} >= 1e-6"
    assert x_tt.is_allocated() and lnw_tt.is_allocated() and gam_tt.is_allocated() and bet_tt.is_allocated()





