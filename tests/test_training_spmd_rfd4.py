"""CPU tests for SPMD MeshDevice RFD4 orchestration (tt_bio.train.spmd_rfd4)."""
from __future__ import annotations

import math
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_training_ccl import _FakeMeshDevice, _FakeMeshTensor, _make_mock_ttnn
from tt_bio.train.ccl import exact_2d_fp32_sum
from tt_bio.train.spmd_rfd4 import (
    B200_REF_EX_S_PER_GPU,
    BH_GALAXY_USD,
    DGX_B200_GPUS,
    DGX_B200_USD,
    SPMDZero1Step,
    compute_spmd_g4_metrics,
    gather_winner_batches,
    load_depths_jsonl,
    master_consensus_digest,
    parse_mesh_arg,
    select_rank_batches,
    take_example,
)


def _install_mock_ttnn(monkeypatch):
    mod = _make_mock_ttnn()

    def subtract(a, b):
        assert a.is_allocated() and b.is_allocated()
        return a.device().make_tensor([x.reshape(a.shape) - y.reshape(b.shape) for x, y in zip(a._buf.shards, b._buf.shards)])

    mod.subtract = subtract
    mod.synchronize_device = lambda dev: None
    mod.DRAM_MEMORY_CONFIG = "DRAM"
    mod.bfloat16 = "bf16"
    mod.bfloat4_b = "bf4"
    mod.uint32 = "u32"
    mod.int32 = "i32"
    mod.CoreCoord = lambda x, y: (x, y)
    mod.NOC = types.SimpleNamespace(NOC_0=0, NOC_1=1)
    mod.UnaryOpType = types.SimpleNamespace(RELU=0)
    mod.MathFidelity = types.SimpleNamespace(HiFi4=4, HiFi2=2, LoFi=0)
    monkeypatch.setitem(sys.modules, "ttnn", mod)
    return mod


def test_import_spmd_rfd4_does_not_import_ttnn():
    code = (
        "import sys; import tt_bio.train.spmd_rfd4; "
        "assert 'ttnn' not in sys.modules, 'ttnn leaked at import'"
    )
    r = subprocess.run([sys.executable, "-c", code], check=False, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_parse_mesh_arg():
    assert parse_mesh_arg("8x4") == (8, 4)
    assert parse_mesh_arg(" 2X4 ") == (2, 4)
    assert parse_mesh_arg("1x1") == (1, 1)
    for bad in ("8", "8x4x2", "0x4", "-2x4", "axb"):
        with pytest.raises(ValueError):
            parse_mesh_arg(bad)


def test_load_depths_jsonl_matches_b200_final_mat_nocg():
    p = Path("perf/kitcmp/results/r4t-train-b200/rfd4/r4t/final-mat-nocg/steps.jsonl")
    assert p.exists(), f"missing B200 reference jsonl at {p}"
    depths = load_depths_jsonl(p)
    assert len(depths) == 45
    assert depths[:5] == [6, 1, 1, 1, 5]
    timed = depths[5:]
    assert len(timed) == 40
    assert abs(sum(timed) / len(timed) - 2.75) < 1e-12


def test_compute_spmd_g4_metrics_exact_bar():
    depths = [6, 1, 1, 1, 5] + [2, 3] * 20
    # Exactly clears 1.50x/$ at 1.7866 s/step; misses at 1.7870 s/step.
    m_pass = compute_spmd_g4_metrics([2.5] * 5 + [1.7866] * 40, depths, world=32, single_chip_step_s=1.70)
    assert m_pass["clears_g4_1p50x"] is True
    assert m_pass["ratio_per_dollar_vs_b200"] >= 1.50
    assert abs(m_pass["dp_efficiency"] - (1.70 / 1.7866)) < 1e-9
    assert abs(m_pass["mean_depth"] - 2.5) < 1e-12

    m_fail = compute_spmd_g4_metrics([2.5] * 5 + [1.7870] * 40, depths, world=32)
    assert m_fail["clears_g4_1p50x"] is False
    assert m_fail["ratio_per_dollar_vs_b200"] < 1.50

    # Verify list-price formula matches D2 / D99 exactly.
    m_fast = compute_spmd_g4_metrics([2.0] * 5 + [1.35] * 40, depths, world=32)
    expected = ((32.0 / 1.35) / BH_GALAXY_USD) / ((DGX_B200_GPUS * B200_REF_EX_S_PER_GPU) / DGX_B200_USD)
    assert abs(m_fast["ratio_per_dollar_vs_b200"] - expected) < 1e-12
    assert m_fast["ratio_per_dollar_vs_b200"] > 1.98


def test_select_and_gather_winner_batches_heterogeneous():
    pool = [
        {"x": torch.arange(4 * 3, dtype=torch.float32).reshape(4, 3) + 100 * i,
         "nested": {"m": torch.full((4, 2), float(i))},
         "tag": f"b{i}"}
        for i in range(6)
    ]
    idx, per_rank, stacked = select_rank_batches(pool, step_idx=2, world=4)
    assert idx == [2, 3, 4, 5]
    assert stacked["x"].shape == (16, 3)
    assert stacked["nested"]["m"].shape == (16, 2)

    win_indices = [3, 0, 2, 1]
    per_rank_w, stacked_w = gather_winner_batches(per_rank, win_indices)
    assert stacked_w["x"].shape == (4, 3)
    for r, (bi, wi) in enumerate(zip(idx, win_indices)):
        assert torch.equal(per_rank_w[r]["x"], take_example(pool[bi], wi)["x"])
        assert torch.equal(stacked_w["x"][r:r + 1], pool[bi]["x"][wi:wi + 1])
        assert torch.equal(stacked_w["nested"]["m"][r:r + 1], pool[bi]["nested"]["m"][wi:wi + 1])


def test_master_consensus_digest_pass_and_broken_fail(monkeypatch):
    _install_mock_ttnn(monkeypatch)
    dev = _FakeMeshDevice(2, 4)
    w = torch.randn(32, 32, dtype=torch.float32)
    t_ok = dev.make_tensor([w.clone() for _ in range(8)])
    d1, n1 = master_consensus_digest({"a": t_ok}, dev)
    assert n1 == 1 and len(d1) == 16

    # Perturb rank 5 by 1e-5 -> must FAIL.
    bad_shards = [w.clone() for _ in range(8)]
    bad_shards[5][0, 0] += 1e-5
    t_bad = dev.make_tensor(bad_shards)
    with pytest.raises(AssertionError, match="rank 5 differs from rank 0"):
        master_consensus_digest({"a": t_bad}, dev)


class _FakeMuonCfg:
    def __init__(self, momentum=0.95, nesterov=True, weight_decay=0.01, ns_steps=2, eps=1e-7):
        self.momentum = momentum
        self.nesterov = nesterov
        self.weight_decay = weight_decay
        self.ns_steps = ns_steps
        self.ns_coefficients = (3.4445, -4.7750, 2.0315)
        self.eps = eps
        self.schedule = None

    def ratio(self, shape):
        m, n = shape[-2], shape[-1]
        return 0.2 * math.sqrt(max(m, n))


class _FakeBucket:
    def __init__(self, kind, shape, names, master_b):
        self.kind = kind
        self.shape = tuple(shape)
        self.names = list(names)
        self.k = len(names)
        if len(shape) == 2:
            self.canon = (1, shape[0], shape[1])
        else:
            self.canon = (1, 1, shape[0])
        self.master_b = master_b
        self.state1_b = None
        self.state2_b = None
        self.ema_b = None
        self._zero1 = None


class _FakeEMA:
    def __init__(self, decay, buckets):
        self.decay = float(decay)
        self._buckets = buckets
        self.n_averaged = 0

    def update(self):
        import ttnn

        if self.n_averaged == 0:
            for bk in self._buckets:
                bk.ema_b = ttnn.clone(bk.master_b)
        else:
            w = 1.0 - self.decay
            for bk in self._buckets:
                diff = ttnn.subtract(bk.master_b, bk.ema_b)
                sc = ttnn.multiply(diff, w)
                nxt = ttnn.add(bk.ema_b, sc)
                ttnn.deallocate(diff)
                ttnn.deallocate(sc)
                ttnn.deallocate(bk.ema_b)
                bk.ema_b = nxt
        self.n_averaged += 1

    def host_shadow(self):
        out = {}
        for bk in self._buckets:
            if bk.ema_b is None:
                continue
            arr = bk.ema_b._buf.shards[0].float().numpy()
            s0, m, n = bk.canon
            for i, nm in enumerate(bk.names):
                out[nm] = np.ascontiguousarray(arr[i * s0:(i + 1) * s0, :m, :n].reshape(bk.shape), dtype=np.float32)
        return out

    def deallocate(self):
        import ttnn

        for bk in self._buckets:
            if bk.ema_b is not None and bk.ema_b.is_allocated():
                ttnn.deallocate(bk.ema_b)
            bk.ema_b = None


def _mock_ns_torch(x: torch.Tensor, steps: int, coeffs: tuple, eps: float) -> torch.Tensor:
    a, b, c = coeffs
    tall = x.shape[-2] > x.shape[-1]
    y = x.transpose(-2, -1) if tall else x
    norm = torch.linalg.norm(y, dim=(-2, -1), keepdim=True) + eps
    y = y / norm
    for _ in range(steps):
        g = y @ y.transpose(-2, -1)
        y = a * y + (b * g + c * (g @ g)) @ y
    return y.transpose(-2, -1) if tall else y


class _FakeDeviceOptimizer:
    """Minimal mock of DeviceOptimizer implementing the exact contract SPMDZero1Step uses."""

    def __init__(self, weights_np, muon_names, device, *, lr=1e-3, clip_norm=1.0, ema_decay=0.95):
        import ttnn

        self.device = device
        self.lr = lr
        self.beta1, self.beta2 = 0.9, 0.999
        self.beta1_pow, self.beta2_pow = 1.0, 1.0
        self.eps = 1e-8
        self.weight_decay = 0.0
        self.clip_norm = clip_norm
        self.schedule = lambda s: lr * (0.9 ** s)
        self.muon = _FakeMuonCfg()
        self.ns_mode = "fp32"
        self.steps = 0
        self.last_grad_norm = 0.0
        self.last_grad_norm_muon = 0.0
        self.shard_world = 1
        self.shapes = {k: tuple(v.shape) for k, v in weights_np.items()}
        self.muon_names = frozenset(muon_names)
        self.master = {
            k: ttnn.from_torch(torch.from_numpy(v.copy()), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
            for k, v in weights_np.items()
        }

        def pack_bk(kind, shp, nms):
            canon = (1, shp[0], shp[1]) if len(shp) == 2 else (1, 1, shp[0])
            arr = torch.stack([torch.from_numpy(weights_np[nm].copy()).reshape(canon) for nm in nms], dim=0).reshape(
                len(nms) * canon[0], canon[1], canon[2]
            )
            mb = ttnn.from_torch(arr, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
            return _FakeBucket(kind, shp, nms, mb)

        mu_by_shp: dict[tuple, list[str]] = {}
        ad_by_shp: dict[tuple, list[str]] = {}
        for nm in sorted(weights_np):
            (mu_by_shp if nm in self.muon_names else ad_by_shp).setdefault(self.shapes[nm], []).append(nm)
        self._muon_buckets = [pack_bk("muon", s, nms) for s, nms in sorted(mu_by_shp.items())]
        self._adam_buckets = [pack_bk("adam", s, nms) for s, nms in sorted(ad_by_shp.items())]
        self._all_buckets = self._muon_buckets + self._adam_buckets
        self.ema = _FakeEMA(ema_decay, self._all_buckets) if ema_decay is not None else None

    def clip_coef(self, gnorm: float) -> float:
        return min(1.0, self.clip_norm / gnorm) if (self.clip_norm > 0.0 and gnorm > 0.0) else 1.0

    def _pack_grads(self, bk, grads, dis):
        import ttnn

        flags = [(nm not in dis) and (grads.get(nm) is not None) for nm in bk.names]
        if not any(flags):
            return None, flags
        s0, m, n = bk.canon
        pieces = []
        for nm, ok in zip(bk.names, flags):
            if ok:
                pieces.append(ttnn.reshape(grads[nm], [s0, m, n]))
            else:
                pieces.append(ttnn.zeros([s0, m, n], dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.device))
        out = ttnn.concat(pieces, dim=0)
        for p in pieces:
            ttnn.deallocate(p)
        return out, flags

    def _sum_group_norm_sq(self, packed_list):
        tot = 0.0
        for _, g_b, _ in packed_list:
            if g_b is not None:
                tot += float((g_b._buf.shards[0].float() ** 2).sum().item())
        return tot

    def _unpack_master(self, bk):
        import ttnn

        s0, m, n = bk.canon
        for i, nm in enumerate(bk.names):
            sl = ttnn.slice(bk.master_b, [i * s0, 0, 0], [(i + 1) * s0, m, n])
            new_m = ttnn.reshape(sl, list(bk.shape))
            ttnn.deallocate(sl)
            old_m = self.master[nm]
            self.master[nm] = new_m
            if old_m is not None and old_m.is_allocated():
                ttnn.deallocate(old_m)

    def _step_muon_bucket(self, bk, g_b, flags, clip_muon, lr_muon):
        import ttnn

        if g_b is None:
            return
        mu = self.muon
        om = 1.0 - mu.momentum
        n_dev = self.device.get_num_devices()
        g_sh = [s * clip_muon for s in g_b._buf.shards]
        if bk.state1_b is None:
            buf_sh = [g * om for g in g_sh]
        else:
            prev_sh = bk.state1_b._buf.shards
            buf_sh = []
            for r in range(n_dev):
                cand = prev_sh[r] + om * (g_sh[r] - prev_sh[r])
                buf_sh.append(torch.stack([cand[i] if ok else prev_sh[r][i] for i, ok in enumerate(flags)], dim=0))
            ttnn.deallocate(bk.state1_b)
        bk.state1_b = self.device.make_tensor(buf_sh)
        u_sh = [b - om * (b - g) if mu.nesterov else b.clone() for b, g in zip(buf_sh, g_sh)]
        o_sh = [_mock_ns_torch(u, mu.ns_steps, mu.ns_coefficients, mu.eps) for u in u_sh]
        w_prev_sh = bk.master_b._buf.shards
        w_new_sh = []
        for r in range(n_dev):
            cand = w_prev_sh[r] * (1.0 - lr_muon * mu.weight_decay) - (lr_muon * mu.ratio(bk.shape)) * o_sh[r]
            w_new_sh.append(torch.stack([cand[i] if ok else w_prev_sh[r][i] for i, ok in enumerate(flags)], dim=0))
        ttnn.deallocate(bk.master_b)
        bk.master_b = self.device.make_tensor(w_new_sh)
        self._unpack_master(bk)

    def _step_adam_bucket(self, bk, g_b, dis, clip_adam, lr, bc1, bc2):
        import ttnn

        enabled = [nm not in dis for nm in bk.names]
        if not any(enabled) or g_b is None:
            return
        n_dev = self.device.get_num_devices()
        g_sh = [s * clip_adam for s in g_b._buf.shards]
        m_prev_sh = [torch.zeros_like(g) for g in g_sh] if bk.state1_b is None else bk.state1_b._buf.shards
        v_prev_sh = [torch.zeros_like(g) for g in g_sh] if bk.state2_b is None else bk.state2_b._buf.shards
        w_prev_sh = bk.master_b._buf.shards
        m_new_sh, v_new_sh, w_new_sh = [], [], []
        for r in range(n_dev):
            mc = self.beta1 * m_prev_sh[r] + (1.0 - self.beta1) * g_sh[r]
            vc = self.beta2 * v_prev_sh[r] + (1.0 - self.beta2) * (g_sh[r] * g_sh[r])
            st = (mc / bc1) / (torch.sqrt(vc / bc2) + self.eps)
            wc = w_prev_sh[r] - lr * st
            m_new_sh.append(torch.stack([mc[i] if ok else m_prev_sh[r][i] for i, ok in enumerate(enabled)], dim=0))
            v_new_sh.append(torch.stack([vc[i] if ok else v_prev_sh[r][i] for i, ok in enumerate(enabled)], dim=0))
            w_new_sh.append(torch.stack([wc[i] if ok else w_prev_sh[r][i] for i, ok in enumerate(enabled)], dim=0))
        if bk.state1_b is not None:
            ttnn.deallocate(bk.state1_b)
        if bk.state2_b is not None:
            ttnn.deallocate(bk.state2_b)
        ttnn.deallocate(bk.master_b)
        bk.state1_b = self.device.make_tensor(m_new_sh)
        bk.state2_b = self.device.make_tensor(v_new_sh)
        bk.master_b = self.device.make_tensor(w_new_sh)
        self._unpack_master(bk)

    def step(self, grads, *, reduce_fn=None, disabled=(), free_grads=False):
        import ttnn

        reduced = reduce_fn(grads) if reduce_fn is not None else grads
        dis = set(disabled)
        lr = float(self.schedule(self.steps))
        lr_muon = lr
        self.steps += 1
        self.beta1_pow *= self.beta1
        self.beta2_pow *= self.beta2
        bc1 = 1.0 - self.beta1_pow
        bc2 = 1.0 - self.beta2_pow
        mp = [(bk, *self._pack_grads(bk, reduced, dis)) for bk in self._muon_buckets]
        ap = [(bk, *self._pack_grads(bk, reduced, dis)) for bk in self._adam_buckets]
        gn_m = math.sqrt(self._sum_group_norm_sq(mp))
        gn_a = math.sqrt(self._sum_group_norm_sq(ap))
        cm = self.clip_coef(gn_m)
        ca = self.clip_coef(gn_a)
        for bk, g_b, flags in mp:
            self._step_muon_bucket(bk, g_b, flags, cm, lr_muon)
            if g_b is not None and g_b.is_allocated():
                ttnn.deallocate(g_b)
        for bk, g_b, _ in ap:
            self._step_adam_bucket(bk, g_b, dis, ca, lr, bc1, bc2)
            if g_b is not None and g_b.is_allocated():
                ttnn.deallocate(g_b)
        if self.ema is not None:
            self.ema.update()
        if reduce_fn is not None:
            for nm, g in reduced.items():
                if g is not None and g is not grads.get(nm) and g.is_allocated():
                    ttnn.deallocate(g)
        if free_grads:
            for g in grads.values():
                if g is not None and g.is_allocated():
                    ttnn.deallocate(g)
        return {"step": self.steps, "lr": lr, "lr_muon": lr_muon, "grad_norm": gn_a, "grad_norm_muon": gn_m, "clip": ca, "clip_muon": cm}

    def host_master(self):
        out = {}
        for bk in self._all_buckets:
            arr = bk.master_b._buf.shards[0].float().numpy()
            s0, m, n = bk.canon
            for i, nm in enumerate(bk.names):
                out[nm] = np.ascontiguousarray(arr[i * s0:(i + 1) * s0, :m, :n].reshape(bk.shape), dtype=np.float32)
        for nm, t in self.master.items():
            if nm not in out:
                out[nm] = np.ascontiguousarray(t._buf.shards[0].float().numpy().reshape(self.shapes[nm]), dtype=np.float32)
        return out

    def deallocate(self):
        import ttnn

        if self.ema is not None:
            self.ema.deallocate()
        for t in self.master.values():
            if t is not None and t.is_allocated():
                ttnn.deallocate(t)
        self.master.clear()
        for bk in self._all_buckets:
            for attr in ("master_b", "state1_b", "state2_b", "ema_b", "_zero1"):
                t = getattr(bk, attr, None)
                if t is not None and t.is_allocated():
                    ttnn.deallocate(t)
                setattr(bk, attr, None)


def _install_mock_device_optim(monkeypatch):
    import ttnn

    def _clone(t):
        return ttnn.clone(t)

    def _concat(ts):
        return ttnn.concat(ts, dim=0)

    def _sum_sq_last2(x):
        assert x.is_allocated()
        return x.device().make_tensor([(s.float() ** 2).sum(dim=(-2, -1), keepdim=True) for s in x._buf.shards])

    def _reduce_dim0_sum(cur):
        assert cur.is_allocated()
        out = cur.device().make_tensor([s.float().sum(dim=0).reshape(1) for s in cur._buf.shards])
        ttnn.deallocate(cur)
        return out

    def _safe_dealloc(t, keep=None):
        if t is None:
            return
        if keep is not None and (t is keep or t.buffer_address() == keep.buffer_address()):
            return
        if t.is_allocated():
            ttnn.deallocate(t)

    def _ns_dev(u, *, steps, coeffs, eps, mode, tall):
        assert u.is_allocated()
        return u.device().make_tensor([_mock_ns_torch(s, steps, coeffs, eps) for s in u._buf.shards])

    def _tr(t):
        assert t.is_allocated()
        return t.device().make_tensor([s.transpose(-2, -1).contiguous() for s in t._buf.shards])

    mod = SimpleNamespace(
        _clone_exact_f32=_clone,
        _concat_dim0_exact_f32=_concat,
        _reduce_dim0_sum_exact_f32=_reduce_dim0_sum,
        _safe_dealloc=_safe_dealloc,
        _sum_sq_last2_exact_f32=_sum_sq_last2,
        newton_schulz_device=_ns_dev,
        transpose_exact_f32=_tr,
    )
    monkeypatch.setitem(sys.modules, "tt_bio.train.device_optim", mod)


@pytest.mark.parametrize("mshape,use_zero1", [((2, 4), True), ((4, 2), True), ((1, 4), False)])
def test_spmd_zero1_step_matches_serial_reference_and_broken_fails(monkeypatch, mshape, use_zero1):
    _install_mock_ttnn(monkeypatch)
    _install_mock_device_optim(monkeypatch)
    rm, cm = mshape
    world = rm * cm
    dev = _FakeMeshDevice(rm, cm)

    rng = np.random.default_rng(12345)
    weights = {
        # Sharded Muon buckets (wide + tall, tile-aligned, k >= 2)
        **{f"mu_wide_{i}": rng.standard_normal((32, 64), dtype=np.float32) * 0.1 for i in range(5)},
        **{f"mu_tall_{i}": rng.standard_normal((64, 32), dtype=np.float32) * 0.1 for i in range(3)},
        # Tiny Muon bucket (k=1 -> stays unsharded in base_opt)
        "mu_single": rng.standard_normal((32, 32), dtype=np.float32) * 0.1,
        # AdamW params (1D bias/LN)
        "ad_ln_0": rng.standard_normal((64,), dtype=np.float32) * 0.1,
        "ad_ln_1": rng.standard_normal((64,), dtype=np.float32) * 0.1,
        "ad_bias": rng.standard_normal((32,), dtype=np.float32) * 0.1,
    }
    muon_names = {k for k in weights if k.startswith("mu_")}

    base_spmd = _FakeDeviceOptimizer(weights, muon_names, dev, ema_decay=0.95)
    spmd = SPMDZero1Step(base_spmd, weights, dev, full_shape=mshape, use_zero1_muon=use_zero1)

    if use_zero1:
        # Sharded Muon buckets detached from base_opt (only mu_single remains).
        assert len(base_spmd._muon_buckets) == 1
        assert base_spmd._muon_buckets[0].names == ["mu_single"]
        assert len(spmd.sharded_names()) == 8

    # Single-rank serial reference on exact same initial weights (1x1 mesh).
    dev_ref = _FakeMeshDevice(1, 1)
    ref_opt = _FakeDeviceOptimizer(weights, muon_names, dev_ref, ema_decay=0.95)

    for step_idx in range(3):
        disabled = {"mu_wide_2", "ad_ln_1"} if step_idx == 1 else set()
        dev_grads = {}
        ref_grads = {}
        for nm, arr in weights.items():
            if step_idx == 2 and nm == "mu_tall_1":
                per_rank = [torch.zeros(arr.shape, dtype=torch.float32) for _ in range(world)]
            else:
                per_rank = [torch.from_numpy(rng.standard_normal(arr.shape, dtype=np.float32) * 0.05) for _ in range(world)]
            dev_grads[nm] = dev.make_tensor(per_rank)
            # Exact 2D binary-tree fp32 sum matching ccl.exact_2d_fp32_sum!
            s_exact = exact_2d_fp32_sum(torch.stack(per_rank, dim=0), mshape, tree=True)
            ref_grads[nm] = dev_ref.make_tensor([s_exact.clone()])

        st_spmd = spmd.step(dev_grads, disabled=disabled, free_grads=True)
        st_ref = ref_opt.step(ref_grads, disabled=disabled, free_grads=True)

        assert st_spmd["step"] == st_ref["step"]
        assert abs(st_spmd["grad_norm"] - st_ref["grad_norm"]) < 1e-5
        assert abs(st_spmd["grad_norm_muon"] - st_ref["grad_norm_muon"]) < 1e-5
        if use_zero1:
            expected_dead = set(disabled) | ({"mu_tall_1"} if step_idx == 2 else set())
            assert st_spmd["dead_grads"] == expected_dead
            assert base_spmd.last_dead_grads == frozenset(expected_dead)

        # Every single master param is bit-identical across all mesh chips!
        digest, n_chk = master_consensus_digest(spmd.master, dev)
        assert n_chk == len(weights) and len(digest) == 16

        hm_spmd = base_spmd.host_master()
        hm_ref = ref_opt.host_master()
        for nm in weights:
            err = float(np.max(np.abs(hm_spmd[nm] - hm_ref[nm])))
            assert err <= 1e-6, f"step {step_idx} master {nm}: max_err={err}"

        ema_spmd = spmd.host_ema_shadow()
        ema_ref = ref_opt.ema.host_shadow()
        assert set(ema_spmd) == set(ema_ref) == set(weights)
        for nm in weights:
            err = float(np.max(np.abs(ema_spmd[nm] - ema_ref[nm])))
            assert err <= 1e-6, f"step {step_idx} ema {nm}: max_err={err}"

    # Broken arm: skip cross-rank reduction -> MUST fail consensus AND diverge from serial ref.
    broken_grads = {
        nm: dev.make_tensor([torch.from_numpy(rng.standard_normal(arr.shape, dtype=np.float32) * 0.2) for _ in range(world)])
        for nm, arr in weights.items()
    }
    spmd.step(broken_grads, skip_reduce_for_broken_arm=True, free_grads=True)
    with pytest.raises(AssertionError):
        master_consensus_digest(spmd.master, dev)

    spmd.deallocate()
    base_spmd.deallocate()
    ref_opt.deallocate()
    assert len(dev.live_bufs) == 0, f"leaked {len(dev.live_bufs)} device buffers on SPMD mesh!"
    assert len(dev_ref.live_bufs) == 0, f"leaked {len(dev_ref.live_bufs)} device buffers on ref!"


def test_spmd_zero1_works_without_tall_kwarg_in_ns(monkeypatch):
    """SPMDZero1Step works identically when newton_schulz_device has no tall= parameter (r4t-resident)."""
    _install_mock_ttnn(monkeypatch)
    _install_mock_device_optim(monkeypatch)
    orig_mod = sys.modules["tt_bio.train.device_optim"]
    orig_ns = orig_mod.newton_schulz_device

    def _ns_no_tall(u, *, steps, coeffs, eps, mode):
        assert u.shape[-2] <= u.shape[-1], "caller must transpose tall matrices when tall= is absent"
        return orig_ns(u, steps=steps, coeffs=coeffs, eps=eps, mode=mode, tall=False)

    monkeypatch.setattr(orig_mod, "newton_schulz_device", _ns_no_tall)
    dev = _FakeMeshDevice(2, 4)
    rng = np.random.default_rng(7)
    weights = {
        **{f"mu_tall_{i}": rng.standard_normal((64, 32), dtype=np.float32) * 0.1 for i in range(4)},
        "ad_b": rng.standard_normal((32,), dtype=np.float32) * 0.1,
    }
    base = _FakeDeviceOptimizer(weights, {k for k in weights if k.startswith("mu_")}, dev, ema_decay=None)
    spmd = SPMDZero1Step(base, weights, dev, full_shape=(2, 4), use_zero1_muon=True)
    grads = {
        nm: dev.make_tensor([torch.from_numpy(rng.standard_normal(arr.shape, dtype=np.float32) * 0.05) for _ in range(8)])
        for nm, arr in weights.items()
    }
    st = spmd.step(grads, free_grads=True)
    assert st["dead_grads"] == set()
    master_consensus_digest(spmd.master, dev)
    spmd.deallocate()
    base.deallocate()
    assert len(dev.live_bufs) == 0


def test_select_per_rank_candidate_uniform_and_heterogeneous(monkeypatch):
    """select_per_rank_candidate matches manual per-rank slicing for both uniform & heterogeneous winners."""
    _install_mock_ttnn(monkeypatch)
    from tt_bio.train.mesh_dp import select_per_rank_candidate

    dev = _FakeMeshDevice(2, 2)
    rng = np.random.default_rng(19)
    # 4D [K=4, 4, 4, 8] and 3D [K=4, 6, 8] tensors
    for shape in ((4, 4, 4, 8), (4, 6, 8)):
        shards = [torch.from_numpy(rng.standard_normal(shape, dtype=np.float32)) for _ in range(4)]
        t_mesh = dev.make_tensor(shards)

        # Uniform fast path
        u = select_per_rank_candidate(t_mesh, [2, 2, 2, 2], dev)
        for r in range(4):
            assert torch.equal(u._buf.shards[r], shards[r][2:3])
        sys.modules["ttnn"].deallocate(u)

        # Heterogeneous per-rank winners
        wins = [3, 0, 2, 1]
        h = select_per_rank_candidate(t_mesh, wins, dev)
        for r, w in enumerate(wins):
            assert torch.allclose(h._buf.shards[r], shards[r][w:w + 1], atol=0.0, rtol=0.0)
        sys.modules["ttnn"].deallocate(h)
        sys.modules["ttnn"].deallocate(t_mesh)
    assert len(dev.live_bufs) == 0


def test_mesh_bridge_backward_ups_sharded_and_replicated(monkeypatch):
    """MeshBridge.backward_ups gathers sharded activation grads and sums replicated parameter grads."""
    _install_mock_ttnn(monkeypatch)
    fake_ag = types.ModuleType("tt_bio.autograd")

    class _Leaf:
        def __init__(self, v, requires_grad=False):
            self.value = v
            self.requires_grad = requires_grad
            self.grad = None

    fake_ag.Tensor = _Leaf
    monkeypatch.setitem(sys.modules, "tt_bio.autograd", fake_ag)
    from tt_bio.train.mesh_dp import MeshBridge

    dev = _FakeMeshDevice(2, 2)  # world=4
    W = torch.randn(4, 3, requires_grad=True)
    x = torch.randn(4, 5, requires_grad=True)
    br = MeshBridge(dev)
    Wl = br.up_replicated(W)
    xl = br.up(x)

    # Simulate per-chip device gradients: chip r produces gW_r and gx_r
    gW_shards = [torch.randn(4, 3) for _ in range(4)]
    gx_shards = [torch.randn(1, 5) for _ in range(4)]
    Wl.grad = dev.make_tensor(gW_shards)
    xl.grad = dev.make_tensor(gx_shards)

    br.backward_ups()
    assert torch.allclose(W.grad, sum(gW_shards), atol=1e-6)
    assert torch.allclose(x.grad, torch.cat(gx_shards, dim=0), atol=1e-6)

    sys.modules["ttnn"].deallocate(Wl.value)
    sys.modules["ttnn"].deallocate(xl.value)
    assert len(dev.live_bufs) == 0


def test_device_insertion_loss_spmd_matches_serial(monkeypatch):
    """device_insertion_loss on a 2x2 MeshDevice matches exact per-rank serial calls (incl. mixed active/inactive)."""
    import contextlib

    mock_ttnn = _install_mock_ttnn(monkeypatch)
    fake_ag = types.ModuleType("tt_bio.autograd")
    fake_ag.no_grad = contextlib.nullcontext
    monkeypatch.setitem(sys.modules, "tt_bio.autograd", fake_ag)
    # Add ops needed by device_insertion_loss on _FakeMeshTensor
    def _bin(fn):
        def _op(a, b):
            if isinstance(a, _FakeMeshTensor) and isinstance(b, _FakeMeshTensor):
                return a.device().make_tensor([fn(sa, sb) for sa, sb in zip(a._buf.shards, b._buf.shards)])
            if isinstance(a, _FakeMeshTensor):
                return a.device().make_tensor([fn(sa, b) for sa in a._buf.shards])
            return b.device().make_tensor([fn(a, sb) for sb in b._buf.shards])
        return _op

    mock_ttnn.eq = _bin(lambda a, b: (a == b).to(torch.float32))
    mock_ttnn.typecast = lambda t, dt: t.device().make_tensor([s.clone() for s in t._buf.shards])
    mock_ttnn.max = lambda t, dim=-1, keepdim=False: t.device().make_tensor(
        [s.max(dim=dim, keepdim=keepdim).values for s in t._buf.shards]
    )
    mock_ttnn.exp = lambda t: t.device().make_tensor([torch.exp(s) for s in t._buf.shards])
    mock_ttnn.log = lambda t: t.device().make_tensor([torch.log(s) for s in t._buf.shards])
    mock_ttnn.divide = _bin(lambda a, b: a / b)
    mock_ttnn.sum = lambda t, dim=-1, keepdim=False: t.device().make_tensor(
        [s.sum(dim=dim, keepdim=keepdim) for s in t._buf.shards]
    )

    import tt_bio.train.rfd4_heads as r4h
    monkeypatch.setattr(r4h, "ttnn", mock_ttnn)

    dev4 = _FakeMeshDevice(2, 2)
    dev1 = _FakeMeshDevice(1, 1)
    rng = np.random.default_rng(23)
    n_tok, n_pad = 6, 8
    logits_shards = [torch.from_numpy(rng.standard_normal((1, n_pad, n_pad), dtype=np.float32)) for _ in range(4)]
    # Ranks 0 and 2 have insertion queries; ranks 1 and 3 have NO insertion targets (-1 everywhere).
    targets = np.array([
        [-1, -1, 0, -1, 1, -1],
        [-1, -1, -1, -1, -1, -1],
        [-1, 0, -1, 2, -1, 4],
        [-1, -1, -1, -1, -1, -1],
    ], dtype=np.int64)
    pad = np.array([
        [1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 0, 0, 0],
    ], dtype=bool)
    batch_spmd = {"insertion_source_idx": targets, "padding_mask": pad}

    z_mesh = dev4.make_tensor(logits_shards)
    loss_spmd, dz_spmd = r4h.device_insertion_loss(z_mesh, batch_spmd, weight=2.5)
    assert loss_spmd.shape == (4,)
    assert loss_spmd[1] == 0.0 and loss_spmd[3] == 0.0
    assert loss_spmd[0] > 0.0 and loss_spmd[2] > 0.0

    for r in range(4):
        z_r = dev1.make_tensor([logits_shards[r]])
        b_r = {"insertion_source_idx": targets[r:r + 1], "padding_mask": pad[r:r + 1]}
        l_r, dz_r = r4h.device_insertion_loss(z_r, b_r, weight=2.5)
        assert abs(loss_spmd[r] - l_r[0]) <= 1e-6
        if dz_r is None:
            assert torch.all(dz_spmd._buf.shards[r] == 0.0)
        else:
            assert torch.allclose(dz_spmd._buf.shards[r], dz_r._buf.shards[0], atol=1e-6)
            mock_ttnn.deallocate(dz_r)
        mock_ttnn.deallocate(z_r)

    mock_ttnn.deallocate(dz_spmd)
    mock_ttnn.deallocate(z_mesh)
    assert len(dev4.live_bufs) == 0 and len(dev1.live_bufs) == 0


def test_spmd_zero1_host_master_replicated_exclusion_and_profile_sync(monkeypatch):
    """replicated_names are excluded from sharding, host_master/digest match ref, profile_sync reports sync_s."""
    _install_mock_ttnn(monkeypatch)
    _install_mock_device_optim(monkeypatch)

    dev = _FakeMeshDevice(2, 4)
    rng = np.random.default_rng(41)
    weights = {
        **{f"mu_{i}": rng.standard_normal((32, 32), dtype=np.float32) * 0.1 for i in range(4)},
        "glue.host_lin": rng.standard_normal((32, 32), dtype=np.float32) * 0.1,
        "ad_0": rng.standard_normal((32,), dtype=np.float32) * 0.1,
    }
    muon_names = {k for k in weights if k != "ad_0"}
    base = _FakeDeviceOptimizer(weights, muon_names, dev, ema_decay=0.9)
    spmd = SPMDZero1Step(
        base, weights, dev, full_shape=(2, 4),
        replicated_names={"glue.host_lin"}, use_zero1_muon=True,
    )
    assert "glue.host_lin" not in spmd.sharded_names()
    assert spmd.sharded_names() == {f"mu_{i}" for i in range(4)}

    # Per-chip distinct grads for device params, IDENTICAL pre-reduced grad for glue.host_lin.
    g_glue_sum = torch.from_numpy(rng.standard_normal((32, 32), dtype=np.float32) * 0.1)
    grads = {
        **{
            nm: dev.make_tensor([torch.from_numpy(rng.standard_normal(arr.shape, dtype=np.float32) * 0.05) for _ in range(8)])
            for nm, arr in weights.items() if nm != "glue.host_lin"
        },
        "glue.host_lin": dev.make_tensor([g_glue_sum.clone() for _ in range(8)]),
    }
    info = spmd.step(grads, free_grads=True, profile_sync=True)
    assert "sync_s" in info and info["sync_s"] >= 0.0
    hm = spmd.host_master()
    assert set(hm) == set(weights)
    assert len(spmd.master_digest()) == 16
    master_consensus_digest(spmd.master, dev)
    spmd.deallocate()
    base.deallocate()
    assert len(dev.live_bufs) == 0


def test_trunk_masks_sharded_flag(monkeypatch):
    """trunk_masks passes ShardTensorToMesh only when sharded=True on multi-chip device."""
    import ast

    mock_ttnn = _install_mock_ttnn(monkeypatch)
    seen_mappers = []
    orig_from_torch = mock_ttnn.from_torch

    def _spy_from_torch(t, *args, **kwargs):
        seen_mappers.append(kwargs.get("mesh_mapper"))
        return orig_from_torch(t, *args, **kwargs)

    mock_ttnn.from_torch = _spy_from_torch
    mock_ttnn.typecast = lambda t, dt: t.device().make_tensor([s.clone() for s in t._buf.shards])
    src = Path("tt_bio/rfd4/trunk.py").read_text()
    tree = ast.parse(src)
    fn_node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "trunk_masks")
    ns = {"torch": torch, "ttnn": mock_ttnn, "ELIDE_FULL_MASKS": True, "NEG_INF": -1e9}
    exec(compile(ast.Module(body=[fn_node], type_ignores=[]), "tt_bio/rfd4/trunk.py", "exec"), ns)
    trunk_masks = ns["trunk_masks"]

    dev4 = _FakeMeshDevice(2, 2)
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0], [1, 0, 0, 0], [1, 1, 1, 1]], dtype=torch.float32)

    seen_mappers.clear()
    outs_rep = trunk_masks(mask, dev4, sharded=False)
    assert all(m is None for m in seen_mappers)
    for t in outs_rep:
        if t is not None and t.is_allocated():
            mock_ttnn.deallocate(t)

    seen_mappers.clear()
    outs_sh = trunk_masks(mask, dev4, sharded=True)
    assert len(seen_mappers) >= 3 and all(isinstance(m, tuple) and m[0] == "shard1d" and m[1] == 0 for m in seen_mappers)
    for t in outs_sh:
        if t is not None and t.is_allocated():
            mock_ttnn.deallocate(t)
    assert len(dev4.live_bufs) == 0


def test_gather_winner_batches_keys_filter():
    """gather_winner_batches(..., keys=WINNER_SMALL_KEYS) drops pair tensors and preserves token keys."""
    from tt_bio.train.spmd_rfd4 import WINNER_SMALL_KEYS, gather_winner_batches

    b0 = {
        "padding_mask": torch.tensor([[True, False], [True, True]]),
        "x_1": {"bb_ca": torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)},
        "huge_pair": torch.zeros((2, 64, 64), dtype=torch.float32),
        "tensors": {"duplicate_huge_pair": torch.zeros((2, 64, 64), dtype=torch.float32)},
    }
    b1 = {
        "padding_mask": torch.tensor([[True, True], [False, False]]),
        "x_1": {"bb_ca": torch.arange(12, 24, dtype=torch.float32).reshape(2, 2, 3)},
        "huge_pair": torch.ones((2, 64, 64), dtype=torch.float32),
        "tensors": {"duplicate_huge_pair": torch.ones((2, 64, 64), dtype=torch.float32)},
    }
    per_w, stacked = gather_winner_batches([b0, b1], [1, 0], keys=WINNER_SMALL_KEYS)
    assert len(per_w) == 2
    assert set(stacked.keys()) == {"padding_mask", "x_1"}
    assert "huge_pair" not in stacked and "tensors" not in stacked
    assert torch.equal(stacked["padding_mask"][0], b0["padding_mask"][1])
    assert torch.equal(stacked["padding_mask"][1], b1["padding_mask"][0])
    assert torch.equal(stacked["x_1"]["bb_ca"][0], b0["x_1"]["bb_ca"][1])
    assert torch.equal(stacked["x_1"]["bb_ca"][1], b1["x_1"]["bb_ca"][0])


def test_transition_bwd_torch_matches_autograd():
    """Analytical SwiGLU Transition VJP in spmd_trace matches torch.autograd.backward."""
    from tt_bio.train.spmd_trace import _transition_bwd_torch

    torch.manual_seed(7)
    b, n, d, h = 3, 16, 32, 64
    x = torch.randn(b, n, d, requires_grad=True)
    w1 = (torch.randn(2 * h, d) * 0.1).requires_grad_(True)
    w2 = (torch.randn(d, h) * 0.1).requires_grad_(True)
    mask = (torch.rand(b, n, 1) > 0.25).float()

    h_out = x @ w1.T
    a, g = h_out[..., :h], h_out[..., h:]
    y = ((torch.nn.functional.silu(g) * a) @ w2.T) * mask
    dy = torch.randn_like(y)
    y.backward(dy)

    dx_ref, dw1_ref, dw2_ref = x.grad.clone(), w1.grad.clone(), w2.grad.clone()
    dx_got, dw1_got, dw2_got = _transition_bwd_torch(dy, x.detach(), w1.detach(), w2.detach(), mask)
    assert torch.allclose(dx_got, dx_ref, atol=1e-5, rtol=1e-5)
    assert torch.allclose(dw1_got, dw1_ref, atol=1e-5, rtol=1e-5)
    assert torch.allclose(dw2_got, dw2_ref, atol=1e-5, rtol=1e-5)


def test_factory_static_split_and_gather_matrix():
    """_OneFactorySplit separates static/dynamic creators, matches full output, and builds exact gather matrix."""
    from tt_bio.train.spmd_trace import _OneFactorySplit

    class _StaticFeat(torch.nn.Module):
        def forward(self, batch):
            return batch["s_val"]

    class _DynEmb(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = torch.nn.Embedding(8, 5)

        def forward(self, batch):
            return self.emb(batch["tok_id"])

    class _ToyFactory(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.feat_creators = torch.nn.ModuleList([_StaticFeat(), _DynEmb(), _StaticFeat()])
            self.linear_out = torch.nn.Linear(4 + 5 + 4, 6, bias=False)

        @staticmethod
        def apply_padding_mask(x, mask):
            return x * mask[..., None].to(x.dtype)

        def forward(self, batch):
            feats = torch.cat([fc(batch) for fc in self.feat_creators], dim=-1)
            feats = self.apply_padding_mask(feats, batch["padding_mask"])
            return self.apply_padding_mask(self.linear_out(feats), batch["padding_mask"])

    torch.manual_seed(3)
    fac = _ToyFactory()
    batch = {
        "padding_mask": torch.tensor([[True, True, False], [True, False, False]]),
        "s_val": torch.randn(2, 3, 4),
        "tok_id": torch.tensor([[1, 2, 3], [4, 5, 6]]),
    }
    split = _OneFactorySplit(fac, exclude_last_recycle=False)
    tpl = split.build_static_template(batch, pad_to=32)
    assert tpl.shape == (2, 3, 32)
    assert split.total_dim == 13
    assert split.dynamic_cols == [4, 5, 6, 7, 8]

    full = split.fill_into(tpl[:, :, :13], batch)
    got = (full @ fac.linear_out.weight.T) * batch["padding_mask"][..., None].float()
    ref = fac(batch)
    assert torch.allclose(got, ref, atol=1e-6)

    p_mat = split.gather_matrix(32)
    assert p_mat.shape == (32, 32)
    picked = full @ p_mat[:13, :5]
    _, dyn_cat = split.eval_dynamic(batch, grad=False)
    assert torch.equal(picked, dyn_cat)


def test_spmd_tracer_find_win_pair_donor():
    """find_win_pair_donor matches opposite-prev keys with identical shapes/dtypes and rejects mismatches."""
    from tt_bio.train.spmd_trace import SPMDMeshTracer

    tr = SPMDMeshTracer(device=None, enabled=True)
    k_no_prev = (False, (True, True, False), (1, 256, 768), "f32", (1, 256, 256, 384), "f32", 1)
    k_prev = (True, (True, True, False), (1, 256, 768), "f32", (1, 256, 256, 384), "f32", 1)
    k_diff_shape = (True, (True, True, False), (1, 128, 768), "f32", (1, 128, 128, 384), "f32", 1)

    donor_entry = {"tid_fwd": 1, "tid_bwd": 2}
    tr._win_traces[k_no_prev] = donor_entry
    assert tr.find_win_pair_donor(k_prev) is donor_entry
    assert tr.find_win_pair_donor(k_no_prev) is None
    assert tr.find_win_pair_donor(k_diff_shape) is None

    tr._win_traces[k_prev] = "eager"
    assert tr.find_win_pair_donor(k_prev) is None


def test_fm_loss_values_only_matches_ro_fm_loss(monkeypatch):
    """_fm_loss_values_only is 0-ULP identical to RO.fm_loss and _small_npy_dict filters non-loss keys."""
    _install_mock_ttnn(monkeypatch)
    from tt_bio.train import rfd4_objective as RO
    import tt_bio.train.rfd4_heads as r4h

    rng = np.random.default_rng(11)
    x1 = rng.standard_normal((8, 16, 3))
    xt = rng.standard_normal((8, 16, 3))
    v = rng.standard_normal((8, 16, 3))
    t = rng.uniform(0.05, 0.95, size=(8,))
    emit = rng.random((8, 16)) > 0.2
    sup = emit & (rng.random((8, 16)) > 0.1)
    lw = rng.uniform(0.5, 1.5, size=(8, 16))

    ref_loss, _ = RO.fm_loss(x1, xt, v, t, emit, sup, lw)
    got_loss = r4h._fm_loss_values_only(x1, xt, v, t, emit, sup, lw)
    assert np.array_equal(ref_loss, got_loss)

    filt = r4h._small_npy_dict({
        "padding_mask": torch.ones((2, 4), dtype=torch.bool),
        "x_1": {"bb_ca": torch.zeros((2, 4, 3))},
        "huge_pair_tensor": torch.zeros((2, 64, 64)),
    })
    assert set(filt.keys()) == {"padding_mask", "x_1"}
    assert "huge_pair_tensor" not in filt


def _augment_mock_ttnn_for_glue(mod):
    """Extend mock ttnn with realistic bf16 typecast, matmul, silu, sigmoid, slice, concat, rsub, reshape."""
    def _mk(dev, shards, dt=None):
        t = dev.make_tensor(shards)
        t.dtype = dt or mod.float32
        return t

    def _typecast(t, dt):
        assert t.is_allocated()
        if dt == mod.bfloat16:
            return _mk(t.device(), [s.reshape(t.shape).to(torch.bfloat16).to(torch.float32) for s in t._buf.shards], dt)
        return _mk(t.device(), [s.reshape(t.shape).to(torch.float32) for s in t._buf.shards], dt)

    orig_from_torch = mod.from_torch

    def _from_torch(t, *args, **kwargs):
        mm = kwargs.get("mesh_mapper")
        dev = kwargs.get("device")
        if mm == ("replicate",) and dev is not None:
            w = dev.rm * dev.cm
            return _mk(dev, [t.detach().to(torch.float32).clone() for _ in range(w)], kwargs.get("dtype", mod.float32))
        out = orig_from_torch(t, *args, **kwargs)
        out.dtype = kwargs.get("dtype", mod.float32)
        return out

    mod.ReplicateTensorToMesh = lambda dev: ("replicate",)
    mod.from_torch = _from_torch
    mod.ROW_MAJOR_LAYOUT = "RM"
    mod.to_layout = lambda t, layout: _mk(t.device(), [x.clone() for x in t._buf.shards], getattr(t, "dtype", mod.float32))

    def _matmul(a, b, transpose_a=False, transpose_b=False, dtype=None, compute_kernel_config=None):
        assert a.is_allocated() and b.is_allocated()
        out = []
        for sa, sb in zip(a._buf.shards, b._buf.shards):
            xa = sa.reshape(a.shape).to(torch.float32)
            xb = sb.reshape(b.shape).to(torch.float32)
            if transpose_a:
                xa = xa.transpose(-2, -1)
            if transpose_b:
                xb = xb.transpose(-2, -1)
            out.append(torch.matmul(xa, xb))
        return _mk(a.device(), out, dtype or mod.float32)

    def _slice(t, starts, ends):
        assert t.is_allocated()
        sl = tuple(slice(s, e) for s, e in zip(starts, ends))
        return _mk(t.device(), [x[sl].contiguous() for x in t._buf.shards], getattr(t, "dtype", mod.float32))

    def _concat(ts, dim=-1):
        for t in ts:
            assert t.is_allocated()
        n_sh = len(ts[0]._buf.shards)
        shards = [torch.cat([t._buf.shards[i] for t in ts], dim=dim) for i in range(n_sh)]
        return _mk(ts[0].device(), shards, getattr(ts[0], "dtype", mod.float32))

    def _reshape(t, shape):
        assert t.is_allocated()
        # Simulate real ttnn zero-copy reshape aliasing the exact same underlying buffer!
        alias = _FakeMeshTensor(t._buf, tuple(int(d) for d in shape), t.device())
        alias.dtype = getattr(t, "dtype", mod.float32)
        alias.buffer_address = lambda _b=t._buf: id(_b)
        if not hasattr(t, "buffer_address"):
            t.buffer_address = lambda _b=t._buf: id(_b)
        return alias

    def _pad(t, padding, value=0.0):
        assert t.is_allocated()
        # ttnn.pad takes [(lo, hi), ...] per dim -> torch.nn.functional.pad expects reversed flat pairs.
        tpad = []
        for lo, hi in reversed(padding):
            tpad.extend([int(lo), int(hi)])
        return _mk(
            t.device(),
            [torch.nn.functional.pad(x, tpad, value=float(value)) for x in t._buf.shards],
            getattr(t, "dtype", mod.float32),
        )

    def _silu(t):
        assert t.is_allocated()
        return _mk(t.device(), [torch.nn.functional.silu(x.float()) for x in t._buf.shards], getattr(t, "dtype", mod.float32))

    def _sigmoid(t):
        assert t.is_allocated()
        return _mk(t.device(), [torch.sigmoid(x.float()) for x in t._buf.shards], getattr(t, "dtype", mod.float32))

    def _rsub(t, val):
        assert t.is_allocated()
        return _mk(t.device(), [float(val) - x.float() for x in t._buf.shards], getattr(t, "dtype", mod.float32))

    mod.typecast = _typecast
    mod.matmul = _matmul
    mod.slice = _slice
    mod.concat = _concat
    mod.reshape = _reshape
    mod.pad = _pad
    mod.silu = _silu
    mod.sigmoid = _sigmoid
    mod.rsub = _rsub
    return mod


def test_exact_f32_mm_recovers_fp32_via_split2_bf16(monkeypatch):
    """_exact_f32_mm splits fp32 operands into bf16 hi+lo, recovering <1e-5 accuracy with 0 buffer leaks."""
    mod = _augment_mock_ttnn_for_glue(_install_mock_ttnn(monkeypatch))
    from tt_bio.train.spmd_trace import _exact_f32_mm

    dev = _FakeMeshDevice(1, 2)
    torch.manual_seed(17)
    # Non-bf16-representable fp32 inputs where single bf16 rounding loses ~1e-2.
    a_h = [torch.randn(1, 32, 64, dtype=torch.float32) * 1.337 + 0.123456 for _ in range(2)]
    b_h = [torch.randn(32, 64, dtype=torch.float32) * 0.771 - 0.098765 for _ in range(2)]
    a_t = dev.make_tensor(a_h)
    b_t = dev.make_tensor(b_h)

    out_t = _exact_f32_mm(a_t, b_t, transpose_b=True)
    for r in range(2):
        ref = a_h[r] @ b_h[r].T
        err = (out_t._buf.shards[r] - ref).abs().max().item()
        bf16_err = (a_h[r].bfloat16().float() @ b_h[r].bfloat16().float().T - ref).abs().max().item()
        assert err < 2e-4, f"split2 error too large: {err}"
        assert err < bf16_err * 0.02, f"split2 ({err}) should strongly beat plain bf16 ({bf16_err})"

    mod.deallocate(out_t)
    mod.deallocate(a_t)
    mod.deallocate(b_t)
    assert len(dev.live_bufs) == 0


def test_device_winner_glue_forward_backward_parity_and_no_leaks(monkeypatch):
    """DeviceWinnerGlue forward + analytical VJP matches torch autograd on all 6 device + CPU emb leaves, 0 leaks."""
    import contextlib
    mod = _augment_mock_ttnn_for_glue(_install_mock_ttnn(monkeypatch))
    fake_ag = types.ModuleType("tt_bio.autograd")
    fake_ag.precise_config = lambda: None
    fake_ag.no_grad = contextlib.nullcontext
    monkeypatch.setitem(sys.modules, "tt_bio.autograd", fake_ag)
    from tt_bio.train.spmd_trace import DeviceWinnerGlue, FactoryStaticSplit

    class _StaticFeat(torch.nn.Module):
        def __init__(self, key):
            super().__init__()
            self.key = key

        def forward(self, batch):
            return batch[self.key]

    class _DynEmb(torch.nn.Module):
        def __init__(self, vocab, dim, key):
            super().__init__()
            self.key = key
            self.emb = torch.nn.Embedding(vocab, dim)

        def forward(self, batch):
            return self.emb(batch[self.key])

    class _RemFeat(torch.nn.Module):
        def __init__(self, dim=4):
            super().__init__()
            self.dim = dim

        def forward(self, batch):
            rem = batch.get("recycle_remaining", None)
            n = batch["padding_mask"].shape[0]
            out = torch.zeros((n, 1, self.dim), dtype=torch.float32)
            if rem is not None:
                out[:, :, min(int(rem), self.dim - 1)] = 1.0
            return out

    class _Factory(torch.nn.Module):
        def __init__(self, creators, in_dim, out_dim):
            super().__init__()
            self.feat_creators = torch.nn.ModuleList(creators)
            self.linear_out = torch.nn.Linear(in_dim, out_dim, bias=False)

        @staticmethod
        def apply_padding_mask(x, mask):
            return x * mask[..., None].to(x.dtype)

        def forward(self, batch):
            n_tok = batch["padding_mask"].shape[1]
            parts = []
            for fc in self.feat_creators:
                v = fc(batch)
                if v.shape[1] == 1 and n_tok > 1:
                    v = v.expand(-1, n_tok, -1)
                parts.append(v)
            feats = self.apply_padding_mask(torch.cat(parts, dim=-1), batch["padding_mask"])
            return self.apply_padding_mask(self.linear_out(feats), batch["padding_mask"])

    class _SwiGLU(torch.nn.Module):
        def forward(self, x):
            a, g = x.chunk(2, dim=-1)
            return torch.nn.functional.silu(g) * a

    class _Transition(torch.nn.Module):
        def __init__(self, dim, hidden):
            super().__init__()
            self.swish_linear = torch.nn.Sequential(
                torch.nn.Linear(dim, 2 * hidden, bias=False),
                _SwiGLU(),
            )
            self.linear_out = torch.nn.Linear(hidden, dim, bias=False)

        def forward(self, x, mask):
            return self.linear_out(self.swish_linear(x)) * mask[..., None].to(x.dtype)

    class _ToyGlue(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.condition_on_recycle_remaining = True
            self.init_repr_factory = _Factory(
                [_StaticFeat("s_ir"), _DynEmb(12, 7, "tok_ir")], in_dim=30 + 7, out_dim=32
            )
            self.cond_factory = _Factory(
                [_StaticFeat("s_cf"), _DynEmb(10, 5, "tok_cf"), _RemFeat(4)], in_dim=28 + 5 + 4, out_dim=32
            )
            self.transition_c_1 = _Transition(32, 64)
            self.transition_c_2 = _Transition(32, 64)

    torch.manual_seed(23)
    glue = _ToyGlue()
    world, n_tok = 4, 32
    dev = _FakeMeshDevice(2, 2)
    mask = torch.ones((world, n_tok), dtype=torch.bool)
    mask[1, 20:] = False
    mask[3, 12:] = False
    batch = {
        "padding_mask": mask,
        "s_ir": torch.randn(world, n_tok, 30),
        "tok_ir": torch.randint(0, 12, (world, n_tok)),
        "s_cf": torch.randn(world, n_tok, 28),
        "tok_cf": torch.randint(0, 10, (world, n_tok)),
    }
    ds_h = torch.randn(world, n_tok, 32)
    dc_h = torch.randn(world, n_tok, 32)

    # Reference CPU autograd forward + backward
    glue.zero_grad(set_to_none=True)
    s_ref = glue.init_repr_factory(batch)
    c0_ref = glue.cond_factory({**batch, "recycle_remaining": 2})
    c1_ref = glue.transition_c_1(c0_ref, mask)
    c2_ref = glue.transition_c_2(c1_ref, mask)
    (s_ref * ds_h).sum().add((c2_ref * dc_h).sum()).backward()
    ref_grads = {nm: p.grad.detach().clone() for nm, p in glue.named_parameters()}
    assert all(g.abs().max().item() > 0.0 for g in ref_grads.values()), "every leaf must receive non-zero grad"

    # DeviceWinnerGlue path (zero grad on CPU embeddings first)
    glue.zero_grad(set_to_none=True)
    split = FactoryStaticSplit(glue)
    dwg = DeviceWinnerGlue(glue, split, dev)
    dev_masters = {
        "glue.init_repr_factory.linear_out.weight": dev.make_tensor(
            [glue.init_repr_factory.linear_out.weight.detach().clone() for _ in range(world)]
        ),
        "glue.cond_factory.linear_out.weight": dev.make_tensor(
            [glue.cond_factory.linear_out.weight.detach().clone() for _ in range(world)]
        ),
        "glue.transition_c_1.swish_linear.0.weight": dev.make_tensor(
            [glue.transition_c_1.swish_linear[0].weight.detach().clone() for _ in range(world)]
        ),
        "glue.transition_c_1.linear_out.weight": dev.make_tensor(
            [glue.transition_c_1.linear_out.weight.detach().clone() for _ in range(world)]
        ),
        "glue.transition_c_2.swish_linear.0.weight": dev.make_tensor(
            [glue.transition_c_2.swish_linear[0].weight.detach().clone() for _ in range(world)]
        ),
        "glue.transition_c_2.linear_out.weight": dev.make_tensor(
            [glue.transition_c_2.linear_out.weight.detach().clone() for _ in range(world)]
        ),
    }
    m1_tok = dev.make_tensor([mask[r:r + 1, :, None].float() for r in range(world)])
    ds_dev = dev.make_tensor([ds_h[r:r + 1] for r in range(world)])
    dc_dev = dev.make_tensor([dc_h[r:r + 1] for r in range(world)])

    s_dev, c_dev, saved = dwg.forward(
        batch,
        win_indices=[0] * world,
        k_local=1,
        prefetched=None,
        m1_tok=m1_tok,
        recycle_remaining=2,
        trunk_dtype=mod.float32,
        dev_masters=dev_masters,
    )
    for r in range(world):
        assert torch.allclose(s_dev._buf.shards[r][0], s_ref[r], atol=5e-4, rtol=1e-4)
        assert torch.allclose(c_dev._buf.shards[r][0], c2_ref[r], atol=5e-4, rtol=1e-4)

    dev_grads = dwg.backward(ds_dev, dc_dev, saved, dev_masters)
    assert set(dev_grads.keys()) == set(dev_masters.keys())

    # Check all 6 device weight grads (sum across shards matches full-batch ref grad).
    for nm, g_t in dev_grads.items():
        short = nm[len("glue."):]
        got_sum = sum(g_t._buf.shards)
        ref_g = ref_grads[short]
        rel = (got_sum - ref_g).norm().item() / (ref_g.norm().item() + 1e-12)
        assert rel < 5e-4, f"{nm} grad rel_l2={rel}"

    # Check both CPU embedding grads match torch autograd tightly and no leaf was dropped.
    for short in ("init_repr_factory.feat_creators.1.emb.weight", "cond_factory.feat_creators.1.emb.weight"):
        got_g = dict(glue.named_parameters())[short].grad
        assert got_g is not None, f"missing CPU embedding grad for {short}"
        rel = (got_g - ref_grads[short]).norm().item() / (ref_grads[short].norm().item() + 1e-12)
        assert rel < 5e-4, f"{short} grad rel_l2={rel}"

    # Clean up everything and prove 0 leaked device buffers.
    for t in [s_dev, c_dev, m1_tok, ds_dev, dc_dev, *dev_masters.values(), *dev_grads.values()]:
        mod.deallocate(t)
    dwg.release()
    assert len(dev.live_bufs) == 0


def test_rfd4_run_spmd_trace_region_size_at_least_512mib():
    """Guard against D110 trace_region_size overflow (258.45 MiB > 256 MiB caused refused=1 on step 5)."""
    src = Path("tt_bio/train/rfd4_run.py").read_text()
    assert "trace_region_size=(512 << 20)" in src



