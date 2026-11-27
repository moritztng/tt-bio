"""CPU tests for SPMD MeshDevice RFD4 orchestration (tt_bio.train.spmd_rfd4)."""
from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_training_ccl import _FakeMeshDevice, _make_mock_ttnn
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
        return a.device().make_tensor([x - y for x, y in zip(a._buf.shards, b._buf.shards)])

    mod.subtract = subtract
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
