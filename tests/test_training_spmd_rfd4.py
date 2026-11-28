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

    def _pack_host_arrays(self, bk, arrays, *, fill_zero=False):
        import ttnn

        s0, m, n = bk.canon
        zero = np.zeros((s0, m, n), dtype=np.float32) if fill_zero else None
        pieces = []
        for nm in bk.names:
            if nm in arrays:
                pieces.append(np.asarray(arrays[nm], dtype=np.float32).reshape(s0, m, n))
            elif fill_zero:
                pieces.append(zero)
            else:
                raise KeyError(nm)
        arr = torch.from_numpy(np.ascontiguousarray(np.concatenate(pieces, axis=0), dtype=np.float32))
        return ttnn.from_torch(arr, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.device)

    def _unpack_bucket_host(self, bk, dev_t):
        if dev_t is None:
            return {nm: np.zeros(bk.shape, dtype=np.float32) for nm in bk.names}
        arr = dev_t._buf.shards[0].float().numpy()
        s0, m, n = bk.canon
        return {
            nm: np.ascontiguousarray(arr[i * s0:(i + 1) * s0, :m, :n].reshape(bk.shape), dtype=np.float32)
            for i, nm in enumerate(bk.names)
        }

    def host_master(self):
        out = {}
        for bk in self._all_buckets:
            out.update(self._unpack_bucket_host(bk, bk.master_b))
        if self.shard_world > 1:
            for nm, t in self.master.items():
                if nm not in out:
                    out[nm] = np.ascontiguousarray(t._buf.shards[0].float().numpy().reshape(self.shapes[nm]), dtype=np.float32)
        return out

    def state_families(self):
        exp_avg, exp_avg_sq = {}, {}
        for bk in self._adam_buckets:
            exp_avg.update(self._unpack_bucket_host(bk, bk.state1_b))
            exp_avg_sq.update(self._unpack_bucket_host(bk, bk.state2_b))
        fams = {"exp_avg": exp_avg, "exp_avg_sq": exp_avg_sq}
        if self._muon_buckets:
            mom = {}
            for bk in self._muon_buckets:
                mom.update(self._unpack_bucket_host(bk, bk.state1_b))
            fams["momentum_buffer"] = mom
        if self.ema is not None and self.ema.n_averaged > 0:
            fams["ema_shadow"] = self.ema.host_shadow()
        return fams

    def state_dict(self):
        d = {
            "steps": self.steps, "lr": self.lr, "beta1": self.beta1, "beta2": self.beta2,
            "eps": self.eps, "weight_decay": self.weight_decay, "clip_norm": self.clip_norm,
            "beta1_pow": self.beta1_pow, "beta2_pow": self.beta2_pow,
        }
        if self.ema is not None:
            d["ema_decay"] = self.ema.decay
            d["ema_n_averaged"] = self.ema.n_averaged
        return d

    def load_state_dict(self, d):
        for k in ("steps", "lr", "beta1", "beta2", "eps", "weight_decay", "clip_norm", "beta1_pow", "beta2_pow"):
            if k in d:
                setattr(self, k, d[k])
        if self.ema is not None and "ema_n_averaged" in d:
            self.ema.n_averaged = int(d["ema_n_averaged"])

    def load_host_state(self, masters, families=()):
        import ttnn

        fams = dict(families) if families else {}
        have, want = set(masters), set(self.master)
        if have != want:
            raise ValueError(f"mismatch: missing {sorted(want - have)}, unexpected {sorted(have - want)}")
        for bk in self._all_buckets:
            new_m = self._pack_host_arrays(bk, masters)
            if bk.master_b is not None and bk.master_b.is_allocated():
                ttnn.deallocate(bk.master_b)
            bk.master_b = new_m
            self._unpack_master(bk)
            if bk.kind == "muon":
                mom = fams.get("momentum_buffer")
                if bk.state1_b is not None and bk.state1_b.is_allocated():
                    ttnn.deallocate(bk.state1_b)
                bk.state1_b = self._pack_host_arrays(bk, mom, fill_zero=True) if (mom and any(nm in mom for nm in bk.names)) else None
            else:
                ea = fams.get("exp_avg")
                if bk.state1_b is not None and bk.state1_b.is_allocated():
                    ttnn.deallocate(bk.state1_b)
                bk.state1_b = self._pack_host_arrays(bk, ea, fill_zero=True) if (ea and any(nm in ea for nm in bk.names)) else None
                eas = fams.get("exp_avg_sq")
                if bk.state2_b is not None and bk.state2_b.is_allocated():
                    ttnn.deallocate(bk.state2_b)
                bk.state2_b = self._pack_host_arrays(bk, eas, fill_zero=True) if (eas and any(nm in eas for nm in bk.names)) else None
            if self.ema is not None:
                es = fams.get("ema_shadow")
                if bk.ema_b is not None and bk.ema_b.is_allocated():
                    ttnn.deallocate(bk.ema_b)
                bk.ema_b = self._pack_host_arrays(bk, es, fill_zero=True) if (es and any(nm in es for nm in bk.names)) else None

    def drop_unowned(self):
        return

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

    def _ns_dev(u, *, steps, coeffs, eps, mode):
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

        hm_spmd = spmd.host_master()
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


def test_spmd_zero1_uses_native_tall_ns_without_transpose(monkeypatch):
    """SPMDZero1Step passes tall matrices directly to native newton_schulz_device (0 transpose calls)."""
    _install_mock_ttnn(monkeypatch)
    _install_mock_device_optim(monkeypatch)
    orig_mod = sys.modules["tt_bio.train.device_optim"]
    orig_ns = orig_mod.newton_schulz_device
    seen_shapes = []

    def _ns_native(u, *, steps, coeffs, eps, mode):
        seen_shapes.append(tuple(u.shape))
        return orig_ns(u, steps=steps, coeffs=coeffs, eps=eps, mode=mode)

    def _forbid_transpose(_t):
        raise AssertionError("SPMDZero1Step must not transpose tall matrices; newton_schulz_device handles tall natively")

    monkeypatch.setattr(orig_mod, "newton_schulz_device", _ns_native)
    monkeypatch.setattr(orig_mod, "transpose_exact_f32", _forbid_transpose)
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
    assert seen_shapes and all(s[-2] == 64 and s[-1] == 32 for s in seen_shapes)
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
        return _mk(t.device(), [x.reshape(t.shape)[sl].contiguous() for x in t._buf.shards], getattr(t, "dtype", mod.float32))

    def _concat(ts, dim=-1):
        for t in ts:
            assert t.is_allocated()
        n_sh = len(ts[0]._buf.shards)
        shards = [torch.cat([t._buf.shards[i].reshape(t.shape) for t in ts], dim=dim) for i in range(n_sh)]
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

    mod._silu_calls = 0
    mod._sigmoid_modes = []
    mod.SigmoidMode = types.SimpleNamespace(Accurate="Accurate", FastApproximate="FastApproximate")

    def _silu(t):
        assert t.is_allocated()
        mod._silu_calls += 1
        return _mk(t.device(), [torch.nn.functional.silu(x.float()) for x in t._buf.shards], getattr(t, "dtype", mod.float32))

    def _sigmoid(t, mode=None):
        assert t.is_allocated()
        mod._sigmoid_modes.append(mode)
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
    assert mod._silu_calls == 0, f"approximate ttnn.silu must not be called (got {mod._silu_calls})"
    assert mod._sigmoid_modes == ["Accurate", "Accurate"], f"expected 2 Accurate sigmoids reused in bwd, got {mod._sigmoid_modes}"

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
    import re

    src = Path("tt_bio/train/rfd4_run.py").read_text()
    m = re.search(r"trace_region_size=\((\d+)\s*<<\s*20\)", src)
    assert m is not None and int(m.group(1)) >= 512


def test_make_rfd4_refold_hook_in_place_refreshes_t1_wgo16(monkeypatch):
    """make_rfd4_refold_hook(in_place=True, sync_host=False) refreshes _t1_w[True] in place using wgo16."""
    import tt_bio.train.folds as F

    mod = _install_mock_ttnn(monkeypatch)
    mod.bfloat8_b = "bfloat8_b"
    mod.bfloat16 = "bfloat16"

    def _tc_inplace(t, dt, *, output_tensor=None):
        shards = [s.to(torch.bfloat16).float() if dt == "bfloat16" else s.clone() for s in t._buf.shards]
        if output_tensor is not None:
            for i in range(len(output_tensor._buf.shards)):
                output_tensor._buf.shards[i].copy_(shards[i])
            return output_tensor
        return t.device().make_tensor(shards)

    def _transpose(t, d0, d1):
        return t.device().make_tensor([s.transpose(d0, d1).contiguous() for s in t._buf.shards])

    monkeypatch.setattr(mod, "typecast", _tc_inplace, raising=False)
    monkeypatch.setattr(mod, "transpose", _transpose, raising=False)
    monkeypatch.setattr(F, "_dev_concat_f32", lambda xs, dim: mod.concat(xs, dim=dim))

    dev = _FakeMeshDevice(1, 2)
    tri = SimpleNamespace(_t1_w={})
    pu = SimpleNamespace(tri_out=tri, tri_in=None)
    trunk = SimpleNamespace(pair_updates={0: pu}, blocks=(), _levers=frozenset({"trimul_t1", "trimul_t1w8"}))
    pfx = "nn.pair_update_layers.0.tri_mult_out."

    def _masters(val):
        return {
            pfx + "linear_a_p.weight": dev.make_tensor([torch.full((4, 4), val) for _ in range(2)]),
            pfx + "linear_b_p.weight": dev.make_tensor([torch.full((4, 4), val + 1.0) for _ in range(2)]),
            pfx + "linear_a_g.weight": dev.make_tensor([torch.full((4, 4), val + 2.0) for _ in range(2)]),
            pfx + "linear_b_g.weight": dev.make_tensor([torch.full((4, 4), val + 3.0) for _ in range(2)]),
            pfx + "linear_g.weight": dev.make_tensor([torch.arange(16, dtype=torch.float32).reshape(4, 4) + val for _ in range(2)]),
        }

    host_p = torch.zeros(4, 4)
    hook = F.make_rfd4_refold_hook(trunk, host_glue_params={"glue.x": host_p}, in_place=True, sync_host=False)
    m1 = _masters(1.0)
    hook(m1)
    wp0, wg0, wgo0 = tri._t1_w[True]
    assert torch.allclose(wgo0._buf.shards[0], (torch.arange(16, dtype=torch.float32).reshape(4, 4) + 1.0).t())
    assert torch.equal(host_p, torch.zeros(4, 4)), "sync_host=False must not sync host_glue_params"

    m2 = _masters(5.0)
    hook(m2)
    wp1, wg1, wgo1 = tri._t1_w[True]
    assert wp1 is wp0 and wg1 is wg0 and wgo1 is wgo0, "in_place=True must preserve _t1_w tensor buffers"
    assert torch.allclose(wp1._buf.shards[0][:4], torch.full((4, 4), 5.0))
    assert torch.allclose(wgo1._buf.shards[0], (torch.arange(16, dtype=torch.float32).reshape(4, 4) + 5.0).t())
    for t in (*m1.values(), *m2.values(), wp0, wg0, wgo0):
        mod.deallocate(t)
    assert len(dev.live_bufs) == 0


def test_rfd4_pair_spmd_candidates_and_winner_three_part_weights(monkeypatch):
    """ResidentPairBuilder.candidates_spmd (bf16 fast + fp32 paths) & winner_taped_spmd match PyTorch ref."""
    mod = _install_mock_ttnn(monkeypatch)
    mod.uint32 = "uint32"
    mod.bfloat16 = "bfloat16"
    mod.bfloat8_b = "bfloat8_b"
    mod.TILE_LAYOUT = "TILE"
    mod.ROW_MAJOR_LAYOUT = "ROW_MAJOR"
    mod.DRAM_MEMORY_CONFIG = "DRAM"
    mod.Tensor = object

    def _mock_emb(idx_tt, tbl_tt, *, layout=None, memory_config=None):
        tbl_s = tbl_tt._buf.shards[0].float()
        return idx_tt.device().make_tensor([tbl_s[s.long()] for s in idx_tt._buf.shards])

    def _mock_rmsnorm(x, *, epsilon=1e-5, weight=None, compute_kernel_config=None):
        w = None if weight is None else weight._buf.shards[0].float()
        shards = []
        for s in x._buf.shards:
            sf = s.float()
            r = torch.rsqrt((sf * sf).mean(dim=-1, keepdim=True) + epsilon)
            y = sf * r
            if w is not None:
                y = y * w
            shards.append(y.to(s.dtype))
        return x.device().make_tensor(shards)

    def _mock_rsqrt(x, *, fast_and_approximate_mode=False):
        return x.device().make_tensor([torch.rsqrt(s.float()) for s in x._buf.shards])

    def _mock_tc(t, dt, *, output_tensor=None):
        shards = [s.to(torch.bfloat16).float() if dt == "bfloat16" else s.float().clone() for s in t._buf.shards]
        if output_tensor is not None:
            for i in range(len(output_tensor._buf.shards)):
                output_tensor._buf.shards[i].copy_(shards[i])
            return output_tensor
        out = t.device().make_tensor(shards)
        out.dtype = dt
        return out

    def _mock_mm(a, b, *, transpose_a=False, transpose_b=False, dtype=None, compute_kernel_config=None, core_grid=None):
        shards = []
        for sa, sb in zip(a._buf.shards, b._buf.shards):
            xa = sa.float().transpose(-2, -1) if transpose_a else sa.float()
            xb = sb.float().transpose(-2, -1) if transpose_b else sb.float()
            shards.append(xa @ xb)
        out = a.device().make_tensor(shards)
        out.dtype = dtype or mod.float32
        return out

    def _mock_from_torch(t, *, dtype=None, layout=None, device=None, memory_config=None, mesh_mapper=None):
        w = device.get_num_devices()
        tf = t.to(torch.bfloat16).float() if dtype == mod.bfloat16 else t.float()
        if mesh_mapper is not None:
            chunks = [c.contiguous() for c in tf.chunk(w, dim=0)]
            out = device.make_tensor(chunks)
        else:
            out = device.make_tensor([tf.clone() for _ in range(w)])
        out.dtype = dtype or mod.float32
        return out

    orig_add = mod.add

    def _mock_add(a, b):
        if isinstance(b, (int, float)):
            return a.device().make_tensor([s.float() + float(b) for s in a._buf.shards])
        return orig_add(a, b)

    monkeypatch.setattr(mod, "add", _mock_add, raising=False)
    monkeypatch.setattr(mod, "rsub", lambda a, s: a.device().make_tensor([float(s) - x.float() for x in a._buf.shards]), raising=False)
    monkeypatch.setattr(mod, "embedding", _mock_emb, raising=False)
    monkeypatch.setattr(mod, "rms_norm", _mock_rmsnorm, raising=False)
    monkeypatch.setattr(mod, "rsqrt", _mock_rsqrt, raising=False)
    monkeypatch.setattr(mod, "typecast", _mock_tc, raising=False)
    monkeypatch.setattr(mod, "matmul", _mock_mm, raising=False)
    monkeypatch.setattr(mod, "from_torch", _mock_from_torch, raising=False)
    monkeypatch.setattr(mod, "bitcast", lambda t, dt: t.device().make_tensor([s.clone() for s in t._buf.shards]), raising=False)
    monkeypatch.setattr(mod, "permute", lambda t, dims: t.device().make_tensor([s.permute(*dims).contiguous() for s in t._buf.shards]), raising=False)
    monkeypatch.setattr(mod, "ShardTensorToMesh", lambda dev, dim=0: ("shard", dim), raising=False)
    monkeypatch.setattr(mod, "ReplicateTensorToMesh", lambda dev: ("rep",), raising=False)
    monkeypatch.setattr(mod, "MeshMapperConfig", lambda *a, **k: ("cfg",), raising=False)
    monkeypatch.setattr(mod, "PlacementShard", lambda dim=0: ("ps", dim), raising=False)

    # Provide a minimal fake tt_bio.autograd so rfd4_pair binds cleanly without real ttnn/tenstorrent.
    import contextlib
    import types
    import tt_bio

    fake_ag = types.ModuleType("tt_bio.autograd")
    fake_ag.no_grad = contextlib.nullcontext
    fake_ag.precise_config = lambda: None
    fake_ag._row_mean = lambda v, cfg, c: v.device().make_tensor(
        [s.float().mean(dim=-1, keepdim=True) for s in v._buf.shards]
    )
    fake_ag._sum_leading = lambda v, shp: v.device().make_tensor(
        [s.float().reshape(-1, shp[-1]).sum(dim=0).reshape(shp) for s in v._buf.shards]
    )

    class _FakeAgTensor:
        def __init__(self, value, requires_grad=False):
            self.value = value
            self.requires_grad = requires_grad
            self.grad = None
            self._bw = None

        def add_grad(self, g):
            if self.grad is None:
                self.grad = g
            else:
                nxt = mod.add(self.grad, g)
                mod.deallocate(self.grad)
                mod.deallocate(g)
                self.grad = nxt

    def _fake_tape(val, leaves, make_fn, reads=()):
        t = _FakeAgTensor(val, requires_grad=True)
        t._bw = make_fn()
        return t

    fake_ag.Tensor = _FakeAgTensor
    fake_ag._tape = _fake_tape
    monkeypatch.setitem(sys.modules, "tt_bio.autograd", fake_ag)
    monkeypatch.setattr(tt_bio, "autograd", fake_ag, raising=False)
    monkeypatch.delitem(sys.modules, "tt_bio.train.rfd4_pair", raising=False)
    import tt_bio.train.rfd4_pair as RP
    monkeypatch.setattr(RP, "ttnn", mod)
    monkeypatch.setattr(RP, "ag", fake_ag)
    EncodedPair = RP.EncodedPair
    ResidentPairBuilder = RP.ResidentPairBuilder
    _PairStepCache = RP._PairStepCache

    dev = _FakeMeshDevice(1, 2)
    world, k, n, c = 2, 2, 4, 32
    nn2 = n * n
    torch.manual_seed(17)
    W = torch.nn.Parameter(torch.randn(c, 16) * 0.2)
    lnw = torch.nn.Parameter(torch.ones(c) + 0.1 * torch.randn(c))

    rb = ResidentPairBuilder.__new__(ResidentPairBuilder)
    rb.device = dev
    rb.c = c
    rb.e1 = 1e-5
    rb.e2 = 1e-5
    rb.step_w = 8
    rb.const_w = 8
    rb.k_pack = 16
    rb.k_orig = 16
    rb._k_orig_pad = 16
    rb._pack_cols = list(range(16))
    rb._orig_cols = list(range(16))
    rb.misc_w = 0
    rb.large_oh = []
    rb.small_oh = []
    rb.scalar_meta = []
    rb._tbl_large_oh = []
    rb._tbl_small_oh = []
    rb._tbl_scalar = None
    step_tbl = torch.eye(8, dtype=torch.float32)
    rb._tbl_step = dev.make_tensor([step_tbl.clone() for _ in range(world)])
    rb._P_pack_to_orig = dev.make_tensor([torch.eye(16, dtype=torch.float32) for _ in range(world)])
    rb.builder = SimpleNamespace(
        init_repr_factory=SimpleNamespace(
            linear_out=SimpleNamespace(weight=W),
            ln_out=SimpleNamespace(weight=lnw),
        )
    )

    encs = []
    for _r in range(world):
        encs.append(EncodedPair(
            k=k, n=n,
            bins_step=torch.randint(0, 8, (k, nn2), dtype=torch.int32),
            idx_onehot=[],
            idx_scalar=torch.zeros((1, nn2), dtype=torch.int32),
            dedup=True,
            gamma=torch.randn(k, c) * 0.5 + 1.0,
            beta=torch.randn(k, c) * 0.1,
            fallback_const=torch.randint(0, 2, (1, nn2, 8)).float(),
        ))

    def _ref_forward(f_all, w_t, l_t, g_t, b_t):
        x = (f_all @ w_t.t()).reshape(-1, n, n, c)
        n1 = x * torch.rsqrt((x * x).mean(dim=-1, keepdim=True) + rb.e1)
        a = n1 * l_t
        bb = a * torch.rsqrt((a * a).mean(dim=-1, keepdim=True) + rb.e2)
        return bb * g_t.reshape(-1, 1, 1, c) + b_t.reshape(-1, 1, 1, c)

    for dt in (mod.bfloat16, mod.float32):
        z_cand, cache = rb.candidates_spmd(encs, dtype=dt)
        assert isinstance(cache, _PairStepCache)
        assert len(cache.ws_parts) == 3
        assert tuple(z_cand.shape) == (k, n, n, c)
        # Verify forward output parity per rank.
        for r in range(world):
            f_step_r = step_tbl[encs[r].bins_step.long()]  # [k, nn2, 8]
            f_const_r = encs[r].fallback_const.expand(k, nn2, 8)
            f_all_r = torch.cat([f_step_r, f_const_r], dim=-1)
            z_ref_r = _ref_forward(f_all_r, W.detach(), lnw.detach(), encs[r].gamma, encs[r].beta)
            tol = 3e-2 if dt == mod.bfloat16 else 1e-4
            assert torch.allclose(z_cand._buf.shards[r].float(), z_ref_r, atol=tol, rtol=tol)
        mod.deallocate(z_cand)

        win_idx = [0, 1]
        gamma_w = torch.stack([encs[r].gamma[win_idx[r]] for r in range(world)], dim=0).detach().requires_grad_(True)
        beta_w = torch.stack([encs[r].beta[win_idx[r]] for r in range(world)], dim=0).detach().requires_grad_(True)
        dz_shards = [torch.randn(1, n, n, c) for _ in range(world)]

        # Reference PyTorch autograd grads for winner.
        W_ref = W.detach().clone().requires_grad_(True)
        lnw_ref = lnw.detach().clone().requires_grad_(True)
        gw_ref = gamma_w.detach().clone().requires_grad_(True)
        bw_ref = beta_w.detach().clone().requires_grad_(True)
        ref_loss = 0.0
        for r in range(world):
            fs_w = step_tbl[encs[r].bins_step[win_idx[r]:win_idx[r] + 1].long()]
            fc_w = encs[r].fallback_const
            fa_w = torch.cat([fs_w, fc_w], dim=-1)
            zw_r = _ref_forward(fa_w, W_ref, lnw_ref, gw_ref[r:r + 1], bw_ref[r:r + 1])
            ref_loss = ref_loss + (zw_r * dz_shards[r]).sum()
        ref_loss.backward()

        leaf_holders = []

        class _MiniBridge:
            def up_replicated(self, t, dtype=None):
                ag_t = fake_ag.Tensor(dev.make_tensor([t.detach().clone() for _ in range(world)]), requires_grad=t.requires_grad)
                leaf_holders.append(ag_t)
                return ag_t

            def up(self, t, dtype=None):
                ag_t = fake_ag.Tensor(dev.make_tensor([t[r:r + 1].detach().clone() for r in range(world)]), requires_grad=t.requires_grad)
                leaf_holders.append(ag_t)
                return ag_t

        z_win = rb.winner_taped_spmd(cache, encs, win_idx, gamma_w, beta_w, _MiniBridge(), dtype=mod.float32)
        assert tuple(z_win.value.shape) == (1, n, n, c)
        dz = dev.make_tensor([d.clone() for d in dz_shards])
        z_win._bw(dz)
        mod.deallocate(dz)
        mod.deallocate(z_win.value)

        w_l, l_l, g_l, b_l = leaf_holders
        dW_got = sum(w_l.grad._buf.shards)
        dL_got = sum(l_l.grad._buf.shards).reshape(c)
        dG_got = torch.cat(g_l.grad._buf.shards, dim=0)
        dB_got = torch.cat(b_l.grad._buf.shards, dim=0)
        assert torch.allclose(dW_got, W_ref.grad, atol=5e-4, rtol=5e-4)
        assert torch.allclose(dL_got, lnw_ref.grad, atol=5e-4, rtol=5e-4)
        assert torch.allclose(dG_got, gw_ref.grad, atol=5e-4, rtol=5e-4)
        assert torch.allclose(dB_got, bw_ref.grad, atol=5e-4, rtol=5e-4)
        for lh in leaf_holders:
            mod.deallocate(lh.value)
            if lh.grad is not None:
                mod.deallocate(lh.grad)

    rb.deallocate()
    assert len(dev.live_bufs) == 0


def test_rfd4_run_cli_depths_flags_no_conflict(tmp_path):
    """rfd4_run.py argparse registers --depths, --depths-from, --depth-override, --mesh without conflict."""
    import ast

    src = Path("tt_bio/train/rfd4_run.py").read_text()
    tree = ast.parse(src)
    # Count add_argument("--depths-from", ...) calls: must be exactly 1.
    depths_from_calls = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            for arg in node.args:
                if isinstance(arg, ast.Constant) and arg.value == "--depths-from":
                    depths_from_calls += 1
    assert depths_from_calls == 1, f"expected 1 --depths-from arg, found {depths_from_calls}"

    from tt_bio.train.rfd4_run import _load_depths
    jf = tmp_path / "steps.jsonl"
    jf.write_text('{"n_recycle": 2}\n{"depth": 1}\n{"n_recycle": 3}\n')
    assert _load_depths("", str(jf)) == [2, 1, 3]
    assert _load_depths("3,1,2", "") == [3, 1, 2]


def _make_ckpt_weights(seed: int = 123):
    rng = np.random.default_rng(seed)
    weights = {
        # 4 sharded Muon matrices (shape 32x64 -> 2 per axis-0 row on 2x4)
        "mu_sh_0": rng.standard_normal((32, 64)).astype(np.float32) * 0.2,
        "mu_sh_1": rng.standard_normal((32, 64)).astype(np.float32) * 0.2,
        "mu_sh_2": rng.standard_normal((32, 64)).astype(np.float32) * 0.2,
        "mu_sh_3": rng.standard_normal((32, 64)).astype(np.float32) * 0.2,
        # 2 sharded tall Muon matrices (shape 64x32 -> 1 per axis-0 row on 2x4)
        "mu_tall_0": rng.standard_normal((64, 32)).astype(np.float32) * 0.2,
        "mu_tall_1": rng.standard_normal((64, 32)).astype(np.float32) * 0.2,
        # 1 tiny unsharded non-tile-aligned Muon matrix (stays in base_opt._muon_buckets)
        "mu_tiny": rng.standard_normal((16, 16)).astype(np.float32) * 0.2,
        # AdamW params (1D and 2D)
        "adam_norm": rng.standard_normal((64,)).astype(np.float32) * 0.5,
        "adam_head": rng.standard_normal((32, 32)).astype(np.float32) * 0.2,
    }
    muon_names = {k for k in weights if k.startswith("mu_")}
    return weights, muon_names


def _install_spmd_step_mocks(monkeypatch):
    mod = _install_mock_ttnn(monkeypatch)
    import tt_bio.train.device_optim as dopt

    monkeypatch.setattr(dopt, "_clone_exact_f32", lambda t: mod.clone(t))
    monkeypatch.setattr(dopt, "_concat_dim0_exact_f32", lambda ts: mod.concat(ts, dim=0))
    monkeypatch.setattr(
        dopt,
        "_sum_sq_last2_exact_f32",
        lambda t: t.device().make_tensor(
            [torch.sum(s.float() * s.float(), dim=(-2, -1)) for s in t._buf.shards]
        ),
    )
    monkeypatch.setattr(
        dopt,
        "newton_schulz_device",
        lambda u, steps=5, coeffs=(3.4445, -4.7750, 2.0315), eps=1e-7, mode="fp32": u.device().make_tensor(
            [_mock_ns_torch(s.float(), steps, coeffs, eps) for s in u._buf.shards]
        ),
    )
    return mod


def test_spmd_zero1_checkpoint_save_resume_exact_and_broken_arms(monkeypatch, tmp_path):
    """SPMDZero1Step save_adapter/load_adapter round-trips all sharded+unsharded state bit-exact; broken arms FAIL."""
    _install_spmd_step_mocks(monkeypatch)
    from safetensors.numpy import load_file, save_file
    from tt_bio.train.checkpoint import Checkpointer, load_adapter, save_adapter

    weights, muon_names = _make_ckpt_weights(seed=77)
    rng = np.random.default_rng(78)

    # Continuous run on 2x4 (rm=2, cm=4) with ZeRO-1 Muon + EMA.
    dev_cont = _FakeMeshDevice(2, 4)
    base_cont = _FakeDeviceOptimizer(weights, muon_names, dev_cont, ema_decay=0.95)
    spmd_cont = SPMDZero1Step(base_cont, weights, dev_cont, full_shape=(2, 4), use_zero1_muon=True)
    assert len(spmd_cont.sharded_names()) == 6
    assert spmd_cont.muon_names == frozenset(muon_names)

    # Step 0 and Step 1 per-rank gradients (identical across continuous and resumed runs).
    step_grads_np = [
        {nm: [rng.standard_normal(arr.shape).astype(np.float32) * 0.25 for _ in range(8)] for nm, arr in weights.items()}
        for _ in range(2)
    ]

    def _to_mesh(dev, g_dict):
        return {nm: dev.make_tensor([torch.from_numpy(a.copy()) for a in shards]) for nm, shards in g_dict.items()}

    # Step 1 (index 0) on continuous run.
    spmd_cont.step(_to_mesh(dev_cont, step_grads_np[0]), free_grads=True)
    ck = Checkpointer(tmp_path / "ckpt_spmd", every=1, keep=2)
    ckpt_path = ck.save(1, spmd_cont, metrics={"loss": 12.5}, force=True)
    assert ckpt_path is not None and ckpt_path.exists()

    # Negative check 1: passing raw base_opt (the pre-fix bug) silently drops all 6 sharded Muon masters!
    bad_path = tmp_path / "bad_base_opt.safetensors"
    save_adapter(bad_path, base_cont, meta={"step": 1})
    bad_blob = load_file(str(bad_path))
    assert len([k for k in bad_blob if k.startswith("master|")]) == len(weights) - len(spmd_cont.sharded_names())

    # Resumed run on fresh 2x4 device.
    dev_res = _FakeMeshDevice(2, 4)
    base_res = _FakeDeviceOptimizer(weights, muon_names, dev_res, ema_decay=0.95)
    spmd_res = SPMDZero1Step(base_res, weights, dev_res, full_shape=(2, 4), use_zero1_muon=True)
    with pytest.raises(ValueError, match="adapter/parameter mismatch"):
        load_adapter(bad_path, weights, dev_res, opt=spmd_res)

    # Negative check 2: removing one sharded Muon momentum_buffer key MUST be refused on load.
    blob_nomom = dict(load_file(str(ckpt_path)))
    blob_nomom.pop("momentum_buffer|mu_sh_0")
    nomom_path = tmp_path / "nomom.safetensors"
    save_file(blob_nomom, str(nomom_path), metadata={"tt_bio_adapter": "1"})
    with pytest.raises(ValueError, match="no Muon momentum"):
        load_adapter(nomom_path, weights, dev_res, opt=spmd_res)

    # Positive check: load_adapter into spmd_res restores ALL masters, moments, EMA, and scalars bit-exact.
    meta = load_adapter(ckpt_path, weights, dev_res, opt=spmd_res)
    assert meta["step"] == 1
    assert spmd_res.state_dict() == spmd_cont.state_dict()
    d_cont_1, n1 = master_consensus_digest(spmd_cont.master, dev_cont)
    d_res_1, n2 = master_consensus_digest(spmd_res.master, dev_res)
    assert n1 == n2 == len(weights) and d_res_1 == d_cont_1
    for nm in weights:
        assert np.array_equal(spmd_res.host_master()[nm], spmd_cont.host_master()[nm]), f"master {nm}"
        assert np.array_equal(spmd_res.host_ema_shadow()[nm], spmd_cont.host_ema_shadow()[nm]), f"ema {nm}"
    fams_cont = spmd_cont.state_families()
    fams_res = spmd_res.state_families()
    assert set(fams_cont) == set(fams_res) == {"exp_avg", "exp_avg_sq", "momentum_buffer", "ema_shadow"}
    for fam, d_c in fams_cont.items():
        assert set(d_c) == set(fams_res[fam])
        for nm in d_c:
            assert np.array_equal(fams_res[fam][nm], d_c[nm]), f"{fam}|{nm}"

    # Step 2 (index 1) on BOTH continuous and resumed runs -> must remain bit-identical!
    spmd_cont.step(_to_mesh(dev_cont, step_grads_np[1]), free_grads=True)
    spmd_res.step(_to_mesh(dev_res, step_grads_np[1]), free_grads=True)
    d_cont_2, _ = master_consensus_digest(spmd_cont.master, dev_cont)
    d_res_2, _ = master_consensus_digest(spmd_res.master, dev_res)
    assert d_res_2 == d_cont_2, f"step-2 resume digest diverged: {d_res_2} vs {d_cont_2}"
    for nm in weights:
        assert float(np.max(np.abs(spmd_res.host_master()[nm] - spmd_cont.host_master()[nm]))) == 0.0
        assert float(np.max(np.abs(spmd_res.host_ema_shadow()[nm] - spmd_cont.host_ema_shadow()[nm]))) == 0.0

    # Negative check 3 (broken arm): perturbing a sharded Muon momentum buffer MUST diverge step-2 digest!
    blob_brk = dict(load_file(str(ckpt_path)))
    blob_brk["momentum_buffer|mu_sh_2"] = blob_brk["momentum_buffer|mu_sh_2"] * 0.0
    brk_path = tmp_path / "broken_mom.safetensors"
    with open(ckpt_path, "rb") as fh:
        n_hdr = int.from_bytes(fh.read(8), "little")
        hdr = __import__("json").loads(fh.read(n_hdr))
    save_file(blob_brk, str(brk_path), metadata=hdr.get("__metadata__", {}))
    dev_brk = _FakeMeshDevice(2, 4)
    base_brk = _FakeDeviceOptimizer(weights, muon_names, dev_brk, ema_decay=0.95)
    spmd_brk = SPMDZero1Step(base_brk, weights, dev_brk, full_shape=(2, 4), use_zero1_muon=True)
    load_adapter(brk_path, weights, dev_brk, opt=spmd_brk)
    spmd_brk.step(_to_mesh(dev_brk, step_grads_np[1]), free_grads=True)
    d_brk_2, _ = master_consensus_digest(spmd_brk.master, dev_brk)
    assert d_brk_2 != d_cont_2
    assert float(np.max(np.abs(spmd_brk.host_master()["mu_sh_2"] - spmd_cont.host_master()["mu_sh_2"]))) > 1e-4

    spmd_cont.deallocate()
    base_cont.deallocate()
    spmd_res.deallocate()
    base_res.deallocate()
    spmd_brk.deallocate()
    base_brk.deallocate()
    assert len(dev_cont.live_bufs) == 0
    assert len(dev_res.live_bufs) == 0
    assert len(dev_brk.live_bufs) == 0


def test_spmd_zero1_checkpoint_cross_world_and_step0(monkeypatch, tmp_path):
    """Checkpoints saved on 2x4 ZeRO-1 resume cleanly on 1x4 unsharded (and vice versa), plus step-0 zero mom."""
    _install_spmd_step_mocks(monkeypatch)
    from tt_bio.train.checkpoint import load_adapter, save_adapter

    weights, muon_names = _make_ckpt_weights(seed=91)
    rng = np.random.default_rng(92)

    # Step-0 save before any optimizer step (empty sh_mom, ema.n_averaged == 0).
    dev_0 = _FakeMeshDevice(2, 4)
    base_0 = _FakeDeviceOptimizer(weights, muon_names, dev_0, ema_decay=0.9)
    spmd_0 = SPMDZero1Step(base_0, weights, dev_0, full_shape=(2, 4), use_zero1_muon=True)
    p0 = tmp_path / "step0.safetensors"
    save_adapter(p0, spmd_0, meta={"step": 0})
    fams0 = spmd_0.state_families()
    assert set(fams0["momentum_buffer"]) == muon_names
    assert "ema_shadow" not in fams0
    for nm in muon_names:
        assert float(np.max(np.abs(fams0["momentum_buffer"][nm]))) == 0.0

    # Step 1 on 2x4 -> save -> resume on 1x4 (unsharded) -> save -> resume on 2x4 (sharded).
    g0 = {nm: dev_0.make_tensor([torch.from_numpy(rng.standard_normal(arr.shape).astype(np.float32) * 0.2) for _ in range(8)])
          for nm, arr in weights.items()}
    spmd_0.step(g0, free_grads=True)
    p1 = tmp_path / "from_2x4.safetensors"
    save_adapter(p1, spmd_0, meta={"step": 1})

    dev_1x4 = _FakeMeshDevice(1, 4)
    base_1x4 = _FakeDeviceOptimizer(weights, muon_names, dev_1x4, ema_decay=0.9)
    spmd_1x4 = SPMDZero1Step(base_1x4, weights, dev_1x4, full_shape=(1, 4), use_zero1_muon=True)
    assert spmd_1x4.use_zero1_muon is False
    load_adapter(p1, weights, dev_1x4, opt=spmd_1x4)
    assert spmd_1x4.master_digest() == spmd_0.master_digest()
    p2 = tmp_path / "from_1x4.safetensors"
    save_adapter(p2, spmd_1x4, meta={"step": 1})

    dev_2x4_b = _FakeMeshDevice(2, 4)
    base_2x4_b = _FakeDeviceOptimizer(weights, muon_names, dev_2x4_b, ema_decay=0.9)
    spmd_2x4_b = SPMDZero1Step(base_2x4_b, weights, dev_2x4_b, full_shape=(2, 4), use_zero1_muon=True)
    load_adapter(p2, weights, dev_2x4_b, opt=spmd_2x4_b)
    assert spmd_2x4_b.master_digest() == spmd_0.master_digest()
    d_0, _ = master_consensus_digest(spmd_0.master, dev_0)
    d_b, n_b = master_consensus_digest(spmd_2x4_b.master, dev_2x4_b)
    assert n_b == len(weights) and d_b == d_0
    fams_0 = spmd_0.state_families()
    fams_b = spmd_2x4_b.state_families()
    assert set(fams_b) == set(fams_0)
    for fam, d_c in fams_0.items():
        for nm in d_c:
            assert np.array_equal(fams_b[fam][nm], d_c[nm]), f"cross-world {fam}|{nm}"

    for sp, ba in ((spmd_0, base_0), (spmd_1x4, base_1x4), (spmd_2x4_b, base_2x4_b)):
        sp.deallocate()
        ba.deallocate()
    assert len(dev_0.live_bufs) == 0 and len(dev_1x4.live_bufs) == 0 and len(dev_2x4_b.live_bufs) == 0


def test_rfd4_run_uses_ckpt_opt_for_all_checkpoint_calls():
    """rfd4_run.py must pass ckpt_opt (SPMDZero1Step when --mesh active) to load_adapter and ck.save."""
    import ast

    src = Path("tt_bio/train/rfd4_run.py").read_text()
    tree = ast.parse(src)
    load_calls = []
    save_calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id == "load_adapter":
            opt_kw = [kw.value for kw in node.keywords if kw.arg == "opt"]
            assert len(opt_kw) == 1 and isinstance(opt_kw[0], ast.Name) and opt_kw[0].id == "ckpt_opt"
            load_calls.append(node.lineno)
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "ck"
            and node.func.attr == "save"
        ):
            assert len(node.args) >= 2 and isinstance(node.args[1], ast.Name) and node.args[1].id == "ckpt_opt"
            save_calls.append(node.lineno)
    assert len(load_calls) == 1 and len(save_calls) == 2


def test_put_sharded_card_tilize_policy(monkeypatch):
    """put_sharded card-tilizes float32 (>=2D) and bf16 with W%32==0 on multi-chip MeshDevice."""
    mod = _augment_mock_ttnn_for_glue(_install_mock_ttnn(monkeypatch))
    from tt_bio.train.mesh_dp import put_sharded

    events = []
    orig_ft = mod.from_torch
    orig_tl = mod.to_layout

    def _rec_ft(tensor, **kw):
        events.append(("from_torch", tuple(tensor.shape), kw.get("dtype"), kw.get("layout")))
        return orig_ft(tensor, **kw)

    def _rec_tl(t, layout):
        events.append(("to_layout", tuple(t.shape), layout))
        return orig_tl(t, layout)

    monkeypatch.setattr(mod, "from_torch", _rec_ft)
    monkeypatch.setattr(mod, "to_layout", _rec_tl)
    dev = _FakeMeshDevice(2, 2)

    # 1. float32 arbitrary trailing dim -> card-tilized (ROW_MAJOR then to_layout TILE)
    events.clear()
    t1 = put_sharded(torch.randn(4, 32, 3), dev, dtype=mod.float32, layout=mod.TILE_LAYOUT)
    assert events[0][3] == mod.ROW_MAJOR_LAYOUT and events[1][0] == "to_layout"
    mod.deallocate(t1)

    # 2. bfloat16 with W % 32 == 0 and H == 1 -> card-tilized
    events.clear()
    t2 = put_sharded(torch.randn(4, 1, 64), dev, dtype=mod.bfloat16, layout=mod.TILE_LAYOUT)
    assert events[0][3] == mod.ROW_MAJOR_LAYOUT and events[1][0] == "to_layout"
    mod.deallocate(t2)

    # 3. bfloat16 with W % 32 != 0 -> direct TILE_LAYOUT (no unaligned ROW_MAJOR)
    events.clear()
    t3 = put_sharded(torch.randn(4, 32, 1), dev, dtype=mod.bfloat16, layout=mod.TILE_LAYOUT)
    assert len(events) == 1 and events[0][3] == mod.TILE_LAYOUT
    mod.deallocate(t3)

    assert len(dev.live_bufs) == 0


def test_cand_conds_row_constant_and_prefetch_parity(monkeypatch):
    """Row-constant cf_static_2d fast path in SPMDPrefetcher, cand_conds_and_s0_device & DeviceWinnerGlue matches 3D exact."""
    import contextlib

    mod = _augment_mock_ttnn_for_glue(_install_mock_ttnn(monkeypatch))
    fake_ag = types.ModuleType("tt_bio.autograd")
    fake_ag.precise_config = lambda: None
    fake_ag.no_grad = contextlib.nullcontext
    monkeypatch.setitem(sys.modules, "tt_bio.autograd", fake_ag)
    from tt_bio.train.mesh_dp import put_sharded, read_sharded
    from tt_bio.train.spmd_trace import (
        DeviceWinnerGlue,
        FactoryStaticSplit,
        SPMDPrefetcher,
        cand_conds_and_s0_device,
    )

    class _RowConstStatic(torch.nn.Module):
        def __init__(self, key):
            super().__init__()
            self.key = key

        def forward(self, batch):
            return batch[self.key]

    class _DynRowConstEmb(torch.nn.Module):
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

    class _ToyGlue(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.condition_on_recycle_remaining = True
            self.init_repr_factory = _Factory(
                [_RowConstStatic("s_ir"), _DynRowConstEmb(10, 6, "tok_ir")], in_dim=26 + 6, out_dim=32
            )
            self.cond_factory = _Factory(
                [_RowConstStatic("s_cf"), _DynRowConstEmb(8, 5, "tok_cf"), _RemFeat(4)], in_dim=23 + 5 + 4, out_dim=32
            )
            self.transition_c_1 = _Transition(32, 64)
            self.transition_c_2 = _Transition(32, 64)

    torch.manual_seed(41)
    glue = _ToyGlue()
    world, k_local, n_tok = 4, 2, 32
    dev = _FakeMeshDevice(2, 2)
    rank_bts = []
    for r in range(world):
        mask = torch.ones((k_local, n_tok), dtype=torch.bool)
        if r % 2 == 1:
            mask[0, 20:] = False
            mask[1, 14:] = False
        # Row-constant s_cf and tok_cf (like real RFD4); general per-token s_ir / tok_ir.
        s_cf_row = torch.randn(k_local, 1, 23)
        tok_cf_row = torch.randint(0, 8, (k_local, 1))
        rank_bts.append({
            "padding_mask": mask,
            "x_t": {"bb_ca": torch.randn(k_local, n_tok, 3)},
            "s_ir": torch.randn(k_local, n_tok, 26),
            "tok_ir": torch.randint(0, 10, (k_local, n_tok)),
            "s_cf": s_cf_row.expand(k_local, n_tok, 23).clone(),
            "tok_cf": tok_cf_row.expand(k_local, n_tok).clone(),
            "seq_mask": mask.to(torch.float32),
            "residue_mask": mask.to(torch.float32),
        })

    # Stub extract_cand_score_meta since toy batches omit full diffusion fields.
    monkeypatch.setattr("tt_bio.train.rfd4_heads.extract_cand_score_meta", lambda bts: {"ok": len(bts)})

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

    split = FactoryStaticSplit(glue)
    pref_exec = SPMDPrefetcher(split, res_pair=None, max_workers=2)
    try:
        pref = pref_exec.get(0, rank_bts)
    finally:
        pref_exec.close()

    assert pref.cf_static_2d is not None and pref.cf_static is None
    assert tuple(pref.cf_static_2d.shape) == (world * k_local, 32)
    assert tuple(pref.ir_static.shape) == (world * k_local, n_tok, 32)

    mask_all = torch.cat([b["padding_mask"] for b in rank_bts], dim=0).to(torch.float32)
    mk_tok = put_sharded(mask_all[..., None], dev, dtype=mod.float32, layout=mod.TILE_LAYOUT)

    for depth in (1, 2, 4):
        s0_fast, ck_fast = cand_conds_and_s0_device(
            glue, rank_bts, mk_tok, depth, mod.float32, dev,
            dev_masters=dev_masters, prefetched=pref, factory_split=split,
        )
        s0_ref, ck_ref = cand_conds_and_s0_device(
            glue, rank_bts, mk_tok, depth, mod.float32, dev,
            dev_masters=dev_masters, prefetched=None, factory_split=None,
        )
        assert float((read_sharded(s0_fast, dev) - read_sharded(s0_ref, dev)).abs().max()) < 1e-6
        assert len(ck_fast) == len(ck_ref) == depth
        for cyc in range(depth):
            diff = float((read_sharded(ck_fast[cyc], dev) - read_sharded(ck_ref[cyc], dev)).abs().max())
            assert diff < 1e-6, f"depth={depth} cyc={cyc} diff={diff}"
        for t in (s0_fast, s0_ref, *ck_fast, *ck_ref):
            mod.deallocate(t)

    # DeviceWinnerGlue.forward consuming cf_static_2d must match 3D cf_static exact.
    dwg = DeviceWinnerGlue(glue, split, dev)
    win_indices = [0, 1, 0, 1]
    rank_bws = [{k: (v[wi:wi + 1] if torch.is_tensor(v) else {kk: vv[wi:wi + 1] for kk, vv in v.items()})
                 for k, v in b.items()} for b, wi in zip(rank_bts, win_indices)]
    stacked_bw = {
        k: (torch.cat([rb[k] for rb in rank_bws], dim=0) if torch.is_tensor(rank_bws[0][k])
            else {kk: torch.cat([rb[k][kk] for rb in rank_bws], dim=0) for kk in rank_bws[0][k]})
        for k in rank_bws[0]
    }
    w_mask = stacked_bw["padding_mask"].to(torch.float32)
    w_mk = put_sharded(w_mask[..., None], dev, dtype=mod.float32, layout=mod.TILE_LAYOUT)
    s_2d, c_2d, sv_2d = dwg.forward(
        stacked_bw, win_indices, k_local, prefetched=pref, m1_tok=w_mk,
        recycle_remaining=2, trunk_dtype=mod.float32, dev_masters=dev_masters,
    )
    s_3d, c_3d, sv_3d = dwg.forward(
        stacked_bw, win_indices, k_local, prefetched=None, m1_tok=w_mk,
        recycle_remaining=2, trunk_dtype=mod.float32, dev_masters=dev_masters,
    )
    assert float((read_sharded(s_2d, dev) - read_sharded(s_3d, dev)).abs().max()) < 1e-6
    assert float((read_sharded(c_2d, dev) - read_sharded(c_3d, dev)).abs().max()) < 1e-6
    ds_w = put_sharded(torch.randn(world, n_tok, 32), dev, dtype=mod.float32, layout=mod.TILE_LAYOUT)
    dc_w = put_sharded(torch.randn(world, n_tok, 32), dev, dtype=mod.float32, layout=mod.TILE_LAYOUT)
    glue.zero_grad(set_to_none=True)
    g_2d = dwg.backward(ds_w, dc_w, sv_2d, dev_masters)
    emb_2d = {nm: p.grad.detach().clone() for nm, p in glue.named_parameters() if p.grad is not None}
    glue.zero_grad(set_to_none=True)
    g_3d = dwg.backward(ds_w, dc_w, sv_3d, dev_masters)
    emb_3d = {nm: p.grad.detach().clone() for nm, p in glue.named_parameters() if p.grad is not None}
    assert set(g_2d) == set(g_3d) and set(emb_2d) == set(emb_3d)
    for k in g_2d:
        assert float((read_sharded(g_2d[k], dev) - read_sharded(g_3d[k], dev)).abs().max()) < 1e-5
        mod.deallocate(g_2d[k])
        mod.deallocate(g_3d[k])
    for k in emb_2d:
        assert float((emb_2d[k] - emb_3d[k]).abs().max()) < 1e-5
    dwg.release()
    for t in (s_2d, c_2d, s_3d, c_3d, w_mk, ds_w, dc_w):
        mod.deallocate(t)

    # Non-row-constant dynamic cf feature must automatically fall back to 3D and still match exact.
    rank_bts_nonconst = [{**b, "tok_cf": torch.randint(0, 8, (k_local, n_tok))} for b in rank_bts]
    s0_fb, ck_fb = cand_conds_and_s0_device(
        glue, rank_bts_nonconst, mk_tok, 2, mod.float32, dev,
        dev_masters=dev_masters, prefetched=pref, factory_split=split,
    )
    s0_fb_ref, ck_fb_ref = cand_conds_and_s0_device(
        glue, rank_bts_nonconst, mk_tok, 2, mod.float32, dev,
        dev_masters=dev_masters, prefetched=None, factory_split=None,
    )
    assert float((read_sharded(s0_fb, dev) - read_sharded(s0_fb_ref, dev)).abs().max()) < 1e-6
    for cyc in range(2):
        assert float((read_sharded(ck_fb[cyc], dev) - read_sharded(ck_fb_ref[cyc], dev)).abs().max()) < 1e-6
    for t in (s0_fb, s0_fb_ref, *ck_fb, *ck_fb_ref, mk_tok, *dev_masters.values()):
        mod.deallocate(t)
    assert len(dev.live_bufs) == 0


def test_device_winner_glue_and_heads_meshtrace_parity_and_no_leaks(monkeypatch):
    """Paired MeshTrace for DeviceWinnerGlue and DeviceHeads matches eager across live in-place weight updates, 0 leaks."""
    import contextlib
    import tt_bio

    mod = _augment_mock_ttnn_for_glue(_install_mock_ttnn(monkeypatch))
    # Add mock execute_trace / release_trace / synchronize_device and mock TraceScratch to test trace wiring end-to-end.
    trace_fns: dict[int, Any] = {}
    next_tid = [1]

    class _FakeScratch:
        def __init__(self, dev):
            self.dev = dev
            self.captures = 0
            self.released = False

        def capture(self, need, size, runs, settle=None):
            tids = []
            try:
                for fn in runs:
                    fn()
                    tid = next_tid[0]
                    next_tid[0] += 1
                    trace_fns[tid] = fn
                    tids.append(tid)
                self.captures += len(tids)
            finally:
                if settle is not None:
                    settle()
            return tids

        def release(self):
            self.released = True

    def _mock_copy(src, dst):
        for s_src, s_dst in zip(src._buf.shards, dst._buf.shards):
            s_dst.copy_(s_src)

    monkeypatch.setattr(mod, "copy", _mock_copy, raising=False)
    monkeypatch.setattr(mod, "execute_trace", lambda dev, tid, cq_id=0, blocking=False: trace_fns[tid](), raising=False)
    monkeypatch.setattr(mod, "release_trace", lambda dev, tid: trace_fns.pop(tid, None), raising=False)
    monkeypatch.setattr(mod, "synchronize_device", lambda dev: None, raising=False)

    fake_tt = types.ModuleType("tt_bio.tenstorrent")
    fake_tt.TraceScratch = _FakeScratch
    monkeypatch.setitem(sys.modules, "tt_bio.tenstorrent", fake_tt)
    monkeypatch.setattr(tt_bio, "tenstorrent", fake_tt, raising=False)

    fake_ag = types.ModuleType("tt_bio.autograd")
    fake_ag.precise_config = lambda: None
    fake_ag.no_grad = contextlib.nullcontext
    fake_ag.tape = contextlib.nullcontext
    fake_ag.release_pins = lambda: None

    class _FakeAgT:
        def __init__(self, value, requires_grad=False):
            self.value = value
            self.requires_grad = requires_grad
            self.grad = None

    fake_ag.Tensor = _FakeAgT
    monkeypatch.setitem(sys.modules, "tt_bio.autograd", fake_ag)
    monkeypatch.setattr(tt_bio, "autograd", fake_ag, raising=False)

    from tt_bio.train.mesh_dp import put_sharded, read_sharded
    from tt_bio.train.spmd_trace import DeviceWinnerGlue, FactoryStaticSplit, SPMDMeshTracer

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

    class _Transition(torch.nn.Module):
        def __init__(self, dim, hidden):
            super().__init__()
            self.swish_linear = torch.nn.Sequential(torch.nn.Linear(dim, 2 * hidden, bias=False))
            self.linear_out = torch.nn.Linear(hidden, dim, bias=False)

    class _ToyGlue(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.condition_on_recycle_remaining = True
            self.init_repr_factory = _Factory([_StaticFeat("s_ir"), _DynEmb(8, 5, "tok_ir")], 27 + 5, 32)
            self.cond_factory = _Factory([_StaticFeat("s_cf"), _DynEmb(6, 4, "tok_cf"), _RemFeat(4)], 24 + 4 + 4, 32)
            self.transition_c_1 = _Transition(32, 64)
            self.transition_c_2 = _Transition(32, 64)

    torch.manual_seed(55)
    glue = _ToyGlue()
    world, n_tok = 4, 32
    dev = _FakeMeshDevice(2, 2)
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

    split = FactoryStaticSplit(glue)
    tracer = SPMDMeshTracer(dev, enabled=True, trace_win_pair=False)
    dwg_tr = DeviceWinnerGlue(glue, split, dev, tracer=tracer)
    dwg_eg = DeviceWinnerGlue(glue, split, dev, tracer=None)

    def _mk_bw(seed):
        g = torch.Generator().manual_seed(seed)
        m = torch.ones((world, n_tok), dtype=torch.bool)
        m[1, 18:] = False
        return {
            "padding_mask": m,
            "s_ir": torch.randn((world, n_tok, 27), generator=g),
            "tok_ir": torch.randint(0, 8, (world, n_tok), generator=g),
            "s_cf": torch.randn((world, n_tok, 24), generator=g),
            "tok_cf": torch.randint(0, 6, (world, n_tok), generator=g),
        }

    # Step 0: captures DeviceWinnerGlue paired trace
    bw0 = _mk_bw(101)
    mk0 = put_sharded(bw0["padding_mask"].float()[..., None], dev, dtype=mod.float32, layout=mod.TILE_LAYOUT)
    ds0 = put_sharded(torch.randn(world, n_tok, 32), dev, dtype=mod.float32, layout=mod.TILE_LAYOUT)
    dc0 = put_sharded(torch.randn(world, n_tok, 32), dev, dtype=mod.float32, layout=mod.TILE_LAYOUT)
    s0, c0, sv0 = dwg_tr.forward(bw0, [0] * world, 1, prefetched=None, m1_tok=mk0, recycle_remaining=1,
                                 trunk_dtype=mod.float32, dev_masters=dev_masters)
    g0 = dwg_tr.backward(ds0, dc0, sv0, dev_masters)
    assert tracer.stats["captures"] == 2 and tracer.stats["refused"] == 0
    for t in (s0, c0, mk0, ds0, dc0, *g0.values()):
        mod.deallocate(t)

    # Mutate all 6 dev_masters IN-PLACE to prove Step 1 trace replay reads LIVE weights!
    for nm, dm in dev_masters.items():
        upd = dev.make_tensor([s + 0.05 for s in dm._buf.shards])
        mod.copy(upd, dm)
        mod.deallocate(upd)

    # Step 1: replays trace on brand-new inputs + mutated weights and matches eager exact!
    bw1 = _mk_bw(202)
    mk1 = put_sharded(bw1["padding_mask"].float()[..., None], dev, dtype=mod.float32, layout=mod.TILE_LAYOUT)
    ds1 = put_sharded(torch.randn(world, n_tok, 32), dev, dtype=mod.float32, layout=mod.TILE_LAYOUT)
    dc1 = put_sharded(torch.randn(world, n_tok, 32), dev, dtype=mod.float32, layout=mod.TILE_LAYOUT)
    glue.zero_grad(set_to_none=True)
    s_tr, c_tr, sv_tr = dwg_tr.forward(bw1, [0] * world, 1, prefetched=None, m1_tok=mk1, recycle_remaining=2,
                                       trunk_dtype=mod.float32, dev_masters=dev_masters)
    g_tr = dwg_tr.backward(ds1, dc1, sv_tr, dev_masters)
    emb_tr = {nm: p.grad.detach().clone() for nm, p in glue.named_parameters() if p.grad is not None}

    glue.zero_grad(set_to_none=True)
    s_eg, c_eg, sv_eg = dwg_eg.forward(bw1, [0] * world, 1, prefetched=None, m1_tok=mk1, recycle_remaining=2,
                                       trunk_dtype=mod.float32, dev_masters=dev_masters)
    g_eg = dwg_eg.backward(ds1, dc1, sv_eg, dev_masters)
    emb_eg = {nm: p.grad.detach().clone() for nm, p in glue.named_parameters() if p.grad is not None}

    assert tracer.stats["win_glue_replay"] == 1
    assert float((read_sharded(s_tr, dev) - read_sharded(s_eg, dev)).abs().max()) == 0.0
    assert float((read_sharded(c_tr, dev) - read_sharded(c_eg, dev)).abs().max()) == 0.0
    for k in g_tr:
        assert float((read_sharded(g_tr[k], dev) - read_sharded(g_eg[k], dev)).abs().max()) == 0.0
    for k in emb_tr:
        assert float((emb_tr[k] - emb_eg[k]).abs().max()) == 0.0
    for t in (s_tr, c_tr, s_eg, c_eg, mk1, ds1, dc1, *g_tr.values(), *g_eg.values()):
        mod.deallocate(t)

    # Now test SPMDMeshTracer head_pair capture + replay into disjoint _head_scratch!
    class _ToyDeviceHeads:
        def __init__(self):
            self.ca_norm_w = dev.make_tensor([torch.ones(1, 32) for _ in range(world)])
            self.ca_w = dev.make_tensor([torch.randn(32, 3) for _ in range(world)])
            self.ll_norm_w = None
            self.ll_w = None
            self.seq_norm_w = None
            self.seq_w = None
            self.ins_q_w = None
            self.ins_k_w = None

        def __call__(self, s_leaf, batch, *, insertion=True, sequence=True, masks_dev=None):
            eca, ell, pm = masks_dev
            ca_m = mod.matmul(s_leaf.value, self.ca_w)
            ca = _FakeAgT(mod.multiply(ca_m, eca), requires_grad=True)
            mod.deallocate(ca_m)
            ll = _FakeAgT(mod.multiply(s_leaf.value, ell), requires_grad=True)
            ins = _FakeAgT(mod.matmul(s_leaf.value, s_leaf.value, transpose_b=True), requires_grad=True)
            out = {"bb_ca": {"v": ca}, "local_latents": {"v": ll}, "insertion": {"logits": ins}}
            if sequence:
                out["sequence"] = {"logits": _FakeAgT(mod.multiply(s_leaf.value, pm), requires_grad=True)}
            self._last_s = s_leaf
            return out

    dh = _ToyDeviceHeads()

    def _fake_ag_bwd(outs, seeds):
        # Compute analytical grad on s_leaf: d_ca @ W^T + d_ll
        ds = mod.matmul(seeds[0], dh.ca_w, transpose_b=True)
        ds2 = mod.add(ds, seeds[1])
        mod.deallocate(ds)
        dh._last_s.grad = ds2

    monkeypatch.setattr(fake_ag, "backward", _fake_ag_bwd, raising=False)

    class _ToyFolds:
        def tape_backward(self, dm):
            return {"glue.ca_linear.1.weight": mod.clone(dh.ca_w)}, []

    s_in0 = put_sharded(torch.randn(world, n_tok, 32), dev, dtype=mod.float32, layout=mod.TILE_LAYOUT)
    m_in0 = put_sharded(torch.ones(world, n_tok, 1), dev, dtype=mod.float32, layout=mod.TILE_LAYOUT)
    dca0 = put_sharded(torch.randn(world, n_tok, 3), dev, dtype=mod.float32, layout=mod.TILE_LAYOUT)
    dll0 = put_sharded(torch.randn(world, n_tok, 32), dev, dtype=mod.float32, layout=mod.TILE_LAYOUT)
    dseq0 = put_sharded(torch.randn(world, n_tok, 32), dev, dtype=mod.float32, layout=mod.TILE_LAYOUT)
    dins0 = put_sharded(torch.randn(world, n_tok, n_tok), dev, dtype=mod.float32, layout=mod.TILE_LAYOUT)
    _t1 = mod.matmul(s_in0, dh.ca_w)
    ca_r = mod.multiply(_t1, m_in0)
    mod.deallocate(_t1)
    ll_r = mod.multiply(s_in0, m_in0)
    seq_r = mod.multiply(s_in0, m_in0)
    ins_r = mod.matmul(s_in0, s_in0, transpose_b=True)
    _t2 = mod.matmul(dca0, dh.ca_w, transpose_b=True)
    ds_r = mod.add(_t2, dll0)
    mod.deallocate(_t2)
    hg_r = {"glue.ca_linear.1.weight": mod.clone(dh.ca_w)}

    hkey = tracer.head_pair_key(s_in0, dh, True)
    assert tracer.head_pair_should_capture(hkey)
    ok = tracer.capture_head_pair(
        hkey, dh=dh, dfolds=_ToyFolds(), dev_masters=dev_masters,
        s_val=s_in0, emit_ca=m_in0, emit_ll=m_in0, pad_m=m_in0,
        ca_ref=ca_r, ll_ref=ll_r, seq_ref=seq_r, ins_ref=ins_r,
        d_ca_ref=dca0, d_ll_ref=dll0, d_seq_ref=dseq0, d_ins_ref=dins0,
        ds_ref=ds_r, head_grads_ref=hg_r, run_seq=True,
    )
    assert ok and tracer.has_head_pair(hkey)
    assert tracer._head_scratch is not None and tracer._head_scratch is not tracer._scratch

    # Mutate dh.ca_w IN-PLACE and replay on new input s_in1 -> matches exact!
    w_upd = dev.make_tensor([s * 1.3 for s in dh.ca_w._buf.shards])
    mod.copy(w_upd, dh.ca_w)
    mod.deallocate(w_upd)
    s_in1 = put_sharded(torch.randn(world, n_tok, 32), dev, dtype=mod.float32, layout=mod.TILE_LAYOUT)
    ca_o, ll_o, seq_o, ins_o = tracer.replay_head_pair_fwd(hkey, s_in1, m_in0, m_in0, m_in0)
    ds_o, hg_o = tracer.replay_head_pair_bwd(hkey, dca0, dll0, dseq0, dins0)
    assert tracer.stats["head_pair_replay"] == 1
    _t3 = mod.matmul(s_in1, dh.ca_w)
    ca_exp = mod.multiply(_t3, m_in0)
    mod.deallocate(_t3)
    _t4 = mod.matmul(dca0, dh.ca_w, transpose_b=True)
    ds_exp = mod.add(_t4, dll0)
    mod.deallocate(_t4)
    assert float((read_sharded(ca_o, dev) - read_sharded(ca_exp, dev)).abs().max()) == 0.0
    assert float((read_sharded(ds_o, dev) - read_sharded(ds_exp, dev)).abs().max()) == 0.0
    assert float((read_sharded(hg_o["glue.ca_linear.1.weight"], dev) - read_sharded(dh.ca_w, dev)).abs().max()) == 0.0
    # Regression guard: replay_win_pair_bwd_fold must NOT deallocate tracer-owned head_pair ds_out slot!
    assert id(ds_o) in tracer._owned_ids
    win_ds_slot = mod.clone(ds_o)
    tracer._win_traces[("fake_win",)] = {
        "tid_bwd": tracer._head_traces[hkey]["tid_bwd"],
        "ds_out": win_ds_slot, "d_s_i": None, "d_z_in": None, "d_c": [], "dev_grads": {}, "idle": [],
    }
    tracer.replay_win_pair_bwd_fold(("fake_win",), ds_o)
    assert ds_o.is_allocated(), "replay_win_pair_bwd_fold freed borrowed head_pair ds_out slot!"
    mod.deallocate(win_ds_slot)
    tracer._win_traces.pop(("fake_win",))

    # Clean up everything and verify ZERO leaked device buffers!
    for t in (
        s_in0, s_in1, m_in0, dca0, dll0, dseq0, dins0, ca_r, ll_r, seq_r, ins_r, ds_r,
        ca_exp, ds_exp, *hg_r.values(), dh.ca_norm_w, dh.ca_w, *dev_masters.values(),
    ):
        mod.deallocate(t)
    dwg_tr.release()
    dwg_eg.release()
    tracer.release()
    assert len(dev.live_bufs) == 0


def test_spmd_zero1_traced_opt_body_and_ag_unpack_bit_exact(monkeypatch):
    """Commit 5: SPMDZero1Step Trace A (opt_body) + Trace B (ag_unpack) match eager bit-exact across
    6 steps (varying LR, clip < 1 and clip == 1, EMA, partial-disabled AdamW) AND --save-at 1/resume."""
    import sys
    import types
    import tt_bio

    mod = _install_mock_ttnn(monkeypatch)

    # Realistic tt-metal capture semantics: begin_trace_capture records without mutating DRAM;
    # execute_trace executes the recorded closure once on hardware.
    trace_fns: dict[int, Any] = {}
    next_tid = [1]

    class _RecordingScratch:
        def __init__(self, dev):
            self.dev = dev
            self.captures = 0
            self.released = False

        def capture(self, need, size, runs, settle=None):
            tids = []
            try:
                for fn in runs:
                    tid = next_tid[0]
                    next_tid[0] += 1
                    trace_fns[tid] = fn
                    tids.append(tid)
                self.captures += len(tids)
            finally:
                if settle is not None:
                    settle()
            return tids

        def release(self):
            self.released = True

    def _mock_copy(src, dst):
        for s_src, s_dst in zip(src._buf.shards, dst._buf.shards):
            s_dst.copy_(s_src.reshape(dst.shape))

    def _mock_typecast(t, dt):
        if dt == mod.bfloat16:
            return t.device().make_tensor([s.reshape(t.shape).to(torch.bfloat16).to(torch.float32) for s in t._buf.shards])
        return t.device().make_tensor([s.reshape(t.shape).to(torch.float32) for s in t._buf.shards])

    def _mock_matmul(a, b, transpose_a=False, transpose_b=False, dtype=None, compute_kernel_config=None, core_grid=None):
        out = []
        for sa, sb in zip(a._buf.shards, b._buf.shards):
            xa = sa.reshape(a.shape).to(torch.float32)
            xb = sb.reshape(b.shape).to(torch.float32)
            if transpose_a:
                xa = xa.transpose(-2, -1)
            if transpose_b:
                xb = xb.transpose(-2, -1)
            out.append(torch.matmul(xa, xb))
        return a.device().make_tensor(out)

    orig_add = mod.add
    monkeypatch.setattr(
        mod, "add",
        lambda a, b: orig_add(a, b) if hasattr(b, "_buf") else a.device().make_tensor([s.reshape(a.shape) + float(b) for s in a._buf.shards]),
        raising=False,
    )
    monkeypatch.setattr(mod, "copy", _mock_copy, raising=False)
    monkeypatch.setattr(mod, "typecast", _mock_typecast, raising=False)
    monkeypatch.setattr(mod, "matmul", _mock_matmul, raising=False)
    monkeypatch.setattr(mod, "execute_trace", lambda dev, tid, cq_id=0, blocking=False: trace_fns[tid](), raising=False)
    monkeypatch.setattr(mod, "release_trace", lambda dev, tid: trace_fns.pop(tid, None), raising=False)
    monkeypatch.setattr(mod, "synchronize_device", lambda dev: None, raising=False)
    monkeypatch.setattr(mod, "Tensor", _FakeMeshTensor, raising=False)
    monkeypatch.setattr(mod, "square", lambda a: a.device().make_tensor([s.reshape(a.shape) ** 2 for s in a._buf.shards]), raising=False)
    monkeypatch.setattr(mod, "sqrt", lambda a: a.device().make_tensor([torch.sqrt(s.reshape(a.shape)) for s in a._buf.shards]), raising=False)
    monkeypatch.setattr(mod, "rsqrt", lambda a: a.device().make_tensor([torch.rsqrt(s.reshape(a.shape)) for s in a._buf.shards]), raising=False)
    monkeypatch.setattr(mod, "clamp", lambda a, min=None, max=None: a.device().make_tensor([torch.clamp(s.reshape(a.shape), min=min, max=max) for s in a._buf.shards]), raising=False)
    monkeypatch.setattr(mod, "reciprocal", lambda a: a.device().make_tensor([torch.reciprocal(s.reshape(a.shape)) for s in a._buf.shards]), raising=False)
    monkeypatch.setattr(
        mod, "divide",
        lambda a, b: a.device().make_tensor(
            [sa.reshape(a.shape) / sb.reshape(b.shape) for sa, sb in zip(a._buf.shards, b._buf.shards)]
            if hasattr(b, "_buf") else [sa.reshape(a.shape) / float(b) for sa in a._buf.shards]
        ),
        raising=False,
    )
    monkeypatch.setattr(
        mod, "sum",
        lambda a, dim=-1, keepdim=False, compute_kernel_config=None: a.device().make_tensor(
            [s.reshape(a.shape).sum(dim=dim, keepdim=keepdim) for s in a._buf.shards]
        ),
        raising=False,
    )
    monkeypatch.setattr(mod, "MathFidelity", types.SimpleNamespace(HiFi4="HiFi4", HiFi2="HiFi2"), raising=False)
    monkeypatch.setattr(mod, "types", types.SimpleNamespace(BlackholeComputeKernelConfig=lambda **k: None), raising=False)
    monkeypatch.setattr(mod, "init_device_compute_kernel_config", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(mod, "WormholeComputeKernelConfig", lambda *a, **k: None, raising=False)

    fake_tt = types.ModuleType("tt_bio.tenstorrent")
    fake_tt.TraceScratch = _RecordingScratch
    monkeypatch.setitem(sys.modules, "tt_bio.tenstorrent", fake_tt)
    monkeypatch.setattr(tt_bio, "tenstorrent", fake_tt, raising=False)

    from tt_bio.train.device_optim import DeviceOptimizer
    from tt_bio.train.optim import Muon
    from tt_bio.train.spmd_rfd4 import SPMDZero1Step
    from tt_bio.train.spmd_trace import SPMDMeshTracer

    rng = np.random.default_rng(77)
    init_w = {
        # 4 sharded Muon matrices (2x2 mesh -> rm=2 -> 2 per rank!)
        "sh_a0": rng.standard_normal((32, 64)).astype(np.float32) * 0.05,
        "sh_a1": rng.standard_normal((32, 64)).astype(np.float32) * 0.05,
        "sh_b0": rng.standard_normal((64, 64)).astype(np.float32) * 0.05,
        "sh_b1": rng.standard_normal((64, 64)).astype(np.float32) * 0.05,
        # 1 unsharded Muon matrix (non-tile-aligned width -> goes to rem_muon!)
        "rem_mu0": rng.standard_normal((32, 24)).astype(np.float32) * 0.05,
        # 2 AdamW vectors in same shape bucket (1 will be disabled -> tests partial-enabled mask cache!)
        "ad_v0": rng.standard_normal((32,)).astype(np.float32) * 0.1,
        "ad_v1": rng.standard_normal((32,)).astype(np.float32) * 0.1,
        # 1 AdamW matrix (all enabled)
        "ad_m0": rng.standard_normal((16, 16)).astype(np.float32) * 0.1,
    }
    disabled = ("ad_v1",)

    def _mk_zero1(dev_cur):
        opt = DeviceOptimizer(
            init_w,
            dev_cur,
            lr=1e-2,
            weight_decay=0.05,
            clip_norm=0.5,
            schedule=lambda st: 1e-2 * (0.25 * st if st <= 4 else 1.0),
            ema_decay=0.95,
            muon=Muon(
                exclude=("ad_.*",),
                weight_decay=0.02,
                ns_steps=2,
                schedule=lambda st: 1.5e-2 * (0.25 * st if st <= 4 else 1.0),
            ),
            ns_mode="fp32",
        )
        return SPMDZero1Step(opt, init_w, dev_cur, full_shape=(2, 2), use_zero1_muon=True)

    dev_eg = _FakeMeshDevice(2, 2)
    dev_tr = _FakeMeshDevice(2, 2)
    z_eg = _mk_zero1(dev_eg)
    z_tr = _mk_zero1(dev_tr)
    tracer = SPMDMeshTracer(dev_tr, enabled=True)
    # Mark hole ready (as after win_pair Step 1 capture on real hardware).
    tracer._hole_ready = True
    tracer._scratch = _RecordingScratch(dev_tr)

    # Pre-generate 6 steps of deterministic sharded gradients (some large to trigger clip < 1, some tiny for clip == 1).
    step_grads = []
    for st in range(6):
        scale = 3.0 if st in (0, 2, 4) else 0.005
        step_grads.append({
            nm: [rng.standard_normal(arr.shape).astype(np.float32) * scale for _ in range(4)]
            for nm, arr in init_w.items()
        })

    def _put_grads(dev_cur, gmap):
        return {nm: dev_cur.make_tensor([torch.from_numpy(a.copy()) for a in shs]) for nm, shs in gmap.items()}

    ckpt_masters = None
    ckpt_fams = None
    ckpt_sd = None
    for st in range(6):
        g_eg = _put_grads(dev_eg, step_grads[st])
        g_tr = _put_grads(dev_tr, step_grads[st])
        r_eg = z_eg.step(g_eg, disabled=disabled, free_grads=True, in_place_masters=True, tracer=None)
        r_tr = z_tr.step(g_tr, disabled=disabled, free_grads=True, in_place_masters=True, tracer=tracer)
        assert r_eg["clip_muon"] == pytest.approx(r_tr["clip_muon"], abs=0.0, rel=0.0)
        assert r_eg["clip_adam"] == pytest.approx(r_tr["clip_adam"], abs=0.0, rel=0.0)
        assert z_eg.master_digest() == z_tr.master_digest(), f"step {st} digest mismatch"
        # Verify EMA shadow parity too!
        f_eg = z_eg.state_families()
        f_tr = z_tr.state_families()
        for fam in ("exp_avg", "exp_avg_sq", "momentum_buffer", "ema_shadow"):
            for nm in f_eg[fam]:
                diff = float(np.max(np.abs(f_eg[fam][nm] - f_tr[fam][nm])))
                assert diff <= 1e-6, f"step {st} fam={fam} param={nm} diff={diff}"
        if st == 0:
            ckpt_masters = z_tr.host_master()
            ckpt_fams = z_tr.state_families()
            ckpt_sd = z_tr.base_opt.state_dict()

    # Steps: 0=eager(init), 1=warm, 2=capture+replay, 3/4/5=replay -> 4 opt_body replays;
    # ag_unpack: 0=warm, 1=capture+replay, 2/3/4/5=replay -> 5 ag_unpack replays; 0 refused!
    assert tracer.stats["opt_body_replay"] == 4
    assert tracer.stats["ag_unpack_replay"] == 5
    assert tracer.stats["refused"] == 0
    assert z_tr.base_opt.ema.n_averaged == 6

    # Verify --save-at 1 / --resume produces BIT-EXACT identical master_digest after Steps 1..5!
    dev_res = _FakeMeshDevice(2, 2)
    z_res = _mk_zero1(dev_res)
    z_res.base_opt.load_state_dict(ckpt_sd)
    z_res.load_host_state(ckpt_masters, ckpt_fams)
    tracer_res = SPMDMeshTracer(dev_res, enabled=True)
    tracer_res._hole_ready = True
    tracer_res._scratch = _RecordingScratch(dev_res)
    for st in range(1, 6):
        g_res = _put_grads(dev_res, step_grads[st])
        z_res.step(g_res, disabled=disabled, free_grads=True, in_place_masters=True, tracer=tracer_res)
    assert z_res.master_digest() == z_eg.master_digest()
    assert tracer_res.stats["opt_body_replay"] == 4
    assert tracer_res.stats["refused"] == 0

    # Clean up and verify ZERO leaked device buffers on all three devices!
    from tt_bio.train import device_optim as _do

    for z_cur, tr_cur, d_cur in ((z_eg, None, dev_eg), (z_tr, tracer, dev_tr), (z_res, tracer_res, dev_res)):
        if tr_cur is not None:
            tr_cur.release()
        z_cur.deallocate()
        z_cur.base_opt.deallocate()
        for cache in (_do._ONES_COL_CACHE, _do._ONES_ROW_CACHE):
            for k in [ck for ck in list(cache) if ck[0] == id(d_cur)]:
                t = cache.pop(k)
                if t.is_allocated():
                    mod.deallocate(t)
        assert len(d_cur.live_bufs) == 0


def test_spmd_zero1_traced_opt_body_8x4_padded_buckets(monkeypatch):
    """D125 #1 regression: on 8x4 (rm=8), k=14->k_pad=16 and k=7->k_pad=8 ALWAYS have padded
    trailing slots (flags[i]=False for i>=k). Trace A (opt_body) MUST fire and match eager bit-exact."""
    import sys
    import types
    import tt_bio

    mod = _install_mock_ttnn(monkeypatch)
    trace_fns: dict[int, Any] = {}
    next_tid = [1]

    class _RecScratch:
        def __init__(self, dev):
            self.dev = dev

        def capture(self, need, size, runs, settle=None):
            tids = []
            try:
                for fn in runs:
                    tid = next_tid[0]
                    next_tid[0] += 1
                    trace_fns[tid] = fn
                    tids.append(tid)
            finally:
                if settle is not None:
                    settle()
            return tids

        def release(self):
            pass

    def _mock_copy(src, dst):
        for s_src, s_dst in zip(src._buf.shards, dst._buf.shards):
            s_dst.copy_(s_src.reshape(dst.shape))

    def _mock_matmul(a, b, transpose_a=False, transpose_b=False, dtype=None, compute_kernel_config=None, core_grid=None):
        out = []
        for sa, sb in zip(a._buf.shards, b._buf.shards):
            xa = sa.reshape(a.shape).to(torch.float32)
            xb = sb.reshape(b.shape).to(torch.float32)
            if transpose_a:
                xa = xa.transpose(-2, -1)
            if transpose_b:
                xb = xb.transpose(-2, -1)
            out.append(torch.matmul(xa, xb))
        return a.device().make_tensor(out)

    orig_add = mod.add
    monkeypatch.setattr(
        mod, "add",
        lambda a, b: orig_add(a, b) if hasattr(b, "_buf") else a.device().make_tensor([s.reshape(a.shape) + float(b) for s in a._buf.shards]),
        raising=False,
    )
    monkeypatch.setattr(mod, "copy", _mock_copy, raising=False)
    monkeypatch.setattr(
        mod, "typecast",
        lambda t, dt: t.device().make_tensor([
            s.reshape(t.shape).to(torch.bfloat16).to(torch.float32) if dt == mod.bfloat16 else s.reshape(t.shape).to(torch.float32)
            for s in t._buf.shards
        ]),
        raising=False,
    )
    monkeypatch.setattr(mod, "matmul", _mock_matmul, raising=False)
    monkeypatch.setattr(mod, "execute_trace", lambda dev, tid, cq_id=0, blocking=False: trace_fns[tid](), raising=False)
    monkeypatch.setattr(mod, "release_trace", lambda dev, tid: trace_fns.pop(tid, None), raising=False)
    monkeypatch.setattr(mod, "synchronize_device", lambda dev: None, raising=False)
    monkeypatch.setattr(mod, "Tensor", _FakeMeshTensor, raising=False)
    monkeypatch.setattr(mod, "square", lambda a: a.device().make_tensor([s.reshape(a.shape) ** 2 for s in a._buf.shards]), raising=False)
    monkeypatch.setattr(mod, "sqrt", lambda a: a.device().make_tensor([torch.sqrt(s.reshape(a.shape)) for s in a._buf.shards]), raising=False)
    monkeypatch.setattr(mod, "rsqrt", lambda a: a.device().make_tensor([torch.rsqrt(s.reshape(a.shape)) for s in a._buf.shards]), raising=False)
    monkeypatch.setattr(mod, "clamp", lambda a, min=None, max=None: a.device().make_tensor([torch.clamp(s.reshape(a.shape), min=min, max=max) for s in a._buf.shards]), raising=False)
    monkeypatch.setattr(mod, "reciprocal", lambda a: a.device().make_tensor([torch.reciprocal(s.reshape(a.shape)) for s in a._buf.shards]), raising=False)
    monkeypatch.setattr(
        mod, "divide",
        lambda a, b: a.device().make_tensor(
            [sa.reshape(a.shape) / sb.reshape(b.shape) for sa, sb in zip(a._buf.shards, b._buf.shards)]
            if hasattr(b, "_buf") else [sa.reshape(a.shape) / float(b) for sa in a._buf.shards]
        ),
        raising=False,
    )
    monkeypatch.setattr(
        mod, "sum",
        lambda a, dim=-1, keepdim=False, compute_kernel_config=None: a.device().make_tensor(
            [s.reshape(a.shape).sum(dim=dim, keepdim=keepdim) for s in a._buf.shards]
        ),
        raising=False,
    )
    monkeypatch.setattr(mod, "MathFidelity", types.SimpleNamespace(HiFi4="HiFi4", HiFi2="HiFi2"), raising=False)
    monkeypatch.setattr(mod, "types", types.SimpleNamespace(BlackholeComputeKernelConfig=lambda **k: None), raising=False)
    monkeypatch.setattr(mod, "init_device_compute_kernel_config", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(mod, "WormholeComputeKernelConfig", lambda *a, **k: None, raising=False)

    fake_tt = types.ModuleType("tt_bio.tenstorrent")
    fake_tt.TraceScratch = _RecScratch
    monkeypatch.setitem(sys.modules, "tt_bio.tenstorrent", fake_tt)
    monkeypatch.setattr(tt_bio, "tenstorrent", fake_tt, raising=False)

    from tt_bio.train.device_optim import DeviceOptimizer
    from tt_bio.train.optim import Muon
    from tt_bio.train.spmd_rfd4 import SPMDZero1Step
    from tt_bio.train.spmd_trace import SPMDMeshTracer

    rng = np.random.default_rng(125)
    init_w = {}
    # 14 matrices of (32, 64) -> on 8x4 (rm=8): k=14, k_pad=16, k_shard=2 (trailing 2 padded!)
    for i in range(14):
        init_w[f"blk14_{i}"] = rng.standard_normal((32, 64)).astype(np.float32) * 0.05
    # 7 matrices of (64, 64) -> on 8x4 (rm=8): k=7, k_pad=8, k_shard=1 (trailing 1 padded!)
    for i in range(7):
        init_w[f"blk7_{i}"] = rng.standard_normal((64, 64)).astype(np.float32) * 0.05
    init_w["ad_v0"] = rng.standard_normal((32,)).astype(np.float32) * 0.1

    def _mk(d):
        opt = DeviceOptimizer(
            init_w, d, lr=1e-2, weight_decay=0.05, clip_norm=0.5, ema_decay=0.95,
            muon=Muon(exclude=("ad_.*",), weight_decay=0.02, ns_steps=2), ns_mode="fp32",
        )
        return SPMDZero1Step(opt, init_w, d, full_shape=(8, 4), use_zero1_muon=True)

    dev_eg = _FakeMeshDevice(8, 4)
    dev_tr = _FakeMeshDevice(8, 4)
    z_eg = _mk(dev_eg)
    z_tr = _mk(dev_tr)
    # Confirm k < k_pad on both buckets (the exact 8x4 geometry!)
    slots = [s for wg in z_tr.muon_plan.width_groups for s in wg.buckets]
    assert sorted((s.k, s.k_pad, s.k_shard) for s in slots) == [(7, 8, 1), (14, 16, 2)]

    tracer = SPMDMeshTracer(dev_tr, enabled=True)
    tracer._hole_ready = True
    tracer._scratch = _RecScratch(dev_tr)

    for st in range(4):
        sc = 2.0 if st % 2 == 0 else 0.01
        gmap = {nm: [rng.standard_normal(arr.shape).astype(np.float32) * sc for _ in range(32)] for nm, arr in init_w.items()}
        g_eg = {nm: dev_eg.make_tensor([torch.from_numpy(a.copy()) for a in shs]) for nm, shs in gmap.items()}
        g_tr = {nm: dev_tr.make_tensor([torch.from_numpy(a.copy()) for a in shs]) for nm, shs in gmap.items()}
        z_eg.step(g_eg, free_grads=True, in_place_masters=True, tracer=None)
        z_tr.step(g_tr, free_grads=True, in_place_masters=True, tracer=tracer)
        assert z_eg.master_digest() == z_tr.master_digest(), f"8x4 padded step {st} mismatch"

    assert tracer.stats["opt_body_replay"] == 2
    assert tracer.stats["ag_unpack_replay"] == 3
    assert tracer.stats["refused"] == 0

    from tt_bio.train import device_optim as _do
    for z_cur, tr_cur, d_cur in ((z_eg, None, dev_eg), (z_tr, tracer, dev_tr)):
        if tr_cur is not None:
            tr_cur.release()
        z_cur.deallocate()
        z_cur.base_opt.deallocate()
        for cache in (_do._ONES_COL_CACHE, _do._ONES_ROW_CACHE):
            for k in [ck for ck in list(cache) if ck[0] == id(d_cur)]:
                t = cache.pop(k)
                if t.is_allocated():
                    mod.deallocate(t)
        assert len(d_cur.live_bufs) == 0




