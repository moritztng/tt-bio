#!/usr/bin/env python3
"""Profile + parity-grade O1 (DeviceOptimizer fast NS + ZeRO-1) and F1 (DeviceFolds) in ONE device open.

Usage:
    TT_VISIBLE_DEVICES=8 python3 perf/r4t_perf/profile_o1_f1.py --steps 20 --out ~/r4t/perf/o1_f1.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tt_bio.main import ensure_p300_mesh_descriptor  # noqa: E402

ensure_p300_mesh_descriptor()

import numpy as np  # noqa: E402
import torch  # noqa: E402
import ttnn  # noqa: E402

from tt_bio.rfd4 import engine as E, network as N  # noqa: E402
from tt_bio.train.device_optim import (  # noqa: E402
    DeviceOptimizer,
    newton_schulz_device,
    rfd4_device_optimizer,
)
from tt_bio.train.rfd4_objective import MUON_EXCLUDE, UPDATE_RULE  # noqa: E402
from tt_bio.train.optim import Muon  # noqa: E402


def _rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    a64 = np.asarray(a, dtype=np.float64).ravel()
    b64 = np.asarray(b, dtype=np.float64).ravel()
    den = float(np.linalg.norm(b64))
    num = float(np.linalg.norm(a64 - b64))
    return num / den if den > 0.0 else num


class Float64MuonAdamEMA:
    """Float64 host reference matching DeviceOptimizer / rfd4_objective.update_rule."""

    def __init__(self, init_w: dict[str, np.ndarray], *, lr=1e-4, betas=(0.9, 0.999),
                 eps=1e-8, wd=0.0, clip=1.0, ns_steps=5, ema_decay=0.999):
        self.master = {k: np.asarray(v, dtype=np.float64).copy() for k, v in init_w.items()}
        self.mu = Muon(exclude=MUON_EXCLUDE, adjust_lr_fn="match_rms_adamw", ns_steps=ns_steps)
        self.muon_names = frozenset(k for k, v in self.master.items() if self.mu.takes(k, v.shape))
        self.lr, self.b1, self.b2, self.eps, self.wd, self.clip = lr, betas[0], betas[1], eps, wd, clip
        self.ema_decay = ema_decay
        self.m1: dict[str, np.ndarray] = {}
        self.m2: dict[str, np.ndarray] = {}
        self.ema: dict[str, np.ndarray] = {}
        self.steps = 0
        self.n_ema = 0

    def _ns(self, x: np.ndarray) -> np.ndarray:
        a, b, c = self.mu.ns_coefficients
        fl = x.shape[0] > x.shape[1]
        cur = x.T if fl else x.copy()
        nrm = float(np.linalg.norm(cur))
        cur = cur / max(nrm, self.mu.eps)
        for _ in range(self.mu.ns_steps):
            g = cur @ cur.T
            p = b * g + c * (g @ g)
            cur = a * cur + p @ cur
        return cur.T if fl else cur

    def step(self, grads: dict[str, np.ndarray]) -> dict:
        self.steps += 1
        bc1 = 1.0 - self.b1 ** self.steps
        bc2 = 1.0 - self.b2 ** self.steps
        sq_m = sum(float(np.sum(grads[k].astype(np.float64) ** 2)) for k in self.muon_names if k in grads)
        sq_a = sum(float(np.sum(grads[k].astype(np.float64) ** 2)) for k in self.master if k not in self.muon_names and k in grads)
        nm, na = math.sqrt(sq_m), math.sqrt(sq_a)
        cm = min(1.0, self.clip / nm) if nm > 0 else 1.0
        ca = min(1.0, self.clip / na) if na > 0 else 1.0
        om = 1.0 - self.mu.momentum
        from concurrent.futures import ThreadPoolExecutor

        def _do_muon(k: str):
            g = grads.get(k)
            if g is None:
                return
            w = self.master[k]
            ge = g.astype(np.float64) * cm
            buf = (ge * om) if k not in self.m1 else (self.m1[k] + (ge - self.m1[k]) * om)
            self.m1[k] = buf
            u = (buf - (buf - ge) * om) if self.mu.nesterov else buf
            o = self._ns(u)
            w[:] = w * (1.0 - self.lr * self.mu.weight_decay) - self.lr * self.mu.ratio(w.shape) * o

        blas_thr = int(os.environ.get("OPENBLAS_NUM_THREADS") or os.environ.get("OMP_NUM_THREADS") or "1")
        n_thr = 1 if blas_thr > 1 else min(16, max(1, os.cpu_count() or 8))
        if n_thr > 1:
            with ThreadPoolExecutor(max_workers=n_thr) as pool:
                list(pool.map(_do_muon, self.muon_names))
        else:
            for k in self.muon_names:
                _do_muon(k)

        for k, w in self.master.items():
            if k in self.muon_names:
                continue
            g = grads.get(k)
            if g is None:
                continue
            ge = g.astype(np.float64) * ca
            if k not in self.m1:
                self.m1[k] = ge * (1.0 - self.b1)
                self.m2[k] = (ge * ge) * (1.0 - self.b2)
            else:
                self.m1[k] = self.b1 * self.m1[k] + (1.0 - self.b1) * ge
                self.m2[k] = self.b2 * self.m2[k] + (1.0 - self.b2) * (ge * ge)
            mh = self.m1[k] / bc1
            vh = self.m2[k] / bc2
            w[:] = w - self.lr * (mh / (np.sqrt(vh) + self.eps))
        if self.n_ema == 0:
            self.ema = {k: v.copy() for k, v in self.master.items()}
        else:
            ew = 1.0 - self.ema_decay
            for k, v in self.master.items():
                self.ema[k] = self.ema[k] + ew * (v - self.ema[k])
        self.n_ema += 1
        return dict(muon_norm=nm, adam_norm=na)


def _aiclk() -> int:
    try:
        card = int(os.environ.get("TT_VISIBLE_DEVICES", "0").split(",")[0])
        for p in (f"/sys/class/tenstorrent/tenstorrent!{card}/tt_aiclk",
                  f"/sys/class/tenstorrent/tenstorrent{card}/tt_aiclk"):
            if os.path.exists(p):
                return int(open(p).read().strip())
    except Exception:
        pass
    return 0


def profile_subphases(dev, init_np: dict[str, np.ndarray], grads_np: dict[str, np.ndarray], ns_mode: str):
    opt = rfd4_device_optimizer(init_np, dev, lr=1e-4, ema_decay=0.999, ns_mode=ns_mode)
    # Warmup 2 steps
    for _ in range(2):
        dg = {k: ttnn.from_torch(torch.from_numpy(v), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
              for k, v in grads_np.items()}
        opt.step(dg, free_grads=True)
    ttnn.synchronize_device(dev)

    # Now time 3 warm steps decomposed by sub-phase + per Muon bucket
    phases = dict(pack=0.0, norm=0.0, muon_total=0.0, adam_total=0.0, ema=0.0, unpack_muon=0.0, unpack_adam=0.0, step_total=0.0)
    per_bucket: dict[str, float] = {}
    reps = 3
    for _ in range(reps):
        dg = {k: ttnn.from_torch(torch.from_numpy(v), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
              for k, v in grads_np.items()}
        ttnn.synchronize_device(dev)
        t_start = time.perf_counter()

        opt.steps += 1
        opt.beta1_pow *= opt.beta1
        opt.beta2_pow *= opt.beta2
        bc1, bc2 = 1.0 - opt.beta1_pow, 1.0 - opt.beta2_pow
        dis = set()

        t0 = time.perf_counter()
        mp = [(bk, *opt._pack_grads(bk, dg, dis)) for bk in opt._muon_buckets]
        ap = [(bk, *opt._pack_grads(bk, dg, dis)) for bk in opt._adam_buckets]
        for g in dg.values():
            if g is not None and g.is_allocated():
                ttnn.deallocate(g)
        ttnn.synchronize_device(dev)
        phases["pack"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        sqm, sqa, _dead = opt._group_norms_and_dead(mp, ap)
        cm = opt.clip_coef(math.sqrt(max(sqm, 0.0)))
        ca = opt.clip_coef(math.sqrt(max(sqa, 0.0)))
        ttnn.synchronize_device(dev)
        phases["norm"] += time.perf_counter() - t0

        t_m0 = time.perf_counter()
        orig_unpack = opt._unpack_master
        um_acc = [0.0]
        def timed_unpack(bk):
            ttnn.synchronize_device(dev)
            tu = time.perf_counter()
            orig_unpack(bk)
            ttnn.synchronize_device(dev)
            um_acc[0] += time.perf_counter() - tu
        opt._unpack_master = timed_unpack
        for bk, g_b, flags in mp:
            tb = time.perf_counter()
            opt._step_muon_bucket(bk, g_b, flags, cm, opt.lr)
            if g_b is not None:
                ttnn.deallocate(g_b)
            ttnn.synchronize_device(dev)
            k_str = f"{bk.shape}x{bk.k}"
            per_bucket[k_str] = per_bucket.get(k_str, 0.0) + (time.perf_counter() - tb)
        phases["muon_total"] += time.perf_counter() - t_m0
        phases["unpack_muon"] += um_acc[0]

        um_acc[0] = 0.0
        t_a0 = time.perf_counter()
        for bk, g_b, _f in ap:
            opt._step_adam_bucket(bk, g_b, dis, ca, opt.lr, bc1, bc2)
            if g_b is not None:
                ttnn.deallocate(g_b)
        ttnn.synchronize_device(dev)
        phases["adam_total"] += time.perf_counter() - t_a0
        phases["unpack_adam"] += um_acc[0]
        opt._unpack_master = orig_unpack

        t0 = time.perf_counter()
        opt.ema.update()
        ttnn.synchronize_device(dev)
        phases["ema"] += time.perf_counter() - t0
        phases["step_total"] += time.perf_counter() - t_start

    # Also measure clean uninstrumented opt.step() wall time (3 warm steps)
    clean_times = []
    for _ in range(3):
        dg = {k: ttnn.from_torch(torch.from_numpy(v), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
              for k, v in grads_np.items()}
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        opt.step(dg, free_grads=True)
        ttnn.synchronize_device(dev)
        clean_times.append(time.perf_counter() - t0)

    opt.deallocate()
    return {
        "ns_mode": ns_mode,
        "clean_ms": round(1000.0 * float(np.median(clean_times)), 2),
        "clean_all_ms": [round(1000.0 * x, 2) for x in clean_times],
        "phases_ms": {k: round(1000.0 * v / reps, 2) for k, v in phases.items()},
        "muon_buckets_ms": {k: round(1000.0 * v / reps, 2) for k, v in sorted(per_bucket.items(), key=lambda kv: -kv[1])},
    }


def grade_parity(dev, init_np: dict[str, np.ndarray], step_grads: list[dict[str, np.ndarray]],
                 ns_mode: str, *, opt_ref=None, ns_steps: int = 5, betas=(0.9, 0.999)):
    opt_dev = rfd4_device_optimizer(init_np, dev, lr=1e-4, ema_decay=0.999,
                                    ns_mode=ns_mode, ns_steps=ns_steps, betas=betas)
    if opt_ref is None:
        opt_ref = Float64MuonAdamEMA(init_np, lr=1e-4, betas=(0.9, 0.999), ns_steps=5, ema_decay=0.999)
        for g_np in step_grads:
            opt_ref.step(g_np)
    for g_np in step_grads:
        dg = {k: ttnn.from_torch(torch.from_numpy(v), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
              for k, v in g_np.items()}
        opt_dev.step(dg, free_grads=True)
    ttnn.synchronize_device(dev)
    hm = opt_dev.host_master()
    he = opt_dev.ema.host_shadow()
    wm, wm_n = 0.0, ""
    wa, wa_n = 0.0, ""
    we, we_n = 0.0, ""
    for k, ref_w in opt_ref.master.items():
        r = _rel_l2(hm[k], ref_w)
        if k in opt_dev.muon_names:
            if r > wm:
                wm, wm_n = r, k
        else:
            if r > wa:
                wa, wa_n = r, k
        re = _rel_l2(he[k], opt_ref.ema[k])
        if re > we:
            we, we_n = re, k
    opt_dev.deallocate()
    return dict(ns_mode=ns_mode, ns_steps=ns_steps, betas=list(betas),
                worst_muon=wm, worst_muon_name=wm_n,
                worst_adam=wa, worst_adam_name=wa_n,
                worst_ema=we, worst_ema_name=we_n,
                pass_1e4=bool(max(wm, wa, we) <= 1e-4))


def grade_zero1(dev, init_np: dict[str, np.ndarray], step_grads: list[dict[str, np.ndarray]],
                ns_mode: str, shard_world: int = 4, *, opt_ref=None, broken_skip_gather: bool = False):
    """Simulate shard_world ZeRO-1 ranks on one device and compare against Float64 reference."""
    ranks = [
        rfd4_device_optimizer(init_np, dev, lr=1e-4, ema_decay=0.999, ns_mode=ns_mode,
                              shard_rank=r, shard_world=shard_world)
        for r in range(shard_world)
    ]
    # Verify exact partition of all 558 params
    all_owned = set()
    for o in ranks:
        assert all_owned.isdisjoint(o.owned_names), "overlapping ZeRO-1 ownership!"
        all_owned |= o.owned_names
    assert all_owned == set(init_np), f"incomplete ZeRO-1 partition: {len(all_owned)} vs {len(init_np)}"

    if opt_ref is None:
        opt_ref = Float64MuonAdamEMA(init_np, lr=1e-4, betas=(0.9, 0.999), ns_steps=5, ema_decay=0.999)
        for g_np in step_grads:
            opt_ref.step(g_np)
    for g_np in step_grads:
        sq_m_total = 0.0
        sq_a_total = 0.0
        for o in ranks:
            dg = {k: ttnn.from_torch(torch.from_numpy(g_np[k]), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
                  for k in o.owned_names if k in g_np}
            mp = [(bk, *o._pack_grads(bk, dg, set())) for bk in o._muon_buckets]
            ap = [(bk, *o._pack_grads(bk, dg, set())) for bk in o._adam_buckets]
            sm, sa, _ = o._group_norms_and_dead(mp, ap)
            sq_m_total += sm
            sq_a_total += sa
            for _, gb, _ in mp + ap:
                if gb is not None:
                    ttnn.deallocate(gb)
            for g in dg.values():
                ttnn.deallocate(g)

        for o in ranks:
            dg = {k: ttnn.from_torch(torch.from_numpy(g_np[k]), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
                  for k in o.owned_names if k in g_np}
            call_idx = [0]
            def _norm_red(local_sq, _ci=call_idx):
                idx = _ci[0]; _ci[0] += 1
                return sq_m_total if idx == 0 else sq_a_total
            o.step(dg, norm_reduce_fn=_norm_red, free_grads=True)

        # Simulate gather_fn: sync every rank's non-owned master from owner (skipped in broken arm)
        if not broken_skip_gather:
            for name, owner_r in ranks[0].owner_of.items():
                src_t = ranks[owner_r].master[name]
                for r, o in enumerate(ranks):
                    if r != owner_r:
                        old = o.master[name]
                        o.master[name] = ttnn.clone(src_t)
                        if old.is_allocated():
                            ttnn.deallocate(old)

    ttnn.synchronize_device(dev)
    hm0 = ranks[0].host_master()
    # Combine owned EMA shadows across ranks
    he_all = {}
    for o in ranks:
        he_all.update(o.ema.host_shadow())
    worst_w = max(_rel_l2(hm0[k], v) for k, v in opt_ref.master.items())
    worst_e = max(_rel_l2(he_all[k], v) for k, v in opt_ref.ema.items())

    # Also measure single-rank step time for shard_world=32 (1/32 work!)
    for o in ranks:
        o.deallocate()

    opt32 = rfd4_device_optimizer(init_np, dev, lr=1e-4, ema_decay=0.999, ns_mode=ns_mode,
                                  shard_rank=0, shard_world=32)
    t32 = []
    for i in range(4):
        dg = {k: ttnn.from_torch(torch.from_numpy(step_grads[0][k]), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
              for k in opt32.owned_names}
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        opt32.step(dg, free_grads=True)
        ttnn.synchronize_device(dev)
        if i >= 1:
            t32.append(time.perf_counter() - t0)
    n_owned_32 = len(opt32.owned_names)
    opt32.deallocate()
    return dict(
        shard_world=shard_world, broken=broken_skip_gather,
        worst_param=worst_w, worst_ema=worst_e, pass_1e4=bool(max(worst_w, worst_e) <= 1e-4),
        zero1_32_rank0_owned=n_owned_32,
        zero1_32_rank0_ms=round(1000.0 * float(np.median(t32)), 2),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    state, config = N.load_checkpoint(E._resolve_weights(E.DEFAULT_WEIGHTS))
    glue = N.FlowGlue(state, config)
    glue.requires_grad_(True)
    glue_keys = {f"nn.{k}" for k, _ in glue.named_parameters()}
    init_np = {k: v.detach().float().numpy().copy()
               for k, v in state.items() if k.startswith("nn.") and k not in glue_keys}
    for k, p in glue.named_parameters():
        init_np[f"glue.{k}"] = p.detach().float().numpy().copy()

    # Build deterministic near-rank-1 + dense realistic gradients (hardest test for Muon NS per D34!)
    rng = np.random.default_rng(42)
    step_grads = []
    for s in range(args.steps):
        g = {}
        for k, w in init_np.items():
            if w.ndim == 2 and min(w.shape) >= 8:
                # Mix rank-2 structure + dense noise so Muon NS is genuinely stressed (like real RFD4 grads)
                u = rng.standard_normal((w.shape[0], 2)).astype(np.float32) * 0.05
                v = rng.standard_normal((2, w.shape[1])).astype(np.float32) * 0.05
                g[k] = (u @ v + rng.standard_normal(w.shape).astype(np.float32) * 0.005).astype(np.float32)
            else:
                g[k] = (rng.standard_normal(w.shape).astype(np.float32) * 0.02).astype(np.float32)
        step_grads.append(g)

    from tt_bio import tenstorrent as TT
    dev = TT.get_device()
    try:
        clk0 = _aiclk()
        print(f"Opened device, aiclk={clk0} MHz, params={len(init_np)}", flush=True)

        profs = {}
        for mode in ("fp32", "hyb_e1", "split2_fast", "split2"):
            p = profile_subphases(dev, init_np, step_grads[0], mode)
            profs[mode] = p
            print(f"[PROF {mode:13s}] clean={p['clean_ms']:6.1f} ms | phases={p['phases_ms']}", flush=True)
            top4 = list(p["muon_buckets_ms"].items())[:5]
            print(f"    top buckets: {top4}", flush=True)

        # Compute float64 CPU reference ONCE (saves snapshots at step 5 and step args.steps)
        t_ref = time.perf_counter()
        ref_full = Float64MuonAdamEMA(init_np, lr=1e-4, betas=(0.9, 0.999), ns_steps=5, ema_decay=0.999)
        ref_5 = None
        for idx, g_np in enumerate(step_grads):
            ref_full.step(g_np)
            if idx + 1 == min(5, len(step_grads)):
                ref_5 = type("_RefSnap", (), dict(
                    master={k: v.copy() for k, v in ref_full.master.items()},
                    ema={k: v.copy() for k, v in ref_full.ema.items()},
                ))()
        print(f"Float64 ref ({args.steps} steps) computed in {time.perf_counter() - t_ref:.1f} s", flush=True)

        clk1 = _aiclk()
        parities = {}
        for mode in ("fp32", "hyb_e1", "split2_fast"):
            r = grade_parity(dev, init_np, step_grads, mode, opt_ref=ref_full)
            parities[mode] = r
            print(f"[PARITY {mode:12s} {args.steps}st] muon={r['worst_muon']:.3e} ({r['worst_muon_name']}) "
                  f"adam={r['worst_adam']:.3e} ema={r['worst_ema']:.3e} PASS={r['pass_1e4']}", flush=True)

        # Broken arms (must FAIL!)
        brk_ns = grade_parity(dev, init_np, step_grads[:5], "fp32", opt_ref=ref_5, ns_steps=4)
        brk_betas = grade_parity(dev, init_np, step_grads[:5], "fp32", opt_ref=ref_5, betas=(0.999, 0.9))
        print(f"[BROKEN ns=4] muon={brk_ns['worst_muon']:.3e} PASS={brk_ns['pass_1e4']} (expect False)", flush=True)
        print(f"[BROKEN betas_swapped] adam={brk_betas['worst_adam']:.3e} PASS={brk_betas['pass_1e4']} (expect False)", flush=True)

        # ZeRO-1 parity (shard_world=4, 5 steps + broken arm + shard_world=32 timing)
        z1_ok = grade_zero1(dev, init_np, step_grads[:5], "fp32", shard_world=4, opt_ref=ref_5,
                            broken_skip_gather=False)
        z1_brk = grade_zero1(dev, init_np, step_grads[:5], "fp32", shard_world=4, opt_ref=ref_5,
                             broken_skip_gather=True)
        print(f"[ZERO1 w=4 OK] param={z1_ok['worst_param']:.3e} ema={z1_ok['worst_ema']:.3e} PASS={z1_ok['pass_1e4']} | "
              f"w=32 rank0 ({z1_ok['zero1_32_rank0_owned']} params)={z1_ok['zero1_32_rank0_ms']} ms", flush=True)
        print(f"[ZERO1 w=4 BROKEN] param={z1_brk['worst_param']:.3e} PASS={z1_brk['pass_1e4']} (expect False)", flush=True)

        res = dict(
            aiclk=[clk0, clk1, _aiclk()],
            profiles=profs,
            parities=parities,
            broken=dict(ns4=brk_ns, betas_swapped=brk_betas),
            zero1=dict(ok=z1_ok, broken=z1_brk),
        )
        if args.out:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            with open(args.out, "w") as f:
                json.dump(res, f, indent=2)
            print(f"Wrote {args.out}", flush=True)
    finally:
        pass


if __name__ == "__main__":
    main()
