"""Pre-registered 20-step OPT-PARITY, D37 loss trajectory, and BROKEN-ARM grader for DeviceOptimizer.

Pre-registered bar (state/r4t/PLAN.md, D34, D37, D89):
1. Same-grads 20-step parity: starting from the RFD4 checkpoint on 20 real RFD4 steps
   (batch00..08 cycled, real forward+backward+fold_bwd gradients including real near-rank-1
   momentum, depth-1/2/3 cycle-proj zeros, and batch08 insertion), apply the EXACT SAME
   per-step gradients to:
     - DeviceOptimizer on card (Muon + Adam + per-group clip 1.0 + EMA 0.999)
     - Independent Float64MuonAdamEMA host reference (all arithmetic in float64)
   Bar: after 20 steps, EVERY parameter (558/558) AND EVERY EMA shadow (558/558) has
   rel_l2 <= 1e-4 vs float64 host reference.
2. Closed-loop 20-step loss trajectory (D37 tol): run 20 closed-loop steps with
   DeviceOptimizer on card vs 20 closed-loop steps where Float64MuonAdamEMA computes the
   parameter updates on host and pushes them into DeviceFolds.refold; every step's loss must
   stay within D37 tol (<= 1e-2 relative) and agree on winners.
3. Broken arms (same 20-step real grad sequence):
     - broken_ns4: ns_steps=4 (off by 1 from 5) -> MUST FAIL (max_rel_l2 > 1e-4)
     - broken_beta2: betas=(0.9, 0.99) (beta2 swapped from 0.999) -> MUST FAIL (max_rel_l2 > 1e-4)
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
import threading
import time

import numpy as np


def _rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    a64 = np.asarray(a, np.float64).ravel()
    b64 = np.asarray(b, np.float64).ravel()
    den = float(np.linalg.norm(b64))
    num = float(np.linalg.norm(a64 - b64))
    return num if den == 0.0 else num / den


class Float64MuonAdamEMA:
    """Independent float64 host reference for Muon + Adam + per-group clip 1.0 + EMA 0.999.

    Matches `tt_bio.train.rfd4_objective.update_rule` exactly, with all math in float64
    (including float64 Newton-Schulz, per-group clip norm, Adam moments, and EMA shadow).
    """

    def __init__(
        self,
        initial_weights: dict[str, np.ndarray],
        *,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        clip_norm: float = 1.0,
        muon_exclude: tuple[str, ...] = (),
        muon_momentum: float = 0.95,
        muon_nesterov: bool = True,
        muon_ns_coeffs: tuple[float, float, float] = (3.4445, -4.775, 2.0315),
        muon_ns_steps: int = 5,
        muon_eps: float = 1e-7,
        muon_weight_decay: float = 0.1,
        ema_decay: float | None = 0.999,
    ):
        self.lr = float(lr)
        self.b1, self.b2 = float(betas[0]), float(betas[1])
        self.eps = float(eps)
        self.wd = float(weight_decay)
        self.clip_norm = float(clip_norm)
        self.muon_momentum = float(muon_momentum)
        self.muon_nesterov = bool(muon_nesterov)
        self.ns_a, self.ns_b, self.ns_c = (float(x) for x in muon_ns_coeffs)
        self.ns_steps = int(muon_ns_steps)
        self.muon_eps = float(muon_eps)
        self.muon_wd = float(muon_weight_decay)
        self.ema_decay = None if ema_decay is None else float(ema_decay)

        self.master: dict[str, np.ndarray] = {
            k: np.asarray(v, np.float64).copy() for k, v in initial_weights.items()
        }
        self.muon_names: list[str] = []
        self.adam_names: list[str] = []
        for k, w in self.master.items():
            takes = (w.ndim == 2 and min(w.shape) > 0
                     and not any(re.search(p, k) for p in muon_exclude))
            if takes:
                self.muon_names.append(k)
            else:
                self.adam_names.append(k)

        self.muon_buf = {k: np.zeros_like(self.master[k]) for k in self.muon_names}
        self.adam_m = {k: np.zeros_like(self.master[k]) for k in self.adam_names}
        self.adam_v = {k: np.zeros_like(self.master[k]) for k in self.adam_names}
        self.ema = (
            {k: w.copy() for k, w in self.master.items()}
            if self.ema_decay is not None else None
        )
        self.step_count = 0

    def _ns_f64(self, u: np.ndarray) -> np.ndarray:
        flip = u.shape[0] > u.shape[1]
        x = u.T.copy() if flip else u.copy()
        nrm = float(np.linalg.norm(x))
        x = x / max(nrm, self.muon_eps)
        a, b, c = self.ns_a, self.ns_b, self.ns_c
        for _ in range(self.ns_steps):
            gram = x @ x.T
            poly = b * gram + c * (gram @ gram)
            x = a * x + poly @ x
        return x.T if flip else x

    def step(self, grads: dict[str, np.ndarray | None]) -> dict[str, float]:
        self.step_count += 1
        t = self.step_count
        lr = self.lr

        m_sq = 0.0
        for k in self.muon_names:
            g = grads.get(k)
            if g is not None:
                g64 = np.asarray(g, np.float64)
                m_sq += float(np.sum(g64 * g64))
        a_sq = 0.0
        for k in self.adam_names:
            g = grads.get(k)
            if g is not None:
                g64 = np.asarray(g, np.float64)
                a_sq += float(np.sum(g64 * g64))

        m_norm = math.sqrt(m_sq)
        a_norm = math.sqrt(a_sq)
        m_scale = min(1.0, self.clip_norm / (m_norm + 1e-6)) if self.clip_norm > 0 else 1.0
        a_scale = min(1.0, self.clip_norm / (a_norm + 1e-6)) if self.clip_norm > 0 else 1.0

        for k in self.muon_names:
            g = grads.get(k)
            if g is None:
                continue
            g64 = np.asarray(g, np.float64) * m_scale
            buf = self.muon_buf[k]
            buf = self.muon_momentum * buf + (1.0 - self.muon_momentum) * g64
            self.muon_buf[k] = buf
            u = (1.0 - self.muon_momentum) * g64 + self.muon_momentum * buf if self.muon_nesterov else buf
            orth = self._ns_f64(u)
            ratio = 0.2 * math.sqrt(max(self.master[k].shape))
            eff_lr = lr * ratio
            self.master[k] = self.master[k] * (1.0 - lr * self.muon_wd) - eff_lr * orth

        bc1 = 1.0 - self.b1 ** t
        bc2 = 1.0 - self.b2 ** t
        step_size = lr / bc1
        inv_sqrt_bc2 = 1.0 / math.sqrt(bc2)
        for k in self.adam_names:
            g = grads.get(k)
            if g is None:
                continue
            g64 = np.asarray(g, np.float64) * a_scale
            m = self.b1 * self.adam_m[k] + (1.0 - self.b1) * g64
            v = self.b2 * self.adam_v[k] + (1.0 - self.b2) * (g64 * g64)
            self.adam_m[k] = m
            self.adam_v[k] = v
            denom = np.sqrt(v) * inv_sqrt_bc2 + self.eps
            if self.wd != 0.0:
                self.master[k] = self.master[k] * (1.0 - lr * self.wd)
            self.master[k] = self.master[k] - step_size * (m / denom)

        if self.ema is not None:
            d = self.ema_decay
            for k in self.master:
                self.ema[k] = d * self.ema[k] + (1.0 - d) * self.master[k]

        return {"muon_norm": m_norm, "adam_norm": a_norm}


def _aiclk_sampler(node: int):
    clk, stop = [], threading.Event()
    path = f"/sys/class/tenstorrent/tenstorrent!{node}/tt_aiclk"

    def run():
        while not stop.is_set():
            try:
                clk.append(int(open(path).read().split()[0]))
            except Exception:
                pass
            time.sleep(0.05)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return clk, stop, t


def _build_resident_env(dev, state, cfg, glue, n_tok: int):
    import torch
    import ttnn
    from tt_bio import autograd as ag
    from tt_bio.rfd4 import engine as E
    from tt_bio.rfd4.heads import DeviceHeads
    from tt_bio.train.folds import DeviceFolds

    glue.requires_grad_(True)
    state_res = dict(state)
    for gn, gp in glue.named_parameters():
        if gp.requires_grad:
            state_res[f"glue.{gn}"] = gp.detach()

    def trimuls(root):
        seen, out, todo = set(), [], [root]
        while todo:
            o = todo.pop()
            if id(o) in seen:
                continue
            seen.add(id(o))
            if hasattr(o, "prewarm") and hasattr(o, "width_variants"):
                out.append(o)
            if isinstance(o, (list, tuple)):
                todo.extend(o)
            elif isinstance(o, dict):
                todo.extend(o.values())
            elif hasattr(o, "__dict__") and not isinstance(o, (torch.Tensor, ttnn.Tensor, type)):
                todo.extend(v for v in vars(o).values() if not isinstance(v, (torch.Tensor, ttnn.Tensor)))
        return out

    dfolds = DeviceFolds(state_res, dev)
    with dfolds.recording() as sd:
        trunk = E.load_trunk(glue, sd, cfg, training=True, attn_fp32=True)
        for m in trimuls(trunk):
            m.prewarm(n_tok, 1)
        dheads = DeviceHeads(glue, dev, state=sd)
        glue._device_heads = dheads
    dfolds.bind(trunk, dheads)
    for spec in dfolds.specs:
        ag.parameter(spec.handle)
    unreached = set(dfolds.unreached())
    folded_names = [nm for nm in dfolds.leaves if nm not in unreached]
    folded_set = set(folded_names)
    host_glue_params = {
        f"glue.{gn}": gp for gn, gp in glue.named_parameters()
        if gp.requires_grad and f"glue.{gn}" not in folded_set
    }
    init_w = {nm: dfolds.leaves[nm].detach().float().clone() for nm in folded_names}
    init_w.update({nm: p.detach().float().clone() for nm, p in host_glue_params.items()})
    return dfolds, trunk, dheads, host_glue_params, init_w


def _run_one_fwd_bwd(dev, trunk, glue, dfolds, dev_masters, host_glue_params, card_pair, batch_tuple):
    import contextlib
    import torch
    import ttnn
    from tt_bio import autograd as ag, tenstorrent as T
    from tt_bio.rfd4 import trunk as TR
    from tt_bio.rfd4.pair import PairInputs
    from tt_bio.train.bridge import Bridge
    from tt_bio.train.rfd4_heads import device_candidate_scores, device_winner_loss
    from tt_bio.train.rfd4_pair import pair_taped

    _nm, bt, meta, _rec, ref_l = batch_tuple
    K = int(bt["padding_mask"].shape[0])
    depth = max(1, int(meta["depth"]))
    levers = trunk._levers

    def put(t, dtype=ttnn.float32):
        return ttnn.from_torch(t.to(torch.float32).contiguous(), dtype=dtype,
                               layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def take(d, i):
        if isinstance(d, dict):
            return {k: take(v, i) for k, v in d.items()}
        return d[i:i + 1] if torch.is_tensor(d) and d.dim() and d.shape[0] == K else d

    def conds_for(b):
        cs = [glue.conditioning(b, None)]
        for cyc in range(1, depth):
            cs.append(glue.conditioning(b, depth - cyc - 1) if glue.condition_on_recycle_remaining else cs[0])
        return cs

    def tmask(b):
        return TR.trunk_masks(b["padding_mask"].float(), dev, dtype=trunk.dtype, residual_dtype=trunk.residual_dtype)

    @contextlib.contextmanager
    def cand_scope():
        mods = [trunk] + list(trunk.pair_updates.values())
        saved = [(m, m.pair_residual_dtype if m is trunk else m.residual_dtype) for m in mods]
        saved_levers = trunk._levers
        cand_levs = levers | {"pair_bf16z", "dit_sigf32", "tok_recycle_mm"}
        trunk.pair_residual_dtype = ttnn.bfloat16
        for m in mods[1:]:
            m.residual_dtype = ttnn.bfloat16
        trunk._levers = cand_levs
        try:
            with T.lever_scope(cand_levs):
                yield
        finally:
            trunk._levers = saved_levers
            for m, d in saved:
                setattr(m, "pair_residual_dtype" if m is trunk else "residual_dtype", d)

    pc = card_pair[0]
    mk = tmask(bt)
    with torch.no_grad():
        ck = conds_for(bt)
        s0, z0 = glue.initial_states(bt, pair_constant=glue.pair_split.constant(bt))
    with ag.suspended(), cand_scope():
        zk0 = pc(z0, 0)
        sk, zk = trunk(put(s0), zk0, [put(c, trunk.dtype) for c in ck], mk)
        win = device_candidate_scores(glue, sk, bt, K)
    for t in (sk, zk):
        ttnn.deallocate(t)

    bw = take(bt, win)
    br = Bridge(dev, trunk.dtype)
    m1 = tmask(bw)
    with T.lever_scope(levers), ag.tape():
        cs = conds_for(bw)
        s_i = glue.init_repr_factory(bw) * bw["padding_mask"][..., None]
        gamma, beta = glue.pair_split.modulation(bw)
        zw = PairInputs(z0.constant, z0.bins[win:win + 1], z0.scale[win:win + 1], gamma, beta)
        z_in = pair_taped(pc, zw, br, bw, trunk.pair_dtype(m1))
        s, z = trunk(br.up(s_i, ttnn.float32), z_in, [br.up(c) for c in cs], m1)

    loss, terms, _ref = device_winner_loss(
        glue, s if isinstance(s, ag.Tensor) else ag.Tensor(s, requires_grad=True),
        bw, bridge=br, ref_terms=ref_l, win_idx=win,
    )
    del s, z
    dev_grads, _idle = dfolds.tape_backward(dev_masters)
    ag.release_pins()
    for gn, gp in host_glue_params.items():
        if gp.grad is not None:
            dev_grads[gn] = ttnn.from_torch(
                gp.grad.detach().to(torch.float32).contiguous(),
                dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev,
            )
            gp.grad = None
    return float(loss), int(win), dev_grads


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", default="/home/ttuser/r4t/step/batch*.npz",
                    help="glob or comma-separated batch*.npz files")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--ns-mode", choices=("split2", "split2_hifi4", "fp32"), default="split2")
    ap.add_argument("--out", default="")
    ap.add_argument("--cpu-selftest", action="store_true",
                    help="pure-CPU self-test of Float64MuonAdamEMA reference only")
    args = ap.parse_args()

    if args.cpu_selftest:
        rng = np.random.default_rng(0)
        w = {"a": rng.standard_normal((16, 32)).astype(np.float32), "b": rng.standard_normal((16,)).astype(np.float32)}
        ref = Float64MuonAdamEMA(w, lr=1e-4)
        for _ in range(3):
            g = {"a": rng.standard_normal((16, 32)).astype(np.float32) * 0.1,
                 "b": rng.standard_normal((16,)).astype(np.float32) * 0.1}
            st = ref.step(g)
        assert ref.step_count == 3 and st["muon_norm"] > 0 and st["adam_norm"] > 0
        print("CPU Float64MuonAdamEMA selftest PASS")
        return

    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import torch
    import ttnn
    from tt_bio import autograd as ag, tenstorrent as T
    from tt_bio.rfd4 import engine as E, network as N
    from tt_bio.rfd4.pair import PairOnCard, PairSplit
    from tt_bio.train.device_optim import rfd4_device_optimizer
    from tt_bio.train.folds import make_rfd4_refold_hook
    from tt_bio.train.provenance import open_nodes
    from tt_bio.train.rfd4_data import load_batch
    from tt_bio.train.rfd4_objective import MUON_EXCLUDE

    if "," in args.batches:
        bfiles = [x.strip() for x in args.batches.split(",") if x.strip()]
    else:
        bfiles = sorted(glob.glob(args.batches))
    bfiles = [f for f in bfiles if os.path.basename(f).startswith("batch") and os.path.basename(f)[5:7].isdigit()]
    assert bfiles, f"no batch files found for {args.batches}"
    batches = [(os.path.basename(f),) + load_batch(f) for f in bfiles]
    n_tok = int(batches[0][1]["padding_mask"].shape[1])
    print(f"Loaded {len(batches)} batches (n={n_tok}), running {args.steps} steps, ns_mode={args.ns_mode}", flush=True)

    dev = T.get_device()
    card = int(os.environ.get("TT_VISIBLE_DEVICES", "0").split(",")[0])
    nodes = open_nodes()
    clk_node = nodes[0] if nodes else card
    clk, stop, t_clk = _aiclk_sampler(clk_node)

    ag.install()
    state, config = N.load_checkpoint(E._resolve_weights(E.DEFAULT_WEIGHTS))
    cfg = config["nn"]

    print(f"\n=== [1/3] 20-step Resident closed-loop + same-grads parity (ns_mode={args.ns_mode}) ===", flush=True)
    glue1 = N.FlowGlue(state, config)
    dfolds1, trunk1, dheads1, host_glue1, init_w = _build_resident_env(dev, state, cfg, glue1, n_tok)
    init_np = {k: v.numpy().copy() for k, v in init_w.items()}
    print(f"  Total params: {len(init_w)} (folded={len(init_w) - len(host_glue1)}, host_glue={len(host_glue1)})", flush=True)

    opt_dev = rfd4_device_optimizer(init_w, dev, lr=1e-4, ema_decay=0.999, ns_mode=args.ns_mode)
    opt_f64_same = Float64MuonAdamEMA(init_np, lr=1e-4, muon_exclude=MUON_EXCLUDE, ema_decay=0.999)

    card_pair1 = []
    glue1.pair_split = PairSplit(glue1.pair_repr_builder)
    card_pair1.append(PairOnCard(glue1.pair_split, dev))
    refold_hook1 = make_rfd4_refold_hook(
        trunk1, glue1, host_glue_params=host_glue1, pair_card_ref=card_pair1, device=dev,
    )

    saved_grads: list[dict[str, np.ndarray]] = []
    res_losses: list[float] = []
    res_winners: list[int] = []

    for step in range(args.steps):
        bt_tup = batches[step % len(batches)]
        loss, win, dev_grads = _run_one_fwd_bwd(
            dev, trunk1, glue1, dfolds1, opt_dev.master, host_glue1, card_pair1, bt_tup,
        )
        res_losses.append(loss)
        res_winners.append(win)

        g_host = {
            nm: ttnn.to_torch(g_tt).float().numpy().reshape(init_np[nm].shape).copy()
            for nm, g_tt in dev_grads.items()
        }
        saved_grads.append(g_host)

        st_dev = opt_dev.step(dev_grads, free_grads=True)
        st_f64 = opt_f64_same.step(g_host)
        dfolds1.refold(opt_dev.master, on_refold=refold_hook1)
        ttnn.synchronize_device(dev)

        if (step + 1) in (1, 5, 10, 15, args.steps):
            print(
                f"  step {step+1:2d}/{args.steps} ({bt_tup[0]}): loss={loss:.5f} win={win} "
                f"muon_norm dev={st_dev['grad_norm_muon']:.4f} f64={st_f64['muon_norm']:.4f} "
                f"adam_norm dev={st_dev['grad_norm']:.4f} f64={st_f64['adam_norm']:.4f}",
                flush=True,
            )

    worst_param_rel, worst_param_name = 0.0, ""
    worst_muon_rel, worst_muon_name = 0.0, ""
    worst_adam_rel, worst_adam_name = 0.0, ""
    worst_ema_rel, worst_ema_name = 0.0, ""
    muon_set = set(opt_dev.muon_names)
    hm_dev = opt_dev.host_master()
    em_dev = opt_dev.ema.host_shadow()

    for nm in init_np:
        w_d = hm_dev[nm]
        w_r = opt_f64_same.master[nm]
        rel = _rel_l2(w_d, w_r)
        if rel > worst_param_rel:
            worst_param_rel, worst_param_name = rel, nm
        if nm in muon_set:
            if rel > worst_muon_rel:
                worst_muon_rel, worst_muon_name = rel, nm
        else:
            if rel > worst_adam_rel:
                worst_adam_rel, worst_adam_name = rel, nm

        e_d = em_dev[nm]
        e_r = opt_f64_same.ema[nm]
        erel = _rel_l2(e_d, e_r)
        if erel > worst_ema_rel:
            worst_ema_rel, worst_ema_name = erel, nm

    print(
        f"  20-step SAME-GRADS OPT-PARITY (bar <= 1e-4):\n"
        f"    worst param: {worst_param_rel:.3e} ({worst_param_name})\n"
        f"    worst Muon : {worst_muon_rel:.3e} ({worst_muon_name})\n"
        f"    worst Adam : {worst_adam_rel:.3e} ({worst_adam_name})\n"
        f"    worst EMA  : {worst_ema_rel:.3e} ({worst_ema_name})",
        flush=True,
    )

    card_pair1.pop().release()
    for spec in dfolds1.specs:
        ttnn.deallocate(spec.handle)
    for c in dfolds1.consts.values():
        ttnn.deallocate(c)
    for pu in trunk1.pair_updates.values():
        for attr in ("tri_out", "tri_in"):
            tri = getattr(pu, attr, None)
            if tri is not None and getattr(tri, "_t1_w", None):
                for triple in tri._t1_w.values():
                    for t in triple:
                        if t is not None:
                            ttnn.deallocate(t)
                tri._t1_w.clear()
    opt_dev.deallocate()
    ag.release_pins()
    ag.forget_parameters()

    print(f"\n=== [2/3] 20-step Closed-loop Float64 Host Optimizer Trajectory (D37 tol) ===", flush=True)
    glue2 = N.FlowGlue(state, config)
    dfolds2, trunk2, dheads2, host_glue2, init_w2 = _build_resident_env(dev, state, cfg, glue2, n_tok)
    opt_f64_cl = Float64MuonAdamEMA(init_np, lr=1e-4, muon_exclude=MUON_EXCLUDE, ema_decay=0.999)

    dev_masters_f64 = {
        nm: ttnn.from_torch(init_w2[nm], dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
        for nm in init_w2
    }
    card_pair2 = []
    glue2.pair_split = PairSplit(glue2.pair_repr_builder)
    card_pair2.append(PairOnCard(glue2.pair_split, dev))
    refold_hook2 = make_rfd4_refold_hook(
        trunk2, glue2, host_glue_params=host_glue2, pair_card_ref=card_pair2, device=dev,
    )

    ref_losses: list[float] = []
    ref_winners: list[int] = []
    max_loss_rel = 0.0
    max_loss_step = 0
    winner_mismatches = 0

    for step in range(args.steps):
        bt_tup = batches[step % len(batches)]
        loss_r, win_r, dev_grads = _run_one_fwd_bwd(
            dev, trunk2, glue2, dfolds2, dev_masters_f64, host_glue2, card_pair2, bt_tup,
        )
        ref_losses.append(loss_r)
        ref_winners.append(win_r)
        lrel = abs(res_losses[step] - loss_r) / max(abs(loss_r), 1.0)
        if lrel > max_loss_rel:
            max_loss_rel, max_loss_step = lrel, step + 1
        if res_winners[step] != win_r:
            winner_mismatches += 1

        g_host = {}
        for nm, g_tt in dev_grads.items():
            g_host[nm] = ttnn.to_torch(g_tt).float().numpy().reshape(init_np[nm].shape).copy()
            ttnn.deallocate(g_tt)
        opt_f64_cl.step(g_host)

        for nm, w64 in opt_f64_cl.master.items():
            old = dev_masters_f64[nm]
            dev_masters_f64[nm] = ttnn.from_torch(
                torch.from_numpy(w64.astype(np.float32)), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev,
            )
            ttnn.deallocate(old)
        dfolds2.refold(dev_masters_f64, on_refold=refold_hook2)
        ttnn.synchronize_device(dev)

    print(
        f"  Closed-loop 20-step trajectory: max_loss_rel={max_loss_rel:.3e} (step {max_loss_step}, bar <= 1e-2), "
        f"winner_mismatches={winner_mismatches}/{args.steps}",
        flush=True,
    )

    card_pair2.pop().release()
    for spec in dfolds2.specs:
        ttnn.deallocate(spec.handle)
    for c in dfolds2.consts.values():
        ttnn.deallocate(c)
    for pu in trunk2.pair_updates.values():
        for attr in ("tri_out", "tri_in"):
            tri = getattr(pu, attr, None)
            if tri is not None and getattr(tri, "_t1_w", None):
                for triple in tri._t1_w.values():
                    for t in triple:
                        if t is not None:
                            ttnn.deallocate(t)
                tri._t1_w.clear()
    for m in dev_masters_f64.values():
        ttnn.deallocate(m)
    ag.release_pins()
    ag.forget_parameters()

    print(f"\n=== [3/3] Broken arms (ns_steps=4, beta2=0.99) on exact same {args.steps}-step grads ===", flush=True)
    broken_results = {}
    for arm_name, kw in (
        ("ns_steps_4", dict(ns_steps=4, betas=(0.9, 0.999))),
        ("adam_beta2_099", dict(ns_steps=5, betas=(0.9, 0.99))),
    ):
        opt_brk = rfd4_device_optimizer(init_w, dev, lr=1e-4, ema_decay=0.999, ns_mode=args.ns_mode, **kw)
        for step in range(args.steps):
            dev_g = {
                nm: ttnn.from_torch(
                    torch.from_numpy(g_arr), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev,
                )
                for nm, g_arr in saved_grads[step].items()
            }
            opt_brk.step(dev_g, free_grads=True)
        ttnn.synchronize_device(dev)

        hm_brk = opt_brk.host_master()
        w_rel, w_nm = 0.0, ""
        for nm in init_np:
            rel = _rel_l2(hm_brk[nm], opt_f64_same.master[nm])
            if rel > w_rel:
                w_rel, w_nm = rel, nm
        opt_brk.deallocate()
        broken_results[arm_name] = {"worst_rel": w_rel, "worst_name": w_nm, "failed_bar": bool(w_rel > 1e-4)}
        print(
            f"  Broken arm {arm_name:16s}: worst_rel={w_rel:.3e} ({w_nm}) -> "
            f"{'FAIL (expected)' if w_rel > 1e-4 else 'UNEXPECTED PASS!'}",
            flush=True,
        )

    stop.set()
    t_clk.join(timeout=1.0)
    aiclk_min = min(clk) if clk else 0
    aiclk_max = max(clk) if clk else 0
    aiclk_med = sorted(clk)[len(clk) // 2] if clk else 0

    opt_pass = (worst_param_rel <= 1e-4 and worst_ema_rel <= 1e-4 and max_loss_rel <= 1e-2 and winner_mismatches == 0)
    brk_pass = all(v["failed_bar"] for v in broken_results.values())

    report = {
        "steps": args.steps,
        "ns_mode": args.ns_mode,
        "n_params": len(init_np),
        "card": card,
        "clk_node": clk_node,
        "aiclk": [aiclk_min, aiclk_max],
        "aiclk_med": aiclk_med,
        "opt_parity_pass": opt_pass,
        "worst_param_rel": worst_param_rel,
        "worst_param_name": worst_param_name,
        "worst_muon_rel": worst_muon_rel,
        "worst_muon_name": worst_muon_name,
        "worst_adam_rel": worst_adam_rel,
        "worst_adam_name": worst_adam_name,
        "worst_ema_rel": worst_ema_rel,
        "worst_ema_name": worst_ema_name,
        "closed_loop_max_loss_rel": max_loss_rel,
        "closed_loop_max_loss_step": max_loss_step,
        "closed_loop_winner_mismatches": winner_mismatches,
        "res_losses": res_losses,
        "ref_losses": ref_losses,
        "broken_arms": broken_results,
        "broken_arms_all_fail": brk_pass,
    }
    print(
        f"\nSUMMARY: OPT-PARITY={'PASS' if opt_pass else 'FAIL'} "
        f"(worst={worst_param_rel:.3e}, ema={worst_ema_rel:.3e}, loss_rel={max_loss_rel:.3e}) | "
        f"BROKEN-ARM={'FAIL (as required)' if brk_pass else 'BAD'} | "
        f"AICLK={aiclk_min}..{aiclk_max} (med={aiclk_med}) MHz",
        flush=True,
    )
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
    sys.exit(0 if (opt_pass and brk_pass) else 1)


if __name__ == "__main__":
    main()
