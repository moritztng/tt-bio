#!/usr/bin/env python3
"""Verify + benchmark P3 (DeviceTrainPair: 0 pair-sized PCIe in fwd + bwd) in ONE device open.

Tests:
  1. Forward z_init parity vs float64 PairSplit.assemble across real batches (K=4 candidates + B=1 winner).
  2. Backward VJP parity (dW, d_lnw, dgamma, dbeta) vs float64 torch autograd + broken arm (must FAIL).
  3. Wall-time A/B: P2 (host constant + 100.66 MB proj + 33.55 MB F) vs P3 (12.6 MB F_const + on-card proj/norms/dW).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tt_bio.main import ensure_p300_mesh_descriptor  # noqa: E402

ensure_p300_mesh_descriptor()

import numpy as np  # noqa: E402
import torch  # noqa: E402
import ttnn  # noqa: E402

from tt_bio import autograd as ag  # noqa: E402
from tt_bio.rfd4 import engine as E, network as N  # noqa: E402
from tt_bio.rfd4.pair import PairInputs, PairOnCard, PairSplit  # noqa: E402
from tt_bio.train.bridge import Bridge  # noqa: E402
from tt_bio.train.rfd4_data import load_batch  # noqa: E402
from tt_bio.train.rfd4_pair import DeviceTrainPair, pair_taped  # noqa: E402


def _metrics(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    a64 = np.asarray(a, dtype=np.float64).ravel()
    b64 = np.asarray(b, dtype=np.float64).ravel()
    num = float(np.linalg.norm(a64 - b64))
    den = float(np.linalg.norm(b64))
    rel = num / den if den > 0 else num
    cos = float(np.dot(a64, b64) / (np.linalg.norm(a64) * den + 1e-30))
    mx = float(np.max(np.abs(a64 - b64)))
    return rel, cos, mx


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


def _take(bt: dict, i: int) -> dict:
    return {k: (_take(v, i) if isinstance(v, dict)
                else (v[i:i + 1] if isinstance(v, torch.Tensor) and v.ndim > 0 else v))
            for k, v in bt.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", default="", help="comma-separated batch npz paths or glob")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.batches:
        bfiles = []
        for pat in args.batches.split(","):
            m = sorted(glob.glob(os.path.expanduser(pat)))
            bfiles.extend(m if m else [os.path.expanduser(pat)])
    else:
        bfiles = sorted(glob.glob(os.path.expanduser("~/r4t/step/batch*.npz")))
        if not bfiles:
            bfiles = sorted(glob.glob(os.path.expanduser("~/fleet/r4t/batches/batch*.npz")))
    assert bfiles, "no batch files found"

    state, config = N.load_checkpoint(E._resolve_weights(E.DEFAULT_WEIGHTS))
    glue = N.FlowGlue(state, config)
    glue.requires_grad_(True)
    builder = glue.pair_repr_builder
    sp = PairSplit(builder)

    from tt_bio import tenstorrent as TT
    ag.install()
    dev = TT.get_device()
    try:
        clk0 = _aiclk()
        dp = DeviceTrainPair(builder, dev)
        pc = PairOnCard(sp, dev)

        w_host = builder.init_repr_factory.linear_out.weight.detach().float()
        lnw_host = builder.init_repr_factory.ln_out.weight.detach().float()
        dev_masters = {
            dp.W_KEY: ttnn.from_torch(w_host.contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev),
            dp.LNW_KEY: ttnn.from_torch(lnw_host.contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev),
        }

        fwd_results = []
        p2_times, p3_times = [], []
        for idx, bf in enumerate(bfiles[:9]):
            bt = load_batch(bf)[0]
            bn = os.path.basename(bf)
            with torch.no_grad():
                c_ref = sp.constant(bt)
                p_ref = sp.inputs(bt, c_ref)
                z_ref_f64 = sp.assemble(p_ref).numpy()  # [K, n, n, 384] float64
                gk, bk = dp.modulation(bt)

            # Time P3 cand + prepare
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            dp.prepare_batch(bt, dev_masters)
            ttnn.synchronize_device(dev)
            t_prep = time.perf_counter() - t0
            if idx == 0:
                proj_p3 = ttnn.to_torch(dp._step_state["proj"]).float().numpy()
                proj_ref = c_ref.proj[:1].numpy()
                pr, pc_cos, pmx = _metrics(proj_p3, proj_ref)
                print(f"[P3 PROJ {bn}] rel={pr:.3e} cos={pc_cos:.8f} max={pmx:.3e}", flush=True)
            t1 = time.perf_counter()
            z_p3_tt = dp.forward_candidates(gk, bk, dtype=ttnn.float32)
            ttnn.synchronize_device(dev)
            t_fwd = time.perf_counter() - t1
            t_p3 = t_prep + t_fwd
            z_p3 = ttnn.to_torch(z_p3_tt).float().numpy()
            ttnn.deallocate(z_p3_tt)
            dp.release_step()

            rel, cos, mx = _metrics(z_p3, z_ref_f64)
            fwd_results.append(dict(batch=bn, shape=list(z_p3.shape), rel_l2=rel, cos=cos, max_abs=mx,
                                    prep_ms=round(1000 * t_prep, 2), fwd_ms=round(1000 * t_fwd, 2),
                                    p3_ms=round(1000 * t_p3, 2)))
            print(f"[P3 FWD {bn}] rel={rel:.3e} cos={cos:.8f} max={mx:.3e} prep={1000*t_prep:.1f}ms fwd={1000*t_fwd:.1f}ms tot={1000*t_p3:.1f}ms", flush=True)
            if idx >= 1:
                p3_times.append(t_p3)

            # Time P2 cand for A/B on same batch
            if idx >= 1:
                ttnn.synchronize_device(dev)
                t0 = time.perf_counter()
                with torch.no_grad():
                    c2 = sp.constant(bt)
                    p2 = sp.inputs(bt, c2)
                z2_tt = pc(p2, 0)
                ttnn.synchronize_device(dev)
                p2_times.append(time.perf_counter() - t0)
                ttnn.deallocate(z2_tt)
                pc.release()

        # VJP grade on batch 0 winner=0 against float64 torch autograd
        bt0 = load_batch(bfiles[0])[0]
        bw0 = _take(bt0, 0)
        n = int(bw0["padding_mask"].shape[-1])
        rng = np.random.default_rng(7)
        g_out_np = rng.standard_normal((1, n, n, dp.c)).astype(np.float32)

        # Float64 torch reference VJP
        feats64 = sp.features(bw0).double().detach()
        W64 = w_host.double().clone().requires_grad_(True)
        lnw64 = lnw_host.double().clone().requires_grad_(True)
        with torch.no_grad():
            gw_h, bw_h = dp.modulation(bw0)
        g64 = gw_h.double().clone().requires_grad_(True)
        b64 = bw_h.double().clone().requires_grad_(True)
        x64 = feats64 @ W64.t()
        r1_64 = torch.rsqrt((x64 * x64).mean(-1, keepdim=True) + dp.e1)
        n1_64 = x64 * r1_64
        a64 = n1_64 * lnw64
        r2_64 = torch.rsqrt((a64 * a64).mean(-1, keepdim=True) + dp.e2)
        bb64 = a64 * r2_64
        z64 = bb64 * g64[:, None, None, :] + b64[:, None, None, :]
        (z64 * torch.from_numpy(g_out_np).double()).sum().backward()

        # P3 device VJP
        def run_p3_vjp(corrupt_fconst: bool = False):
            dp.grads.clear()
            for p in builder.parameters():
                p.grad = None
            dp.prepare_batch(bt0, dev_masters)
            if corrupt_fconst:
                # Broken arm: zero out F_const on device before backward
                old_f = dp._step_state["f_const_hi"]
                dp._step_state["f_const_hi"] = ttnn.zeros_like(old_f)
                ttnn.deallocate(old_f)
            br = Bridge(dev, ttnn.bfloat16)
            with ag.tape():
                gw_t = gw_h.detach().clone().requires_grad_(True)
                bw_t = bw_h.detach().clone().requires_grad_(True)
                z_w = dp.forward_winner_taped(0, gw_t, bw_t, br, dtype=ttnn.float32)
                g_tt = ttnn.from_torch(torch.from_numpy(g_out_np), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
                ag.backward([z_w], [g_tt])
                ttnn.deallocate(g_tt)
            # Trigger bridge backward to get dgamma, dbeta on host tensors
            br.backward(torch.zeros((), dtype=torch.float32))
            grads_dev = dp.pop_grads()
            dW_dev = ttnn.to_torch(grads_dev[dp.W_KEY]).float().numpy()
            dln_dev = ttnn.to_torch(grads_dev[dp.LNW_KEY]).float().numpy()
            for t in grads_dev.values():
                ttnn.deallocate(t)
            ttnn.deallocate(z_w.value)
            ag.release_pins()
            out = dict(
                dW=_metrics(dW_dev, W64.grad.numpy()),
                d_lnw=_metrics(dln_dev, lnw64.grad.numpy()),
                dgamma=_metrics(gw_t.grad.numpy(), g64.grad.numpy()),
                dbeta=_metrics(bw_t.grad.numpy(), b64.grad.numpy()),
            )
            return out

        # Wait: Bridge.backward expects a loss tensor connected to bridge's torch leaves;
        # read Gl.grad / Bl.grad directly from bridge's leaves!
        def run_p3_vjp_direct(corrupt_fconst: bool = False):
            dp.grads.clear()
            dp.prepare_batch(bt0, dev_masters)
            if corrupt_fconst:
                old_f = dp._step_state["f_const_hi"]
                dp._step_state["f_const_hi"] = ttnn.zeros_like(old_f)
                ttnn.deallocate(old_f)
            br = Bridge(dev, ttnn.bfloat16)
            gw_t = gw_h.detach().clone().requires_grad_(True)
            bw_t = bw_h.detach().clone().requires_grad_(True)
            with ag.tape():
                z_w = dp.forward_winner_taped(0, gw_t, bw_t, br, dtype=ttnn.float32)
                g_tt = ttnn.from_torch(torch.from_numpy(g_out_np), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
                ag.backward([z_w], [g_tt])
                ttnn.deallocate(g_tt)
            # Bridge holds pairs (host_tensor, dev_leaf) in br.ups
            dg_host = ttnn.to_torch(br.ups[0][1].grad).float().numpy().reshape(g64.shape)
            db_host = ttnn.to_torch(br.ups[1][1].grad).float().numpy().reshape(b64.shape)
            grads_dev = dp.pop_grads()
            dW_dev = ttnn.to_torch(grads_dev[dp.W_KEY]).float().numpy()
            dln_dev = ttnn.to_torch(grads_dev[dp.LNW_KEY]).float().numpy()
            for t in grads_dev.values():
                ttnn.deallocate(t)
            for _, dl in br.ups:
                if dl.grad is not None:
                    ttnn.deallocate(dl.grad)
                ttnn.deallocate(dl.value)
            ttnn.deallocate(z_w.value)
            ag.release_pins()
            return dict(
                dW=_metrics(dW_dev, W64.grad.numpy()),
                d_lnw=_metrics(dln_dev, lnw64.grad.numpy()),
                dgamma=_metrics(dg_host, g64.grad.numpy()),
                dbeta=_metrics(db_host, b64.grad.numpy()),
            )

        vjp_ok = run_p3_vjp_direct(corrupt_fconst=False)
        vjp_brk = run_p3_vjp_direct(corrupt_fconst=True)
        for k, (rel, cos, mx) in vjp_ok.items():
            print(f"[P3 VJP OK {k:6s}] rel={rel:.3e} cos={cos:.8f} max={mx:.3e}", flush=True)
        print(f"[P3 VJP BROKEN dW] rel={vjp_brk['dW'][0]:.3e} cos={vjp_brk['dW'][1]:.6f} (expect FAIL)", flush=True)

        # Also time P2 vs P3 full fwd+bwd (prepare + cand + win fwd + win bwd)
        t_p3_full = []
        for rep in range(4):
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            dp.prepare_batch(bt0, dev_masters)
            zk = dp.forward_candidates(gk, bk, dtype=ttnn.bfloat16)
            ttnn.deallocate(zk)
            br = Bridge(dev, ttnn.bfloat16)
            gw_t = gw_h.detach().clone().requires_grad_(True)
            bw_t = bw_h.detach().clone().requires_grad_(True)
            with ag.tape():
                zw = dp.forward_winner_taped(0, gw_t, bw_t, br, dtype=ttnn.bfloat16)
                g_tt = ttnn.from_torch(torch.from_numpy(g_out_np), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
                ag.backward([zw], [g_tt])
                ttnn.deallocate(g_tt)
            ttnn.synchronize_device(dev)
            if rep >= 1:
                t_p3_full.append(time.perf_counter() - t0)
            for t in dp.pop_grads().values():
                ttnn.deallocate(t)
            for _, dl in br.ups:
                if dl.grad is not None:
                    ttnn.deallocate(dl.grad)
                ttnn.deallocate(dl.value)
            ttnn.deallocate(zw.value)
            ag.release_pins()

        t_p2_full = []
        for rep in range(4):
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            with torch.no_grad():
                c2 = sp.constant(bt0)
                p2 = sp.inputs(bt0, c2)
            zk = pc(p2, 0)
            ttnn.deallocate(zk)
            br = Bridge(dev, ttnn.bfloat16)
            with ag.tape():
                gw_t, bw_t = sp.modulation(bw0)
                zw_in = PairInputs(p2.constant, p2.bins[0:1], p2.scale[0:1], gw_t, bw_t)
                zw = pair_taped(pc, zw_in, br, bw0, dtype=ttnn.bfloat16)
                g_tt = ttnn.from_torch(torch.from_numpy(g_out_np), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
                ag.backward([zw], [g_tt])
                ttnn.deallocate(g_tt)
            ttnn.synchronize_device(dev)
            if rep >= 1:
                t_p2_full.append(time.perf_counter() - t0)
            for _, dl in br.ups:
                if dl.grad is not None:
                    ttnn.deallocate(dl.grad)
                ttnn.deallocate(dl.value)
            ttnn.deallocate(zw.value)
            pc.release()
            ag.release_pins()

        print(f"[P2 vs P3 FULL PAIR (cand+win fwd+bwd)] P2={1000*np.median(t_p2_full):.1f} ms -> P3={1000*np.median(t_p3_full):.1f} ms", flush=True)

        dp.release()
        for t in dev_masters.values():
            ttnn.deallocate(t)

        res = dict(
            aiclk=[clk0, _aiclk()],
            fwd=fwd_results,
            p2_cand_ms=round(1000 * float(np.median(p2_times)), 2) if p2_times else None,
            p3_cand_ms=round(1000 * float(np.median(p3_times)), 2) if p3_times else None,
            p2_full_ms=round(1000 * float(np.median(t_p2_full)), 2),
            p3_full_ms=round(1000 * float(np.median(t_p3_full)), 2),
            vjp_ok={k: dict(rel=v[0], cos=v[1], max_abs=v[2]) for k, v in vjp_ok.items()},
            vjp_broken={k: dict(rel=v[0], cos=v[1], max_abs=v[2]) for k, v in vjp_brk.items()},
            pass_all=bool(
                all(r["rel_l2"] <= 1e-2 and r["cos"] >= 0.9999 for r in fwd_results)
                and all(v[0] <= 1e-2 and v[1] >= 0.9999 for v in vjp_ok.values())
                and vjp_brk["dW"][0] > 0.1
            ),
        )
        print(f"OVERALL P3 PASS={res['pass_all']}", flush=True)
        if args.out:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            with open(args.out, "w") as f:
                json.dump(res, f, indent=2)
            print(f"Wrote {args.out}", flush=True)
    finally:
        pass


if __name__ == "__main__":
    main()
