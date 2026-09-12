#!/usr/bin/env python3
"""The atom attention's other window question: what is projected FOUR TIMES because it is
projected after the gather instead of before it.

`TT_BIO_ATOM_KEY_WINDOW` proved the key window is a contiguous gather of H=128 atoms per 32-atom
query window. Each atom therefore appears in exactly H/W = 4 key windows, and the shipped code
runs the K/V projection on the GATHERED tensor:

    s_kv = window(s)                      (1, 140, 128, 128)   17920 rows
    kv   = linear(s_kv, kv_weight)        17920 x 128 x 256

A linear is per-row, so `linear(window(x)) == window(linear(x))` value for value, and the second
form projects 4480 rows instead of 17920 -- a quarter of the matmul -- at the price of gathering
a 256-wide tensor instead of a 128-wide one. Whether that trade wins is a measurement, and whether
it is BIT-EXACT is a second measurement: ttnn picks a matmul's blocking off the operand shape, so
changing M can reorder nothing (the reduction is over K) but must still be shown, not assumed
(CONTEXT 4-E).

Also screens the q padding beside it: q is padded 32 -> 128 rows purely so it can enter
`nlp_create_qkv_heads` beside a 128-row kv, and is sliced straight back to 32 afterwards.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

B, K, W, H, D = 1, 140, 32, 128, 128
N_HEADS = 4


def timed(ttnn, dev, fn, reps=20, blocks=5):
    for _ in range(4):
        fn()
    ttnn.synchronize_device(dev)
    meds = []
    for _ in range(blocks):
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        ttnn.synchronize_device(dev)
        meds.append((time.perf_counter() - t0) / reps)
    return round(1e6 * st.median(meds), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    import tt_bio.tenstorrent as T
    from tt_bio.boltz2 import get_indexing_matrix

    dev = T.get_device()
    kernel_cls = (ttnn.types.WormholeComputeKernelConfig
                  if dev.arch() == ttnn.Arch.WORMHOLE_B0
                  else ttnn.types.BlackholeComputeKernelConfig)
    ckc = kernel_cls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                     fp32_dest_acc_en=True, packer_l1_acc=True)
    out = {"env": {"card": os.environ.get("TT_VISIBLE_DEVICES"),
                   "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                   "commit": os.popen(f"git -C {ROOT} rev-parse --short HEAD").read().strip(),
                   "arch": T.arch_name(), "grid": str(T.CORE_GRID_MAIN),
                   "shape": {"B": B, "K": K, "W": W, "H": H, "D": D},
                   "loadavg": open("/proc/loadavg").read().split()[:3]}}

    src = torch.randn(B, K, W, D)
    s = ttnn.from_torch(src, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=dev)
    ki = get_indexing_matrix(K, W, H, torch.device("cpu"))
    ki_tt = ttnn.from_torch(ki, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat4_b, device=dev)
    kvw = ttnn.from_torch(torch.randn(D, 2 * D) * 0.05, layout=ttnn.TILE_LAYOUT,
                          dtype=ttnn.bfloat16, device=dev)

    plan = T._atom_window_plan((B, K, W, D), ki_tt)

    def win(x):
        return T._atom_window_gather(x, plan)

    def lin(x, w):
        return ttnn.linear(x, w, compute_kernel_config=ckc, core_grid=T.CORE_GRID_MAIN,
                           dtype=ttnn.bfloat16)

    s_kv = win(s)
    out["window_shape"] = list(s_kv.shape)

    # --- P1: project after the gather (shipped) vs before it (candidate) -------------------
    def shipped():
        return lin(win(s), kvw)

    def preproj():
        return win(lin(s, kvw))

    r_ship, r_pre = shipped(), preproj()
    t_ship, t_pre = ttnn.to_torch(r_ship), ttnn.to_torch(r_pre)
    out["P1_kv_preprojection"] = {
        "shape_shipped": list(r_ship.shape), "shape_preproj": list(r_pre.shape),
        "bit_exact": bool(torch.equal(t_ship, t_pre)),
        "max_abs": float((t_ship.float() - t_pre.float()).abs().max()),
        "us_shipped_chain": timed(ttnn, dev, shipped),
        "us_preproj_chain": timed(ttnn, dev, preproj),
        "us_window_128": timed(ttnn, dev, lambda: win(s)),
        "us_window_256": timed(ttnn, dev, lambda: win(lin(s, kvw))) - 0,  # includes the linear
        "us_kv_linear_17920": timed(ttnn, dev, lambda: lin(s_kv, kvw)),
        "us_kv_linear_4480": timed(ttnn, dev, lambda: lin(s, kvw)),
    }
    p1 = out["P1_kv_preprojection"]
    p1["us_window_256_alone"] = round(p1["us_preproj_chain"] - p1["us_kv_linear_4480"], 3)
    p1["speedup"] = round(p1["us_shipped_chain"] / p1["us_preproj_chain"], 5)
    print("P1", json.dumps(p1, indent=1), flush=True)

    # negative control: the same two chains with a DIFFERENT weight must NOT agree
    kvw2 = ttnn.from_torch(torch.randn(D, 2 * D) * 0.05, layout=ttnn.TILE_LAYOUT,
                           dtype=ttnn.bfloat16, device=dev)
    ctrl = ttnn.to_torch(win(lin(s, kvw2)))
    p1["negative_control_differs"] = bool(not torch.equal(t_ship, ctrl))

    # --- P2: the q pad, and whether the head split needs it -------------------------------
    qw = ttnn.from_torch(torch.randn(D, D) * 0.05, layout=ttnn.TILE_LAYOUT,
                         dtype=ttnn.bfloat16, device=dev)
    q0 = lin(s, qw)
    kv0 = r_pre

    def shipped_heads():
        q = ttnn.pad(q0, [[0, 0], [0, 0], [0, T.ATOM_DIM - W], [0, 0]], 0.0)
        q = ttnn.reshape(q, (B * K, 1, T.ATOM_DIM, -1))
        kv = ttnn.reshape(kv0, (B * K, 1, T.ATOM_DIM, -1))
        qq, kk, vv = ttnn.experimental.nlp_create_qkv_heads(
            q, kv, num_heads=N_HEADS, num_kv_heads=N_HEADS, transpose_k_heads=False)
        _, Hh, S, Dq = qq.shape
        qq = ttnn.reshape(qq, (B, K * Hh, S, Dq))
        return qq[:, :, :W, :], kk, vv

    def unpadded_heads():
        q = ttnn.reshape(q0, (B * K, 1, W, -1))
        kv = ttnn.reshape(kv0, (B * K, 1, T.ATOM_DIM, -1))
        qq, kk, vv = ttnn.experimental.nlp_create_qkv_heads(
            q, kv, num_heads=N_HEADS, num_kv_heads=N_HEADS, transpose_k_heads=False)
        _, Hh, S, Dq = qq.shape
        return ttnn.reshape(qq, (B, K * Hh, S, Dq)), kk, vv

    rec = {}
    try:
        a_q, a_k, a_v = shipped_heads()
        rec["shipped_q_shape"] = list(a_q.shape)
        rec["us_shipped"] = timed(ttnn, dev, lambda: shipped_heads(), reps=10)
        b_q, b_k, b_v = unpadded_heads()
        rec["unpadded_q_shape"] = list(b_q.shape)
        rec["bit_exact_q"] = bool(torch.equal(ttnn.to_torch(a_q), ttnn.to_torch(b_q)))
        rec["us_unpadded"] = timed(ttnn, dev, lambda: unpadded_heads(), reps=10)
        rec["speedup"] = round(rec["us_shipped"] / rec["us_unpadded"], 5)
    except Exception as e:                                                  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {str(e)[:500]}"
    out["P2_q_pad"] = rec
    print("P2", json.dumps(rec, indent=1), flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    print("DONE", a.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
