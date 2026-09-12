#!/usr/bin/env python3
"""The atom attention's second piece of padding: q is grown 32 -> 128 rows to enter the head split.

`nlp_create_qkv_heads` refuses a kv whose sequence length differs from q's ("KV tensor seq_len dim
must be same as Q tensor seq_len"), and the atom attention's q is one 32-atom query window against
a 128-atom key window. So the shipped code pads q to 128 rows, splits heads on 4x the rows it
needs, and slices the 96 pad rows straight back off. In the committed step census that is
pad 68.42 + nlp_create_qkv_heads 159.57 + slice 23.52 = 251.51 us per atom layer, six times a step.

A head split is a reshape and a permute. This screens doing it that way -- separately for q and for
the two halves of kv, at their own row counts -- against the shipped pad/split/slice, for
bit-exactness first and time second.
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
N_HEADS, D_H = 4, 32


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

    dev = T.get_device()
    out = {"env": {"card": os.environ.get("TT_VISIBLE_DEVICES"),
                   "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                   "commit": os.popen(f"git -C {ROOT} rev-parse --short HEAD").read().strip(),
                   "arch": T.arch_name(),
                   "loadavg": open("/proc/loadavg").read().split()[:3]}}

    q0 = ttnn.from_torch(torch.randn(B, K, W, D), layout=ttnn.TILE_LAYOUT,
                         dtype=ttnn.bfloat16, device=dev)
    kv0 = ttnn.from_torch(torch.randn(B, K, H, 2 * D), layout=ttnn.TILE_LAYOUT,
                          dtype=ttnn.bfloat16, device=dev)

    def shipped():
        q = ttnn.pad(q0, [[0, 0], [0, 0], [0, T.ATOM_DIM - W], [0, 0]], 0.0)
        q = ttnn.reshape(q, (B * K, 1, T.ATOM_DIM, -1))
        kv = ttnn.reshape(kv0, (B * K, 1, T.ATOM_DIM, -1))
        qq, kk, vv = ttnn.experimental.nlp_create_qkv_heads(
            q, kv, num_heads=N_HEADS, num_kv_heads=N_HEADS, transpose_k_heads=False)
        _, Hh, S, Dq = qq.shape
        qq = ttnn.reshape(qq, (B, K * Hh, S, Dq))
        kk = ttnn.reshape(kk, (B, K * Hh, S, Dq))
        vv = ttnn.reshape(vv, (B, K * Hh, S, Dq))
        return qq[:, :, :W, :], kk, vv

    def split(x, rows):
        """(B, K, rows, n_heads*d) -> (B, K*n_heads, rows, d)."""
        x = ttnn.reshape(x, (B * K, rows, N_HEADS, D_H))
        x = ttnn.permute(x, (0, 2, 1, 3))
        return ttnn.reshape(x, (B, K * N_HEADS, rows, D_H))

    def unpadded():
        k_half = kv0[..., :N_HEADS * D_H]
        v_half = kv0[..., N_HEADS * D_H:]
        return split(q0, W), split(k_half, H), split(v_half, H)

    rec = {}
    a_q, a_k, a_v = shipped()
    b_q, b_k, b_v = unpadded()
    rec["shapes_shipped"] = [list(t.shape) for t in (a_q, a_k, a_v)]
    rec["shapes_unpadded"] = [list(t.shape) for t in (b_q, b_k, b_v)]
    for nm, x, y in (("q", a_q, b_q), ("k", a_k, b_k), ("v", a_v, b_v)):
        tx, ty = ttnn.to_torch(x), ttnn.to_torch(y)
        rec[f"bit_exact_{nm}"] = bool(torch.equal(tx, ty))
        rec[f"max_abs_{nm}"] = float((tx.float() - ty.float()).abs().max())
    rec["negative_control_differs"] = bool(not torch.equal(ttnn.to_torch(a_k), ttnn.to_torch(b_v)))
    rec["us_shipped"] = timed(ttnn, dev, shipped, reps=10)
    rec["us_unpadded"] = timed(ttnn, dev, unpadded, reps=10)
    rec["speedup"] = round(rec["us_shipped"] / rec["us_unpadded"], 5)
    out["head_split"] = rec
    print("HEADS", json.dumps(rec, indent=1), flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    print("DONE", a.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
