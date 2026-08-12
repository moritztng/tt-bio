#!/usr/bin/env python3
"""Does a fused SDPA with fp32 CBs run at all, and what does it cost?

`ab_of3_hifi_512` showed the plDDT loss is not DST precision and not fidelity: `fp32_dest_acc`
governs the accumulator inside an op, not the data format the flash kernel's circular buffers hold,
so the exponentiated scores are stored bf16 before `attn @ v` whatever DST does. The one way to make
those CBs fp32 without touching a kernel is to hand the op fp32 TENSORS -- the CB formats follow the
tensor formats.

This is the feasibility and cost half only. It answers three things before a fold is spent:
  1. does `ttnn.transformer.scaled_dot_product_attention` accept fp32 q/k/v/bias on this build,
  2. what does it cost per call against the 1.4048 ms bf16 fused and the 62.5789 ms fp32-softmax
     path already measured at this exact shape,
  3. does the K2 persistent-mask kernel accept it, or does it decline (`_common_ok` demands bf16),
     in which case the fp32 route runs the stock op and loses K2's mask win.

Accuracy is NOT decided here. The previous synthetic screen proved random q/k/v cannot discriminate
these configs -- attention goes near-uniform and output bf16 rounding swamps a 0.0009 spread. The
accuracy verdict comes from the fold's plDDT.
"""
from __future__ import annotations

import json, os, statistics as st, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch, ttnn                                                            # noqa: E402
import tt_bio.tenstorrent as T                                                # noqa: E402
import tt_bio.triatt_sdpa as PM                                               # noqa: E402

from tt_bio.main import _detect_p300_devices, _find_ttnn_mesh_graph_descriptor  # noqa: E402
if _detect_p300_devices() and not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
    mgd = _find_ttnn_mesh_graph_descriptor("p150_mesh_graph_descriptor.textproto")
    if mgd:
        os.environ["TT_MESH_GRAPH_DESC_PATH"] = mgd

dev = T.get_device()
g = dev.compute_with_storage_grid_size()
B, H, S, D = 512, 4, 512, 32
scale = float(D) ** -0.5
torch.manual_seed(0)


def bench(fn, n=5, warm=2):
    for _ in range(warm):
        r = fn(); ttnn.synchronize_device(dev); ttnn.deallocate(r)
    ts = []
    for _ in range(n):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        r = fn()
        ttnn.synchronize_device(dev)
        ts.append(time.perf_counter() - t0)
        ttnn.deallocate(r)
    return round(st.median(ts) * 1e3, 4)


def dev_t(x, dt):
    return ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, dtype=dt, device=dev,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)


def main():
    qt = torch.randn(B, H, S, D, dtype=torch.float32)
    kt = torch.randn(B, H, S, D, dtype=torch.float32)
    vt = torch.randn(B, H, S, D, dtype=torch.float32)
    bt = torch.randn(1, H, S, S, dtype=torch.float32) * 0.5

    res = {"host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
           "grid": [g.x, g.y], "shape": {"B": B, "H": H, "S": S, "D": D}, "rows": []}
    k_chunk = T._sdpa_chunks_shipped(S, S)[1]
    fits = [qc for qc in T._tri_att_q_chunks(S, S) if (S, S, qc) not in T._SDPA_Q_CHUNK_OVER_L1]
    q_chunk = fits[0]
    print(f"q_chunk={q_chunk} k_chunk={k_chunk}", flush=True)

    for name, dt in (("bf16", ttnn.bfloat16), ("fp32", ttnn.float32)):
        q, k, v, bias = dev_t(qt, dt), dev_t(kt, dt), dev_t(vt, dt), dev_t(bt, dt)
        row = {"dtype": name, "bytes_per_call_GB": round(
            (3 * B * H * S * D + H * S * S + B * H * S * D) * (2 if dt == ttnn.bfloat16 else 4) / 1e9, 4)}

        # --- K2 fused path ---------------------------------------------------------------
        PM.REJECTS.clear(); PM.STATS[0] = PM.STATS[1] = 0
        o = PM.sdpa(q, k, v, bias, scale, q_chunk, k_chunk)
        if o is None:
            row["k2"] = {"served": False,
                         "rejects": {f"{r}:{sh}": n for (r, sh), n in PM.REJECTS.items()}}
        else:
            ttnn.deallocate(o)
            row["k2"] = {"served": True,
                         "ms": bench(lambda: PM.sdpa(q, k, v, bias, scale, q_chunk, k_chunk))}

        # --- stock ttnn SDPA -------------------------------------------------------------
        def stock():
            return ttnn.transformer.scaled_dot_product_attention(
                q, k, v, attn_mask=bias, is_causal=False, scale=scale,
                program_config=T._sdpa_program_config(q_chunk, k_chunk))

        try:
            o = stock(); ttnn.deallocate(o)
            row["stock"] = {"ok": True, "ms": bench(stock)}
        except Exception as e:                                                # noqa: BLE001
            row["stock"] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}"}

        res["rows"].append(row)
        print(json.dumps(row), flush=True)
        for t in (q, k, v, bias):
            ttnn.deallocate(t)

    Path(sys.argv[1]).write_text(json.dumps(res, indent=1))
    print("wrote", sys.argv[1], flush=True)
    T.cleanup()


if __name__ == "__main__":
    main()
