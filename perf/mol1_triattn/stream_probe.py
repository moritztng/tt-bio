#!/usr/bin/env python3
"""Stock ttnn SDPA at triangle-attention shapes: unmasked (the streaming v2 compute path is eligible) vs a
provided [1, H, S, S] mask (the standard path, the one tt-bio serves). Sizes the lever "give the streaming path
the mask". Timing only; prints one json line per cell with the AICLK sampled during it.

A third arm, `served`, times tt-bio's served `_tri_att_sdpa` on operands captured from a served TriangleAttention
(sdpa_sweep.capture). Arms are interleaved over --rounds so a sagging clock hits all of them alike.

Usage: stream_probe.py --cells 1024x10:256:1024,512x10:512:512 --reps 5 --rounds 3
"""
import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "perf"))
import torch
import ttnn
import tt_bio.tenstorrent as T
import clocksample
from sdpa_sweep import capture


def timed(fn, reps, dev):
    fn(); ttnn.synchronize_device(dev)
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); out = fn(); ttnn.synchronize_device(dev); ts.append(time.perf_counter() - t)
        out.deallocate()
    return sorted(ts)[len(ts) // 2] * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="1024x10:256:1024,512x10:512:512")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=3)
    a = ap.parse_args()
    dev = T.get_device()
    for cell in a.cells.split(","):
        sh, qc, kc = cell.split(":")
        S, H = map(int, sh.split("x"))
        put = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)  # noqa: E731
        g = torch.Generator().manual_seed(0)
        q, k, v = (put(torch.randn(S, H, S, 32, generator=g, dtype=torch.bfloat16)) for _ in range(3))
        mask = put(torch.randn(1, H, S, S, generator=g, dtype=torch.bfloat16))
        cfg = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=T.COMPUTE_GRID_MAIN, exp_approx_mode=False,
                                     q_chunk_size=int(qc), k_chunk_size=int(kc))
        ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi2,
                                                     math_approx_mode=True, fp32_dest_acc_en=False, packer_l1_acc=False)
        sq, sk, sv, sb, ss, sckc, spad = capture(dev, S, H * 32, H)
        arms = {"served": lambda: T._tri_att_sdpa(sq, sk, sv, sb, ss, sckc, spad),
                "nomask": lambda: ttnn.transformer.scaled_dot_product_attention(
                    q, k, v, is_causal=False, scale=1.0, program_config=cfg, compute_kernel_config=ckc),
                "mask": lambda: ttnn.transformer.scaled_dot_product_attention(
                    q, k, v, attn_mask=mask, is_causal=False, scale=1.0, program_config=cfg, compute_kernel_config=ckc)}
        for name, fn in [kv for _ in range(a.rounds) for kv in arms.items()]:
            try:
                with clocksample.during(period=0.5) as clk:
                    ms = timed(fn, a.reps, dev)
                print(json.dumps({"S": S, "H": H, "q": int(qc), "k": int(kc), "arm": name, "ms": round(ms, 3),
                                  "aiclk": clk.summary()}), flush=True)
            except Exception as e:  # noqa: BLE001
                print(json.dumps({"S": S, "H": H, "arm": name, "error": str(e)[:300]}), flush=True)
        for t in (q, k, v, mask, sq, sk, sv, sb):
            t.deallocate()


if __name__ == "__main__":
    main()
