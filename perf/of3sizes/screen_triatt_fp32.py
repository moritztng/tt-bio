#!/usr/bin/env python3
"""Screen: OpenFold3 TriangleAttention's fp32-softmax attention at 512 / 768 / 1024 aa.

OpenFold3 sets `fp32_softmax=True` on every TriangleAttention it owns (openfold3_trunk.py:130,
openfold3_template.py:110, msa, confidence), so the whole SDPA lever family (K2, wide-q) is off the
path and `_fp32_softmax_attention` (tenstorrent.py:745) is what runs. That function is UNCHUNKED: it
materialises `sc = q @ k^T` of shape [S, n_heads, S, S] and then holds fp32 copies of it. Elements
are n_heads * S**3, i.e. CUBIC in the token count, so this screen asks two questions at the real
trunk shapes (n_heads=4, head_dim=32, _PF_DIMS in openfold3_trunk.py:54):

  1. CAPACITY. Does the unchunked call still allocate at S=768 and S=1024 on a 32 GB p150a?
  2. PRICE. What does a row-blocked variant cost, and is it bit-exact? Rows of the leading dim are
     independent (the softmax reduces over the last dim only), so blocking them cannot change a
     value -- `torch.equal` is the bar, not a tolerance.

Writes incrementally so a refusal at one size keeps every earlier row.
"""
import json, os, sys, time
from pathlib import Path

import torch, ttnn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tt_bio.tenstorrent import get_device, _fp32_softmax_attention, batched_matmul  # noqa: E402

SIZES = tuple(int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else ("512", "768", "1024")))
OUT = Path(__file__).resolve().parent / (
    "screen_triatt_fp32_qb1c%s" % os.environ.get("TT_VISIBLE_DEVICES", "0")
    + ("" if SIZES == (512, 768, 1024) else "_" + "_".join(str(x) for x in SIZES)) + ".json")
H, DH = 4, 32                 # _PF_DIMS = (head_dim=32, n_heads=4, ...)
SCALE_INV = DH ** -0.5
BLOCK_BUDGET = 1 << 30        # 1 GiB of live fp32 score per block


def blocked(q, k, v, bias, scale_inv, ckc, rows):
    """`_fp32_softmax_attention` over row blocks of the leading dim. Bit-exact by construction."""
    S = int(q.shape[0])
    parts = []
    for s in range(0, S, rows):
        e = min(s + rows, S)
        parts.append(_fp32_softmax_attention(
            q[s:e], k[s:e], v[s:e], bias, scale_inv=scale_inv,
            compute_kernel_config=ckc, out_dtype=ttnn.bfloat16, bias_scale_inv=1.0))
    if len(parts) == 1:
        return parts[0]
    o = ttnn.concat(parts, dim=0)
    for p in parts:
        ttnn.deallocate(p)
    return o


def timed(fn, reps=3):
    ttnn.synchronize_device(get_device())
    fn_out = fn()                       # warm (compile + program cache)
    ttnn.synchronize_device(get_device())
    ttnn.deallocate(fn_out)
    ts = []
    for _ in range(reps):
        ttnn.synchronize_device(get_device())
        t0 = time.perf_counter()
        out = fn()
        ttnn.synchronize_device(get_device())
        ts.append(time.perf_counter() - t0)
        if _ < reps - 1:
            ttnn.deallocate(out)
    ts.sort()
    return ts[len(ts) // 2], ts, out


def main():
    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    res = {"host": "qb1", "card": os.environ.get("TT_VISIBLE_DEVICES", "0"),
           "ttnn": "0.67.4", "n_heads": H, "head_dim": DH, "rows": []}
    OUT.write_text(json.dumps(res, indent=1))

    for S in SIZES:
        rows = max(32, int(BLOCK_BUDGET // (H * S * S * 4)) // 32 * 32)
        rows = min(rows, S)
        row = {"S": S, "block_rows": rows,
               "score_elems": H * S ** 3,
               "score_fp32_GiB": H * S ** 3 * 4 / 2 ** 30,
               "score_bf16_GiB": H * S ** 3 * 2 / 2 ** 30}
        torch.manual_seed(S)
        try:
            qh, kh, vh = (torch.randn(S, H, S, DH, dtype=torch.bfloat16) * 0.1 for _ in range(3))
            bh = torch.randn(1, H, S, S, dtype=torch.bfloat16) * 0.1
            mk = lambda t: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev,
                                           dtype=ttnn.bfloat16,
                                           memory_config=ttnn.DRAM_MEMORY_CONFIG)
            q, k, v, bias = mk(qh), mk(kh), mk(vh), mk(bh)
            row["qkv_MiB_each"] = S * H * S * DH * 2 / 2 ** 20
        except Exception as e:                                          # noqa: BLE001
            row["operands"] = f"REFUSED: {type(e).__name__}: {str(e)[:300]}"
            res["rows"].append(row); OUT.write_text(json.dumps(res, indent=1)); continue

        # blocked first, so a refusal of the unchunked form cannot cost the blocked number
        try:
            t, ts, ob = timed(lambda: blocked(q, k, v, bias, SCALE_INV, ckc, rows))
            row["blocked_ms"] = t * 1e3
            row["blocked_all_ms"] = [x * 1e3 for x in ts]
            ob_h = ttnn.to_torch(ob)
            ttnn.deallocate(ob)
        except Exception as e:                                          # noqa: BLE001
            row["blocked_ms"] = f"REFUSED: {type(e).__name__}: {str(e)[:300]}"
            ob_h = None
        res["rows"].append(row); OUT.write_text(json.dumps(res, indent=1))

        try:
            t, ts, of = timed(lambda: _fp32_softmax_attention(
                q, k, v, bias, scale_inv=SCALE_INV, compute_kernel_config=ckc,
                out_dtype=ttnn.bfloat16, bias_scale_inv=1.0))
            row["full_ms"] = t * 1e3
            row["full_all_ms"] = [x * 1e3 for x in ts]
            of_h = ttnn.to_torch(of)
            ttnn.deallocate(of)
            if ob_h is not None:
                row["torch_equal"] = bool(torch.equal(of_h, ob_h))
                row["max_abs_diff"] = float((of_h.float() - ob_h.float()).abs().max())
            if isinstance(row.get("blocked_ms"), float):
                row["blocked_over_full"] = row["blocked_ms"] / row["full_ms"]
        except Exception as e:                                          # noqa: BLE001
            row["full_ms"] = f"REFUSED: {type(e).__name__}: {str(e)[:400]}"

        for t_ in (q, k, v, bias):
            try: ttnn.deallocate(t_)
            except Exception: pass
        OUT.write_text(json.dumps(res, indent=1))
        print(json.dumps(row, indent=1), flush=True)

    print("wrote", OUT)
    return 0


sys.exit(main())
