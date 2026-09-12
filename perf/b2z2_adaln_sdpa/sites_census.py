#!/usr/bin/env python3
"""What the AdaLN norm of `a` and the token SDPA are actually made of, per call, on this build.

The brief hands this row two sites off `site_rank.py`, which clusters by op code and cost. That
cannot separate `AdaLN.s_terms`'s norm from `AdaLN.__call__`'s: both are `[1, 512, 768]` and they
are 4 us apart. So the first thing here is a per-CALL join, not a cluster: wrap `ttnn.layer_norm`
and `ttnn.transformer.scaled_dot_product_attention` for exactly one settled `Diffusion.__call__`,
record the caller, the shapes, the dtypes and the memory configs, then replay each distinct
(site, shape) standalone to price it and put it on the two roofs.

Isolated replays over-price: `b2z2-step-program-fusion` over-predicted its own step win by 38 %
from an off-fold probe. Every ms here is therefore labelled ISOLATED and nothing is claimed off it
except the SPLIT between arithmetic, bytes and per-program constant, which is a ratio and transfers.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "perf" / "b2z2_step_fusion"))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))
sys.path.insert(0, str(ROOT / "perf" / "b2x_difflayer"))

ELEM = {"BFLOAT16": 2, "FLOAT32": 4, "BFLOAT8_B": 1.0625, "BFLOAT4_B": 0.5625, "UINT32": 4}


def nbytes(shape, dtype) -> float:
    n = 1
    for d in shape:
        n *= int(d)
    return n * ELEM.get(str(dtype).split(".")[-1].upper(), 2)


def caller(skip_prefixes=("ttnn",)) -> str:
    for fr in reversed(traceback.extract_stack()[:-2]):
        if "tenstorrent.py" in fr.filename:
            return f"{Path(fr.filename).name}:{fr.lineno} {fr.name}"
    return "?"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--keep-blocks", type=int, default=2)
    ap.add_argument("--reps", type=int, default=30)
    a = ap.parse_args()

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    import tt_bio.tenstorrent as T
    import tt_baseline as B
    import step_probe as SP

    SP.OUT_PATH = a.out
    SP.OUT["env"] = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                     "card": os.environ.get("TT_VISIBLE_DEVICES"), "size": a.size,
                     "commit": os.popen(f"git -C {ROOT} rev-parse --short HEAD").read().strip(),
                     "profiler": os.environ.get("TT_METAL_DEVICE_PROFILER", "0"),
                     "loadavg": open("/proc/loadavg").read().split()[:3]}
    out = SP.OUT
    a.out.parent.mkdir(parents=True, exist_ok=True)

    dev, g = SP.grab_step(ttnn, T, B, a.size, a.keep_blocks)
    out["env"]["arch"] = str(dev.arch())
    out["env"]["cores"] = f"{dev.compute_with_storage_grid_size().x}x{dev.compute_with_storage_grid_size().y}"
    fence = SP.make_fence(ttnn, dev)

    def run():
        return g["obj"](*g["args"], **g["kwargs"])

    run()                                   # settle before the recording replay
    ttnn.synchronize_device(dev)

    # ---- one recorded replay -------------------------------------------------------------
    ln_calls, sdpa_calls = [], []
    keep = {"ln": [], "sdpa": []}
    orig_ln = ttnn.layer_norm
    orig_sdpa = ttnn.transformer.scaled_dot_product_attention

    def ln_rec(x, **kw):
        rec = {"site": caller(), "shape": list(x.shape), "dtype": str(x.dtype),
               "mc": str(x.memory_config().buffer_type).split(".")[-1],
               "weight": kw.get("weight") is not None, "bias": kw.get("bias") is not None,
               "bytes": 2 * nbytes(x.shape, x.dtype)}
        ln_calls.append(rec)
        if len(keep["ln"]) < 200:
            keep["ln"].append((rec, x, dict(kw)))
        return orig_ln(x, **kw)

    def sdpa_rec(q, k, v, **kw):
        b = kw.get("attn_mask")
        rec = {"site": caller(), "q": list(q.shape), "k": list(k.shape), "v": list(v.shape),
               "bias": list(b.shape) if b is not None else None,
               "dtype": str(q.dtype), "bias_dtype": str(b.dtype) if b is not None else None,
               "qkv_bytes": nbytes(q.shape, q.dtype) + nbytes(k.shape, k.dtype) + nbytes(v.shape, v.dtype),
               "bias_bytes": nbytes(b.shape, b.dtype) if b is not None else 0.0,
               "out_bytes": nbytes(q.shape, q.dtype)}
        h, s, d = int(q.shape[1]), int(q.shape[2]), int(q.shape[3])
        rec["flops"] = 4.0 * h * s * int(k.shape[2]) * d
        sdpa_calls.append(rec)
        if len(keep["sdpa"]) < 60:
            keep["sdpa"].append((rec, (q, k, v), dict(kw)))
        return orig_sdpa(q, k, v, **kw)

    ttnn.layer_norm = ln_rec
    ttnn.transformer.scaled_dot_product_attention = sdpa_rec
    try:
        run()
        ttnn.synchronize_device(dev)
    finally:
        ttnn.layer_norm = orig_ln
        ttnn.transformer.scaled_dot_product_attention = orig_sdpa

    def group(calls, keyf):
        agg = defaultdict(lambda: {"n": 0})
        for c in calls:
            k = keyf(c)
            e = agg[k]
            e["n"] += 1
            e.update({kk: vv for kk, vv in c.items() if kk != "site"})
            e["site"] = c["site"]
        return [dict(key=k, **v) for k, v in sorted(agg.items(), key=lambda kv: -kv[1]["n"])]

    out["layer_norm"] = {"n_calls": len(ln_calls),
                         "sites": group(ln_calls, lambda c: (c["site"], tuple(c["shape"]),
                                                             c["weight"], c["bias"]))}
    out["sdpa"] = {"n_calls": len(sdpa_calls),
                   "sites": group(sdpa_calls, lambda c: (c["site"], tuple(c["q"]),
                                                         tuple(c["bias"] or ())))}
    SP.dump()
    print(f"  layer_norm calls {len(ln_calls)}  sdpa calls {len(sdpa_calls)}", flush=True)
    for s in out["layer_norm"]["sites"]:
        print(f"    LN  n={s['n']:3d} {s['site']:44s} {s['shape']} w={s['weight']} b={s['bias']}", flush=True)
    for s in out["sdpa"]["sites"]:
        print(f"    SDPA n={s['n']:3d} {s['site']:44s} q={s['q']} bias={s['bias']} {s['bias_dtype']}", flush=True)

    # ---- isolated price per distinct (site, shape) ---------------------------------------
    def price(tag, entries, call):
        seen, rows = set(), []
        for rec, operands, kw in entries:
            key = (rec["site"], tuple(rec.get("shape") or rec.get("q")))
            if key in seen:
                continue
            seen.add(key)
            t = SP.timed_reps(ttnn, dev, call, operands, kw, a.reps, fence, n_med=5)
            rows.append({**{kk: vv for kk, vv in rec.items()}, **t})
            print(f"    {tag} ISOLATED {rec['site']:44s} {t['ms_per_call']*1e3:8.2f} us", flush=True)
        return rows

    out["ln_isolated"] = price("LN", keep["ln"], lambda x, **kw: orig_ln(x, **kw))
    out["sdpa_isolated"] = price("SDPA", keep["sdpa"],
                                 lambda qkv, **kw: orig_sdpa(*qkv, **kw))
    SP.dump()

    # ---- A/A floor on the step, so every later ratio has one beside it -------------------
    for _ in range(3):
        run()
    ttnn.synchronize_device(dev)
    aa = []
    for _ in range(6):
        fence()
        t0 = time.perf_counter()
        for _ in range(10):
            run()
        ttnn.synchronize_device(dev)
        aa.append(1e3 * (time.perf_counter() - t0) / 10)
    out["step_ms"] = round(st.median(aa), 4)
    out["step_aa_blocks"] = [round(x, 4) for x in aa]
    out["step_aa_floor"] = round(max(aa) / min(aa), 5)
    SP.dump()
    print(f"\n  step {out['step_ms']:.4f} ms   A/A floor {out['step_aa_floor']:.5f}  {out['step_aa_blocks']}")
    print("DONE", a.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
