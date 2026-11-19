"""Ceiling of sequence packing for the ESMC page batch: device ms of one layer's row-wise ops at R rows.

The page runs 7 sequences of 514 tokens, each padded to 544 (3808 rows, 119 tiles). Packed end to end they are
3598 rows, 113 tiles. Every op of a layer except rope/head split and SDPA works row by row, so packing saves
whatever these ops lose between 3808 and 3616 rows. This times them through the shipped paths on random
activations, one block's real weights:

    attention LN + qkv split (Attention._qkv), out_proj + residual (_lin_resid), FFN + residual (SwiGLUFFN).

    python3 perf/spd_m_esmc/packrows.py --model esmc-300m [--fast] --out packrows.jsonl
"""
import argparse
import json
import statistics
import time

import torch
import ttnn

from tt_bio import esmc


def dev_ms(fn, dev, n=20):
    """Device ms a call: n calls captured in one trace and replayed."""
    for _ in range(2):
        for t in fn():
            ttnn.deallocate(t)
    ttnn.synchronize_device(dev)
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    try:
        outs = [fn() for _ in range(n)]
    finally:   # a throw inside an open capture otherwise hangs the process at exit (drain refuses during capture)
        ttnn.end_trace_capture(dev, tid, cq_id=0)
    ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
    t = []
    for _ in range(7):
        t0 = time.perf_counter()
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
        t.append((time.perf_counter() - t0) * 1000 / n)
    ttnn.release_trace(dev, tid)
    for o in outs:
        for x in o:
            ttnn.deallocate(x)
    return round(statistics.median(t), 4)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="esmc-300m")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--rows", default="3808,3616,3584,3456")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    model = esmc.load_esmc(args.model, fast=args.fast, trace=True)
    mod = model.module
    blk = mod.blocks[len(mod.blocks) // 2]
    attn, ffn = blk.attn, blk.ffn
    d = int(ffn.fc2_weight.shape[-1])
    n_layers = len(mod.blocks)
    dev = ffn.fc2_weight.device()
    ln_mc = ttnn.L1_MEMORY_CONFIG if (esmc._ESMC_QK_L1 and attn.qk_l1) else None
    torch.manual_seed(0)
    rec = []
    for rows in [int(r) for r in args.rows.split(",")]:
        x = ttnn.from_torch(torch.randn(1, rows, d), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        o = ttnn.from_torch(torch.randn(1, rows, d) * 0.1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)

        def qkv():
            return attn._qkv(x, ln_mc)

        def out_resid():
            r = attn._lin_resid(o, attn.out_weight, x)
            return (r if r is not None else ttnn.add(x, attn._lin(o, attn.out_weight))),

        def ffn_resid():
            r, added = ffn(x, resid=True)
            return (r if added else ttnn.add(x, r)),

        row = dict(model=args.model, fast=args.fast, rows=rows, tiles=rows // 32, n_layers=n_layers)
        for name, fn in (("qkv", qkv), ("out_resid", out_resid), ("ffn_resid", ffn_resid)):
            row[name] = dev_ms(fn, dev)
        row["layer"] = round(row["qkv"] + row["out_resid"] + row["ffn_resid"], 4)
        row["page_ms"] = round(row["layer"] * n_layers, 3)
        rec.append(row)
        print(json.dumps(row), flush=True)
        ttnn.deallocate(x)
        ttnn.deallocate(o)
    base = rec[0]["page_ms"]
    for r in rec[1:]:
        print(f"{r['rows']} rows: {base - r['page_ms']:+.3f} ms a page saved on the row ops", flush=True)
    with open(args.out, "a") as f:
        for r in rec:
            f.write(json.dumps(r) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
