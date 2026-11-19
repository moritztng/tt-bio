"""Ceiling of moving ESMC's RoPE and head split into the qkv matmul (tt_bio/mm_rope, as SaProt runs it).

Today a layer runs LayerNorm + minimal_matmul_split (q, k, v rows to DRAM) + qk_rope_heads (q/k LayerNorm, RoPE, head
split; reads them back). mm_rope writes roped heads straight from the matmul epilogue. ESMC's q/k LayerNorm is a per-row
scale after centring, which commutes with RoPE, so a real lever folds the centring into the weight and applies the row
rstd inside SDPA; this probe leaves the norm out (q, k unnormalised) and times only what the fusion can save:

    today   = attn._qkv (LN + qkv split) + qk_rope_heads, the shipped dtypes
    mmrope  = the same LN + mm_rope.qkv_rope_heads on the device's own (LN-folded) qkv weight, columns permuted
    pad18   = 300M only: 3 zero heads added to each of q, k, v so the qkv splits whole heads over all 9 in1 cores

v has no norm, so mmrope's v heads must match today's (rel_l2 printed). Device ms a call by trace replay.
*_stats: the same arm with MM_ROPE_STATS_PROBE, the q/k LayerNorm producer's epilogue math (g multiply, diag(y y^T)
accumulation) with its results discarded: what the real producer will cost on top, minus its tiny stats write.

    python3 perf/spd_m_esmc/ropeceil.py --model esmc-600m [--fast] --out ropeceil.jsonl
"""
import argparse
import json

import torch
import ttnn

from tt_bio import esmc, mm_rope
from tt_bio.qk_rope_heads import qk_rope_heads

from packrows import dev_ms

BATCH = {"esmc-300m": 7, "esmc-600m": 7, "esmc-6b": 1}
SEQ = 544


def rel(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="esmc-600m", choices=sorted(BATCH))
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    model = esmc.load_esmc(args.model, fast=args.fast, trace=True)
    mod = model.module
    attn = mod.blocks[len(mod.blocks) // 2].attn
    dev = attn.out_weight.device()
    d = int(attn.out_weight.shape[-1])
    H = attn.n_heads
    B = BATCH[args.model]
    rows = B * SEQ
    ln_mc = ttnn.L1_MEMORY_CONFIG if (esmc._ESMC_QK_L1 and attn.qk_l1) else None
    torch.manual_seed(0)
    x = ttnn.from_torch(torch.randn(B, SEQ, d), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    cos, sin = esmc.rope_tables(SEQ, 64, device=dev)
    od, qd = esmc._qkrh_dtypes(attn.out_weight, attn.mm_exact_acc, d, rows)

    def today():
        q, k, v = attn._qkv(x, ln_mc)
        heads = qk_rope_heads(q, k, v, attn.q_ln_weight, attn.k_ln_weight, cos, sin, out_dtype=od, q_dtype=qd)
        assert heads is not None, "qk_rope_heads declined"
        for t in (q, k, v):
            ttnn.deallocate(t)
        return heads

    w = attn.qkv_weight
    wt = ttnn.to_torch(w).float().reshape(int(w.shape[-2]), int(w.shape[-1]))
    bt = (ttnn.to_torch(attn.qkv_bias).float().reshape(-1)[: 3 * d] if attn.qkv_bias is not None
          else torch.zeros(3 * d))
    mmcfg = esmc._mm_config(x, w, attn.mm_exact_acc)["config"]
    cfg = (mmcfg.M_block_size, mmcfg.K_block_size, mmcfg.N_block_size, mmcfg.subblock_h, mmcfg.subblock_w)
    g = dev.compute_with_storage_grid_size()
    ck = attn._mm_ck(w)

    def arm(wt_, bt_, heads):
        n = wt_.shape[-1]
        transposed = rows > n
        axis = g.y if transposed else g.x
        nc = mm_rope.cols(n // 32, axis, cfg[2])
        if nc is None:
            return None, None
        wd = ttnn.from_torch(mm_rope.permute(wt_, nc, cfg[2]), dtype=w.dtype, layout=ttnn.TILE_LAYOUT, device=dev)
        bd = ttnn.from_torch(mm_rope.permute(bt_[None], nc, cfg[2]), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                             device=dev)
        grid = (g.x, nc) if transposed else (nc, g.y)

        def fn():
            xn = esmc._ln(x, *((None, None) if attn.ln_fold else (attn.in_norm_weight, attn.in_norm_bias)),
                          attn.compute_kernel_config, ln_mc)
            out = mm_rope.qkv_rope_heads(xn, wd, bd, cos, sin, heads, cfg, grid, ck, ncols=nc)
            ttnn.deallocate(xn)
            assert out is not None, "mm_rope declined"
            return out
        return fn, (nc, grid)

    rec = dict(model=args.model, fast=args.fast, rows=rows, cfg=cfg, grid=(g.x, g.y), w_dtype=str(w.dtype),
               qkrh_dtypes=[str(od), str(qd)], n_layers=len(mod.blocks))
    rec["today"] = dev_ms(today, dev)
    ref = today()
    fn, where = arm(wt, bt, H)
    if fn is not None:
        rec["mmrope_cols"] = where
        rec["mmrope"] = dev_ms(fn, dev)
        got = fn()
        rec["v_rel_l2"] = rel(ttnn.to_torch(got[2]).float(), ttnn.to_torch(ref[2]).float())
        for t in got:
            ttnn.deallocate(t)
    else:
        rec["mmrope"] = None
    def stats(fn):   # the same program plus the q/k LayerNorm producer's math (MM_ROPE_STATS_PROBE, wrong output)
        mm_rope.PROBE_DEFINES["MM_ROPE_STATS_PROBE"] = 1
        try:
            return dev_ms(fn, dev)
        finally:
            mm_rope.PROBE_DEFINES.clear()

    if rec["mmrope"] is not None:
        fn, _ = arm(wt, bt, H)
        rec["mmrope_stats"] = stats(fn)
    if args.model == "esmc-300m":   # 15 heads split over 6 of 9 in1 cores; 18 (3 zero) over all 9
        hp = 18
        pad = lambda t: torch.cat([torch.cat([c, torch.zeros(*c.shape[:-1], (hp - H) * 64)], -1)
                                   for c in t.split(d, -1)], -1)
        fn, where = arm(pad(wt), pad(bt[None])[0], hp)
        if fn is not None:
            rec["pad18_cols"] = where
            rec["pad18"] = dev_ms(fn, dev)
            rec["pad18_stats"] = stats(fn)
    for t in ref:
        ttnn.deallocate(t)
    for k in ("mmrope", "pad18", "mmrope_stats", "pad18_stats"):
        if rec.get(k):
            rec[f"{k}_saved_page_ms"] = round((rec["today"] - rec[k]) * rec["n_layers"], 3)
    print(json.dumps(rec), flush=True)
    with open(args.out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
