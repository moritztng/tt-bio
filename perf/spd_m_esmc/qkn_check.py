"""ESMC's q/k LayerNorm moved into mm_rope (tt_bio/mm_rope.qkv_rope_heads(qk_norm=True)): accuracy and device time.

Today a layer runs LayerNorm + qkv split + qk_rope_heads (q/k LayerNorm, RoPE, head split). The fused arm folds the
norm's centring into the qkv weight, multiplies by the norm weight and rotates in the matmul epilogue, and writes each
row's partial sums of q^2 / k^2; qk_rstd turns them into the rows' rstd, which SDPA will apply (here: on the host, to
grade). Both arms are graded against a float64 reference of the same layer from the same bf16 input:

    rel_l2(today q, ref), rel_l2(fused q * rstd_q, ref), the same for k, and v against today's v
    device ms a call (trace replay): today, fused (mm_rope + qk_rstd)

    python3 perf/spd_m_esmc/qkn_check.py --model esmc-300m [--fast] [--pad 18] --out qkn.jsonl
"""
import argparse
import json

import torch
import ttnn

from tt_bio import esmc, mm_rope, shsdpa
from tt_bio.qk_rope_heads import qk_rope_heads

from packrows import dev_ms

BATCH = {"esmc-300m": 7, "esmc-600m": 7, "esmc-6b": 1}
SEQ = 544


def rel(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm())


def ln(t, g=None, eps=1e-5):
    t = (t - t.mean(-1, keepdim=True)) / torch.sqrt(t.var(-1, unbiased=False, keepdim=True) + eps)
    return t if g is None else t * g


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="esmc-300m", choices=sorted(BATCH))
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--pad", type=int, default=0, help="pad q, k, v to this many heads (300M: 18 fills 9 in1 cores)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--diag", action="store_true", help="split the producer's cost into its parts, skip the SDPA stage")
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
    xt = torch.randn(B, SEQ, d).bfloat16()
    x = ttnn.from_torch(xt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    cos, sin = esmc.rope_tables(SEQ, 64, device=dev)
    od, qd = esmc._qkrh_dtypes(attn.out_weight, attn.mm_exact_acc, d, rows)
    gq = ttnn.to_torch(attn.q_ln_weight).double().reshape(-1)[:d]
    gk = ttnn.to_torch(attn.k_ln_weight).double().reshape(-1)[:d]

    def today():
        q, k, v = attn._qkv(x, ln_mc)
        heads = qk_rope_heads(q, k, v, attn.q_ln_weight, attn.k_ln_weight, cos, sin, out_dtype=od, q_dtype=qd)
        assert heads is not None, "qk_rope_heads declined"
        for t in (q, k, v):
            ttnn.deallocate(t)
        return heads

    w = attn.qkv_weight
    wt = ttnn.to_torch(w).double().reshape(int(w.shape[-2]), int(w.shape[-1]))
    bt = (ttnn.to_torch(attn.qkv_bias).double().reshape(-1)[: 3 * d] if attn.qkv_bias is not None
          else torch.zeros(3 * d, dtype=torch.float64))

    # float64 reference of the layer the device runs (its LN-folded weight when ln_fold)
    if attn.ln_fold:
        xn = ln(xt.double())
    else:
        xn = ln(xt.double(), ttnn.to_torch(attn.in_norm_weight).double().reshape(-1)[:d])
        xn = xn + ttnn.to_torch(attn.in_norm_bias).double().reshape(-1)[:d]
    y = xn.reshape(rows, d) @ wt + bt
    c = ttnn.to_torch(cos).double().reshape(-1, 64)[:SEQ]
    s = ttnn.to_torch(sin).double().reshape(-1, 64)[:SEQ]

    def heads_ref(t):   # [rows, d] -> [B, H, L, 64] after NeoX RoPE
        t = t.reshape(B, SEQ, H, 64).permute(0, 2, 1, 3)
        rot = torch.cat([-t[..., 32:], t[..., :32]], -1)
        return t * c + rot * s
    ref = (heads_ref(ln(y[:, :d], gq)), heads_ref(ln(y[:, d:2 * d], gk)))

    mmcfg = esmc._mm_config(x, w, attn.mm_exact_acc)["config"]
    cfg = (mmcfg.M_block_size, mmcfg.K_block_size, mmcfg.N_block_size, mmcfg.subblock_h, mmcfg.subblock_w)
    g = dev.compute_with_storage_grid_size()
    ck = attn._mm_ck(w)
    wf, bg = mm_rope.qk_norm_fold(wt, bt, gq, gk, d)
    hp = args.pad or H
    if hp != H:
        pad = lambda t: torch.cat([torch.cat([u, torch.zeros(*u.shape[:-1], (hp - H) * 64)], -1)
                                   for u in t.split(d, -1)], -1)
        wf, bg = pad(wf), pad(bg)
    n = wf.shape[-1]
    transposed = rows > n
    nc = mm_rope.cols(n // 32, g.y if transposed else g.x, cfg[2])
    assert nc is not None, "mm_rope cannot split this qkv"
    wd = ttnn.from_torch(mm_rope.permute(wf, nc, cfg[2]), dtype=w.dtype, layout=ttnn.TILE_LAYOUT, device=dev)
    bd = ttnn.from_torch(mm_rope.permute(bg, nc, cfg[2]), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    grid = (g.x, nc) if transposed else (nc, g.y)

    def fused():
        xn_ = esmc._ln(x, *((None, None) if attn.ln_fold else (attn.in_norm_weight, attn.in_norm_bias)),
                       attn.compute_kernel_config, ln_mc)
        out = mm_rope.qkv_rope_heads(xn_, wd, bd, cos, sin, hp, cfg, grid, ck, ncols=nc, qk_norm=True)
        ttnn.deallocate(xn_)
        assert out is not None, "mm_rope declined"
        q, k, v, st = out
        r = mm_rope.qk_rstd(st, d)
        ttnn.deallocate(st)
        return q, k, v, r

    rec = dict(model=args.model, fast=args.fast, pad=hp, rows=rows, cfg=cfg, grid=grid, ncols=nc, w_dtype=str(w.dtype),
               n_layers=len(mod.blocks))
    tq, tk, tv = [ttnn.to_torch(t).double() for t in today()]
    q, k, v, r = fused()
    r = ttnn.to_torch(r).double().reshape(rows, -1)
    rq, rk = r[:, 0].reshape(B, 1, SEQ, 1), r[:, 32].reshape(B, 1, SEQ, 1)
    fq, fk, fv = [ttnn.to_torch(t).double()[:, :H] for t in (q, k, v)]
    for t in (q, k, v):
        ttnn.deallocate(t)
    ref_r = (1 / torch.sqrt(y[:, :d].var(-1, unbiased=False) + 1e-5)).reshape(B, 1, SEQ, 1)
    rec.update(today_q=rel(tq, ref[0]), today_k=rel(tk, ref[1]), fused_q=rel(fq * rq, ref[0]),
               fused_k=rel(fk * rk, ref[1]), rstd_q=rel(rq, ref_r), v_vs_today=rel(fv, tv))
    print(json.dumps(rec), flush=True)
    rec["today_ms"] = dev_ms(today, dev)
    rec["fused_ms"] = dev_ms(fused, dev)
    rec["saved_page_ms"] = round((rec["today_ms"] - rec["fused_ms"]) * rec["n_layers"], 3)
    print(json.dumps(rec), flush=True)
    with open(args.out, "a") as f:
        f.write(json.dumps(rec) + "\n")

    if args.diag:   # device ms a layer of each producer part; the QKN_DIAG_* arms give wrong output on purpose
        wp = ttnn.from_torch(mm_rope.permute(wf, nc, cfg[2]), dtype=w.dtype, layout=ttnn.TILE_LAYOUT, device=dev)
        bp = ttnn.from_torch(mm_rope.permute(bg[:1], nc, cfg[2]), dtype=ttnn.bfloat16,
                             layout=ttnn.TILE_LAYOUT, device=dev)
        drec = {k_: rec[k_] for k_ in ("model", "fast", "pad", "rows", "n_layers")}
        drec["stage"] = "diag"
        # LN first: an L1-resident xn0 left alive sits inside layer_norm's static CB region (TT_THROW, qd1 11-19)
        drec["ln_ms"] = dev_ms(lambda: (esmc._ln(x, *((None, None) if attn.ln_fold else (attn.in_norm_weight,
                               attn.in_norm_bias)), attn.compute_kernel_config, ln_mc),), dev)
        xn0 = esmc._ln(x, *((None, None) if attn.ln_fold else (attn.in_norm_weight, attn.in_norm_bias)),
                       attn.compute_kernel_config, ln_mc)

        def rope_only(qkn, defs=()):
            def fn():
                mm_rope.PROBE_DEFINES.update({k_: 1 for k_ in defs})
                try:
                    out = mm_rope.qkv_rope_heads(xn0, wd if qkn else wp, bd if qkn else bp, cos, sin, hp, cfg, grid,
                                                 ck, ncols=nc, qk_norm=qkn)
                finally:
                    mm_rope.PROBE_DEFINES.clear()
                return tuple(out)
            return fn

        drec["rope_plain_ms"] = dev_ms(rope_only(False), dev)
        drec["rope_qkn_ms"] = dev_ms(rope_only(True), dev)
        drec["rope_qkn_nostat_ms"] = dev_ms(rope_only(True, ("QKN_DIAG_NOSTAT",)), dev)
        drec["rope_qkn_nosq_ms"] = dev_ms(rope_only(True, ("QKN_DIAG_NOSQ",)), dev)
        drec["rope_qkn_nosq_nostat_ms"] = dev_ms(rope_only(True, ("QKN_DIAG_NOSQ", "QKN_DIAG_NOSTAT")), dev)
        drec["rope_probe_ms"] = dev_ms(rope_only(False, ("MM_ROPE_STATS_PROBE",)), dev)
        q_, k_, v_, st_ = rope_only(True)()
        for t in (q_, k_, v_, xn0):
            ttnn.deallocate(t)
        drec["rstd_ms"] = dev_ms(lambda: (mm_rope.qk_rstd(st_, d),), dev)
        print(json.dumps(drec), flush=True)
        with open(args.out, "a") as f:
            f.write(json.dumps(drec) + "\n")
        return 0

    # The consumer: shsdpa with qk_rstd (Q *= rq, K *= rk in the kernel) on the fused q, k, v against today's shsdpa on
    # today's, both graded against float64 attention over the page's 514 real keys; plus the fused heads without rstd
    # (wrong output) to isolate the scale's own cost.
    L = 514
    table, row = shsdpa.key_pad([L] * B, SEQ, dev)
    pad_ = (table, row)
    scale = 64 ** -0.5
    vref = y[:, 2 * d:].reshape(B, SEQ, H, 64).permute(0, 2, 1, 3)
    sc = (ref[0] @ ref[1].transpose(-1, -2)) * scale
    sc[..., L:] = float("-inf")
    oref = torch.softmax(sc, -1) @ vref

    def att_today():
        q, k, v = today()
        o = shsdpa.sdpa(q, k, v, scale, pad_, out_dtype=ttnn.bfloat16)
        for t in (q, k, v):
            ttnn.deallocate(t)
        return (o,)

    def att_fused(use_r=True):
        q, k, v, r = fused()
        o = shsdpa.sdpa(q, k, v, scale, pad_, out_dtype=ttnn.bfloat16, qk_rstd=r if use_r else None)
        for t in (q, k, v, r):
            ttnn.deallocate(t)
        return (o,)

    rec2 = {k_: rec[k_] for k_ in ("model", "fast", "pad", "rows", "n_layers")}
    rec2["stage"] = "sdpa"
    ot = ttnn.to_torch(att_today()[0]).double()[:, :H, :L]
    of = ttnn.to_torch(att_fused()[0]).double()[:, :H, :L]
    rec2.update(att_today=rel(ot, oref[:, :, :L]), att_fused=rel(of, oref[:, :, :L]), att_fused_vs_today=rel(of, ot))
    print(json.dumps(rec2), flush=True)
    rec2["att_today_ms"] = dev_ms(att_today, dev)
    rec2["att_fused_ms"] = dev_ms(att_fused, dev)
    rec2["att_fused_nor_ms"] = dev_ms(lambda: att_fused(False), dev)
    rec2["saved_page_ms"] = round((rec2["att_today_ms"] - rec2["att_fused_ms"]) * rec2["n_layers"], 3)
    rec2["rstd_cost_page_ms"] = round((rec2["att_fused_ms"] - rec2["att_fused_nor_ms"]) * rec2["n_layers"], 3)
    print(json.dumps(rec2), flush=True)
    with open(args.out, "a") as f:
        f.write(json.dumps(rec2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
