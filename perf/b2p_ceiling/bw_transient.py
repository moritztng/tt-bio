#!/usr/bin/env python3
"""What the triangle-attention backward's chunked-recompute fallback adds to the device, per call.

On a Wormhole Galaxy chip `triatt_bw` declines every call from 352 up (its bias + fp32 dbias
buffers are Nt^2 tiles each whatever the query chunk), so the backward runs
`autograd.triangle_attention`'s recompute. That loop bounds each recomputed score block by
`taped_ttnn.SDPA_SCORE_BUDGET`, a fixed byte count, and a block is live alongside its softmax,
dP, dS and the softmax-backward temporaries. This measures the transient that costs at BC2's
shape, q/k/v [N, 4, N, 32] with the full leading axis, per budget: DRAM in use after the forward
(the resident floor) against the high-water mark during the backward, sampled before every
`ttnn.deallocate`, which is where each block is at its largest. A second pass per budget is timed
with sampling off. Gradients are compared across budgets (they must agree: chunking is a memory
decision) and against float64: dq/dk/dv on the first `--grade-rows` leading rows,
dbias over all of them, since it is the one gradient summed across leading chunks.
"""
import argparse, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ["TT_BIO_TAPED_KERNELS"] = "tri_att_sdpa_hifi"
H, D = 4, 32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=544)
    ap.add_argument("--budgets-mb", default="256,128,64,32")
    ap.add_argument("--grade-rows", type=int, default=16)
    ap.add_argument("--out", type=Path, default=Path(__file__).with_name("bw_transient.json"))
    a = ap.parse_args()
    import torch, ttnn
    import tt_bio.tenstorrent as T
    import tt_bio.autograd as ag
    import tt_bio.taped_ttnn as TT
    import tt_bio.triatt_bw as TBW
    assert Path(T.__file__).resolve().is_relative_to(ROOT), T.__file__
    dev = T.get_device()
    n, scale = a.n, D ** -0.5
    up = lambda t: ttnn.from_torch(t.float(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                   device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    g_ = torch.Generator().manual_seed(n)
    mk = lambda *s: torch.randn(*s, generator=g_, dtype=torch.float32).to(torch.bfloat16)
    q, k, v, g = (mk(n, H, n, D) for _ in range(4))
    bias = mk(1, H, n, n)

    def used():
        mv = ttnn.get_memory_view(dev, ttnn.BufferType.DRAM)
        nb = int(mv.num_banks)
        return (int(mv.total_bytes_per_bank) - int(mv.total_bytes_free_per_bank)) * nb

    peak = {"v": 0, "on": False}
    real_dealloc = ttnn.deallocate

    def dealloc(t, *aa, **kw):
        if peak["on"]:
            peak["v"] = max(peak["v"], used())
        return real_dealloc(t, *aa, **kw)
    ttnn.deallocate = dealloc

    def one(budget, sample):
        TT.SDPA_SCORE_BUDGET = budget
        leaves = [ag.Tensor(up(t), requires_grad=True) for t in (q, k, v, bias)]
        gd = up(g)
        with ag.tape():
            out = T._tri_att_sdpa_hifi(*leaves, scale)
            assert out is not None, "fused forward declined"
        ttnn.synchronize_device(dev)
        floor = used()
        peak["v"], peak["on"] = floor, sample
        s0 = dict(TBW.STATS)
        t0 = time.perf_counter()
        ag.backward([out], [gd])
        ttnn.synchronize_device(dev)
        dt = time.perf_counter() - t0
        peak["on"] = False
        grads = [ttnn.to_torch(t.grad).float() for t in leaves]
        for t in leaves:
            for x in (t.value, t.grad):
                if x is not None:
                    real_dealloc(x)
        real_dealloc(out.value if hasattr(out, "value") else out)
        real_dealloc(gd)
        return floor, peak["v"], dt, grads, {kk: TBW.STATS.get(kk, 0) - s0.get(kk, 0) for kk in TBW.STATS}

    # float64 reference on the first rows. dq/dk/dv are per leading row; dbias sums over rows, so it
    # is graded against the device's dbias only through dq/dk/dv here and across budgets below.
    r = a.grade_rows
    qr, kr, vr, br = (t[:r].double().clone().requires_grad_(True) if t.shape[0] == n
                      else t.double().clone().requires_grad_(True) for t in (q, k, v, bias))
    o = torch.softmax((qr @ kr.transpose(-1, -2) + br) * scale, -1) @ vr
    o.backward(g[:r].double())
    ref = {"dq": qr.grad, "dk": kr.grad, "dv": vr.grad}
    # dbias sums every leading row, so its reference does too, built in row chunks:
    # d/dbias of softmax((qk + bias) * scale) is scale * sum_b P * (dP - rowsum(dP * P)).
    db64 = torch.zeros(1, H, n, n, dtype=torch.float64)
    for b0 in range(0, n, 32):
        qb, kb, vb, gb = (t[b0:b0 + 32].double() for t in (q, k, v, g))
        P = torch.softmax((qb @ kb.transpose(-1, -2) + bias.double()) * scale, -1)
        dP = gb @ vb.transpose(-1, -2)
        db64 += scale * (P * (dP - (dP * P).sum(-1, keepdim=True))).sum(0, keepdim=True)
    ref_db = db64

    rows, base = [], None
    for mb in [int(x) for x in a.budgets_mb.split(",")]:
        b = mb << 20
        cB, cQ = TT._sdpa_chunking(n, H, n, n, 2, budget=b)
        floor, pk, _dt, grads, st = one(b, True)
        _f2, _p2, dt, _g2, _s2 = one(b, False)
        dq, dk, dv, db = grads
        f64 = {nm: float((x[:r].double() - ref[nm]).norm() / ref[nm].norm())
               for nm, x in (("dq", dq), ("dk", dk), ("dv", dv))}
        f64["dbias_all_rows"] = float((db.double() - ref_db).norm() / ref_db.norm())
        if base is None:
            base = grads
        vs = {nm: float((x - y).norm() / y.norm()) for nm, x, y in
              zip(("dq", "dk", "dv", "dbias"), grads, base)}
        rec = {"n": n, "budget_mb": mb, "chunk_leading": cB, "chunk_query": cQ,
               "resident_after_fwd_gb": round(floor / 1e9, 4), "peak_in_bw_gb": round(pk / 1e9, 4),
               "transient_gb": round((pk - floor) / 1e9, 4), "bw_s_unsampled": round(dt, 3),
               "triatt_bw_stats": st, "rel_l2_vs_f64_first_rows": f64,
               "rel_l2_vs_first_budget": vs}
        rows.append(rec)
        print(json.dumps(rec), flush=True)
        a.out.write_text(json.dumps({"heads": H, "head_dim": D, "rows": rows}, indent=1))
    T.cleanup()


if __name__ == "__main__":
    main()
