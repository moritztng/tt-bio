"""tt_bio.add_ln against ttnn.add + ttnn.layer_norm at SaProt's page shape (7 x 544 rows): the sum must be bit-identical,
the norm is scored against float64 of the same bf16 sum, both paths timed on device in the memory placement --fast ships
(stream, sum and norm in L1, projection in DRAM) and in DRAM.

    python3 perf/spd_m_saprot/add_ln_bench.py --model saprot-650m --out add_ln.jsonl
"""
import argparse
import json
import time

import torch
import ttnn

WIDTH = {"saprot-35m": 480, "saprot-650m": 1280, "saprot-1.3b": 1280}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="saprot-650m", choices=sorted(WIDTH))
    ap.add_argument("--rows", type=int, default=7 * 544)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--inner", type=int, default=20)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_grad_enabled(False)

    from tt_bio import tenstorrent as T
    from tt_bio import saprot as S
    from tt_bio.add_ln import add_ln, STATS, REJECTS
    dev = T.get_device()
    cfg, _ = S.CONFIGS[a.model]
    ck = S.Saprot(**cfg).compute_kernel_config
    d = WIDTH[a.model]
    gen = torch.Generator().manual_seed(0)
    B, L = 7, a.rows // 7
    xa = (torch.randn(B, L, d, generator=gen) * 4).to(torch.bfloat16)   # the residual stream
    xb = torch.randn(B, L, d, generator=gen).to(torch.bfloat16)         # a sublayer's projection
    g = (1 + 0.2 * torch.randn(d, generator=gen)).to(torch.bfloat16)
    be = (0.2 * torch.randn(d, generator=gen)).to(torch.bfloat16)
    up = lambda t, mc: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,  # noqa: E731
                                       memory_config=mc)
    out = open(a.out, "a")

    def timed(fn):
        r = fn()
        ttnn.synchronize_device(dev)
        ts = []
        for _ in range(a.repeat):
            t = time.perf_counter()
            for _ in range(a.inner):
                for o in fn():
                    ttnn.deallocate(o)
            ttnn.synchronize_device(dev)
            ts.append((time.perf_counter() - t) / a.inner * 1e3)
        return r, sorted(ts)[len(ts) // 2]

    def score(y, ref):
        got = ttnn.to_torch(y).double().reshape(ref.shape)
        return (float((got - ref).norm() / ref.norm()),
                float(((got - ref) * ref.sign()).mean() / ref.abs().mean()))

    for place in ("l1", "dram"):
        mc = ttnn.L1_MEMORY_CONFIG if place == "l1" else ttnn.DRAM_MEMORY_CONFIG
        ta, tb = up(xa, mc), up(xb, ttnn.DRAM_MEMORY_CONFIG)   # as the model: stream in L1 (--fast), projection in DRAM
        tg, tbe = (ttnn.from_torch(v.reshape(1, d), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
                   for v in (g, be))
        try:
            (h0, y0), ms0 = timed(lambda: (lambda h: (h, ttnn.layer_norm(h, weight=tg, bias=tbe, epsilon=1e-5,
                                                                         compute_kernel_config=ck, memory_config=mc)))(
                ttnn.add(ta, tb, memory_config=mc)))
            (h1, y1), ms1 = timed(lambda: add_ln(ta, tb, tg, tbe, h_memory_config=mc, y_memory_config=mc))
        except Exception as e:  # noqa: BLE001 -- the reason is the data point
            rec = dict(model=a.model, place=place, error=str(e).splitlines()[0][:300], stats=STATS, rejects=REJECTS)
            print(json.dumps(rec), flush=True); out.write(json.dumps(rec) + "\n"); out.flush()
            continue
        h_ref = ttnn.to_torch(h0).double().reshape(B, L, d)
        h_same = bool(torch.equal(ttnn.to_torch(h1).reshape(B, L, d), ttnn.to_torch(h0).reshape(B, L, d)))
        ref = torch.nn.functional.layer_norm(h_ref, (d,), g.double(), be.double(), eps=1e-5)
        rel0, bias0 = score(y0, ref)
        rel1, bias1 = score(y1, ref)
        rec = dict(model=a.model, place=place, ms_ttnn=round(ms0, 4), ms_add_ln=round(ms1, 4), h_identical=h_same,
                   rel_l2_ttnn=rel0, rel_l2_add_ln=rel1, bias_ttnn=bias0, bias_add_ln=bias1, stats=list(STATS))
        print(json.dumps(rec), flush=True)
        out.write(json.dumps(rec) + "\n"); out.flush()
        for t in (ta, tb, h0, y0, h1, y1):
            ttnn.deallocate(t)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
