"""Triangle-attention SDPA at BoltzGen's padded token length, incumbent against every
(q_chunk, k_chunk) that fits L1.

The incumbent arm is `_tri_att_sdpa_at` itself under the shipped env, so the pair it reports in
`SDPA_CHUNK_PICKS` is the pair the fold runs -- screening `_sdpa_chunks_shipped`'s return value
instead measures a config the fold never executes (`_sdpa_chunks_shipped` docstring, the K4
inversion). Candidate arms name their pair explicitly and go through the same two routes the
ladder uses: the fused K1/K2 kernel in `triatt_sdpa`, and the stock ttnn op.

Arms are interleaved round-robin, the first iteration of each is discarded, and every timed
region ends on `ttnn.synchronize_device`. Accuracy is the arm's output against an fp32 torch
evaluation of the SAME bf16 operands, on the first `--acc-rows` batch rows only (the full score
tensor at S=2208 is 86 TB).

    python3 perf/bgsdpa/sdpa_ab.py --seq 2208 --iters 3 --out ab2208.json
"""
import argparse
import json
import os
import sys
import time

import torch
import ttnn

sys.path.insert(0, os.environ.get("WT", os.getcwd()))
from tt_bio import tenstorrent as T          # noqa: E402
from tt_bio import triatt_sdpa as TS         # noqa: E402
from tt_bio.sdpa_generic import plan         # noqa: E402

TILE = 32
_RESERVE = 109056   # measured, see perf/bgsdpa/cb_model.py


def cb_bytes(p, q, k, v, mask, out):
    tb = {ttnn.bfloat16: 2048, ttnn.bfloat8_b: 1088, ttnn.float32: 4096}
    im = 2048
    return (p["q_tiles"] * tb[q.dtype] + p["k_tiles"] * tb[k.dtype]
            + p["v_tiles"] * tb[v.dtype] + p["mask_tiles"] * tb[mask.dtype]
            + 3 * im + p["qk_tiles"] * im + 2 * p["out_im_tiles"] * im
            + 5 * p["statistics_tiles"] * im + p["out0_t"] * tb[out.dtype])


def divisors_32(padded):
    return [padded // n for n in range(1, padded // TILE + 1)
            if padded % n == 0 and (padded // n) % TILE == 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=2208)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--head-dim", type=int, default=32)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--acc-rows", type=int, default=2)
    ap.add_argument("--card", default="2")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    S, H, D = args.seq, args.heads, args.head_dim
    dev = T.get_device()
    torch.manual_seed(0)
    scale = D ** -0.5

    def mk(shape):
        t = torch.randn(shape, dtype=torch.float32) * 0.5
        return t.to(torch.bfloat16), ttnn.from_torch(
            t.to(torch.bfloat16), device=dev, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    qh, q = mk([S, H, S, D])
    kh, k = mk([S, H, S, D])
    vh, v = mk([S, H, S, D])
    bh, bias = mk([1, H, S, S])
    ttnn.synchronize_device(dev)

    # fp32 reference on the first rows only.
    n = args.acc_rows
    with torch.no_grad():
        sc = (qh[:n].float() @ kh[:n].float().transpose(-1, -2)) * scale + bh.float()
        ref = torch.softmax(sc, dim=-1) @ vh[:n].float()
    del sc

    def err(o):
        got = ttnn.to_torch(o)[:n].float()
        return float(((got - ref) ** 2).mean().sqrt() / ref.std())

    # ---- the arms -------------------------------------------------------------------------
    padded = T._padded_sdpa_len(S)
    divs = sorted(set(divisors_32(padded)))
    cands = []
    for kc in sorted(divs, reverse=True):
        for qc in sorted(divs, reverse=True):
            p = plan(q, k, v, bias, q, qc, kc, T.COMPUTE_GRID_MAIN,
                     (ttnn.MathFidelity.HiFi2, True, False, False), scale)
            stock = cb_bytes(p, q, k, v, bias, q) + _RESERVE
            pers = p["k_num_chunks"] * p["Sq_chunk_t"] * p["Sk_chunk_t"]
            fused = stock + (pers - p["mask_tiles"]) * 2048
            if stock <= 1572864:
                cands.append(("stock", qc, kc, stock))
            if fused <= 1572864:
                cands.append(("fused", qc, kc, fused))

    def run_incumbent():
        return T._tri_att_sdpa_at(q, k, v, bias, scale)

    def run_stock(qc, kc):
        return ttnn.transformer.scaled_dot_product_attention(
            q, k, v, attn_mask=bias, is_causal=False, scale=scale,
            program_config=T._sdpa_program_config(qc, kc))

    def run_fused(qc, kc):
        prev = TS._Q_SPLIT_MAX_S
        TS._Q_SPLIT_MAX_S = max(prev, padded)
        try:
            return TS.sdpa(q, k, v, bias, scale, qc, kc)
        finally:
            TS._Q_SPLIT_MAX_S = prev

    arms = [("incumbent", run_incumbent)]
    for route, qc, kc, _b in cands:
        arms.append((f"{route}:q{qc}k{kc}",
                     (lambda a, b: (lambda: run_fused(a, b)))(qc, kc) if route == "fused"
                     else (lambda a, b: (lambda: run_stock(a, b)))(qc, kc)))

    print(f"S={S} padded={padded} divisors={divs}", flush=True)
    print(f"{len(cands)} candidate configs fit L1", flush=True)
    for route, qc, kc, b in cands:
        print(f"  fits {route:5s} q={qc:5d} k={kc:5d}  {b} B", flush=True)

    res = {a: {"ms": [], "err": None, "note": None} for a, _ in arms}
    live = dict(arms)
    for it in range(args.iters + 1):
        for name, fn in arms:
            if name not in live:
                continue
            try:
                t0 = time.perf_counter()
                o = fn()
                if o is None:
                    res[name]["note"] = "declined"
                    del live[name]
                    continue
                ttnn.synchronize_device(dev)
                dt = (time.perf_counter() - t0) * 1e3
            except Exception as exc:                                    # noqa: BLE001
                res[name]["note"] = str(exc).splitlines()[0][:200]
                del live[name]
                continue
            if it > 0:
                res[name]["ms"].append(round(dt, 3))
                if res[name]["err"] is None:
                    res[name]["err"] = round(err(o), 6)
            if name == "incumbent":
                res[name]["pick"] = {str(kk): vv for kk, vv in T.SDPA_CHUNK_PICKS.items()}
            ttnn.deallocate(o)
        print(f"iter {it} done", flush=True)

    base = None
    for name in res:
        if res[name]["ms"]:
            res[name]["mean_ms"] = round(sum(res[name]["ms"]) / len(res[name]["ms"]), 3)
            res[name]["min_ms"] = min(res[name]["ms"])
    base = res["incumbent"].get("mean_ms")
    for name in res:
        if res[name].get("mean_ms") and base:
            res[name]["speedup"] = round(base / res[name]["mean_ms"], 4)

    out = {"seq": S, "padded": padded, "heads": H, "head_dim": D, "iters": args.iters,
           "arms": res, "fits": [{"route": r, "q": a, "k": b, "l1_b": c} for r, a, b, c in cands],
           "route_counts": dict(T.SDPA_ROUTE_COUNTS),
           "triatt_stats": list(TS.STATS),
           "triatt_rejects": {str(kk): vv for kk, vv in TS.REJECTS.items()},
           "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    for name, r in sorted(res.items(), key=lambda kv: kv[1].get("mean_ms") or 1e9):
        print(f"{name:22s} mean={str(r.get('mean_ms')):>9}  x={str(r.get('speedup')):>7}  "
              f"err={str(r['err']):>9}  {r['note'] or ''}", flush=True)
    ttnn.close_device(dev)


main()
