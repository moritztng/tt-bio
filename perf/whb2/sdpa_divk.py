"""Does a k_chunk that DIVIDES the padded sequence bring the fused K1/K2 kernel back?

Diagnosis, from `sdpa_generic.plan` line 112:

    use_padded_mask = (padded_Sk != Sk) or (padded_Sq != Sq)
    padded_Sk = div_up(Sk, k_chunk_size) * k_chunk_size

and `triatt_sdpa.sdpa` rejects on `fill_preconditions` when `use_padded_mask` is true. So the fused
kernel declines whenever k_chunk does not divide the padded sequence. `_capped_sdpa_chunk_size`
returns `min(SDPA_CHUNK_MAX=256, S)`, and 256 divides only every fourth multiple of 64. Above the
64/64 band's 384 ceiling that leaves the fused kernel declining at padded 448, 576, 640, 704, 832,
896 and 960 -- most of the range JapanFold serves -- while 256, 512, 768 and 1024 are served.

Measured at 320 and 384 (`perf/whb2/out/band_b2_wh.json`), the fused kernel is 1.75-2.70x faster
than the stock op at Boltz-2's shape, so a decline is expensive.

This probes one thing: at each reachable padded size (multiples of PAIRFORMER_PAD_MULTIPLE), does
the fused kernel serve at the shipped k_chunk, and does it serve at the largest 32-aligned DIVISOR
of the padded sequence that is <= SDPA_CHUNK_MAX? When both serve, the two are timed interleaved
against each other with an A/A control, and the reduction-order difference is reported -- a
dividing k_chunk is NOT bit-exact against the shipped one.

Run with TT_VISIBLE_DEVICES=<free id>.
"""
import argparse, json, os, statistics, sys, time
from pathlib import Path

H, D = 4, 32


def div_k(S, cap, tile=32):
    """Largest 32-aligned divisor of S that is <= cap, or None if only the trivial ones exist."""
    best = None
    for c in range(tile, min(cap, S) + 1, tile):
        if S % c == 0:
            best = c
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[256, 320, 384, 448, 512, 576, 640, 704, 768, 896, 1024])
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--blocks", type=int, default=5)
    a = ap.parse_args()

    tree = a.tree.resolve()
    sys.path.insert(0, str(tree))
    import torch, ttnn
    import tt_bio.tenstorrent as T
    import tt_bio.triatt_sdpa as F
    assert Path(T.__file__).resolve().is_relative_to(tree)

    device = T.get_device()
    grid = tuple(int(x) for x in T.COMPUTE_GRID_MAIN)
    out = {"host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
           "grid": list(grid), "cores": grid[0] * grid[1], "arch": T.arch_name(),
           "h": H, "d": D, "sdpa_chunk_max": T.SDPA_CHUNK_MAX,
           "pad_multiple": T.PAIRFORMER_PAD_MULTIPLE,
           "warmup": a.warmup, "iters": a.iters, "blocks": a.blocks, "cases": []}

    def timed_pair(fa, fb):
        for f in (fa, fb):
            for _ in range(a.warmup):
                o = f()
                if o is None:
                    return None, None
                ttnn.deallocate(o)
        ttnn.synchronize_device(device)
        ms = {"a": [], "b": []}
        for _ in range(a.blocks):
            for tag, f in (("a", fa), ("b", fb)):
                t0 = time.perf_counter()
                outs = [f() for _ in range(a.iters)]
                ttnn.synchronize_device(device)
                ms[tag].append((time.perf_counter() - t0) * 1e3 / a.iters)
                for o in outs:
                    ttnn.deallocate(o)
        return tuple({"best": round(min(x), 4), "median": round(statistics.median(x), 4),
                      "all": [round(y, 4) for y in x]} for x in (ms["a"], ms["b"]))

    for S in a.sizes:
        shipped_k = T._sdpa_chunks_shipped(S, S)[1]
        dk = div_k(S, T.SDPA_CHUNK_MAX)
        rec = {"padded": S, "shipped_k": shipped_k, "div_k": dk,
               "shipped_k_divides": S % shipped_k == 0,
               "q_ladder": list(T._tri_att_q_chunks(S, S))}
        try:
            def mk(shape):
                t = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
                return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                       device=device)
            q, k, v = mk((S, H, S, D)), mk((S, H, S, D)), mk((S, H, S, D))
            bias = mk((1, H, S, S))
            scale = D ** -0.5

            def first_q(kc):
                for qc in rec["q_ladder"]:
                    o = F.sdpa(q, k, v, bias, scale, qc, kc)
                    if o is not None:
                        ttnn.deallocate(o)
                        return qc
                return None

            rec["q_shipped"] = first_q(shipped_k)
            rec["q_divk"] = first_q(dk) if dk else None
            rec["shipped_served"] = rec["q_shipped"] is not None
            rec["divk_served"] = rec["q_divk"] is not None

            if rec["divk_served"] and not rec["shipped_served"]:
                # The case this script exists for: shipped declines, a dividing k_chunk serves.
                # Time the fused kernel at div_k against the stock op the fold falls back to today.
                qd = rec["q_divk"]
                qs = rec["q_ladder"][0]
                rec["divk_fused"], rec["stock_today"] = timed_pair(
                    lambda: F.sdpa(q, k, v, bias, scale, qd, dk),
                    lambda: ttnn.transformer.scaled_dot_product_attention(
                        q, k, v, attn_mask=bias, is_causal=False, scale=scale,
                        program_config=T._sdpa_program_config(qs, shipped_k)))
                if rec["divk_fused"] and rec["stock_today"]:
                    rec["speedup_divk_over_today"] = round(
                        rec["stock_today"]["median"] / rec["divk_fused"]["median"], 4)
            elif rec["divk_served"] and rec["shipped_served"] and dk != shipped_k:
                qd, qs = rec["q_divk"], rec["q_shipped"]
                rec["divk_fused"], rec["shipped_fused"] = timed_pair(
                    lambda: F.sdpa(q, k, v, bias, scale, qd, dk),
                    lambda: F.sdpa(q, k, v, bias, scale, qs, shipped_k))
                if rec["divk_fused"] and rec["shipped_fused"]:
                    rec["speedup_divk_over_today"] = round(
                        rec["shipped_fused"]["median"] / rec["divk_fused"]["median"], 4)
            elif rec["shipped_served"] and dk == shipped_k:
                # A/A control: same config down both legs, must read 1.00x.
                qs = rec["q_shipped"]
                a1, a2 = timed_pair(lambda: F.sdpa(q, k, v, bias, scale, qs, shipped_k),
                                    lambda: F.sdpa(q, k, v, bias, scale, qs, shipped_k))
                rec["aa_control"] = {"a": a1, "b": a2,
                                     "ratio": round(a1["median"] / a2["median"], 4)
                                     if a1 and a2 else None}
            for t in (q, k, v, bias):
                ttnn.deallocate(t)
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        out["cases"].append(rec)
        print(json.dumps(rec), flush=True)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(out, indent=1))

    out["rejects"] = {str(kk): int(vv) for kk, vv in getattr(F, "REJECTS", {}).items()}
    a.out.write_text(json.dumps(out, indent=1))
    print("wrote", a.out, flush=True)
    T.cleanup()


if __name__ == "__main__":
    sys.exit(main())
