"""Lever B screened against the op the fold actually runs, at Boltz-2's shape.

The audit measured `_sdpa_chunks_shipped`'s 64/64 band by timing
`ttnn.transformer.scaled_dot_product_attention` under two program configs, (64,64) against
(256,256), and found the band 1.46x slow at padded 384 on Wormhole. A Boltz-2 fold runs neither
arm of that comparison:

  * `_tri_att_sdpa_at` (tenstorrent.py:571) takes only `[1]` from the band -- the k_chunk. q comes
    from `_tri_att_q_chunks`, widest first, and the band's q is merely the last fallback. At padded
    384 the ladder's widest candidate is 384, not 64.
  * the (q_chunk, k_chunk) pair is handed to the FUSED K1/K2 kernel first, and the gate census says
    that kernel served 1120 of 1120 triangle-attention calls at 384 aa on both architectures. The
    stock ttnn op is never reached at this size.

So the reachable comparison is the fused kernel at (q_ladder_pick, k=64) against the same kernel at
(q_ladder_pick, k=capped). This script measures exactly that, using the shipped functions rather
than restating their constants, and reports for each arm whether the fused kernel was still SERVED.

That last part is the point. The fused kernel's persistent mask CB is
`k_num_chunks * Sq_chunk_t * Sk_chunk_t` tiles (triatt_sdpa.py:141). Going from k=64 to k=256 at
padded 384 takes k_num_chunks 6 -> 2 and Sk_chunk_t 2 -> 8, i.e. 12*Sq tiles -> 16*Sq. The wider
k_chunk costs MORE L1, so it can push the kernel over its budget and drop the call to the stock op,
which would be a loss dressed as a win. A wider k_chunk is not free here and the screen has to say so.

Chunking changes the online-softmax reduction order, so the two arms are NOT bit-exact. The script
reports max abs diff and rmsd/std between them; the fold-level accuracy question is pLDDT's.

Run with TT_VISIBLE_DEVICES=<free id>. Uses tt_bio's own get_device so the grid-derived
thresholds are applied exactly as a fold applies them.
"""
import argparse, json, os, statistics, sys, time
from pathlib import Path

H, D = 4, 32          # Boltz-2 triangle attention: tri_att_n_heads=4 (boltz2.py:5144), d=32
SIZES = [256, 288, 320, 352, 384, 448, 512]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
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
           "grid": list(grid), "cores": grid[0] * grid[1], "h": H, "d": D,
           "arch": T.arch_name(), "sdpa_chunk_max": T.SDPA_CHUNK_MAX,
           "pad_multiple": T.PAIRFORMER_PAD_MULTIPLE,
           "warmup": a.warmup, "iters": a.iters, "blocks": a.blocks, "cases": []}

    def mk(shape):
        t = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
        return t, ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    def timed(fn):
        for _ in range(a.warmup):
            o = fn()
            if o is None:
                return None, None
            o.deallocate()
        ttnn.synchronize_device(device)
        ms = []
        for _ in range(a.blocks):
            t0 = time.perf_counter()
            for _ in range(a.iters):
                o = fn()
            ttnn.synchronize_device(device)
            ms.append((time.perf_counter() - t0) * 1e3 / a.iters)
            o.deallocate()
        return {"best": round(min(ms), 4), "median": round(statistics.median(ms), 4),
                "all": [round(x, 4) for x in ms]}, True

    for N in SIZES:
        S = T._padded_sdpa_len(N)
        band = T._sdpa_chunks_shipped(S, S)
        capped = (T._capped_sdpa_chunk_size(S), T._capped_sdpa_chunk_size(S))
        rec = {"N": N, "padded": S, "band_chunks": list(band), "capped_chunks": list(capped),
               "in_band": band != capped, "q_ladder": list(T._tri_att_q_chunks(S, S))}
        try:
            _tq, q = mk((S, H, S, D))
            _tk, k = mk((S, H, S, D))
            _tv, v = mk((S, H, S, D))
            _tb, bias = mk((1, H, S, S))
            scale = D ** -0.5

            # The fold's own q pick: widest candidate the fused kernel accepts, band k_chunk.
            q_used = None
            for qc in rec["q_ladder"]:
                o = F.sdpa(q, k, v, bias, scale, qc, band[1])
                if o is not None:
                    q_used = qc
                    o.deallocate()
                    break
            rec["q_used_by_fused"] = q_used

            if q_used is None:
                rec["note"] = "fused kernel declines every ladder q at this shape; band is stock-op only"
            else:
                for leg, kc in (("band_k", band[1]), ("capped_k", capped[1])):
                    served = F.sdpa(q, k, v, bias, scale, q_used, kc)
                    rec[leg + "_served"] = served is not None
                    if served is None:
                        rec[leg] = None
                        continue
                    ref = ttnn.to_torch(served).float()
                    served.deallocate()
                    rec[leg + "_out_mean"] = round(float(ref.mean()), 6)
                    rec[leg + "_ref"] = ref
                    rec[leg], _ok = timed(lambda kc=kc: F.sdpa(q, k, v, bias, scale, q_used, kc))
                if rec.get("band_k") and rec.get("capped_k"):
                    rec["ratio_band_over_capped"] = round(
                        rec["band_k"]["median"] / rec["capped_k"]["median"], 4)
                    x, y = rec.pop("band_k_ref"), rec.pop("capped_k_ref")
                    d = (x - y).abs()
                    rec["max_abs_diff"] = round(float(d.max()), 6)
                    rec["rmsd_over_std"] = round(float(((x - y) ** 2).mean().sqrt() / x.std()), 6)
                    rec["bit_exact"] = bool((x == y).all())
                else:
                    rec.pop("band_k_ref", None); rec.pop("capped_k_ref", None)
            for t in (q, k, v, bias):
                t.deallocate()
        except Exception as e:                      # a config the wheel refuses is itself a result
            rec["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        rec.pop("band_k_ref", None); rec.pop("capped_k_ref", None)
        out["cases"].append(rec)
        print(json.dumps(rec), flush=True)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(out, indent=1))

    out["fused_stats"] = {"served": int(F.STATS[0]), "declined": int(F.STATS[1])}
    out["rejects"] = {str(k): int(v) for k, v in getattr(F, "REJECTS", {}).items()}
    a.out.write_text(json.dumps(out, indent=1))
    print("wrote", a.out, flush=True)
    T.cleanup()


if __name__ == "__main__":
    sys.exit(main())
