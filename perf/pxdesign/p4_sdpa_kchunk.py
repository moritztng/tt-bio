"""The 848-token triangle-attention SDPA, screened over every chunk pair the shape allows.

`pxdesign-perf` pass 3 left "widen the q_chunk ceiling past 864" as the one live prize and framed
it as a circular-buffer sizing question. The arithmetic says the framing needs one correction
before any measurement: 848 tokens tile-pad to 864 = 2^5 * 27, whose only 32-aligned divisors are
32, 96, 288 and 864. `_tri_att_q_chunks` offers only divisors (a non-dividing q_chunk pays the
padded mask twice, measured 0.797x at seq 768), so the ladder is exactly (864, 288, 256) and there
is NO intermediate step between the current fallback of 288 and the refused 864. "A smaller q_chunk
step that still fits L1" does not exist at this token count.

What does exist is the k half. `_dividing_sdpa_chunk_size` looks for a 32-aligned divisor of the
padded sequence down to a `cap/2 = 128` floor; 864's divisors below the 256 cap are 96 and 32, both
under the floor, so it hands back 256, which does not divide 864. Two consequences at this cell,
both visible in pass 3's census:

  * `sdpa_generic.plan` sets `use_padded_mask` when k_chunk does not divide padded Sk, and
    `triatt_sdpa.sdpa` rejects on `fill_preconditions` when it is set -- so the fused K1/K2 kernel
    declines 100% of calls here (288 declines per fold) while at the 768-token probe cell, where
    256 divides 768, it SERVES 100%. That is the probe/filter lever-set difference.
  * the stock op's mask CB is `2 * Sq_chunk_t * Sk_chunk_t` tiles, so k=256 makes it 8 tiles wide.
    That is what the q_chunk=864 candidate overflows on (2005504 B against 1572864 B).

The K3 comment in `tenstorrent.py` says the cap/2 floor is "a precaution, not a measured refusal"
and "lower the floor only behind that measurement". This is that measurement, at this shape.

Every arm is (q_chunk, k_chunk) with both dividing 864, plus the incumbent (288, 256). For each,
the fused kernel is offered the pair first (as the fold does) and the stock op is timed too, so a
config that only wins by dropping the fused kernel cannot look like a win. Arms are interleaved
round-robin block by block, and the incumbent appears TWICE so the instrument's own A/A floor is
in the record.

k_chunk sets the online-softmax reduction order, so arms with a different k are NOT bit-exact;
max abs diff and rmsd/std against the incumbent are reported per arm and the fold-level accuracy
question is pLDDT's.
"""
import argparse, json, os, statistics, sys, time
from pathlib import Path

H, D = 4, 32   # protenix c_z=128 / TRI_HEAD_DIM=32; pass 3's reject key was (848, 4, 848, 32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tokens", type=int, default=848)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--no-compare", action="store_true")
    a = ap.parse_args()

    tree = a.tree.resolve()
    sys.path.insert(0, str(tree))
    import torch, ttnn
    import tt_bio.tenstorrent as T
    import tt_bio.triatt_sdpa as F
    assert Path(T.__file__).resolve().is_relative_to(tree), T.__file__

    N = a.tokens
    padded = T._padded_sdpa_len(N)
    divisors = [padded // m for m in range(1, padded // T.SDPA_CHUNK_TILE + 1)
                if padded % m == 0 and (padded // m) % T.SDPA_CHUNK_TILE == 0]
    prod_q, prod_k = T._sdpa_chunks_shipped(N, N)
    ladder = list(T._tri_att_q_chunks(N, N))

    device = T.get_device()
    grid = tuple(int(x) for x in T.COMPUTE_GRID_MAIN)
    rec = {"host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
           "grid": list(grid), "cores": grid[0] * grid[1], "arch": T.arch_name(),
           "tokens": N, "padded": padded, "h": H, "d": D,
           "aligned_divisors_of_padded": divisors,
           "shipped_chunks": [prod_q, prod_k], "q_ladder": ladder,
           "dividing_k": T._dividing_sdpa_chunk_size(N), "sdpa_chunk_max": T.SDPA_CHUNK_MAX,
           "warmup": a.warmup, "iters": a.iters, "blocks": a.blocks, "arms": []}

    def mk(shape):
        t = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    torch.manual_seed(0)
    q, k, v = mk((N, H, N, D)), mk((N, H, N, D)), mk((N, H, N, D))
    bias = mk((1, H, N, N))
    scale = D ** -0.5

    # Candidate arms. `label` is what the fold would run; `incumbent` is today's actual pick,
    # reached after (848,848,864) is retired over L1 and the fused kernel declines everything.
    # The incumbent is NOT `_sdpa_chunks_shipped`. That pair is the last fallback;
    # `_tri_att_sdpa_at` walks the q ladder widest-first and at this shape lands on q=288 with the
    # shipped k=256, after 864 is retired over L1 and the fused kernel declines all three. The first
    # version of this screen baselined on (256, 256) and overstated every arm.
    fold_q = next(x for x in ladder if x != 864)
    arms = [("incumbent", fold_q, prod_k), ("incumbent_aa", fold_q, prod_k),
            ("shipped_fallback", prod_q, prod_k)]
    for qc in [x for x in divisors if x >= 96]:
        for kc in [x for x in divisors if x >= 32]:
            if (qc, kc) not in ((prod_q, prod_k), (fold_q, prod_k)):
                arms.append((f"q{qc}_k{kc}", qc, kc))

    def stock(qc, kc):
        return ttnn.transformer.scaled_dot_product_attention(
            q, k, v, attn_mask=bias, is_causal=False, scale=scale,
            program_config=T._sdpa_program_config(qc, kc))

    def fused(qc, kc):
        return F.sdpa(q, k, v, bias, scale, qc, kc)

    # Which arms are even runnable, and on which path. One call each, exceptions captured, before
    # any timing -- an L1 refusal must be recorded rather than kill the sweep.
    live = []
    for label, qc, kc in arms:
        e = {"arm": label, "q_chunk": qc, "k_chunk": kc,
             "q_divides": padded % qc == 0, "k_divides": padded % kc == 0}
        before = dict(F.REJECTS)
        try:
            o = fused(qc, kc)
            e["fused_served"] = o is not None
            if o is not None:
                ttnn.deallocate(o)
        except Exception as exc:                                    # noqa: BLE001
            e["fused_served"] = False
            e["fused_error"] = f"{type(exc).__name__}: {exc}"[:200]
        e["fused_rejects"] = {str(kk): int(F.REJECTS[kk] - before.get(kk, 0))
                              for kk in F.REJECTS if F.REJECTS[kk] - before.get(kk, 0)}
        try:
            o = stock(qc, kc)
            e["stock_ok"] = True
            ttnn.deallocate(o)
        except Exception as exc:                                    # noqa: BLE001
            e["stock_ok"] = False
            e["stock_error"] = f"{type(exc).__name__}: {exc}"[:240]
            e["stock_over_l1"] = "circular buffers" in str(exc)
        rec["arms"].append(e)
        print(json.dumps(e), flush=True)
        # The fold prefers the fused kernel and falls back to the stock op, so that is the path
        # timed for each arm.
        if e.get("fused_served"):
            live.append((label, qc, kc, "fused", lambda qc=qc, kc=kc: fused(qc, kc)))
        elif e["stock_ok"]:
            live.append((label, qc, kc, "stock", lambda qc=qc, kc=kc: stock(qc, kc)))

    # Round-robin blocks over every live arm, so no arm inherits another's allocator state and
    # ordering cannot be mistaken for the effect.
    ms = {label: [] for label, *_ in live}
    for label, _q, _k, _p, fn in live:
        for _ in range(a.warmup):
            ttnn.deallocate(fn())
    ttnn.synchronize_device(device)
    for _ in range(a.blocks):
        for label, _q, _k, _p, fn in live:
            t0 = time.perf_counter()
            outs = [fn() for _ in range(a.iters)]
            ttnn.synchronize_device(device)
            ms[label].append((time.perf_counter() - t0) * 1e3 / a.iters)
            for o in outs:
                ttnn.deallocate(o)
    for label, _q, _k, path, _fn in live:
        v_ = ms[label]
        e = next(x for x in rec["arms"] if x["arm"] == label)
        e.update(path=path, ms_median=round(statistics.median(v_), 4),
                 ms_best=round(min(v_), 4), ms_all=[round(x, 4) for x in v_])

    base = next((x for x in rec["arms"] if x["arm"] == "incumbent"), None)
    if base and "ms_median" in base:
        for e in rec["arms"]:
            if "ms_median" in e:
                e["speedup_vs_incumbent"] = round(base["ms_median"] / e["ms_median"], 4)

    # Numerics. Two references, because "differs from the incumbent" and "worse than the
    # incumbent" are different claims: the device incumbent's own output, and a torch fp32 SDPA on
    # the first BSL batch rows. Every arm gets both, including the incumbent itself against torch,
    # so the incumbent's own bf16 error is in the record as the scale to read the others against.
    BSL = 8
    if not a.no_compare and base:
        qt = ttnn.to_torch(q)[:BSL].float()
        kt = ttnn.to_torch(k)[:BSL].float()
        vt = ttnn.to_torch(v)[:BSL].float()
        bt = ttnn.to_torch(bias).float()
        tref = torch.nn.functional.scaled_dot_product_attention(
            qt, kt, vt, attn_mask=bt.expand(BSL, -1, -1, -1), scale=scale)
        del qt, kt, vt, bt
        ref = ttnn.to_torch(next(fn for lb, _q, _k, _p, fn in live if lb == "incumbent")())
        for e in rec["arms"]:
            if "ms_median" not in e or e["arm"] == "incumbent_aa":
                continue
            fn = next(f for lb, _q, _k, _p, f in live if lb == e["arm"])
            o = ttnn.to_torch(fn())
            d = o.float() - ref.float()
            e["max_abs_diff"] = round(float(d.abs().max()), 6)
            e["rmsd_over_std"] = round(float(d.pow(2).mean().sqrt() / ref.float().std()), 6)
            e["bit_exact_vs_incumbent"] = bool(torch.equal(o, ref))
            dt = o[:BSL].float() - tref
            e["rmsd_over_std_vs_torch"] = round(
                float(dt.pow(2).mean().sqrt() / tref.std()), 6)
            e["max_abs_diff_vs_torch"] = round(float(dt.abs().max()), 6)
            del o, d, dt
        del ref, tref

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(rec, f, indent=1)
    print(json.dumps({k_: rec[k_] for k_ in ("aligned_divisors_of_padded", "shipped_chunks",
                                             "q_ladder", "dividing_k")}), flush=True)
    for e in sorted((x for x in rec["arms"] if "ms_median" in x), key=lambda x: x["ms_median"]):
        print(f'{e["arm"]:>16} {e["path"]:>6} q{e["q_chunk"]:>4} k{e["k_chunk"]:>4} '
              f'{e["ms_median"]:>9.3f} ms  {e.get("speedup_vs_incumbent", 0):>6.3f}x  '
              f'rmsd/std vs inc {e.get("rmsd_over_std", "-")}  '
              f'vs torch {e.get("rmsd_over_std_vs_torch", "-")}', flush=True)
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
