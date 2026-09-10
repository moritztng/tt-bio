"""Which padded token lengths the fused triangle-attention kernel can serve at all, and what the
shipped ladder picks instead where it cannot. Host only, no device.

Every gate this asks is the shipped one, imported rather than restated: `_sdpa_chunks_shipped`,
`_tri_att_q_chunks`, `_dividing_k_chunks` and `triatt_sdpa.q_parallel_factor`. The L1 arithmetic
is `perf/bgsdpa/cb_model.py`, exact on 10 measured refusals. What it cannot know is which of two
configs that both fit is FASTER; `sdpa_ab.py` measures that.

    python3 perf/bgsdpa/fused_reach.py --from 512 --to 2592
    TT_BIO_SDPA_WIDE_K=1 TT_BIO_TRIATT_NARROW_Q_FALLBACK=1 python3 perf/bgsdpa/fused_reach.py
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cb_model as M                                    # noqa: E402

from tt_bio import tenstorrent as T                      # noqa: E402
from tt_bio import triatt_sdpa as TS                     # noqa: E402

HEADS, HEAD_DIM = 4, 32     # boltz2 / boltzgen / nesso1 trunk; rf3 and protenix are 8 and 12


def ladder(S, heads, cores, q_split_max):
    """The (route, q_chunk, k_chunk) the shipped ladder settles on at padded length S, by
    replaying `_tri_att_sdpa_at`'s preference order against the L1 model."""
    prev, TS._Q_SPLIT_MAX_S = TS._Q_SPLIT_MAX_S, q_split_max
    try:
        k_chunks = T._tri_att_k_chunks(S, S)
        tried = []

        def consider(qc, kc):
            p = M.plan_for(S, heads, HEAD_DIM, qc, kc)
            pers = p["k_num_chunks"] * p["Sq_chunk_t"] * p["Sk_chunk_t"]
            q_pf = TS.q_parallel_factor(S, heads, qc, cores)
            fp = M.plan_for(S, heads, HEAD_DIM, qc, kc,
                            split=(max(cores // (heads * q_pf), 1), heads, q_pf))
            fused_ok = (fp["q_per_core"] == 1 and fp["nh_per_core"] == 1
                        and not fp["use_padded_mask"]
                        and M.fits(fp, mask_cb_tiles=pers))
            tried.append((qc, kc, "fused" if fused_ok else
                          ("stock" if M.fits(p) else "over_l1")))
            if fused_ok:
                return ("fused", qc, kc)
            if M.fits(p):
                return ("stock", qc, kc)
            return None

        if len(k_chunks) > 1:
            qs = tuple(q for q in T._tri_att_q_chunks(S, S) if S % q == 0)
            for kc in k_chunks[:-1]:
                for qc in qs:
                    got = consider(qc, kc)
                    if got:
                        return got, tried
        kc = k_chunks[-1]
        for qc in T._tri_att_q_chunks(S, S):
            got = consider(qc, kc)
            if got:
                return got, tried
        return ("none", 0, 0), tried
    finally:
        TS._Q_SPLIT_MAX_S = prev


def best_fused(S, heads, cores, q_split_max):
    """The fused pair with the widest k that fits, over the 32-aligned divisors of S. Widest k
    first because K5 measured widest-k winning at every size it screened, and one k chunk needs
    no online-softmax rescale at all."""
    divs = sorted({S // n for n in range(1, S // 32 + 1)
                   if S % n == 0 and (S // n) % 32 == 0}, reverse=True)
    prev, TS._Q_SPLIT_MAX_S = TS._Q_SPLIT_MAX_S, q_split_max
    try:
        for kc in divs:
            for qc in divs:
                q_pf = TS.q_parallel_factor(S, heads, qc, cores)
                fp = M.plan_for(S, heads, HEAD_DIM, qc, kc,
                                split=(max(cores // (heads * q_pf), 1), heads, q_pf))
                pers = fp["k_num_chunks"] * fp["Sq_chunk_t"] * fp["Sk_chunk_t"]
                if (fp["q_per_core"] == 1 and fp["nh_per_core"] == 1
                        and not fp["use_padded_mask"] and M.fits(fp, mask_cb_tiles=pers)):
                    return qc, kc, M.reported_bytes(fp, mask_cb_tiles=pers)
        return None
    finally:
        TS._Q_SPLIT_MAX_S = prev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="lo", type=int, default=512)
    ap.add_argument("--to", dest="hi", type=int, default=2592)
    ap.add_argument("--step", type=int, default=32)
    ap.add_argument("--heads", type=int, default=HEADS)
    ap.add_argument("--grid", default="11x10")
    ap.add_argument("--raised", type=int, default=4096,
                    help="the _Q_SPLIT_MAX_S to test against the shipped 1024")
    ap.add_argument("--out")
    args = ap.parse_args()
    gx, gy = (int(x) for x in args.grid.split("x"))
    cores = gx * gy
    shipped = TS._Q_SPLIT_MAX_S

    print(f"grid {args.grid} = {cores} cores, heads {args.heads}, head_dim {HEAD_DIM}, "
          f"shipped _Q_SPLIT_MAX_S = {shipped}, "
          f"WIDE_K={T._sdpa_wide_k()} NARROW_Q={T._SDPA_NARROW_Q_FALLBACK}")
    print(f"{'padded':>7} {'shipped pick':>22} {'raised pick':>22} {'widest fused fit':>22}")
    rows = []
    for S in range(args.lo, args.hi + 1, args.step):
        a, _ = ladder(S, args.heads, cores, shipped)
        b, _ = ladder(S, args.heads, cores, args.raised)
        bf = best_fused(S, args.heads, cores, args.raised)
        rows.append({"padded": S, "shipped": a, "raised": b,
                     "widest_fused": None if bf is None else
                     {"q": bf[0], "k": bf[1], "l1_b": bf[2]}})
        print(f"{S:7d} {a[0]+' q'+str(a[1])+' k'+str(a[2]):>22} "
              f"{b[0]+' q'+str(b[1])+' k'+str(b[2]):>22} "
              f"{('-' if bf is None else 'q'+str(bf[0])+' k'+str(bf[1])):>22}")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"grid": args.grid, "cores": cores, "heads": args.heads,
                       "shipped_q_split_max": shipped, "raised": args.raised,
                       "wide_k": T._sdpa_wide_k(),
                       "narrow_q": T._SDPA_NARROW_Q_FALLBACK, "rows": rows}, fh, indent=2)
    n_fused_now = sum(r["shipped"][0] == "fused" for r in rows)
    n_fused_raised = sum(r["raised"][0] == "fused" for r in rows)
    print(f"\nfused on the shipped ladder: {n_fused_now}/{len(rows)} lengths; "
          f"with _Q_SPLIT_MAX_S={args.raised}: {n_fused_raised}/{len(rows)}")


main()
