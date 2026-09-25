#!/usr/bin/env python3
"""Why the fused-HiFi triangle attention declines at 288, BindCraft 2's own length.

FIRING question, not a timing one: which (q_chunk, k_chunk) rungs the ladder actually reaches and
what each one answers. Valid on a loud or a Wormhole box -- no wall clock is quoted and no weights
are needed, because serve/decline is a function of shape, grid and L1 only.

`bcx-forward`'s sweep (qb2 card 0, p300c, commit 69a0a6bdc) served 192/224/256/320/384 and
declined 288 with 8 `fill_preconditions` rejects and an EMPTY l1_refusals list. Read on this tree,
that count is exact rather than suggestive: `_tri_att_sdpa_hifi_inner` builds

    k_chunks = (padded_k, shipped_k) if one_k_chunk and padded_k != shipped_k else (shipped_k,)

so at 288 it is (64,), `wide` is False, q_chunks is the whole ladder (288, 96, 32, 64) and kv_bf is
(2,) -- 4 rungs, every one carrying k_chunk 64, which does not divide 288. 4 rungs x 2 triangle
attentions in one Evoformer block = 8. It never calls `_dividing_k_chunks`, whose own docstring
says that on the fused-only path a dividing chunk is a PRECONDITION and not a preference.

This replays that ladder on real device operands at 256 (control, serves), 288 (the arm) and 320
(control, serves), with and without the one-line fix, and checks the fixed 288 output against the
same float64 reference the declining path is graded on.

    khole.py --ns 256,288,320 --arms shipped,dividing
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch                                                          # noqa: E402
import ttnn                                                           # noqa: E402
from tt_bio import tenstorrent as T, triatt_sdpa as TS                # noqa: E402

HEADS, HEAD_DIM = 4, 32


class Rungs:
    """Records every (q_chunk, k_chunk, kv_buffer_factor) the ladder offers and its verdict.

    `TS.REJECTS` keys on (reason, shape), so at one shape every rung collapses into one counter
    and "which rung did it try" has no answer in the artifact. That is the whole question here.
    """

    def __init__(self, real):
        self.real = real
        self.seen = []
        self.on = False

    def __call__(self, q, k, v, bias, scale, q_chunk, k_chunk, *a, **kw):
        out = self.real(q, k, v, bias, scale, q_chunk, k_chunk, *a, **kw)
        if self.on:
            self.seen.append({"q_chunk": q_chunk, "k_chunk": k_chunk,
                              "kv_bf": kw.get("kv_buffer_factor", 2),
                              "k_divides": int(q.shape[2]) % k_chunk == 0,
                              "served": out is not None})
        return out


def operands(dev, n, seed):
    """One triangle-attention call's operands at BC2's head count, as the engine hands them over:
    q/k/v [S, heads, S, head_dim] and an additive bias [1, heads, S, S]."""
    g = torch.Generator().manual_seed(seed)
    mk = lambda *s: torch.randn(*s, generator=g, dtype=torch.float32)  # noqa: E731
    q, k, v = (mk(n, HEADS, n, HEAD_DIM) for _ in range(3))
    bias = mk(1, HEADS, n, n)
    up = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    return (q, k, v, bias), (up(q), up(k), up(v), up(bias))


def reference(host, scale):
    """float64 triangle attention on the SAME operands, so a served output is graded against the
    exact answer rather than against the path it replaced."""
    q, k, v, bias = (t.double() for t in host)
    s = torch.einsum("shqd,shkd->shqk", q, k) * scale + bias.double()
    return torch.einsum("shqk,shkd->shqd", s.softmax(-1), v)


def rmsd(a, b):
    return float(torch.sqrt(torch.mean((a - b) ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default="256,288,320")
    ap.add_argument("--arms", default="shipped,dividing")
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--card", default=os.environ.get("TT_VISIBLE_DEVICES", "0"))
    ap.add_argument("--out", default="out/khole.json")
    args = ap.parse_args()

    from tt_bio import device_lease as DL
    dev = ttnn.open_device(device_id=0)
    print(f"opened visible 0; physical_card()={DL.physical_card()} "
          f"grant={os.environ.get('TT_BIO_LEASE_CARDS')} "
          f"grid={dev.compute_with_storage_grid_size().x}x{dev.compute_with_storage_grid_size().y}",
          flush=True)

    rungs = Rungs(TS.sdpa)
    TS.sdpa = rungs
    shipped_k_ladder = T._tri_att_sdpa_hifi_inner.__globals__["_sdpa_chunks_shipped"]
    rows = []
    try:
        for n in [int(x) for x in args.ns.split(",")]:
            host, (q, k, v, bias) = operands(dev, n, args.seed)
            scale = HEAD_DIM ** -0.5
            ref = reference(host, scale)
            row = {"n": n, "padded": T._padded_sdpa_len(n),
                   "shipped_k": shipped_k_ladder(n, n)[1],
                   "dividing_k": list(T._dividing_k_chunks(n, n)),
                   "q_ladder": list(T._tri_att_q_chunks(n, n)), "arms": {}}
            for arm in args.arms.split(","):
                T._TRIATT_HIFI_OVER_L1.clear()
                TS.REJECTS.clear()
                T.TRIATT_FUSED_HIFI_STATS.update(served=0, declined=0, too_short=0)
                rungs.seen, rungs.on = [], True
                os.environ["TT_BIO_TRIATT_DIVIDING_K"] = "1" if arm == "dividing" else "0"
                o = T._tri_att_sdpa_hifi(q, k, v, bias, scale ** -1)
                rungs.on = False
                got = None
                if o is not None:
                    got = rmsd(ttnn.to_torch(o).double(), ref)
                    ttnn.deallocate(o)
                row["arms"][arm] = {
                    "served": o is not None, "rmsd_vs_f64": got,
                    "stats": dict(T.TRIATT_FUSED_HIFI_STATS),
                    "rejects": {str(kk): vv for kk, vv in TS.REJECTS.items()},
                    "rungs": list(rungs.seen)}
                print(f"n={n} arm={arm} served={o is not None} rmsd={got} "
                      f"rungs={[(r['q_chunk'], r['k_chunk'], r['served']) for r in rungs.seen]}",
                      flush=True)
            # What a decline actually falls through to. `_tri_att_sdpa_hifi`'s docstring is explicit
            # that None sends the caller to `_fp32_softmax_attention` and NOT to the stock bf16 op,
            # so the fused route's accuracy has to be read against that and not against float64
            # alone -- at 288 the fix replaces this path, and it is the more accurate one by design.
            fb = T._fp32_softmax_attention(
                q, k, v, bias, scale_inv=scale ** -1,
                compute_kernel_config=T._SOFTMAX_PRECISE_CKC, out_dtype=ttnn.bfloat16,
                bias_scale_inv=1.0, accurate_softmax=False)
            row["fallback_rmsd_vs_f64"] = rmsd(ttnn.to_torch(fb).double(), ref)
            ttnn.deallocate(fb)
            print(f"n={n} fp32-softmax fall-back rmsd={row['fallback_rmsd_vs_f64']}", flush=True)
            for t in (q, k, v, bias):
                ttnn.deallocate(t)
            rows.append(row)
    finally:
        TS.sdpa = rungs.real
        ttnn.close_device(dev)

    out = Path(args.out)
    if not out.is_absolute():
        out = Path(__file__).resolve().parent / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"heads": HEADS, "head_dim": HEAD_DIM,
                               "card": args.card, "rows": rows}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
