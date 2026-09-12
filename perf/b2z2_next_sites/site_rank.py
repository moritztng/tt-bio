#!/usr/bin/env python3
"""Every site in the diffusion step, ranked, with the row that already owns it named.

`b2z2-step-fusion`'s `site_cost.py` prices the 1066 device programs of one settled step. It does
not say which of them a wave-2 row has already taken, and four rows have taken the top of the list.
This reads its output, clusters the programs into SITES -- a site is the set of programs that come
off the same line of `tenstorrent.py`, which the cost map identifies by op code plus cost, because
the same line costs the same on every one of its calls to well under a per cent -- and joins each
site to its owner.

No device. It runs on the committed capture and reproduces `site_cost.py`'s own op-code totals.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

# first ttnn index -> (what the site is, which row owns it, "" when nobody does)
SITES = {
    # --- atom attention, 6 layers a step ---------------------------------------------------
    10:  ("atom key gather: reshape", "b2z2-layout-op-elision / b2z2-step-program-fusion"),
    11:  ("atom key gather: permute", "b2z2-layout-op-elision / b2z2-step-program-fusion"),
    12:  ("atom key gather: one-hot matmul", "b2z2-layout-op-elision / b2z2-step-program-fusion"),
    13:  ("atom key gather: inverse permute", "b2z2-layout-op-elision / b2z2-step-program-fusion"),
    14:  ("atom key gather: reshape back", "b2z2-layout-op-elision / b2z2-step-program-fusion"),
    15:  ("atom q projection", "b2z2-step-fusion-next-sites (TT_BIO_ATOM_L1)"),
    16:  ("atom kv projection", "b2z2-step-fusion-next-sites (TT_BIO_ATOM_L1)"),
    17:  ("atom query pad 32 -> 128", "b2z2-step-fusion-next-sites (TT_BIO_ATOM_HEADS_UNPADDED)"),
    20:  ("atom head split", "b2z2-step-fusion-next-sites (TT_BIO_ATOM_HEADS_UNPADDED)"),
    24:  ("atom query unpad", "b2z2-step-fusion-next-sites (TT_BIO_ATOM_HEADS_UNPADDED)"),
    26:  ("atom SDPA", ""),
    28:  ("atom head concat", ""),
    31:  ("atom gate projection", "b2z2-step-fusion-next-sites (TT_BIO_ATOM_L1)"),
    34:  ("atom out projection", "b2z2-step-fusion-next-sites (TT_BIO_ATOM_L1)"),
    # --- token DiT, 24 layers a step ------------------------------------------------------
    # NOT one site: cost clustering cannot separate `AdaLN.s_terms`'s 48 norms (31.94 us) from
    # `AdaLN.__call__`'s 48 (27.88 us) at the same [1, 512, 768]. The per-call join says 48 of
    # these 97 belong to `b2z2-step-layernorm-fusion` and only 60 programs / 1.566 ms are free.
    300: ("AdaLN norm of a (48 of these are s_terms, owned)",
          "b2z2-step-adaln-sdpa (NAMED NOT BUILT: core-starved at 16 of 72 cores, no bit-exact fix)"),
    301: ("AdaLN s_scale / s_bias projections", "b2z2-step-matmul-group (N-stacking REFUTED)"),
    308: ("fused QKV projection", ""),
    310: ("token head split", "b2z2-step-layout-elision (arm B, screened not built)"),
    312: ("token SDPA",
          "b2z2-step-adaln-sdpa (TT_BIO_SDPA_GRID_Q_CHUNK, 1.3117x on the op, bit-exact)"),
    316: ("token attention epilogue", "b2z2-step-layout-elision (arm A, BUILT 1.02574x)"),
    320: ("attention out projection", ""),
    328: ("AdaLN conditioning norm of s", "b2z2-step-layernorm-fusion (BUILT 1.03932x)"),
    337: ("transition swish gate chunks", "b2z2-step-matmul-group (roofline: no single term)"),
    345: ("transition b_to_a", "b2z2-step-matmul-group (roofline: no single term)"),
    # --- elementwise, everywhere ----------------------------------------------------------
    303: ("AdaLN scale/shift elementwise", "b2z2-step-binaryng-fusion (NO-GO, measured slower)"),
    # --- once a step ----------------------------------------------------------------------
    162: ("atom -> token pooling matmul (no core_grid)", ""),
}


def cluster(table):
    """Programs off one source line have the same op code and the same cost. Group on that."""
    groups = defaultdict(list)
    for code in sorted({r["code"] for r in table if r["code"]}):
        rows = sorted([r for r in table if r["code"] == code], key=lambda r: r["us"])
        cur = [rows[0]]
        for r in rows[1:]:
            if r["us"] <= cur[0]["us"] * 1.15 + 1.0:
                cur.append(r)
            else:
                groups[code].append(cur)
                cur = [r]
        groups[code].append(cur)
    return [g for gs in groups.values() for g in gs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site-cost", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    table = json.loads(a.site_cost.read_text())["table"]

    rows = []
    for g in cluster(table):
        idx = sorted(r["i"] for r in g)
        label, owner = "", ""
        for i in idx:
            if i in SITES:
                label, owner = SITES[i]
                break
        rows.append({"code": g[0]["code"], "n": len(g),
                     "us_each": round(sum(r["us"] for r in g) / len(g), 2),
                     "ms_step": round(sum(r["us"] for r in g) / 1e3, 4),
                     "first_idx": idx[:4], "site": label, "owner": owner})
    rows.sort(key=lambda r: -r["ms_step"])
    total = round(sum(r["ms_step"] for r in rows), 4)
    free = [r for r in rows if not r["owner"]]
    out = {"kernel_sum_ms": total,
           "n_programs": sum(r["n"] for r in rows),
           "ms_unowned": round(sum(r["ms_step"] for r in free), 4),
           "sites": rows}
    a.out.write_text(json.dumps(out, indent=1))
    print(f"{total:.4f} ms over {out['n_programs']} programs; "
          f"{out['ms_unowned']:.4f} ms carries no owner\n")
    print(f"{'ms/step':>8} {'n':>4} {'us/ea':>8}  {'site':44s} owner")
    for r in rows:
        if r["ms_step"] < 0.2:
            continue
        print(f"{r['ms_step']:8.3f} {r['n']:4d} {r['us_each']:8.2f}  "
              f"{(r['site'] or '?' + str(r['first_idx'][:2])):44s} {r['owner'] or '-- FREE --'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
