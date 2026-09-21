#!/usr/bin/env python3
"""Score two aux_heads output sets against each other, per head.

A relative L2 alone cannot identify the direction of an error, so every row carries the norm
ratio r = ||a||/||b|| and the cosine of a against b beside it (A25 and the fleet's standing
rule). `max_abs` is there because a relative L2 over a padded tensor dilutes a structural
difference and barely dilutes a numerical one (D95); a max over the same tensor does not.

usage: score.py <A.pt> <B.pt> <OUT.json> [--label-a X --label-b Y] [--mask-real PATH]
"""
import argparse, json
from pathlib import Path
import torch


def stats(a, b):
    a = a.double().flatten(); b = b.double().flatten()
    na, nb = float(a.norm()), float(b.norm())
    d = a - b
    return {
        "rel_l2": float(d.norm() / (nb + 1e-300)),
        "norm_ratio_r": (na / nb) if nb else float("nan"),
        "cosine": float((a @ b) / (na * nb)) if na and nb else float("nan"),
        "max_abs_diff": float(d.abs().max()),
        "norm_a": na, "norm_b": nb,
        "bit_identical": bool(torch.equal(a, b)),
        "n_elem": int(a.numel()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a", type=Path); ap.add_argument("b", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--label-a", default="a"); ap.add_argument("--label-b", default="b")
    ap.add_argument("--real-tokens", type=int, default=56,
                    help="D95: score the real block as well as the padded tensor")
    a = ap.parse_args()
    A = torch.load(a.a, map_location="cpu", weights_only=False)
    B = torch.load(a.b, map_location="cpu", weights_only=False)
    rows = {}
    n = a.real_tokens
    for k in sorted(set(A) & set(B)):
        ta, tb = A[k], B[k]
        row = {"padded": stats(ta, tb)}
        # the token axes are the ones of length 384; slice every one of them to the real block
        if ta.dim() >= 4 and ta.shape[-2] == ta.shape[-3]:
            row["real_block"] = stats(ta[..., :n, :n, :], tb[..., :n, :n, :])
        rows[k] = row
    rep = {"a": str(a.a), "b": str(a.b), "label_a": a.label_a, "label_b": a.label_b,
           "real_tokens": n, "heads": rows,
           "worst_padded_rel_l2": max(v["padded"]["rel_l2"] for v in rows.values()),
           "n_bit_identical": sum(1 for v in rows.values() if v["padded"]["bit_identical"]),
           "n_heads": len(rows)}
    a.out.write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
