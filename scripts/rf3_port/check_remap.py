#!/usr/bin/env python3
"""Check the RF3 -> tt-bio Pairformer weight remap against a real checkpoint.

Card-free. Verifies three things, in the order they go wrong:

1. every key tt-bio's shared `PairformerLayer` reads is produced,
2. every RF3 key is consumed, and anything left over is reported rather than
   dropped — that is how the triangle-attention bias gap below was found,
3. the leftovers that are known-unsupported are still non-zero, i.e. reusing the
   shared block as-is would lose real signal rather than a formality.

    python scripts/rf3_port/check_remap.py --ckpt /path/to/rf3_latest.ckpt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

#: Keys the remap produces that tt-bio's shared TriangleAttention does not read.
#: RF3's gate and output projections are biased; tt-bio's are not.
KNOWN_UNSUPPORTED = {
    "tri_att_start.linear_g.bias", "tri_att_end.linear_g.bias",
    "tri_att_start.linear_o.bias", "tri_att_end.linear_o.bias",
}

STACK = "shadow.recycler.pairformer_stack."


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--block", type=int, default=0)
    args = ap.parse_args()

    from tt_bio.rf3.remap import check_coverage, remap_pairformer_block

    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)["model"]
    n_blocks = len({k[len(STACK):].split(".")[0] for k in sd if k.startswith(STACK)})
    pre = f"{STACK}{args.block}."
    block = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
    if not block:
        print(f"no weights at {pre}")
        return 1

    rep = check_coverage(block)
    unexpected = sorted(set(rep["extra"]) - KNOWN_UNSUPPORTED)

    # Are the unsupported leftovers actually carrying signal?
    magnitudes = {}
    for key in sorted(KNOWN_UNSUPPORTED):
        vals = []
        for b in range(n_blocks):
            p = f"{STACK}{b}."
            got = remap_pairformer_block(
                {k[len(p):]: v for k, v in sd.items() if k.startswith(p)}
            )
            if key in got:
                vals.append(float(got[key].abs().max()))
        if vals:
            magnitudes[key] = {
                "blocks": len(vals),
                "nonzero": sum(1 for v in vals if v > 0),
                "absmax_max": round(max(vals), 4),
            }

    out = {
        "blocks": n_blocks,
        "rf3_keys_per_block": rep["rf3_keys"],
        "produced": rep["produced"],
        "expected_by_tt_bio": rep["expected"],
        "missing": rep["missing"],
        "unsupported_by_shared_block": sorted(KNOWN_UNSUPPORTED),
        "unsupported_magnitudes": magnitudes,
        "unexpected_leftovers": unexpected,
    }
    print(json.dumps(out, indent=2))
    ok = not rep["missing"] and not unexpected
    print("REMAP OK" if ok else "REMAP INCOMPLETE")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
