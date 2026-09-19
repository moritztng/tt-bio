#!/usr/bin/env python3
"""Compare two gradient artifacts per tensor. This is the detector a reference has to survive.

A gradient artifact that has been produced once is a measurement. It becomes a reference only
when a second, independent production of it agrees. PROTOCOL A13 is that check, and it is the one
that caught the defect this script exists because of: BUNDLE-MIN was first taped with the
Pairformer's rate-0.25 Dropout live, so two tapings of the same boundary disagreed at worst 1.167
/ median 0.550 with 47 of 57 tensors over the 5.0e-02 bar, while the same pair with dropout off
agreed to 0.000e+00 on 57 of 57.

Per-tensor relative L2 is the comparison, NOT a file hash. Two runs of the reference that agree to
1.9e-16 produce different `grads_f64.pt` hashes, because a hash sees the last bit of a float. The
hashes in the manifest verify a TRANSFER; this verifies a computation.

Presence is compared before values (PROTOCOL 3b): a parameter with no gradient must have no
gradient on both sides. Filling an absent gradient with zeros is
`zero-filled-missing-gradient-hides-an-untrained-model`, and it would pass a value comparison.
"""
import argparse
import json
import sys
from pathlib import Path

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("--bar", type=float, default=5.0e-02, help="PROTOCOL 3d per-tensor bar")
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    ga = torch.load(args.a, map_location="cpu", weights_only=False)
    gb = torch.load(args.b, map_location="cpu", weights_only=False)

    only_a = sorted(set(ga) - set(gb))
    only_b = sorted(set(gb) - set(ga))
    presence_mismatch = sorted(k for k in set(ga) & set(gb)
                               if (ga[k] is None) != (gb[k] is None))

    rows, bit_identical, both_absent = [], 0, 0
    for k in sorted(set(ga) & set(gb)):
        x, y = ga[k], gb[k]
        if x is None and y is None:
            both_absent += 1
            continue
        if x is None or y is None:
            continue
        x, y = x.double(), y.double()
        if torch.equal(x, y):
            bit_identical += 1
            rows.append((k, 0.0))
            continue
        denom = max(float(torch.linalg.vector_norm(x)), float(torch.linalg.vector_norm(y)), 1e-300)
        rows.append((k, float(torch.linalg.vector_norm(x - y)) / denom))

    vals = sorted(r[1] for r in rows)
    worst = max(rows, key=lambda r: r[1]) if rows else (None, None)
    over = [r for r in rows if r[1] > args.bar]
    out = {
        "a": str(args.a), "b": str(args.b), "bar": args.bar,
        "n_compared": len(rows),
        "n_bit_identical": bit_identical,
        "n_both_absent": both_absent,
        "n_over_bar": len(over),
        "worst_rel_l2": worst[1], "worst_tensor": worst[0],
        "median_rel_l2": vals[len(vals) // 2] if vals else None,
        "presence_mismatch": presence_mismatch,
        "only_in_a": only_a[:8], "only_in_b": only_b[:8],
        "n_only_in_a": len(only_a), "n_only_in_b": len(only_b),
        "over_bar_sample": [{"param": k, "rel_l2": v} for k, v in over[:16]],
    }
    body = json.dumps(out, indent=2)
    print(body)
    if args.json_out:
        args.json_out.write_text(body + "\n")
    ok = (not presence_mismatch and not only_a and not only_b and not over)
    print("VERDICT:", "REPRODUCED" if ok else "DISAGREES")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
