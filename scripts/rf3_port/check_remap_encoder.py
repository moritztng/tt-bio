#!/usr/bin/env python3
"""Account for every RF3 atom-transformer weight: remapped, caller-side, or missed.

Necessary and NOT sufficient. This sees weights only; three gaps on this port
(`no_residual`, `use_inv_dist_squared`, the 2-factor SwiGLU) were weightless config
flags that a key check reports clean. Pair it with a read of the reference's forward.

    python scripts/rf3_port/check_remap_encoder.py --ckpt ...
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

PREFIX = "shadow.feature_initializer.input_feature_embedder.atom_attention_encoder."


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n_block", type=int, default=3)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    from tt_bio.rf3.remap_encoder import (
        atom_transformer_bias_weights,
        atom_transformer_unmapped,
        remap_atom_transformer,
    )

    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)["model"]
    src = {k[len(PREFIX):]: v.float() for k, v in sd.items() if k.startswith(PREFIX)}

    n_src = sum(1 for k in src if k.startswith("atom_transformer."))
    out = remap_atom_transformer(src, args.n_block)
    bias = sum(len(atom_transformer_bias_weights(src, i)) for i in range(args.n_block))
    missed = atom_transformer_unmapped(src, args.n_block)

    print(f"rf3 atom-transformer weights : {n_src}")
    print(f"  -> tt-bio block names      : {len(out)}")
    print(f"  -> caller-side pair bias   : {bias}")
    print(f"  -> unmapped                : {len(missed)}")
    for k in missed:
        print(f"     MISSED {k}")
    if args.verbose:
        for k, v in sorted(out.items()):
            print("   %-52s %s" % (k, tuple(v.shape)))
    return 0 if not missed else 1


if __name__ == "__main__":
    sys.exit(main())
