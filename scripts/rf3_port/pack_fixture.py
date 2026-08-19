#!/usr/bin/env python3
"""Pack a raw RF3 reference capture into a committable parity fixture.

Two all-zero tensors dominate the raw capture: ``feats/atom_level_embedding``
[n_conf, L, 384] and ``feats/mean_atom_level_embedding`` [L, 384]. They are the
MLFF (MACE) conformer-embedding track, which is inert on any public run --
``has_atom_level_embedding`` is all-False and the cache path baked into
atomworks (``/net/tukwila/...``) is IPD-internal, so the values are exactly
zero. We store them as a shape-only stub; the gate asserts the ported
featurizer also emits all-zeros of that shape, which is the real contract.

Everything else is stored verbatim, bit-for-bit.
"""
from __future__ import annotations

import argparse
import json
import os

import torch

ZERO_STUB_KEYS = ("feats/atom_level_embedding", "feats/mean_atom_level_embedding")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="raw capture dir (ref_f.pt + meta)")
    ap.add_argument("--dst", required=True, help="fixture dir to write")
    args = ap.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    raw = torch.load(os.path.join(args.src, "ref_f.pt"), weights_only=False)
    meta = json.load(open(os.path.join(args.src, "ref_f.meta.json")))

    stubs = {}
    for k in ZERO_STUB_KEYS:
        if k not in raw:
            continue
        t = raw.pop(k)
        assert torch.count_nonzero(t) == 0, f"{k} is not all-zero; stub is invalid"
        stubs[k] = {"shape": list(t.shape), "dtype": str(t.dtype)}
    meta["__zero_stub_keys__"] = stubs

    torch.save(raw, os.path.join(args.dst, "ref_f.pt"))
    with open(os.path.join(args.dst, "ref_f.meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
    size = os.path.getsize(os.path.join(args.dst, "ref_f.pt"))
    print(f"{args.dst}: {len(raw)} tensors + {len(stubs)} zero-stubs, {size/1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
