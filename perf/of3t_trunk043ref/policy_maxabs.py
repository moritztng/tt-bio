#!/usr/bin/env python3
"""Elementwise max-abs between the three attention-precision arms, as an artifact.

The row's ATTNPOLICY reading rests on two claims that were only ever prose: that selecting the
tree's own policy through the patching machinery is bit-identical to an unpatched run, and that
switching the policy is live. `arm_sha256` in ATTNPOLICY_c64.json cannot settle either -- it is a
sha256 of the torch file, which carries serialization metadata, so two byte-identical tensor sets
can digest differently (they do here: 43d6cb4e vs b63fec0d). A rel_l2 that agrees to 9 printed
digits is also not bit-identity.

So this compares the tensors themselves and writes the max-abs out, both masked to the real
tokens and over the padded window. The control direction is native32-vs-keep, which must be
exactly 0.0; the live direction is hp32, which must not be.
"""
import argparse
import hashlib
import json

import torch

ARMS = {
    "keep": "/home/ttuser/of3t_trunk043ref/ref_043_bf16_attnkeep_c64.pt",
    "native32": "/home/ttuser/of3t_trunk043ref/ref_043_bf16_attnnative32_c64.pt",
    "hp32": "/home/ttuser/of3t_trunk043ref/ref_043_bf16_attnhp32_c64.pt",
}
PAIRS = [("native32", "keep", "control: must be exactly 0.0"),
         ("hp32", "keep", "live: the policy switched"),
         ("hp32", "native32", "live: the policy switched, against the control arm")]


def sha256_file(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boundary", default="/home/ttuser/of3t_trunk043ref/boundary_c64.pt")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    b = torch.load(a.boundary, map_location="cpu", weights_only=False)
    real = b["single_mask"].to(torch.bool).reshape(-1)
    n_real, n_tot = int(real.sum()), int(real.numel())

    loaded = {k: torch.load(v, map_location="cpu", weights_only=False) for k, v in ARMS.items()}
    rep = {
        "what": __doc__.strip().splitlines()[0],
        "why": "arm_sha256 is a digest of the torch FILE, not of the tensors; it cannot decide "
               "bit-identity, and a rel_l2 printed to 9 digits cannot either",
        "real_over_total_single": f"{n_real} of {n_tot}",
        "arms": {k: {"path": v, "sha256": sha256_file(v)} for k, v in ARMS.items()},
        "pairs": {},
    }
    for x, y, why in PAIRS:
        e = {}
        for t in ("s", "z"):
            u, v = loaded[x][t].to(torch.float64), loaded[y][t].to(torch.float64)
            d = (u - v).abs()
            if t == "s":
                dm = d.reshape(-1, d.shape[-1])[real]
            else:
                dm = d.reshape(d.shape[-3], d.shape[-2], -1)[real][:, real]
            e[t] = {"max_abs_padded": float(d.max()), "max_abs_masked": float(dm.max()),
                    "bit_identical_padded": bool(torch.equal(u, v)),
                    "n_differing_elements_padded": int((u != v).sum())}
        rep["pairs"][f"{x}__minus__{y}"] = {"why": why, **e}

    with open(a.out, "w") as fh:
        json.dump(rep, fh, indent=2)
    print(json.dumps(rep["pairs"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
