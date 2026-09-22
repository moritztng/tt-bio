#!/usr/bin/env python3
"""Score the single-block arms leaf by leaf, masked, against the f64 reference.

Norm ratio and error cosine beside every relative L2: this defect was identified BY its norm
ratio and a bare relative would have hidden it. Medians per leaf, not the worst tensor.

The padded token positions are excluded from every figure. The single transition is the one op
where this matters on its own: upstream masks it (`_mask_trans=True`), tt-bio does not, so the
UNMASKED update norms of the two sides are not comparable and only the masked ones are.
"""
from __future__ import annotations

import argparse
import json

import torch

W = "/home/ttuser/of3t_trunk043ref"
C = "/home/ttuser/of3t_trunkcliff"
PAIR = {"z1", "z2", "z3", "z4", "z5", "z_out"}


def metrics(a, r):
    a = a.reshape(-1).double()
    r = r.reshape(-1).double()
    na, nr = a.norm().item(), r.norm().item()
    d = nr if nr > 0 else 1e-300
    return {"rel": ((a - r).norm() / d).item(), "ratio": na / d,
            "cos": ((a @ r) / ((na * nr) or 1e-300)).item(), "ref_norm": nr}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", default="44,45,46")
    ap.add_argument("--arms", default="REF64Q,REFBF16,OURS")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    b = torch.load(W + "/boundary_c64.pt", map_location="cpu", weights_only=False)
    sm, pm = b["single_mask"], b["pair_mask"]
    N = int(sm.shape[1])
    msk = sm.reshape(1, N, 1).double()
    pmk = pm.reshape(1, N, N, 1).double()
    rep = {"what": __doc__.strip().splitlines()[0], "tokens": N, "real_tokens": int(sm.sum()),
           "padding_fraction_single": 1 - int(sm.sum()) / N,
           "padding_fraction_pair": 1 - float(pm.sum()) / (N * N),
           "blocks": {}}

    for blk in [int(x) for x in a.blocks.split(",")]:
        ref = torch.load(f"{C}/REF64_b{blk}.pt", map_location="cpu", weights_only=False)
        R = dict(ref["shared"])
        R["s_out"], R["z_out"] = ref["s_out"], ref["z_out"]
        rows = {}
        for arm in a.arms.split(","):
            try:
                o = torch.load(f"{C}/{arm}_b{blk}.pt", map_location="cpu", weights_only=False)
            except FileNotFoundError:
                continue
            S = dict(o["shared"])
            S["s_out"], S["z_out"] = o["s_out"], o["z_out"]
            r = {}
            for k in ("z1", "z2", "z3", "z4", "z5", "sn", "u1", "s1", "u2", "s2"):
                if k not in S or k not in R:
                    continue
                m = pmk if k in PAIR else msk
                r[k] = metrics(S[k] * m, R[k] * m)
            rows[arm] = r
        rep["blocks"][blk] = rows

    with open(a.out, "w") as fh:
        json.dump(rep, fh, indent=2)

    keys = ["z1", "z2", "z3", "z4", "z5", "sn", "u1", "s1", "u2", "s2"]
    for blk, rows in rep["blocks"].items():
        print(f"\n=== block {blk}, masked, vs REF64 (upstream 0.4.3, f64 params and "
              f"activations, no cast on the path) ===")
        print(f"{'leaf':>5} " + "".join(f"| {arm:^34}" for arm in rows))
        print(f"{'':>5} " + "".join("| " + f"{'rel':>10}{'ratio':>12}{'cos':>12} " for _ in rows))
        for k in keys:
            line = f"{k:>5} "
            for arm in rows:
                v = rows[arm].get(k)
                line += ("| " + (f"{v['rel']:>10.3e}{v['ratio']:>12.6f}{v['cos']:>12.6f} "
                                 if v else f"{'-':>34} "))
            print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
