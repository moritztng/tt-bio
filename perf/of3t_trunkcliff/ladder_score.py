#!/usr/bin/env python3
"""The depth ladder again, with the corrected pair-bias scale, against the same f64 reference.

Same rungs as `of3t-trunk043ref/LADDER_c64*.json`, same cached reference tensors, same masking.
The shipped arm's numbers are re-read from that row's own device outputs, not retyped.

Upstream's own bf16 recipe (float32 parameters under torch.autocast('cpu', bfloat16)) is scored
on the same rungs where it exists, as the floor. A27: that sentence is the dtype policy.
"""
from __future__ import annotations

import json
import os

import torch

W = "/home/ttuser/of3t_trunk043ref"
C = "/home/ttuser/of3t_trunkcliff"


def metrics(a, r):
    a = a.reshape(-1).double()
    r = r.reshape(-1).double()
    na, nr = a.norm().item(), r.norm().item()
    d = nr if nr > 0 else 1e-300
    return {"rel": ((a - r).norm() / d).item(), "ratio": na / d,
            "cos": ((a @ r) / ((na * nr) or 1e-300)).item()}


def main() -> int:
    b = torch.load(W + "/boundary_c64.pt", map_location="cpu", weights_only=False)
    sm, pm = b["single_mask"], b["pair_mask"]
    N = int(sm.shape[1])
    msk = sm.reshape(1, N, 1).double()
    pmk = pm.reshape(1, N, N, 1).double()
    ks = [1, 8, 16, 24, 32, 42, 44, 45, 46, 47, 48]
    rep = {"what": __doc__.strip().splitlines()[0], "tokens": N, "real_tokens": int(sm.sum()),
           "rows": []}
    print(f"{'k':>3} | {'SHIPPED s':>11}{'ratio':>9} | {'SCALED s':>11}{'ratio':>9}"
          f"{'cos':>10} | {'UPSTREAM bf16 s':>16}{'ratio':>9} | {'SCALED z':>11}{'ratio':>9}")
    for k in ks:
        r = torch.load(f"{W}/ladder_c64/ladder_ref043_k{k}.pt", map_location="cpu",
                       weights_only=False)
        row = {"k": k}
        sh = f"{W}/ladder_c64/ladder_dev_k{k}/device_shipped.pt"
        if os.path.isfile(sh):
            d = torch.load(sh, map_location="cpu", weights_only=False)
            row["shipped"] = {"s": metrics(d["s"] * msk, r["s"] * msk),
                              "z": metrics(d["z"] * pmk, r["z"] * pmk)}
        sc = f"{C}/ladder/SCALED_k{k}.pt"
        if os.path.isfile(sc):
            d = torch.load(sc, map_location="cpu", weights_only=False)
            row["scaled"] = {"s": metrics(d["s"] * msk, r["s"] * msk),
                             "z": metrics(d["z"] * pmk, r["z"] * pmk)}
        bf = f"{W}/ladder_c64/bf16_k{k}.pt"
        if os.path.isfile(bf):
            d = torch.load(bf, map_location="cpu", weights_only=False)
            row["upstream_bf16"] = {"s": metrics(d["s"] * msk, r["s"] * msk),
                                    "z": metrics(d["z"] * pmk, r["z"] * pmk)}
        rep["rows"].append(row)
        g = lambda a, t, f: (row[a][t][f] if a in row else float("nan"))
        print(f"{k:>3} | {g('shipped','s','rel'):>11.4e}{g('shipped','s','ratio'):>9.5f} | "
              f"{g('scaled','s','rel'):>11.4e}{g('scaled','s','ratio'):>9.5f}"
              f"{g('scaled','s','cos'):>10.6f} | "
              f"{g('upstream_bf16','s','rel'):>16.4e}{g('upstream_bf16','s','ratio'):>9.5f} | "
              f"{g('scaled','z','rel'):>11.4e}{g('scaled','z','ratio'):>9.5f}")
    with open(f"{C}/LADDER_SCALED_c64.json", "w") as fh:
        json.dump(rep, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
