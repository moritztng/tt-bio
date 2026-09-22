#!/usr/bin/env python3
"""A18 for the trunk: both tracks, masked, padding fraction beside each.

Bars: 5.0e-02 per tensor; the A26 bar an independent bf16 port can reach is
sqrt(2) x 5.0e-02 = 7.0711e-02. The reference is upstream 0.4.3 with every parameter and every
activation in float64 and no cast on the path (A27 -- that is the dtype policy, "float64" is
only a width), on the same captured boundary and the same `of3-p2-155k` weights.
"""
from __future__ import annotations

import argparse
import json

import torch

W = "/home/ttuser/of3t_trunk043ref"
BAR = 5.0e-02
BAR_A26 = 2 ** 0.5 * BAR


def metrics(a, r):
    a = a.reshape(-1).double()
    r = r.reshape(-1).double()
    na, nr = a.norm().item(), r.norm().item()
    d = nr if nr > 0 else 1e-300
    return {"rel": ((a - r).norm() / d).item(), "ratio": na / d,
            "cos": ((a @ r) / ((na * nr) or 1e-300)).item()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True, help="name=path.pt,...")
    ap.add_argument("--ref", default=W + "/ref_043_f64_c64.pt")
    ap.add_argument("--boundary", default=W + "/boundary_c64.pt")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    b = torch.load(a.boundary, map_location="cpu", weights_only=False)
    sm, pm = b["single_mask"], b["pair_mask"]
    N = int(sm.shape[1])
    msk = sm.reshape(1, N, 1).double()
    pmk = pm.reshape(1, N, N, 1).double()
    R = torch.load(a.ref, map_location="cpu", weights_only=False)

    rep = {"what": __doc__.strip().splitlines()[0], "reference": a.ref,
           "reference_dtype_policy": "upstream 0.4.3, every parameter and every activation "
                                     "float64, checkpoint upcast once at load, no cast on the "
                                     "path",
           "bar": BAR, "bar_a26_reachable": BAR_A26,
           "tokens": N, "real_tokens": int(sm.sum()),
           "padding_fraction_single": 1 - int(sm.sum()) / N,
           "padding_fraction_pair": 1 - float(pm.sum()) / (N * N),
           "arms": {}}
    for spec in a.arms.split(","):
        name, _, path = spec.partition("=")
        o = torch.load(path, map_location="cpu", weights_only=False)
        row = {"s_masked": metrics(o["s"] * msk, R["s"] * msk),
               "z_masked": metrics(o["z"] * pmk, R["z"] * pmk),
               "config": o.get("config"), "overrides": o.get("overrides")}
        row["s_verdict"] = ("PASS" if row["s_masked"]["rel"] <= BAR else
                            "PASS-A26" if row["s_masked"]["rel"] <= BAR_A26 else "FAIL")
        row["z_verdict"] = ("PASS" if row["z_masked"]["rel"] <= BAR else
                            "PASS-A26" if row["z_masked"]["rel"] <= BAR_A26 else "FAIL")
        rep["arms"][name] = row
    with open(a.out, "w") as fh:
        json.dump(rep, fh, indent=2)
    print(f"reference: upstream 0.4.3 f64 params and activations, no cast on the path")
    print(f"bar 5.0e-02 per tensor, A26 reachable {BAR_A26:.4e};  "
          f"padding single {rep['padding_fraction_single']:.3f} "
          f"({rep['real_tokens']} of {N}), pair {rep['padding_fraction_pair']:.3f}")
    print(f"{'arm':<22}{'s rel':>12}{'s ratio':>10}{'s cos':>10}{'s':>10}"
          f"{'z rel':>12}{'z ratio':>10}{'z cos':>10}{'z':>10}")
    for n, r in rep["arms"].items():
        s, z = r["s_masked"], r["z_masked"]
        print(f"{n:<22}{s['rel']:>12.6e}{s['ratio']:>10.5f}{s['cos']:>10.6f}"
              f"{r['s_verdict']:>10}{z['rel']:>12.6e}{z['ratio']:>10.5f}{z['cos']:>10.6f}"
              f"{r['z_verdict']:>10}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
