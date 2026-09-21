#!/usr/bin/env python3
"""Deliverable 1: score the cotangent ENTERING each block against the reference's.

The discriminator this row exists for. The weight gradient at block k is a function of two
things -- the cotangent that arrived there and the leaf backward that consumes it -- and the
per-block error profile alone cannot separate them. The cotangent can:

  (A) cotangent degrades with depth  -> the error is injected per block and carried, and the
      leaf gradients are downstream of it;
  (B) cotangent at the floor at every rung -> the propagation is fine and the leaf backward is
      the defect;
  (C) neither -- flat early and degrading late, a regime-dependent injection.

The floor at every rung is upstream's OWN bf16 recipe's cotangent at that rung against its own
float64, measured here rather than assumed, so each row carries its own denominator (A28) and
names how it was built (A27). `norm_ratio` and `cos` sit beside every `rel_l2` (D35).

Masked and unmasked both: a cotangent lives in activation space, where D95 applies, and the pad
rows carry no information about the function. The verdict is read on the MASKED column.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch


def triple(m, r):
    m = m.reshape(-1).to(torch.float64)
    r = r.reshape(-1).to(torch.float64)
    nr = float(torch.linalg.vector_norm(r))
    nm = float(torch.linalg.vector_norm(m))
    if nr == 0.0:
        return None
    rel = float(torch.linalg.vector_norm(m - r) / nr)
    cos = float((m @ r) / (nm * nr)) if nm else 0.0
    return {"rel_l2": rel, "norm_ratio": nm / nr, "cos": cos, "ref_norm": nr, "our_norm": nm}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-f64", required=True)
    ap.add_argument("--ref-bf16", required=True)
    ap.add_argument("--boundary", required=True)
    ap.add_argument("--arm", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    b = torch.load(a.boundary, map_location="cpu", weights_only=False)
    sm = (b["single_mask"].reshape(-1) > 0)
    N = int(sm.numel())
    real = int(sm.sum())
    msk_s = sm.to(torch.float64).reshape(N, 1)
    msk_z = (sm.to(torch.float64).reshape(N, 1) * sm.to(torch.float64).reshape(1, N)
             ).reshape(N, N, 1)

    f64 = torch.load(a.ref_f64, map_location="cpu", weights_only=False)
    bf = torch.load(a.ref_bf16, map_location="cpu", weights_only=False)
    arms = {}
    for spec in a.arm:
        n, _, p = spec.partition("=")
        arms[n] = torch.load(p, map_location="cpu", weights_only=False)

    def shaped(t, track):
        if t is None:
            return None
        t = t.to(torch.float64)
        return t.reshape(N, -1) if track == "ds" else t.reshape(N, N, -1)

    rungs = sorted(int(k) for k in f64["cot"])
    out = {"what": __doc__.strip().splitlines()[0],
           "rung_definition": "rung k is the cotangent ENTERING block k; rung 0 is the stack "
                              "input and rung 48 is the captured seed itself",
           "masking": f"{real} real of {N} tokens",
           "reference": {"float64": f64["policy"],
                         "floor": "upstream 0.4.3, fp32 parameters under "
                                  "torch.autocast('cpu', bfloat16), their own recipe, "
                                  "differentiated in this row's own process"},
           "rungs": {}}
    for k in rungs:
        row = {}
        for track, msk in (("ds", msk_s), ("dz", msk_z)):
            r = shaped(f64["cot"][k][track], track)
            fl = shaped(bf["cot"][k][track], track)
            if r is None:
                continue
            ent = {"floor_masked": triple(fl * msk, r * msk) if fl is not None else None,
                   "floor_unmasked": triple(fl, r) if fl is not None else None,
                   "ref_norm_masked": float((r * msk).norm())}
            for nm, d in arms.items():
                o = shaped(d["cot"].get(k, {}).get(track), track)
                if o is None:
                    continue
                ent[nm] = {"masked": triple(o * msk, r * msk), "unmasked": triple(o, r)}
                if ent["floor_masked"] and ent[nm]["masked"]:
                    ent[nm]["over_floor_masked"] = (ent[nm]["masked"]["rel_l2"]
                                                    / ent["floor_masked"]["rel_l2"])
            row[track] = ent
        out["rungs"][k] = row

    # the pre-registered discriminator, evaluated mechanically
    verdict = {}
    for nm in arms:
        for track in ("ds", "dz"):
            xs = [(k, out["rungs"][k][track][nm].get("over_floor_masked"))
                  for k in rungs if track in out["rungs"][k]
                  and nm in out["rungs"][k][track]
                  and out["rungs"][k][track][nm].get("over_floor_masked") is not None]
            xs = [(k, v) for k, v in xs if k < max(rungs)]     # the seed rung is trivially 1.0
            if not xs:
                continue
            xs.sort(key=lambda t: -t[0])                       # backward order: 47 -> 0
            vals = [v for _, v in xs]
            growth = vals[-1] / vals[0] if vals[0] else float("nan")
            rises = sum(1 for i in range(1, len(vals)) if vals[i] > vals[i - 1])
            verdict[f"{nm}.{track}"] = {
                "over_floor_at_the_top": vals[0], "over_floor_at_the_bottom": vals[-1],
                "end_to_end_growth": growth,
                "max_over_floor": max(vals), "min_over_floor": min(vals),
                "rung_of_the_max": xs[int(np.argmax(vals))][0],
                "monotone_rising_steps": rises, "steps": len(vals) - 1,
                "all_at_or_under_2.5x": max(vals) <= 2.5,
                "branch": ("B: cotangent at the floor at every rung" if max(vals) <= 2.5 else
                           "A: cotangent degrades with depth"
                           if rises >= 0.8 * (len(vals) - 1) and growth >= 4.0 else
                           "C: neither clean -- see the profile")}
    out["discriminator"] = verdict
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2)

    print(f"{'rung':>4} {'ds ours':>11} {'ds floor':>11} {'x':>7} "
          f"{'dz ours':>11} {'dz floor':>11} {'x':>7}")
    nm = list(arms)[0]
    for k in sorted(rungs, reverse=True):
        r = out["rungs"][k]
        def cell(tr):
            e = r.get(tr)
            if not e or nm not in e:
                return (float("nan"),) * 3
            return (e[nm]["masked"]["rel_l2"], e["floor_masked"]["rel_l2"],
                    e[nm].get("over_floor_masked", float("nan")))
        ds, dz = cell("ds"), cell("dz")
        print(f"{k:>4} {ds[0]:11.4e} {ds[1]:11.4e} {ds[2]:7.2f} "
              f"{dz[0]:11.4e} {dz[1]:11.4e} {dz[2]:7.2f}")
    print(json.dumps(verdict, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
