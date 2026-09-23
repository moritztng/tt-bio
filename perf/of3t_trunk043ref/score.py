#!/usr/bin/env python3
"""Score every arm against every reference, both tracks, masked and padded side by side.

D99 exists because this comparison was quoted on one track for six passes while the other track
sat two keys away in the same file. Nothing here reports one track.

Relative L2 alone cannot name the direction, so every reading carries the norm ratio
r = ||ours|| / ||reference|| and the cosine.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

import torch

BAR = 5.0e-02
BAR_A26 = 5.0e-02 * (2 ** 0.5)


def sha256_file(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def metrics(ours, ref):
    a = ours.to(torch.float64).reshape(-1)
    b = ref.to(torch.float64).reshape(-1)
    na = float(torch.linalg.vector_norm(a))
    nb = float(torch.linalg.vector_norm(b))
    den = nb if nb > 0 else 1e-300
    return {"rel_l2": float(torch.linalg.vector_norm(a - b) / den),
            "norm_ratio": na / den,
            "cos": float((a @ b) / ((na * nb) or 1e-300)),
            "ours_norm": na, "ref_norm": nb}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boundary", required=True)
    ap.add_argument("--arm", action="append", default=[], metavar="TAG=PATH")
    ap.add_argument("--ref", action="append", default=[], metavar="TAG=PATH")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    b = torch.load(a.boundary, map_location="cpu", weights_only=False)
    sm, pm = b["single_mask"], b["pair_mask"]
    N = int(sm.shape[1])
    real = int(sm.sum())
    msk = sm.reshape(1, N, 1).to(torch.float64)
    pmk = pm.reshape(1, N, N, 1).to(torch.float64)

    def load(spec):
        tag, path = spec.split("=", 1)
        d = torch.load(path, map_location="cpu", weights_only=False)
        return tag, {"path": path, "sha256": sha256_file(path),
                     "s": d["s"].to(torch.float64), "z": d["z"].to(torch.float64)}

    arms = dict(load(x) for x in a.arm)
    refs = dict(load(x) for x in a.ref)
    # A16: a model that emits nothing, scored on this exact boundary rather than assumed to be 1
    z0 = {"path": "(A16 zero model)", "sha256": "-",
          "s": torch.zeros_like(next(iter(refs.values()))["s"]),
          "z": torch.zeros_like(next(iter(refs.values()))["z"])}
    arms["zero_model_A16"] = z0
    # the captured 0.5.0 reference as it was published, so the old figures are reproduced here
    refs.setdefault("captured_050", {"path": a.boundary + "[captured]",
                                     "sha256": sha256_file(a.boundary),
                                     "s": b["s_ref_050_captured"],
                                     "z": b["z_ref_050_captured"]})

    table = {}
    for rt, r in refs.items():
        for at, m in arms.items():
            table[f"{at}__vs__{rt}"] = {
                "arm": at, "ref": rt, "arm_sha256": m["sha256"], "ref_sha256": r["sha256"],
                "s_masked": metrics(m["s"] * msk, r["s"] * msk),
                "z_masked": metrics(m["z"] * pmk, r["z"] * pmk),
                "s_padded": metrics(m["s"], r["s"]),
                "z_padded": metrics(m["z"], r["z"]),
            }

    rep = {"what": __doc__.strip().splitlines()[0],
           "tokens": N, "real_tokens": real,
           "real_over_total_single": f"{real} of {N}",
           "padding_fraction_single": 1.0 - real / N,
           "real_over_total_pair": f"{int(pm.sum())} of {N * N}",
           "padding_fraction_pair": 1.0 - float(pm.sum()) / (N * N),
           "bars": {"per_tensor": BAR, "A26_reachable": BAR_A26, "mass_weighted": 2.0e-02},
           "table": table}
    with open(a.out, "w") as fh:
        json.dump(rep, fh, indent=2)

    w = max(len(k) for k in table)
    print(f"{'comparison':<{w}}  {'s_masked':>12} {'z_masked':>12} | "
          f"{'s_padded':>12} {'z_padded':>12}")
    for k, v in table.items():
        print(f"{k:<{w}}  {v['s_masked']['rel_l2']:12.6e} {v['z_masked']['rel_l2']:12.6e} | "
              f"{v['s_padded']['rel_l2']:12.6e} {v['z_padded']['rel_l2']:12.6e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
