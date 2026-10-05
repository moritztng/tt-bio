#!/usr/bin/env python3
"""Grade a tt-bio `{name}_pae.npz` against an upstream reference fold of the same input.

Contact probabilities come from the trunk alone, so tt-bio's and upstream's are compared
directly. PAE and PDE are read off a diffusion sample, so their device-vs-upstream deviation
is printed beside the upstream seed floor: the same deviation between two upstream seeds.
The diagonal is excluded from contacts (tt-bio defines it as 1, upstream leaves it to the head).

    python perf/fdx_confidence/compare.py --ours out/.../9bk6_pae.npz \
        --ref /tmp/fdxconf/of3_9bk6/upstream_seed0.npz --floor /tmp/fdxconf/of3_9bk6/upstream_seed1.npz
"""
import argparse
import json

import numpy as np


def ref_arrays(path):
    z = np.load(path)
    out = {}
    for k in ("pae", "pde", "contact_probs"):
        if k in z.files:
            v = z[k]
            while v.ndim > 2:          # upstream keeps [batch, sample, N, N]; one of each here
                v = v[0]
            out[k] = v.astype(np.float64)
    return out


def dev(a, b, offdiag):
    m = ~np.eye(len(a), dtype=bool) if offdiag else np.ones(a.shape, bool)
    d = np.abs(a - b)[m]
    return {"max": round(float(d.max()), 4), "mean": round(float(d.mean()), 4),
            "pearson": round(float(np.corrcoef(a[m], b[m])[0, 1]), 5)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--floor")
    a = ap.parse_args()
    ours = {k: v.astype(np.float64) for k, v in np.load(a.ours).items() if np.ndim(v) == 2}
    ref = ref_arrays(a.ref)
    floor = ref_arrays(a.floor) if a.floor else {}
    rep = {}
    for k in sorted(set(ours) & set(ref)):
        if ours[k].shape != ref[k].shape:
            rep[k] = {"error": f"shape {ours[k].shape} vs upstream {ref[k].shape}"}
            continue
        rep[k] = {"vs_upstream": dev(ours[k], ref[k], k == "contact_probs")}
        if k in floor:
            rep[k]["upstream_seed_floor"] = dev(floor[k], ref[k], k == "contact_probs")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
