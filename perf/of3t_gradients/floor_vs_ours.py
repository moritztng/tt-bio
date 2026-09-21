#!/usr/bin/env python3
"""Put our device gradient and upstream's own CPU replay side by side against the SAME reference.

The published r = 0 reference does not reproduce exactly on this host: upstream's own float64
CPU replay of their step, on the identical boundary with every recorded draw restored, scores
median 6.2244e-02 against it (`replay_vs_r0.json`). That is a floor -- no stack of ours can be
scored tighter than upstream scores against itself -- and quoting our number without it invites
the reader to attribute the whole gap to the port.

So for the 171 tensors where both exist (their blocks 0, 23 and 47), this joins the two errors
per tensor and asks whether they are the SAME tensors. If our error sits where the reference's
own irreproducibility sits, the two are largely one phenomenon. If it sits elsewhere, the part
that sits elsewhere is ours and is the finding.
"""
from __future__ import annotations

import argparse
import json
import os

OUT = "perf/of3t_gradients"
REF_NORM_FLOOR = 1e-12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--floor", default=os.path.join(OUT, "replay_vs_r0.json"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import numpy as np
    ours = json.load(open(os.path.join(OUT, f"instrument_a_bundle_{a.tag}.json")))
    floor = json.load(open(a.floor))["refs"]["republished_r0"]["per_tensor"]
    by_name = {r["their_tensor"]: r for r in ours["per_parameter"]}

    rows = []
    for n, fv in floor.items():
        r = by_name.get(n)
        if r is None or fv is None or r["ref_norm"] < REF_NORM_FLOOR:
            continue
        rows.append({"tensor": n, "ours": r["rel_l2"], "upstream_replay": fv,
                     "ref_norm": r["ref_norm"], "ratio": r["rel_l2"] / fv if fv else None})
    rows.sort(key=lambda d: -d["ours"])
    o = np.array([r["ours"] for r in rows])
    f = np.array([r["upstream_replay"] for r in rows])

    # Spearman without scipy: Pearson on the ranks.
    def rank(x):
        order = np.argsort(x)
        rk = np.empty_like(order, dtype=float)
        rk[order] = np.arange(len(x), dtype=float)
        return rk
    rho = float(np.corrcoef(rank(o), rank(f))[0, 1]) if len(o) > 2 else None

    rep = {
        "source": f"instrument_a_bundle_{a.tag}.json",
        "floor_source": a.floor,
        "joined": len(rows),
        "a14_denominator_floor": REF_NORM_FLOOR,
        "ours": {"median": float(np.median(o)), "worst": float(o.max()),
                 "over_bar_5e-2": int((o > 5e-2).sum())},
        "upstream_replay_floor": {"median": float(np.median(f)), "worst": float(f.max()),
                                  "over_bar_5e-2": int((f > 5e-2).sum())},
        "ours_over_floor": {"median_ratio": float(np.median(o / f)),
                            "n_ours_below_floor": int((o < f).sum())},
        "rank_correlation_spearman": rho,
        "reading": ("the two errors fall on the same tensors -- our disagreement is largely "
                    "where the reference is already irreproducible"
                    if rho is not None and rho > 0.5 else
                    "the two errors fall on DIFFERENT tensors -- ours is not the reference's "
                    "own irreproducibility wearing a different name"),
        "per_tensor": rows,
    }
    path = a.out or os.path.join(OUT, f"floor_vs_ours_{a.tag}.json")
    json.dump(rep, open(path, "w"), indent=1)
    print(f"joined {len(rows)} tensors")
    print(f"  ours            median {rep['ours']['median']:.4e} "
          f"over bar {rep['ours']['over_bar_5e-2']}/{len(rows)}")
    print(f"  upstream replay median {rep['upstream_replay_floor']['median']:.4e} "
          f"over bar {rep['upstream_replay_floor']['over_bar_5e-2']}/{len(rows)}")
    print(f"  median ratio ours/floor {rep['ours_over_floor']['median_ratio']:.3f}, "
          f"spearman {rho:.3f}")
    print(" ", rep["reading"])
    print("->", path)


if __name__ == "__main__":
    main()
