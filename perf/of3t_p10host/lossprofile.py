#!/usr/bin/env python3
"""Where the 6.309 s of "loss heads" actually goes, term by term, on the host. No card.

`of3t-PLAN.md` prices the loss heads at 6.309 s of a 39.886 s step and `of3t-gpugap`
projects them at 22-38 s at upstream's 48 diffusion samples. Both numbers come out of
`fullstep.py::host_losses`, whose timed region is NOT just `af3_loss`: it also draws three
384x384x64 float64 logit arrays from `rng.standard_normal` and builds the labels
(`_pdist` twice, `atom_bespoke_lddt`, `lddt_mask`) inside the same stopwatch.

So the first question is not how to port the loss set, it is how much of the reading is the
loss set. This splits the region into fixture / labels / af3_loss and then times each term
inside `af3_loss` separately, at the harness's own shapes, so a port can be aimed at the
terms that are both real and large -- and at 48 samples, at the ones that actually repeat
per sample.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(REPO))

from tt_bio.train import losses as L
from tt_bio.train.objectives import af3_loss
from tt_bio.train.losses import of3_loss_weights


def build(n, rng):
    """Exactly `fullstep.py::host_losses`'s fixture, split into its three parts."""
    t = {}
    t0 = time.perf_counter()
    pred = rng.standard_normal((n, 3)) * 10.0
    true_xyz = pred + rng.standard_normal(pred.shape) * 1.0
    lg = lambda *s: rng.standard_normal(s) * 0.5
    logits = {"distogram_logits": lg(n, n, 64), "pde_logits": lg(n, n, 64),
              "pae_logits": lg(n, n, 64), "plddt_logits": lg(n, 50),
              "resolved_logits": lg(n, 2)}
    t["fixture_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    coord_mask = np.ones(n)
    is_nuc = np.zeros(n, bool)
    is_poly = np.ones(n, bool)
    true_dist, pred_dist = L._pdist(true_xyz), L._pdist(pred)
    pair_mask = coord_mask[:, None] * coord_mask[None, :]
    lddt, lddt_w = L.atom_bespoke_lddt(pred, true_xyz, is_nuc, is_poly,
                                       coord_mask.astype(bool))
    idxs = np.arange(n)
    labels = {"true_xyz": true_xyz, "coord_mask": coord_mask, "true_dist": true_dist,
              "lddt_pair_mask": L.lddt_mask(true_dist, pair_mask, is_nuc),
              "bond_mask": np.zeros((n, n)),
              "per_atom_lddt": lddt, "per_atom_weight": lddt_w,
              "frame_atom_index": np.stack(
                  [np.clip(idxs - 1, 0, n - 1), idxs, np.clip(idxs + 1, 0, n - 1)], -1),
              "is_dna": np.zeros(n), "is_rna": np.zeros(n), "is_ligand": np.zeros(n)}
    outputs = {"pred_xyz": pred, "pred_dist": pred_dist, **logits}
    t["labels_s"] = time.perf_counter() - t0
    return labels, outputs, t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=384)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--stage", default="initial_training")
    ap.add_argument("--out", type=Path,
                    default=REPO / "perf/of3t_p10host/out/lossprofile.json")
    a = ap.parse_args()

    n = a.tokens
    weights = of3_loss_weights(a.stage)
    rng = np.random.default_rng(0)
    rows = []
    for rep in range(a.reps):
        labels, outputs, t = build(n, rng)
        t0 = time.perf_counter()
        total, breakdown, seeds = af3_loss(labels, outputs, weights)
        t["af3_loss_s"] = time.perf_counter() - t0

        # Term by term, on the SAME inputs af3_loss just ran on, so the per-term seconds
        # sum to af3_loss's own reading rather than to a differently shaped rerun.
        from tt_bio.train import objectives as O
        b2, _ = O._with_entity_flags(labels)
        per = {}
        for term, w in weights.items():
            if w == 0.0:
                per[term] = {"s": 0.0, "skipped": "weight zero"}
                continue
            need = O._NEEDS[term]
            if not all(k in outputs or k in b2 for k in need):
                per[term] = {"s": 0.0, "skipped": "inputs absent"}
                continue
            t1 = time.perf_counter()
            O._TERMS[term](b2, outputs)
            per[term] = {"s": time.perf_counter() - t1, "weight": w}
        t["per_term"] = per
        t["per_term_sum_s"] = sum(v["s"] for v in per.values())
        t["region_s"] = t["fixture_s"] + t["labels_s"] + t["af3_loss_s"]
        t["rep"] = rep
        t["total"] = float(total)
        rows.append(t)
        print(f"rep {rep}: region {t['region_s']:.3f}s = fixture {t['fixture_s']:.3f} + "
              f"labels {t['labels_s']:.3f} + af3_loss {t['af3_loss_s']:.3f}", flush=True)
        for k, v in sorted(per.items(), key=lambda kv: -kv[1]["s"]):
            print(f"    {k:14s} {v['s']:7.3f}s  {v.get('skipped','')}", flush=True)

    steady = rows[-1]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"tokens": n, "stage": a.stage, "reps": rows,
                                 "steady": steady}, indent=2, default=float))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
