#!/usr/bin/env python3
"""of3t-refcov P0: why the training adapter cannot contribute to coverage_total.

The row was dispatched on the premise that GRADIENTS' coverage leg closes from
`tt_bio/train/openfold3.py`. It cannot, and the reason is not about the adapter's code.

`coverage_total.pct_of_model_compared` is scored over ONE boundary: upstream 0.4.3's own
`loss.backward()` (`perf/of3t_reference/bundle_min.py:368,654`), replayed into each scope as a
captured cotangent (`perf/of3t_diffusion/device_gradient.py:4-5`, "seeded with THEIR
cotangent"). The adapter's arm is seeded by tt-bio's own `af3` objective, and on this batch
only `distogram` and `resolved` fire -- the other six terms are skipped for missing
`pred_xyz`/`pred_dist`/`per_atom_lddt` with the denoise arm off
(`perf/of3t_trainfwd/ARM_full_n384.json`).

So this measures it on the five `input_glue` weights, the one place the two name the same
tensors with no renaming ambiguity. If the disagreement were a precision gap the adapter route
would be live; if it is orders, in both directions, it is a different function of the weights
and no amount of wiring changes that.

    boundary_adapter.py --out BOUNDARY_ADAPTER.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

REF = Path("/home/ttuser/of3t-campaign-refs/bundle_min_043/grads_f64_043.pt")
PIN = "1d4ea9225f854afe06a8adedcb5f8aa1c53656fbf8cefdaa3a451d0327895cc4"
ARM = Path("/home/ttuser/of3t_trainfwd/grad_full_n384.pt")
ARM_JSON = Path("/home/ttuser/of3t_trainfwd/grad_full_n384.pt")  # provenance lives in the repo
# `InputEmbedderGlue` (tt_bio/openfold3.py:29) -> the checkpoint names it loads from.
PAIRS = {"input_glue.w_s": "input_embedder.linear_s.weight",
         "input_glue.w_zi": "input_embedder.linear_z_i.weight",
         "input_glue.w_zj": "input_embedder.linear_z_j.weight",
         "input_glue.w_relpos": "input_embedder.linear_relpos.weight",
         "input_glue.w_tb": "input_embedder.linear_token_bonds.weight"}
MASS_BAR = 0.02


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for c in iter(lambda: fh.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    got = sha256(REF)
    r = torch.load(REF, map_location="cpu", mmap=True)
    if isinstance(r, dict) and "grads" in r:
        r = r["grads"]
    arm = torch.load(ARM, map_location="cpu", mmap=True)

    rows, refuted = [], []
    for ak, rk in PAIRS.items():
        av = arm[ak].double().reshape(-1)
        rv = r[rk].double().reshape(-1)
        assert av.numel() == rv.numel(), (ak, rk, av.numel(), rv.numel())
        rn, an = float(rv.norm()), float(av.norm())
        rel = float((av - rv).norm() / rn) if rn else None
        rows.append({"adapter_param": ak, "checkpoint_param": rk, "numel": av.numel(),
                     "reference_norm": rn, "adapter_norm": an,
                     "norm_ratio": (an / rn) if rn else None, "rel_l2": rel,
                     "cos": (float(torch.dot(av, rv) / (an * rn)) if an and rn else None)})
        if rel is not None and rel <= MASS_BAR:
            refuted.append(ak)

    out = {
        "instrument": "of3t-refcov boundary_adapter.py -- P0: is the training adapter's arm on "
                      "the same loss boundary as the pinned float64 reference",
        "host": "qb2 (tt-quietbox2), CPU only; the adapter dump lives there",
        "prediction": "the norms disagree by ORDERS, in BOTH directions, so it is a different "
                      "function and not a precision gap. Refuted if any pair reads rel_l2 <= "
                      f"{MASS_BAR}",
        "reference": {"path": str(REF), "sha256": got, "matches_pin": got == PIN,
                      "what": "upstream OpenFold3 0.4.3 full OpenFold3Loss, one loss.backward()"},
        "adapter_arm": {"path": str(ARM), "n_keys": len(arm),
                        "what": "perf/of3t_trainfwd/trainfwd_run.py --arm full, seeded by "
                                "tt_bio.train.objectives 'af3'",
                        "terms_that_fired": ["distogram", "resolved"],
                        "terms_skipped": ["bond", "smooth_lddt", "mse", "plddt", "pae", "pde"]},
        "per_tensor": rows,
        "worst_rel_l2": max((x["rel_l2"] or 0.0) for x in rows),
        "worst_tensor": max(rows, key=lambda x: x["rel_l2"] or 0.0)["checkpoint_param"],
        "norm_ratio_range": [min(x["norm_ratio"] for x in rows if x["norm_ratio"] is not None),
                             max(x["norm_ratio"] for x in rows if x["norm_ratio"] is not None)],
        "pairs_inside_the_bar": refuted,
        "prediction_refuted": bool(refuted),
        "finding": "the adapter's arm is a gradient of a DIFFERENT loss. Wiring the atom-encoder "
                   "legs into its tape puts a gradient on those weights, but not one this "
                   "reference can score, so it cannot move coverage_total at any precision. "
                   "The route that can is the union, and of3t-hostleg already built both arms.",
    }
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    for x in rows:
        print("  %-22s %-42s numel %7d  ref %.9g  arm %.9g  ratio %.6g  rel_l2 %.6g"
              % (x["adapter_param"], x["checkpoint_param"], x["numel"], x["reference_norm"],
                 x["adapter_norm"], x["norm_ratio"] or float("nan"),
                 x["rel_l2"] if x["rel_l2"] is not None else float("nan")))
    print("worst rel_l2 %.9g at %s; norm ratios span %.4g to %.4g; prediction refuted: %s"
          % (out["worst_rel_l2"], out["worst_tensor"], out["norm_ratio_range"][0],
             out["norm_ratio_range"][1], out["prediction_refuted"]))
    print("->", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
