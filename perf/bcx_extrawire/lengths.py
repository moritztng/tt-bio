#!/usr/bin/env python3
"""What "n" means on a BindCraft 2 round, which is three different numbers.

This row's own write-up said n=275, `bcx-extramsa` said n=211, and `design_state` says 263. They
are not in conflict; they name different layers, and nothing on the campaign said which was which.
Pinning it down matters because every per-token cost, every padded-region test and every
matched-pair grade quotes one of them:

  263   the residues that exist. Binder 148 (sampled from the campaign seed) + target hPDL1 115.
  275   what BindCraft 2 hands the model. `padded_prediction_complex` (`bindcraft/af2.py:55`) pads
        the DESIGN chain to the length bucket and leaves the target alone, so the binder goes
        148 -> 160 and the complex is 160 + 115. Note 275 is not itself a multiple of 32.
  288   what the card sees. tt-bio pads the token axis to a multiple of 32, so 275 -> 288, which
        is the 13 rows `tests/test_bindcraft2.py` covers.

Opens no device, needs no weights: the whole chain falls out of the settings and the trajectory's
first state. Run it before quoting an n.
"""
import argparse
import json
import os
import pathlib
import sys

os.environ.setdefault("TT_VISIBLE_DEVICES", "")
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(ROOT), str(ROOT / "perf" / "bcx_predictor")):
    if p not in sys.path:
        sys.path.insert(0, p)

import bc2_state as B                                                  # noqa: E402
from bindcraft.af2 import campaign_length_bucket, padded_prediction_length   # noqa: E402

#: The override list hostshare.py, grade_host.py and round_ab.py all share. The length is sampled
#: from the campaign seed, so a harness that drifts from this list can silently measure another
#: complex, and then 22.202 s of host is a cost for a round nobody else ran.
SHARED_OVERRIDES = ["max_trajectories=1", "validation_model=monomer",
                    'design_models=["model_1_ptm"]', 'validation_models=["model_2_ptm"]']


def chain(seed, tile=32):
    settings = B.campaign_settings(
        overrides=[f"campaign_seed={seed}", *SHARED_OVERRIDES,
                   "project_folder=/tmp/bcx_extrawire_lengths"])
    _ds, states, _losses = B.design_state(settings)
    (state,) = states.values()
    chains = {name: len(protein.residue_index) for name, protein in state.items()}
    bucket = campaign_length_bucket(settings)
    # BindCraft 2 pads the design chain only; the target keeps its own length.
    handed_over = sum(padded_prediction_length(n, bucket) if name == "binder" else n
                      for name, n in chains.items())
    return {"seed": seed, "chains": chains, "residues_that_exist": sum(chains.values()),
            "bucket": bucket, "handed_to_the_model": handed_over,
            "on_tile": -(-handed_over // tile) * tile, "tile": tile}


def region_from_mask_fraction(zero_fraction, width):
    """The real region implied by a pair mask's zero fraction, as an independent cross-check.

    `mask_2d` is the outer product of the token mask, so a width-w array with a real region of r
    is zero on w**2 - r**2 of its entries. Inverting that recovers r from a number the gradient
    capture already recorded, with no settings involved at all.
    """
    return ((1.0 - zero_fraction) ** 0.5) * width


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = chain(args.seed)

    grade = ROOT / "perf/bcx_extrawire/runs/grade_hostarms/grade_host.json"
    if grade.is_file():
        g = json.loads(grade.read_text())
        if g.get("seed") == args.seed and "mask_2d_zero_fraction" in g:
            r = region_from_mask_fraction(g["mask_2d_zero_fraction"], g["n"])
            out["cross_check"] = {
                "source": str(grade.relative_to(ROOT)), "capture_width": g["n"],
                "region_from_mask_fraction": round(r, 2),
                "agrees_with_residues_that_exist": abs(r - out["residues_that_exist"]) < 0.5}
    print(json.dumps(out, indent=1))
    if args.out:
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
