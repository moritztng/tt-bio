#!/usr/bin/env python3
"""Which complex a BindCraft 2 number was taken on, which is not one question but two.

There are two different PD-L1 complexes in this campaign's numbers, and for four passes this file
claimed they were three layers of one. They are not. Each has its own residues / handed-over /
on-tile triple, and which one you get depends on how the harness reached BindCraft 2:

  the campaign's trajectory 1     186 residues -> 211 handed over -> 224 on tile   binder 71
  the reference state             263 residues -> 275 handed over -> 288 on tile   binder 148

`campaign.py` folds the trajectory number into the campaign key and splits it again before it
builds a trajectory (`campaign.py:162`, `trajectory.py:280-282`), so trajectory 1 at seed 100
draws a 71-residue binder. `bc2_state.design_state()` calls `initialize_design_trajectory` on the
bare `PRNGKey(seed)` instead, which is `campaign.py:239` -- a reference state, not a trajectory --
and draws 148. No seed makes them agree: the key derivations differ by a `fold_in` and a `split`.

So `round_ab.py`, which drives `campaign.py`, measures the 186-residue complex, and
`hostshare.py` and `grade_host.py`, which go through `design_state()`, measured the 263-residue
one. `bcx-extramsa`'s n=211 is trajectory 1's handed-over width, so its round A/B and this row's
are the same complex; this row's own 22.202 s host share and gradient grade are the other.

Within a complex the three numbers still mean what they always did:

  residues        what exists, binder + target hPDL1 115.
  handed over     `padded_prediction_complex` (`bindcraft/af2.py:55`) pads the DESIGN chain to
                  the length bucket and leaves the target alone. 211 and 275 are both odd sizes,
                  neither a multiple of 32.
  on tile         tt-bio pads the token axis to a multiple of 32.

Opens no device, needs no weights. Run it before quoting an n.
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

import jax                                                             # noqa: E402

import bc2_state as B                                                  # noqa: E402
from bindcraft.settings import build_design_settings                   # noqa: E402
from bindcraft.protein_preparation import initialize_design_trajectory  # noqa: E402
from bindcraft.af2 import campaign_length_bucket, padded_prediction_length   # noqa: E402

#: The override list hostshare.py, grade_host.py and round_ab.py all share. The length is sampled
#: from the campaign seed, so a harness that drifts from this list can silently measure another
#: complex, and then 22.202 s of host is a cost for a round nobody else ran.
SHARED_OVERRIDES = ["max_trajectories=1", "validation_model=monomer",
                    'design_models=["model_1_ptm"]', 'validation_models=["model_2_ptm"]']


def _settings(seed):
    return B.campaign_settings(
        overrides=[f"campaign_seed={seed}", *SHARED_OVERRIDES,
                   "project_folder=/tmp/bcx_extrawire_lengths"])


def _triple(chains, settings, tile):
    bucket = campaign_length_bucket(settings)
    # BindCraft 2 pads the design chain only; the target keeps its own length.
    handed_over = sum(padded_prediction_length(n, bucket) if name == "binder" else n
                      for name, n in chains.items())
    return {"chains": chains, "residues_that_exist": sum(chains.values()), "bucket": bucket,
            "handed_to_the_model": handed_over,
            "on_tile": -(-handed_over // tile) * tile, "tile": tile}


def _chains(state):
    (complex_,) = state.values()
    return {name: len(protein.residue_index) for name, protein in complex_.items()}


def campaign_chain(seed, trajectory=1, tile=32):
    """The complex `campaign.py` designs as trajectory `trajectory` -- what a user's run folds.

    The key derivation is BindCraft 2's, not a reimplementation of it: `campaign.py:162` folds
    the trajectory number into `PRNGKey(campaign_seed)`, and `run_trajectory` splits that once
    more before handing it to `initialize_design_trajectory` (`trajectory.py:280-282`).
    """
    settings = _settings(seed)
    design_settings = build_design_settings(settings)
    key = jax.random.fold_in(jax.random.PRNGKey(settings.get("campaign_seed") or 0), trajectory)
    state, _multi, _losses = initialize_design_trajectory(
        design_settings, jax.random.split(key)[0])
    return {"seed": seed, "trajectory": trajectory,
            "source": "bindcraft/campaign.py:162 -> trajectory.py:280-282",
            **_triple(_chains(state), settings, tile)}


def reference_chain(seed, tile=32):
    """The complex `bc2_state.design_state()` builds -- `campaign.py:239`, not a trajectory."""
    settings = _settings(seed)
    _ds, states, _losses = B.design_state(settings)
    return {"seed": seed, "trajectory": None,
            "source": "bcx_predictor/bc2_state.py:design_state -> campaign.py:239",
            **_triple(_chains(states), settings, tile)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--trajectory", type=int, default=1,
                    help="which trajectory campaign.py is on; round_ab.py runs trajectory 1")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    campaign = campaign_chain(args.seed, args.trajectory)
    reference = reference_chain(args.seed)
    out = {"campaign": campaign, "reference": reference,
           "same_complex": campaign["chains"] == reference["chains"]}

    # The capture the gradient grade was taken on says which of the two it belongs to, from a
    # number no settings were involved in producing.
    grade = ROOT / "perf/bcx_extrawire/runs/grade_hostarms/grade_host.json"
    if grade.is_file():
        g = json.loads(grade.read_text())
        if g.get("seed") == args.seed and "mask_2d_zero_fraction" in g:
            r = region_from_mask_fraction(g["mask_2d_zero_fraction"], g["n"])
            out["cross_check"] = {
                "source": str(grade.relative_to(ROOT)), "capture_width": g["n"],
                "region_from_mask_fraction": round(r, 2),
                "matches": next((k for k, v in (("campaign", campaign), ("reference", reference))
                                 if abs(r - v["residues_that_exist"]) < 0.5), None)}
    print(json.dumps(out, indent=1))
    if args.out:
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.out).write_text(json.dumps(out, indent=1))


def region_from_mask_fraction(zero_fraction, width):
    """The real region implied by a pair mask's zero fraction, as an independent cross-check.

    `mask_2d` is the outer product of the token mask, so a width-w array with a real region of r
    is zero on w**2 - r**2 of its entries. Inverting that recovers r from a number the gradient
    capture already recorded, with no settings involved at all.
    """
    return ((1.0 - zero_fraction) ** 0.5) * width


if __name__ == "__main__":
    main()
