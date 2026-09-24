#!/usr/bin/env python3
"""Which length_bucket_size did each arm actually run at?

The reference streams on qb1 were launched `--bucket 1` and their arm_stamp.json says
`length_bucket_size: 1`. The harness copy they run from (/dev/shm/bcx-ref/harness/run_arm.py)
never appends the override to the settings; it only puts args.bucket in the stamp. So the
stamp records the FLAG, not the setting, and `campaign_length_bucket` (bindcraft/af2.py:50)
falls through `settings.get('length_bucket_size') or DEFAULT_LENGTH_BUCKET` to 32.

The trajectory name carries the proof. `design_hash` (design_identity.py:59) hashes the
settings dict, and `length_bucket_size` is NOT in EXCLUDED_SETTING_NAMES, so a settings dict
with the key absent hashes differently from one carrying 1 and from one carrying 32, while
`drawn` -- the actual draw -- is identical in all three. Recomputing all three candidates and
matching against the name each reference run printed says which one ran.

Run: /home/ttuser/bcx_e2e_venv/bin/python perf/bcx_seeds/ref_bucket_forensics.py
"""
import json
import pathlib
import sys

R = pathlib.Path(__file__).resolve().parents[2]
for p in (str(R / "perf" / "bcx_predictor"), str(R)):
    if p not in sys.path:
        sys.path.insert(0, p)

import bc2_state as B
import jax
from bindcraft.af2 import campaign_length_bucket
from bindcraft.design_identity import design_hash
from bindcraft.protein_preparation import sampled_trajectory_values
from bindcraft.settings import build_design_settings

# The trajectory-1 name each reference stream printed, read off its own run.log.
OBSERVED_REFERENCE = {0: "d2bfded440698f32", 100: "c019673d64134f64", 200: "09b8e7a3d08bc7b6",
                      300: "aaf0e3cad8e56a88", 400: "ffb9c87873af6d36"}
BASE = ["max_trajectories=1", "validation_model=monomer", 'design_models=["model_1_ptm"]',
        'validation_models=["model_2_ptm"]']
CANDIDATES = {"key_absent": [], "key_1": ["length_bucket_size=1"],
              "key_32": ["length_bucket_size=32"]}


def main():
    out = {}
    for seed, observed in OBSERVED_REFERENCE.items():
        row, drawn_seen = {}, []
        for tag, extra in CANDIDATES.items():
            settings = B.campaign_settings(overrides=[f"campaign_seed={seed}"] + BASE + extra)
            ds = build_design_settings(settings)
            drawn, targets = sampled_trajectory_values(
                ds, jax.random.fold_in(jax.random.PRNGKey(ds.seed), 1))
            ident, _ = design_hash(settings, {**drawn, "design_models": ["model_1_ptm"]}, targets)
            row[tag] = {"hash": ident, "effective_bucket": campaign_length_bucket(settings),
                        "key_present": "length_bucket_size" in settings,
                        "matches_reference_run": ident == observed}
            drawn_seen.append(drawn)
        out[str(seed)] = {"observed_reference_hash": observed, "candidates": row,
                          "drawn": drawn_seen[0],
                          "draw_identical_across_candidates": all(d == drawn_seen[0]
                                                                  for d in drawn_seen),
                          "reference_actually_ran": next(
                              (t for t, v in row.items() if v["matches_reference_run"]), None)}
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
