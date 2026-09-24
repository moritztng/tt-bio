#!/usr/bin/env python3
"""Is the device arm on the SAME DRAW as the reference arm?

The two arms differ in one setting by design: the device runs BindCraft 2 default
length_bucket_size 32, the reference runs 1. `length_bucket_size` is not in
design_identity.EXCLUDED_SETTING_NAMES, so the trajectory NAME hash moves even when the
draw does not. This diffs `design_hash`s own basis dict between the two buckets, which is
the thing the name is computed from, and reports every key that differs.
"""
import json
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(ROOT / "perf" / "bcx_predictor"), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import bc2_state as B
import jax
from bindcraft.design_identity import design_hash
from bindcraft.design_identity import design_name
from bindcraft.protein_preparation import sampled_trajectory_values
from bindcraft.settings import build_design_settings
from bindcraft.campaign import trajectory_design_label


def basis_for(seed, bucket, trajectory_number):
    ov = [f"campaign_seed={seed}", "max_trajectories=1", "validation_model=monomer",
          "design_models=[\"model_1_ptm\"]", "validation_models=[\"model_2_ptm\"]",
          f"length_bucket_size={bucket}"]
    settings = B.campaign_settings(overrides=ov)
    ds = build_design_settings(settings)
    key = jax.random.PRNGKey(ds.seed)
    drawn, targets = sampled_trajectory_values(ds, jax.random.fold_in(key, trajectory_number))
    ident, basis = design_hash(settings, {**drawn, "design_models": ["model_1_ptm"]}, targets)
    label = trajectory_design_label(settings, tuple(s.objective for s in ds.prepared_states))
    return ident, basis, drawn, design_name(label, drawn["binder_length"], ident, True, trajectory_number)


out = {}
for seed in (0, 100, 200, 300, 400):
    for tn in (1,):
        i1, b1, d1, n1 = basis_for(seed, 1, tn)
        i32, b32, d32, n32 = basis_for(seed, 32, tn)
        diff = sorted(k for k in set(b1) | set(b32) if b1.get(k) != b32.get(k))
        out[f"seed{seed}_traj{tn}"] = {
            "name_bucket1": n1, "name_bucket32": n32,
            "basis_keys_differing": diff,
            "basis_diff": {k: {"b1": b1.get(k), "b32": b32.get(k)} for k in diff},
            "drawn_identical": d1 == d32,
            "drawn": {k: (v if isinstance(v, (int, float, str)) else str(v)) for k, v in d1.items()},
        }
print(json.dumps(out, indent=1, default=str))
