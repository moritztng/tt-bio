"""What the allocator sees for each drawn binder length, from BindCraft 2's own padding code.

Three paddings sit between the draw and a ttnn buffer, and only the last one is ours:

  1. `pad_design_chains` rounds the DESIGN chain up to `length_bucket_size` (32);
  2. `_predict_complex` rounds the whole complex, binder plus target, up to the same bucket;
  3. `splice._pad_inputs` rounds the token axis up to a multiple of 32 for tt-bio.

With bucket 32 the third is a no-op after the second. Run on CPU, no device.
"""
import argparse, json, pathlib, sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(HERE), str(ROOT / "perf" / "bcx_predictor")):
    sys.path.insert(0, p)

import bc2_state as B                                                   # noqa: E402
from bindcraft.af2 import (campaign_length_bucket, pad_design_chains,   # noqa: E402
                           padded_prediction_length)
from bindcraft.settings import build_design_settings                    # noqa: E402
from bindcraft.protein_preparation import initialize_design_trajectory, sampled_trajectory_values  # noqa: E402
import jax                                                              # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--lengths", default="")
ap.add_argument("--out", default=str(HERE / "padmap.json"))
args = ap.parse_args()
lengths = ([int(x) for x in args.lengths.split(",")] if args.lengths
           else list(range(60, 181)))

rows = []
for L in lengths:
    ov = [f"binder_lengths=[{L}]", "campaign_seed=200", "max_trajectories=1",
          "validation_model=monomer", 'design_models=["model_1_ptm"]',
          'validation_models=["model_2_ptm"]', "autotune=false"]
    settings = B.campaign_settings(overrides=ov)
    ds = build_design_settings(settings)
    bucket = campaign_length_bucket(settings)
    key = jax.random.fold_in(jax.random.PRNGKey(200), 1)
    drawn, targets = sampled_trajectory_values(ds, key)
    binder_key, _ = jax.random.split(key)
    states, _mcb, _losses = initialize_design_trajectory(ds, binder_key, targets)
    padded_states = pad_design_chains(states, bucket, 0)
    per_state = {}
    for name, complex_ in padded_states.items():
        chain_lengths = {c: len(p) for c, p in complex_.items()}
        residue_count = sum(chain_lengths.values())
        per_state[name] = {"chains": chain_lengths, "residue_count": residue_count,
                           "padded_residue_count": padded_prediction_length(residue_count, bucket)}
    biggest = max(s["padded_residue_count"] for s in per_state.values())
    row = {"binder_length": int(drawn["binder_length"]), "requested": L, "bucket": bucket,
           "binder_padded": padded_prediction_length(L, bucket),
           "states": per_state, "padded_tokens_max": biggest}
    rows.append(row)
    print(json.dumps({k: row[k] for k in ("requested", "binder_padded", "padded_tokens_max")}),
          flush=True)

pathlib.Path(args.out).write_text(json.dumps(rows, indent=1))
buckets = {}
for r in rows:
    buckets.setdefault(r["padded_tokens_max"], []).append(r["requested"])
print(json.dumps({str(k): f"{min(v)}..{max(v)} ({len(v)} draws)"
                  for k, v in sorted(buckets.items())}, indent=1))
