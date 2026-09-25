#!/usr/bin/env python3
"""Both sides of the route, live, in one device process on the shipped multimer pool.

The end-to-end proof this belongs to is a whole trajectory, which is hours. This is the
device half of it in minutes: the five `model_*_multimer_v3` trunks resident on card, the
splice installed exactly as `run_arm.py` installs it, and then the two folds that used to
be mutually exclusive --

  * a design model, which the card holds, folds ON CARD;
  * `model_1_ptm` and `model_2_ptm`, the monomer validation pool `campaign.py:77-83` moves
    validation to on the shipped example, fold on BindCraft 2's own JAX trunk instead of
    raising `KeyError` out of `multimer_pool.use`.

Both in the same process, in that order, so the card fold is not merely first but survives
the route being taken after it: the jit cache is keyed on model FAMILY
(`bindcraft/af2.py:273`) and this is what shows the two entries do not collide in a live
process rather than in an argument about the cache key.

AICLK is sampled DURING the card fold, on a thread.

Run (card 1):
  TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=1 TT_BIO_LEASE_HOLDER=worker:bcx-mutate \
    python3 perf/bcx_mutate/validation_route_device.py
"""
import argparse
import json
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parents[1] / "bcx_predictor"
ROOT = HERE.parents[1]
for _p in (str(HERE), str(ROOT), str(ROOT / "perf"), str(ROOT / "perf" / "bcx_afgrad"),
           str(ROOT / "perf" / "bcx_stack")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

import bc2_state as B                                       # noqa: E402
import clocksample                                          # noqa: E402
from multimer_pool import POOL, MultimerPool                # noqa: E402
from splice import EvoformerOnDevice, evoformer_on_device    # noqa: E402
from ttbio_predictor import TTBioAlphaFoldDesignModel        # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--params", default="/home/ttuser/bcx_e2e/af2_params")
ap.add_argument("--binder", type=int, default=73, help="the length bcx-accept's arm died on")
ap.add_argument("--bucket", type=int, default=32)
ap.add_argument("--design-model", default="model_2_multimer_v3")
ap.add_argument("--recycle", type=int, default=0)
ap.add_argument("--json", default="/home/ttuser/bcx_mutate_art/validation_route_device.json")
args = ap.parse_args()

settings = B.campaign_settings(overrides=[f"length_bucket_size={args.bucket}",
                                          "campaign_seed=0",
                                          f"binder_lengths=[{args.binder}]"])
_ds, states, _losses = B.design_state(settings)
n_res = sum(sum(c.values()) for c in B.state_shape(states).values())
n32 = (n_res + args.bucket - 1) // args.bucket * args.bucket
print(f"PD-L1 design state: binder {args.binder}, {n_res} residues, padded {n32}", flush=True)

pool = MultimerPool(args.params, resident=1)
pool.use(pool.models[0])
evo = EvoformerOnDevice(pool, k_evo=48)
print(f"card holds {evo.checkpoints}", flush=True)

out = {"n_residues": n_res, "padded": n32, "binder": args.binder,
       "card_checkpoints": list(evo.checkpoints), "recycles": args.recycle, "folds": []}


def fold(model, where):
    m = TTBioAlphaFoldDesignModel(presets=(model,), data_dir=args.params, models=(model,),
                                  num_recycle=args.recycle, length_bucket_size=args.bucket,
                                  max_cache_size=2, trunk="device", pool=pool)
    m.dropout = False
    before = dict(evo.calls)
    t0 = time.time()
    with clocksample.during() as clk:
        pred = m.predict(states, model=model)
    secs = time.time() - t0
    target = next(iter(pred))
    met = pred[target].metrics
    plddt = np.asarray(met["plddt"])
    row = {"model": model, "expected": where, "seconds": round(secs, 1),
           "device_primal_calls": evo.calls["primal"] - before["primal"],
           "ptm": round(float(met["ptm"]), 6),
           "plddt_mean": round(float(plddt.mean()), 6),
           "clock": clk.summary().get(0), "clock_line": clk.line()}
    out["folds"].append(row)
    print(json.dumps(row), flush=True)
    return row


with evoformer_on_device(evo) as swapped:
    on_card = fold(args.design_model, "card")
    routed = [fold(name, "host") for name in ("model_1_ptm", "model_2_ptm")]

out["evoformer_blocks_spliced"] = list(swapped)
out["device_calls"] = dict(evo.calls)
out["host_trunk_folds"] = dict(evo.host_folds)
out["pool"] = pool.stamp()
pathlib.Path(args.json).parent.mkdir(parents=True, exist_ok=True)
pathlib.Path(args.json).write_text(json.dumps(out, indent=1))
print("wrote", args.json, flush=True)

failures = []
if on_card["device_primal_calls"] != 1:
    failures.append(f"the design model did not fold on card: {on_card['device_primal_calls']} "
                    f"device calls")
if not swapped:
    failures.append("the Evoformer stack was never spliced, so nothing ran on card")
for row in routed:
    if row["device_primal_calls"] != 0:
        failures.append(f"{row['model']} touched the card: {row['device_primal_calls']} calls")
    if not np.isfinite(row["plddt_mean"]) or not 0.0 < row["plddt_mean"] <= 1.0:
        failures.append(f"{row['model']} returned pLDDT {row['plddt_mean']}")
if set(evo.host_folds) != {"model_1_ptm", "model_2_ptm"}:
    failures.append(f"host_folds did not record both validation models: {evo.host_folds}")
if not on_card["clock"]:
    failures.append("AICLK was not sampled during the card fold, so its time is unclocked")

print()
print(f"on card : {on_card['model']} {on_card['seconds']}s  {on_card['clock_line']}")
for row in routed:
    print(f"on host : {row['model']} {row['seconds']}s  pTM {row['ptm']:.4f} "
          f"pLDDT {row['plddt_mean']:.4f}")
if failures:
    print("FAIL: " + "; ".join(failures))
    sys.exit(1)
print("PASS: the card ran the design model and the validation pool ran on the host trunk, "
      "one process, no refusal")
