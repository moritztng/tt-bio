#!/usr/bin/env python3
"""Does main's device arm actually support BindCraft 2's shipped five-model pool?

`run_arm.py --arm device --shipped` used to be refused by an argparse guard. Dropping the guard
is only right if the arm behind it constructs, so this builds exactly what that invocation
builds -- `campaign_predictor(trunk="device", validation="device", resident=N)` against the
five `model_N_multimer_v3` checkpoints -- and reports what landed on card and what did not.
It runs no campaign and no gradient step: the question is the pool, not the loop.
"""
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
# run_arm.py's own preamble, in its own order: bc2_state is what puts BindCraft 2 on the path,
# and `evoformer_on_device` imports `bindcraft.af.alphafold.model` at enter time.
sys.path.insert(0, str(ROOT / "perf" / "bcx_predictor"))
import bc2_state as B                                                  # noqa: E402
sys.path.insert(0, str(ROOT))
if str(B.BC2) not in sys.path:
    sys.path.insert(0, str(B.BC2))
from tt_bio import bindcraft2                                          # noqa: E402

PARAMS = "/home/ttuser/bcx_e2e/af2_params"
SHIPPED = [f"model_{i}_multimer_v3" for i in range(1, 6)]
MONOMER = ["model_1_ptm", "model_2_ptm"]

out = {"params": PARAMS, "asked_shipped": SHIPPED, "asked_monomer": MONOMER}
with bindcraft2.campaign_predictor(trunk="device", validation="device",
                                   checkpoints=PARAMS, resident=1) as build:
    pool = build.pool
    out["resident"] = pool.resident
    pool.require(SHIPPED)
    out["after_shipped"] = {"on_card": sorted(pool.paths), "absent": sorted(pool.absent)}
    # A campaign designs on the five multimer checkpoints and validates on the monomer pool, so
    # the real ask is the union. This is the set `require` is called with in a live run.
    pool.require(MONOMER)
    out["after_both"] = {"on_card": sorted(pool.paths), "absent": sorted(pool.absent)}
    out["evoformer_built"] = build.evoformer is not None
    out["device_card"] = os.environ.get("TT_VISIBLE_DEVICES")

print(json.dumps(out, indent=1))
assert out["resident"] == 1, out["resident"]
assert not out["after_both"]["absent"], f"unsupplied: {out['after_both']['absent']}"
assert set(SHIPPED) <= set(out["after_both"]["on_card"]), out["after_both"]
print("SHIPPED_POOL_DEVICE_OK")
