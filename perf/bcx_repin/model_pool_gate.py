#!/usr/bin/env python3
"""Which AlphaFold checkpoints the shipped VHH example asks for, and which ours runs.

`aa_bias_delta.py` asked what PR #17 changes. This asks the question that decides whether a
monomer-pinned acceptance count can be read against the lab's ~0.33 per trajectory at all:
for `examples/pdl1_vhh.json`, what does BindCraft 2's own selector return under its shipped
settings, and what does it return under the pin `run_arm.py` applies so a monomer trunk can
run it?

Three configurations, because the pin turns out to have two separable halves and only the
pair of them works:

  shipped              no overrides, BindCraft 2's real pools.
  monomer_pin          what the live device arm runs: the three setting overrides AND
                       `run_arm.py:87`'s `campaign.MULTIMER_POOL = MONOMER` rebind.
                       `campaign.py:249` passes that module global positionally, so
                       passing the rebound tuple here is the same call the campaign makes.
  overrides_only       the three overrides against the real pools, i.e. the pin with the
                       rebind left out. This one RAISES, which is the point: BindCraft 2
                       enforces pool membership at `settings.py:541`, so naming a monomer
                       checkpoint in `design_models` is refused and the rebind is load
                       bearing rather than belt-and-braces.

The answers come from BindCraft 2's own loader in a subprocess with that tree first on
sys.path, not from the JSON: the example ships `design_models: null` and the pool is chosen
by whether the binder is scaffolded.
"""
import json
import os
import pathlib
import subprocess

HERE = pathlib.Path(__file__).resolve().parent
VENV = "/home/ttuser/bcx_e2e_venv/bin/python"
AF2 = "/home/ttuser/bcx_e2e/af2_params"

#: run_arm.py's monomer pin, verbatim from its `overrides` list and its module constant.
PIN = ['validation_model=monomer', 'design_models=["model_1_ptm"]',
       'validation_models=["model_2_ptm"]']
MONOMER = ['model_1_ptm', 'model_2_ptm']

PROBE = r'''
import json, os, sys
BC2 = os.environ["BCX_BC2"]
sys.path.insert(0, BC2)
from bindcraft.af2 import MONOMER_POOL, MULTIMER_POOL
from bindcraft.preflight import CampaignPreflightError, cleaned_campaign_settings, preflight_campaign
from bindcraft.settings import (parse_setting_overrides, read_settings,
                                select_design_and_validation_models, validates_on_multimer)

AF2 = os.environ["BCX_AF2"]
MPNN = os.path.join(BC2, "bindcraft", "weights", "proteinmpnn", "weights_neutral")
example = os.path.join(BC2, "examples", os.environ["BCX_EXAMPLE"])

def preflight(settings):
    # preflight_campaign raises on a problem and returns warnings otherwise, so both
    # outcomes have to be read for the configuration-level problems to be visible.
    try:
        return {"raised": False, "messages": list(preflight_campaign(settings, "/tmp/bcx_repin_preflight", AF2, MPNN))}
    except CampaignPreflightError as exc:
        return {"raised": True, "messages": str(exc).splitlines()}

out = {"multimer_pool": list(MULTIMER_POOL), "monomer_pool": list(MONOMER_POOL), "configs": {}}
for label, spec in json.loads(os.environ["BCX_CONFIGS"]).items():
    settings = cleaned_campaign_settings(read_settings(example, parse_setting_overrides(spec["overrides"])))
    design_pool = tuple(spec["multimer_pool"]) if spec["multimer_pool"] else MULTIMER_POOL
    entry = {
        "overrides": spec["overrides"],
        "campaign_multimer_pool": list(design_pool),
        "declared_design_models": settings.get("design_models"),
        "declared_validation_model": settings.get("validation_model"),
        "validates_on_multimer": bool(validates_on_multimer(settings)),
    }
    try:
        selected = select_design_and_validation_models(settings, design_pool, MONOMER_POOL)
        entry.update(selected_design_models=list(selected.design_models),
                     selected_validation_models=list(selected.validation_models),
                     validation_pool_exhausted=bool(selected.validation_pool_exhausted),
                     selection_error=None,
                     preflight=preflight(settings))
    except Exception as exc:
        entry.update(selected_design_models=None, selected_validation_models=None,
                     validation_pool_exhausted=None,
                     selection_error=f"{type(exc).__name__}: {exc}", preflight=None)
    out["configs"][label] = entry
print(json.dumps(out))
'''


def main():
    tree = os.environ.get("BCX_BC2", "/home/ttuser/bcx_repin/bc2")
    configs = {
        "shipped": {"overrides": [], "multimer_pool": None},
        "monomer_pin": {"overrides": PIN, "multimer_pool": MONOMER},
        "overrides_only": {"overrides": PIN, "multimer_pool": None},
    }
    env = dict(os.environ, BCX_BC2=tree, BCX_EXAMPLE="pdl1_vhh.json", BCX_AF2=AF2,
               BCX_CONFIGS=json.dumps(configs))
    done = subprocess.run([VENV, "-c", PROBE], env=env, capture_output=True, text=True)
    if done.returncode:
        raise SystemExit(f"probe failed on {tree}:\n{done.stderr}")
    # BindCraft 2 prints a "campaign preflight:" banner to stdout before we print,
    # so the JSON is the last line rather than the whole stream.
    got = json.loads(done.stdout.strip().splitlines()[-1])
    got["tree"] = tree
    got["tree_head"] = subprocess.run(["git", "-C", tree, "rev-parse", "--short", "HEAD"],
                                      capture_output=True, text=True).stdout.strip()
    (HERE / "model_pool_gate.json").write_text(json.dumps(got, indent=1) + "\n")
    print(json.dumps(got, indent=1))


if __name__ == "__main__":
    main()
