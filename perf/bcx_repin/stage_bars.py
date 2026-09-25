#!/usr/bin/env python3
"""The bar each design stage judges against, for the arm that is actually running.

Trajectories 1-3 of the VHH device arm died `at screen design stage ... due to [pLDDT]`.
The log prints the failing term, not its threshold, so how far the fold missed by is not
readable from the run. `filters.py:822` and `:826` show the stage bars are settings keys,
`min_iptm_<stage>` and `min_plddt_<stage>`, so this asks BindCraft 2's loader for them under
the SAME settings file and the SAME overrides `run_arm.py` builds. That matters: a bar read
out of the example JSON is not necessarily the bar the live arm ran.

Host-only: resolves settings, opens no device.
"""
import json
import os
import sys

BC2 = os.environ.get("BCX_BC2", "/home/ttuser/bcx_repin/bc2")
sys.path.insert(0, BC2)
from bindcraft.preflight import cleaned_campaign_settings  # noqa: E402
from bindcraft.settings import read_settings, parse_setting_overrides  # noqa: E402
import bindcraft.campaign as campaign  # noqa: E402

SETTINGS = os.environ.get("BCX_SETTINGS", os.path.join(BC2, "examples", "pdl1_vhh.json"))
#: The live arm: run_arm.py --arm device --trajectories 10 --seed 0, monomer pin.
OVERRIDES = ["campaign_seed=0", "validation_model=monomer",
             'design_models=["model_1_ptm"]', 'validation_models=["model_2_ptm"]',
             "max_trajectories=10"]
STAGES = ("screen", "refine", "anneal", "harden", "mutate", "final")

settings = cleaned_campaign_settings(read_settings(SETTINGS, parse_setting_overrides(OVERRIDES)))
budget = getattr(campaign, "campaign_trajectory_budget", None)

print(json.dumps({
    "settings_file": SETTINGS,
    "bc2": BC2,
    "max_trajectories_setting": settings.get("max_trajectories"),
    "max_trajectories_resolved": budget(settings) if budget else None,
    "number_of_final_designs": settings.get("number_of_final_designs"),
    "stage_bars": {stage: {"min_plddt": settings.get(f"min_plddt_{stage}"),
                           "min_iptm": settings.get(f"min_iptm_{stage}")}
                   for stage in STAGES},
    "min_monomer_plddt_final": settings.get("min_monomer_plddt_final"),
}, indent=1, default=str))
