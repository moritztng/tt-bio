import json, os, pathlib, sys
HERE = pathlib.Path("perf/bcx_predictor").resolve()
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(pathlib.Path.cwd()))
import bc2_state as B
import bindcraft.campaign as campaign
from bindcraft.af2 import MONOMER_POOL, MULTIMER_POOL
from bindcraft.campaign import select_design_and_validation_models
from bindcraft.settings import parse_setting_overrides, read_settings
from bindcraft.preflight import cleaned_campaign_settings
PDL1 = os.path.join(B.BC2, "examples", "pdl1.json")

def build(mp):
    ov = ["campaign_seed=0", "max_trajectories=1", "project_folder=/tmp/poolcheck_project"]
    if not mp:
        ov += ["validation_model=monomer", 'design_models=["model_1_ptm"]',
               'validation_models=["model_2_ptm"]']
    ov.append("length_bucket_size=1")
    return cleaned_campaign_settings(read_settings(PDL1, parse_setting_overrides(ov)))

def legacy():
    ov = ["campaign_seed=0", "max_trajectories=1", "validation_model=monomer",
          'design_models=["model_1_ptm"]', 'validation_models=["model_2_ptm"]',
          "project_folder=/tmp/poolcheck_project", "length_bucket_size=1"]
    return cleaned_campaign_settings(read_settings(PDL1, parse_setting_overrides(ov)))

d = lambda s: json.dumps(s, sort_keys=True, default=str)
off, on, was = build(False), build(True), legacy()
print("1. default settings vs the pre-flag override list:",
      "IDENTICAL" if d(off) == d(was) else "DIFFERENT")
MONOMER = ("model_1_ptm", "model_2_ptm")
for label, s, pool in (("off", off, MONOMER), ("on ", on, MULTIMER_POOL)):
    # run_arm rebinds campaign.MULTIMER_POOL to MONOMER on the default arm; mirror that here.
    sel = select_design_and_validation_models(s, pool, MONOMER_POOL)
    print(f"2. --multimer-pool {label}: {len(sel.design_models)} design "
          f"{list(sel.design_models)}")
    print(f"{'':23s} {len(sel.validation_models)} validation {list(sel.validation_models)}")
print("3. MULTIMER_POOL =", list(MULTIMER_POOL))
