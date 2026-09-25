import os, sys, pathlib, json
HERE = pathlib.Path("/home/ttuser/.coworker/wt/land-standing/perf/bcx_predictor")
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1]))
import bc2_state as B
from bindcraft.settings import parse_setting_overrides, read_settings
from bindcraft.preflight import cleaned_campaign_settings
base = os.path.join(B.BC2, "examples", "pdl1.json")

def build(ov):
    return cleaned_campaign_settings(read_settings(base, parse_setting_overrides(ov)))

# what main built BEFORE this change
old = ["campaign_seed=0", "max_trajectories=1", "validation_model=monomer",
       'design_models=["model_1_ptm"]', 'validation_models=["model_2_ptm"]',
       "project_folder=/tmp/_sc", "length_bucket_size=1"]
# what it builds now with the flag OFF
new = ["campaign_seed=0", "max_trajectories=1", "project_folder=/tmp/_sc"]
new[2:2] = ["validation_model=monomer", 'design_models=["model_1_ptm"]',
            'validation_models=["model_2_ptm"]']
new.append("length_bucket_size=1")
assert old == new, (old, new)
print("OVERRIDE_LISTS_IDENTICAL")
a, b = build(old), build(new)
assert json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)
print("SETTINGS_IDENTICAL")

shipped = ["campaign_seed=0", "max_trajectories=1", "project_folder=/tmp/_sc",
           "length_bucket_size=1"]
c = build(shipped)
print("default design_models:", a.get("design_models"), "validation_model:", a.get("validation_model"))
print("shipped design_models:", c.get("design_models"), "validation_model:", c.get("validation_model"))
assert a.get("validation_model") != c.get("validation_model") or a.get("design_models") != c.get("design_models")
print("SHIPPED_DIFFERS_OK")
