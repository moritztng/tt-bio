import json, os, sys
bc2 = os.environ["BCX_BC2"]
sys.path.insert(0, bc2)
os.chdir(bc2)
from bindcraft.settings import read_settings, parse_setting_overrides, select_design_and_validation_models, validates_on_multimer
from bindcraft.af2 import MONOMER_POOL, MULTIMER_POOL
out = {"bc2": bc2, "MULTIMER_POOL": list(MULTIMER_POOL), "MONOMER_POOL": list(MONOMER_POOL)}
for ex in ("examples/pdl1.json", "examples/pdl1_vhh.json"):
    s = read_settings(os.path.join(bc2, ex), {})
    sel = select_design_and_validation_models(s, MULTIMER_POOL, MONOMER_POOL)
    out[ex] = {"design_models": list(sel.design_models),
               "validation_models": list(sel.validation_models),
               "validates_on_multimer": validates_on_multimer(s),
               "design_models_key_in_settings": s.get("design_models", "<absent>"),
               "binder_scaffold": bool(s.get("binder_scaffold"))}
print(json.dumps(out, indent=1))
