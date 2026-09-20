"""Whole-model key audit: what does each upstream revision do with the p2 checkpoint?

The four-line check that D23 institutionalises, applied at full model scope rather than to the
one block that reading diffs happened to surface. Reports missing and unexpected key sets.
"""
import sys, json, torch
TREE, TAG, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, TREE)
import openfold3
assert openfold3.__file__.startswith(TREE), openfold3.__file__

from openfold3.projects.of3_all_atom.config.model_config import model_config
from openfold3.projects.of3_all_atom.model import OpenFold3

torch.manual_seed(0)
m = OpenFold3(model_config)
own = set(m.state_dict().keys())

sd = torch.load("/home/moritz/.boltz/of3-p2-155k.pt", map_location="cpu", weights_only=False)
sd = sd.get("state_dict", sd)
ck = set(sd.keys())

missing = sorted(own - ck)       # model wants it, checkpoint does not have it
unexpected = sorted(ck - own)    # checkpoint has it, model has nowhere to put it
json.dump({"tag": TAG, "model_params": len(own), "ckpt_tensors": len(ck),
           "missing": missing, "unexpected": unexpected},
          open(OUT, "w"), indent=1)
print(f"[{TAG}] model={len(own)} ckpt={len(ck)}  missing={len(missing)}  unexpected={len(unexpected)}")
