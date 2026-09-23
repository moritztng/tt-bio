"""Capture the REAL pairformer-stack boundary from upstream 0.4.3's own forward.

Stage 1 of the revision arm pre-registered in REVISION_ARM_PREREGISTERED.json.
Builds the 0.4.3 model with the bundle's own p2 checkpoint, aborts the forward the
instant the trunk is entered, and saves exactly the tensors the stack was called with.
"""
import sys, json, hashlib, torch

TREE = "/home/moritz/.coworker/scratch/of3t-reference/upstream"
CKPT = "/home/moritz/.boltz/of3-p2-155k.pt"
BATCH = "/home/moritz/.coworker/scratch/of3t-reference/bundle/batches/batch_step003.pt"
OUT = "/tmp/of3t/revarm/trunk_entry_043.pt"

sys.path.insert(0, TREE)
import openfold3
assert openfold3.__file__.startswith(TREE), openfold3.__file__
from openfold3.projects.of3_all_atom.config.model_config import model_config
from openfold3.projects.of3_all_atom.model import OpenFold3

torch.manual_seed(0)
torch.set_num_threads(4)              # pinned: a digest is a function of the thread count
model = OpenFold3(model_config).eval()

sd = torch.load(CKPT, map_location="cpu", weights_only=False)
sd = sd.get("state_dict", sd)
r = model.load_state_dict(sd, strict=False)
print(f"load: missing={list(r.missing_keys)} unexpected={len(r.unexpected_keys)}")

batch = torch.load(BATCH, map_location="cpu", weights_only=False)

class Captured(Exception):
    pass

grab = {}

def pre_hook(mod, args, kwargs):
    for k in ("s", "z", "single_mask", "pair_mask"):
        grab[k] = kwargs[k].detach().clone()
    raise Captured

model.pairformer_stack.register_forward_pre_hook(pre_hook, with_kwargs=True)

try:
    with torch.no_grad():
        model(batch)
except Captured:
    pass

assert grab, "the trunk was never entered -- the hook did not fire"
meta = {k: [list(v.shape), str(v.dtype), float(v.double().norm())] for k, v in grab.items()}
torch.save(grab, OUT)
h = hashlib.sha256(open(OUT, "rb").read()).hexdigest()
print(json.dumps({"out": OUT, "sha256": h, "tensors": meta}, indent=1))
