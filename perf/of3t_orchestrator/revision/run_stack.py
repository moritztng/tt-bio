"""Run one revision's PairFormerStack on the captured 0.4.3 trunk boundary.

Stage 2 of the arm pre-registered in REVISION_ARM_PREREGISTERED.json. One tree per
process -- the two release trees cannot share an interpreter. Weights are the bundle's
own p2 checkpoint in both cases; nothing about them is a free choice.

usage: run_stack.py <TREE> <TAG> <OUT.pt> <fp32|bf16> <tb_keep|tb_off>
"""
import sys, json, torch

TREE, TAG, OUT, DTYPE, TB = sys.argv[1:6]
BOUND = "/tmp/of3t/revarm/trunk_entry_043.pt"
CKPT = "/home/moritz/.boltz/of3-p2-155k.pt"

sys.path.insert(0, TREE)
import openfold3
assert openfold3.__file__.startswith(TREE), openfold3.__file__

if TB == "tb_off":
    # Force 0.5.0 back onto 0.4.3's bias orientation. The CONTROL: base_blocks.py
    # differs by exactly one added line, so with this neutralised the two stacks must
    # agree bit-exactly, or another change on the path is live and the arm is void.
    import openfold3.core.model.layers.triangular_attention as _ta
    _orig = _ta.TriangleAttention.forward

    def _patched(self, x, mask=None, transpose_bias=False, **kw):
        return _orig(self, x, mask=mask, transpose_bias=False, **kw)

    _ta.TriangleAttention.forward = _patched

from openfold3.projects.of3_all_atom.config.model_config import model_config
from openfold3.core.model.latent.pairformer import PairFormerStack

torch.manual_seed(0)
torch.set_num_threads(4)

cfg = dict(model_config.architecture.pairformer)
cfg["blocks_per_ckpt"] = None
cfg["tune_chunk_size"] = False
stack = PairFormerStack(**cfg).eval()

sd = torch.load(CKPT, map_location="cpu", weights_only=False)
sd = sd.get("state_dict", sd)
pre = "pairformer_stack."
sub = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
r = stack.load_state_dict(sub, strict=True)
n_loaded = len(sub)

b = torch.load(BOUND, map_location="cpu", weights_only=False)
dt = torch.bfloat16 if DTYPE == "bf16" else torch.float32
s = b["s"].to(dt)
z = b["z"].to(dt)
stack = stack.to(dt)

with torch.no_grad():
    s_out, z_out = stack(
        s=s, z=z,
        single_mask=b["single_mask"].to(dt),
        pair_mask=b["pair_mask"].to(dt),
        chunk_size=None, inplace_safe=False, _mask_trans=True,
    )

torch.save({"s": s_out.float(), "z": z_out.float()}, OUT)
print(json.dumps({"tag": TAG, "dtype": DTYPE, "tb": TB, "n_weights": n_loaded,
                  "s_norm": float(s_out.double().norm()),
                  "z_norm": float(z_out.double().norm())}))
