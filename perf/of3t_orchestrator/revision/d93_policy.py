"""Quantify D93 on CPU by implementing the dtype policies EXPLICITLY.

D93 said the trunk's second revision difference could not be sized without CUDA, because both
halves of it act through `torch.amp.autocast("cuda", ...)`, which CPU torch disables outright.
That is true of the flag as written; it is NOT true of the arithmetic the flag selects. The flag
chooses a dtype policy, and a policy can be written out by hand:

  A  0.4.3's trunk as 0.4.3 ships it -- use_high_precision is never passed, so scores, softmax
     and the score-value matmul all run at the input dtype.
  B  high precision with 0.4.3's PLACEMENT -- scores and softmax in fp32, then `.to(value.dtype)`
     before the score-value matmul, which is exactly 0.4.3's `_attention`.
  C  high precision with 0.5.0's PLACEMENT -- scores, softmax AND the score-value matmul in fp32,
     downcast once on the way out. This is what the reference bundle runs.

A vs C is the whole of D93. B vs C isolates the moved einsum from the flag.

usage: d93_policy.py <TREE> <A|B|C> <OUT.pt>
"""
import sys, json, torch

TREE, POLICY, OUT = sys.argv[1:4]
BOUND = "/tmp/of3t/revarm/trunk_entry_043.pt"
CKPT = "/home/moritz/.boltz/of3-p2-155k.pt"

sys.path.insert(0, TREE)
import openfold3
assert openfold3.__file__.startswith(TREE), openfold3.__file__
import openfold3.core.model.primitives.attention as A
from openfold3.core.model.primitives.attention import softmax_no_cast


def _attention_explicit(query, key, value, biases, use_high_precision=False):
    """`_attention` with the dtype policy written out instead of delegated to a CUDA autocast."""
    in_dtype = query.dtype
    hp = POLICY in ("B", "C")
    wd = torch.float32 if hp else in_dtype

    q, k = query.to(wd), key.to(wd)
    scores = torch.einsum("...qc, ...kc->...qk", q, k)
    for b in biases:
        scores = scores + b.to(wd)
    scores = softmax_no_cast(scores, dim=-1)

    if POLICY == "C":                       # 0.5.0: stay in fp32 through the value matmul
        out = torch.einsum("...qk, ...kc->...qc", scores, value.to(wd))
    else:                                   # 0.4.3: downcast the scores first
        out = torch.einsum("...qk, ...kc->...qc", scores.to(value.dtype), value)
    return out.to(in_dtype)


A._attention = _attention_explicit

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
stack.load_state_dict({k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}, strict=True)

b = torch.load(BOUND, map_location="cpu", weights_only=False)
dt = torch.bfloat16
stack = stack.to(dt)
with torch.no_grad():
    s_out, z_out = stack(s=b["s"].to(dt), z=b["z"].to(dt),
                         single_mask=b["single_mask"].to(dt),
                         pair_mask=b["pair_mask"].to(dt),
                         chunk_size=None, inplace_safe=False, _mask_trans=True)

torch.save({"s": s_out.float(), "z": z_out.float()}, OUT)
print(json.dumps({"policy": POLICY, "s_norm": float(s_out.double().norm()),
                  "z_norm": float(z_out.double().norm())}))
