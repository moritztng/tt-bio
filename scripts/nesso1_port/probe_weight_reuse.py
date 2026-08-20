#!/usr/bin/env python3
"""Does the Nesso-1 checkpoint load into tt-bio's existing classes?

The port's whole premise is that the trunk and both affinity pairformers are the
PairformerNoSeqModule already in tt_bio/reference.py, and that RelativePositionEncoder
and PairwiseConditioning are already in tt_bio/boltz2.py. That is a claim about weight
shapes, so check it against the real safetensors rather than against a similarity score.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from safetensors.torch import load_file

REPO = Path("/home/ttuser/.coworker/wt/nesso1-port-p1-parity")
sys.path.insert(0, str(REPO))

CKPT = Path(sys.argv[1])
HP = json.loads((CKPT.parent / "hparams.json").read_text())
sd = load_file(str(CKPT), device="cpu")
print(f"{len(sd)} tensors, {sum(v.numel() for v in sd.values()) / 1e6:.1f} M params")

top = Counter(k.split(".")[0] for k in sd)
for name, n in sorted(top.items(), key=lambda kv: -kv[1]):
    tot = sum(v.numel() for k, v in sd.items() if k.split(".")[0] == name)
    print(f"  {name:22s} {n:5d} tensors  {tot / 1e6:8.2f} M")


def sub(prefix):
    p = prefix + "."
    return {k[len(p):]: v for k, v in sd.items() if k.startswith(p)}


def check(label, module, weights):
    missing, unexpected = module.load_state_dict(weights, strict=False)
    own = set(module.state_dict())
    extra = set(weights) - own
    print(f"\n{label}")
    print(f"  tt-bio module params : {len(own)}")
    print(f"  checkpoint params    : {len(weights)}")
    print(f"  missing in ckpt      : {len(missing)}  {sorted(missing)[:4]}")
    print(f"  unexpected in ckpt   : {len(extra)}  {sorted(extra)[:4]}")
    ok = not missing and not extra
    print(f"  VERDICT: {'EXACT MATCH' if ok else 'MISMATCH'}")
    return ok


from tt_bio.boltz2 import PairwiseConditioning, RelativePositionEncoder
from tt_bio.reference import PairformerNoSeqModule

token_s, token_z = HP["token_s"], HP["token_z"]
results = {}

results["trunk pairformer (48 blocks)"] = check(
    "trunk pairformer_module  <- tt_bio.reference.PairformerNoSeqModule",
    PairformerNoSeqModule(token_z=token_z, **HP["pairformer_model_args"]),
    sub("pairformer_module"),
)

results["affinity pairformer (8 blocks)"] = check(
    "affinity_module.pairformer_stack  <- tt_bio.reference.PairformerNoSeqModule",
    PairformerNoSeqModule(token_z=token_z, **HP["affinity_model_args"]["pairformer_args"]),
    sub("affinity_module.pairformer_stack"),
)

results["rel_pos"] = check(
    "rel_pos  <- tt_bio.boltz2.RelativePositionEncoder",
    RelativePositionEncoder(token_z),
    sub("rel_pos"),
)

results["affinity pairwise_conditioner"] = check(
    "affinity_module.pairwise_conditioner  <- tt_bio.boltz2.PairwiseConditioning",
    PairwiseConditioning(token_z=token_z, dim_token_rel_pos_feats=token_z, num_transitions=2),
    sub("affinity_module.pairwise_conditioner"),
)

reused = sum(
    v.numel() for k, v in sd.items()
    if k.startswith(("pairformer_module.", "rel_pos.",
                     "affinity_module.pairformer_stack.", "affinity_module2.pairformer_stack.",
                     "affinity_module.pairwise_conditioner.", "affinity_module2.pairwise_conditioner."))
)
total = sum(v.numel() for v in sd.values())
print(f"\nparameters covered by classes tt-bio already ships: "
      f"{reused / 1e6:.1f} M of {total / 1e6:.1f} M  ({100 * reused / total:.1f}%)")
print("all checks exact:", all(results.values()))
