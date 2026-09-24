#!/usr/bin/env python3
"""of3t-confpfe bisect test: the probe's resolved logits against upstream's captured ones.

    bisect_quick.py BOUNDARY PROBE_DUMP   -> prints rel; exit 0 if rel < 0.05 (good), 1 (bad)
"""
import sys
import torch
from openfold3.core.utils.atomize_utils import broadcast_token_feat_to_atoms

B = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
kw, pos = B["inputs"]["kwargs"], B["inputs"]["args"]
batch = kw["batch"] if "batch" in kw else pos[0]
tok = batch["token_mask"]
n = int(tok.shape[-1])
m = broadcast_token_feat_to_atoms(token_mask=tok, num_atoms_per_token=batch["num_atoms_per_token"],
                                  token_feat=tok, max_num_atoms_per_token=23).reshape(-1).bool()
up = B["outputs"]["experimentally_resolved_logits"].reshape(-1, 2).double()
dv = torch.load(sys.argv[2], weights_only=False)["outputs"]["experimentally_resolved_logits"]
dv = dv.reshape(n * 23, 2)[m].double()
rel = float((dv - up).norm() / up.norm())
print(f"resolved rel {rel:.5e}")
sys.exit(0 if rel < 0.05 else 1)
