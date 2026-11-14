"""Upstream Mol-1 on CPU fp32: the trunk single track after every PairformerBlock, up to the confidence head.

    python -I layer_dump.py OUT.pt -- <mol/runner/inference.py arguments>
Saves a list of (N_token, c_s) fp32 tensors in call order, the same order tt-bio's hook (WHL3_LAYERS) writes layers.pt.
"""
import os, sys, torch
out = sys.argv[1]
rest = [a for a in sys.argv[2:] if a != "--"]
torch.set_num_threads(10)
import mol.runner.inference as inf
from mol.model.mol import Mol
from mol.model.modules.pairformer import PairformerBlock
layers = []
fwd = PairformerBlock.forward
def forward(self, s, *a, **kw):
    so, zo = fwd(self, s, *a, **kw)
    if so is not None:
        layers.append(so.detach().float().reshape(-1, so.shape[-1]).clone())
    return so, zo
PairformerBlock.forward = forward
def run_confidence_head(self, *a, **kw):
    torch.save(layers, out)
    print("LAYERS-DUMPED", out, len(layers), flush=True)
    os._exit(0)
Mol.run_confidence_head = run_confidence_head
sys.argv = ["inference.py"] + rest
inf.run()
